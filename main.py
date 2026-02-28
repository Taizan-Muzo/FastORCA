"""
主入口文件
启动高通量量子化学特征提取流水线
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from multiprocessing import Process, Event, Manager
from typing import List

from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from producer.dft_calculator import DFTCalculator
from consumer.feature_extractor import FeatureExtractor
from taskqueue.task_queue import TaskQueue

# Poison pill 标记（表示生产者已完成）
POISON_PILL = {"__poison_pill__": True}


def setup_logging(log_dir: str = "logs"):
    """配置日志系统"""
    Path(log_dir).mkdir(exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # 添加文件输出
    logger.add(
        f"{log_dir}/pipeline_{{time:YYYY-MM-DD}}.log",
        rotation="00:00",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )


def producer_worker(
    smiles_list: List[str],
    queue_config: dict,
    dft_config: dict,
    stop_event: Event,
    shared_queue=None,  # 传入共享队列
    start_idx: int = 0,  # 全局起始索引，用于生成唯一分子ID
):
    """
    GPU 生产者工作进程
    
    Args:
        smiles_list: SMILES 字符串列表
        queue_config: 队列配置
        dft_config: DFT 计算配置
        stop_event: 停止事件
        shared_queue: 共享队列实例
        start_idx: 全局起始索引，确保分子ID唯一
    """
    logger.info(f"Producer worker started (start_idx: {start_idx})")
    
    # 在子进程中初始化 CUDA（必须在导入 gpu4pyscf 之前）
    try:
        import cupy
        cupy.cuda.Device(0).use()
        logger.info("CUDA initialized in producer process")
    except Exception as e:
        logger.warning(f"CUDA initialization failed: {e}")
    
    # 初始化组件（使用共享队列）
    calculator = DFTCalculator(**dft_config)
    if shared_queue is not None:
        queue = TaskQueue(mp_queue=shared_queue)
    else:
        queue = TaskQueue(**queue_config)
    
    success_count = 0
    failed_count = 0
    
    try:
        for i, smiles in enumerate(smiles_list):
            if stop_event.is_set():
                logger.info("Producer received stop signal")
                break
            
            molecule_id = f"mol_{(start_idx + i):06d}"
            
            try:
                logger.info(f"[{molecule_id}] Processing: {smiles[:50]}...")
                
                # 从 SMILES 创建分子
                mol = calculator.from_smiles(smiles)
                
                # 执行 DFT 计算并导出
                result = calculator.calculate_and_export(
                    molecule_id=molecule_id,
                    mol_obj=mol,
                    output_dir="temp/",
                )
                
                if result["success"]:
                    # 将任务放入队列
                    task = {
                        "molecule_id": molecule_id,
                        "pkl_path": result["pkl_file"],
                        "metadata": {
                            "smiles": smiles,
                            "energy": result["energy"],
                        },
                    }
                    queue.put(task)
                    success_count += 1
                    logger.success(f"[{molecule_id}] Queued for feature extraction")
                else:
                    failed_count += 1
                    logger.error(f"[{molecule_id}] DFT failed: {result['error']}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"[{molecule_id}] Processing failed: {e}")
        
        # 注意：poison pill 由主进程统一发送，避免多生产者重复发送
        logger.info(f"Producer finished: {success_count} succeeded, {failed_count} failed")
        
    except Exception as e:
        logger.error(f"Producer error: {e}")
        raise


def consumer_worker(
    queue_config: dict,
    extractor_config: dict,
    output_dir: str,
    stop_event: Event,
    shared_queue=None,  # 传入共享队列
):
    """
    CPU 消费者工作进程
    
    Args:
        queue_config: 队列配置
        extractor_config: 特征提取配置
        output_dir: 输出目录
        stop_event: 停止事件
        shared_queue: 共享队列实例
    """
    logger.info("Consumer worker started")
    
    # 初始化组件（使用共享队列）
    extractor = FeatureExtractor(**extractor_config)
    if shared_queue is not None:
        queue = TaskQueue(mp_queue=shared_queue)
    else:
        queue = TaskQueue(**queue_config)
    
    success_count = 0
    failed_count = 0
    poison_pill_received = False
    
    try:
        while not stop_event.is_set() and not poison_pill_received:
            # 获取任务（阻塞，但会检查 poison pill）
            task = queue.get(block=True, timeout=1.0)
            
            if task is None:
                continue
            
            # 检查 poison pill
            if task.get("__poison_pill__"):
                logger.info("Consumer received poison pill, exiting...")
                poison_pill_received = True
                queue.task_done()
                break
            
            molecule_id = task["molecule_id"]
            pkl_path = task["pkl_path"]
            
            try:
                logger.info(f"[{molecule_id}] Extracting features...")
                
                # 提取特征
                features = extractor.extract_all_features(pkl_path, molecule_id)
                
                if features["success"]:
                    # 保存特征
                    output_path = Path(output_dir) / molecule_id
                    extractor.save_features(features, str(output_path))
                    
                    # 删除临时 pickle 文件
                    try:
                        Path(pkl_path).unlink()
                        logger.debug(f"[{molecule_id}] Removed temp file: {pkl_path}")
                    except Exception as e:
                        logger.warning(f"[{molecule_id}] Failed to remove temp file: {e}")
                    
                    success_count += 1
                    logger.success(f"[{molecule_id}] Features extracted and saved")
                else:
                    failed_count += 1
                    logger.error(f"[{molecule_id}] Feature extraction failed: {features['error']}")
                
                queue.task_done()
                
            except Exception as e:
                failed_count += 1
                logger.error(f"[{molecule_id}] Consumer error: {e}")
                queue.task_done()
    
    finally:
        queue.close()
        logger.info(f"Consumer finished: {success_count} succeeded, {failed_count} failed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="高通量量子化学特征提取流水线"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入 SMILES 文件（每行一个）",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/",
        help="输出目录",
    )
    parser.add_argument(
        "--functional",
        default="B3LYP",
        help="DFT 泛函",
    )
    parser.add_argument(
        "--basis",
        default="def2-SVP",
        help="基组",
    )
    parser.add_argument(
        "--queue-backend",
        default="mp",
        choices=["mp", "redis"],
        help="队列后端（mp=multiprocessing, redis=Redis）",
    )
    parser.add_argument(
        "--n-consumers",
        type=int,
        default=4,
        help="消费者进程数",
    )
    parser.add_argument(
        "--n-producers",
        type=int,
        default=1,
        help="生产者进程数（GPU计算并行度），建议 5-20 以充分利用 A800 显存",
    )
    parser.add_argument(
        "--feature-format",
        default="json",
        choices=["json", "hdf5"],
        help="特征输出格式",
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 读取输入
    with open(args.input, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(smiles_list)} molecules from {args.input}")
    
    # 创建目录
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path("temp").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # 配置
    dft_config = {
        "functional": args.functional,
        "basis": args.basis,
        "verbose": 3,
    }
    
    queue_config = {
        "backend": args.queue_backend,
        "n_consumers": args.n_consumers,  # 用于 poison pill 数量
    }
    
    extractor_config = {
        "output_format": args.feature_format,
    }
    
    # 停止事件
    stop_event = Event()
    
    def signal_handler(signum, frame):
        logger.info("Received interrupt signal, shutting down...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建共享队列（关键修复）
    manager = Manager()
    shared_queue = manager.Queue()
    logger.info("Created shared queue via Manager")
    
    # 启动进程
    processes = []
    
    # 将 SMILES 列表分片给多个生产者
    n_producers = args.n_producers
    chunk_size = (len(smiles_list) + n_producers - 1) // n_producers  # 向上取整
    
    logger.info(f"Distributing {len(smiles_list)} molecules to {n_producers} producers (chunk size: ~{chunk_size})")
    
    for i in range(n_producers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(smiles_list))
        chunk = smiles_list[start_idx:end_idx]
        
        if len(chunk) == 0:
            continue
        
        # 生产者进程（传入分片的分子列表和全局起始索引）
        producer = Process(
            target=producer_worker,
            args=(chunk, queue_config, dft_config, stop_event, shared_queue, start_idx),
            name=f"Producer-{i}"
        )
        producer.start()
        processes.append(producer)
        logger.info(f"Started Producer-{i} with {len(chunk)} molecules [{start_idx}:{end_idx}]")
    
    # 消费者进程（传入共享队列）
    for i in range(args.n_consumers):
        consumer = Process(
            target=consumer_worker,
            args=(queue_config, extractor_config, args.output, stop_event, shared_queue),
            name=f"Consumer-{i}"
        )
        consumer.start()
        processes.append(consumer)
    
    logger.info(f"Pipeline started with {len([p for p in processes if p.name and 'Producer' in p.name])} producers and {args.n_consumers} consumers")
    
    # 等待完成
    try:
        # 先等待所有生产者完成
        producers = [p for p in processes if p.name and 'Producer' in p.name]
        consumers = [p for p in processes if p.name and 'Consumer' in p.name]
        
        for p in producers:
            p.join()
        
        # 所有生产者完成后，发送 poison pill 通知消费者退出
        logger.info("All producers finished, sending poison pills to consumers...")
        for _ in range(args.n_consumers):
            shared_queue.put(POISON_PILL)
        
        # 等待所有消费者完成
        for p in consumers:
            p.join()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_event.set()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    
    logger.info("Pipeline completed")


if __name__ == "__main__":
    # 设置多进程启动方式为 spawn（CUDA 要求）
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了
    main()

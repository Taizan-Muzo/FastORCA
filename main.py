"""
主入口文件
启动高通量量子化学特征提取流水线
"""

import argparse
import sys
import time
import signal
from pathlib import Path
from multiprocessing import Process, Event
from typing import List

from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from producer.dft_calculator import DFTCalculator
from consumer.feature_extractor import FeatureExtractor
from queue.task_queue import TaskQueue


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
):
    """
    GPU 生产者工作进程
    
    Args:
        smiles_list: SMILES 字符串列表
        queue_config: 队列配置
        dft_config: DFT 计算配置
        stop_event: 停止事件
    """
    logger.info("Producer worker started")
    
    # 初始化组件
    calculator = DFTCalculator(**dft_config)
    queue = TaskQueue(**queue_config)
    
    success_count = 0
    failed_count = 0
    
    try:
        for i, smiles in enumerate(smiles_list):
            if stop_event.is_set():
                logger.info("Producer received stop signal")
                break
            
            molecule_id = f"mol_{i:06d}"
            
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
                        "fchk_path": result["fchk_path"],
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
    
    finally:
        queue.close()
        logger.info(f"Producer finished: {success_count} succeeded, {failed_count} failed")


def consumer_worker(
    queue_config: dict,
    extractor_config: dict,
    output_dir: str,
    stop_event: Event,
):
    """
    CPU 消费者工作进程
    
    Args:
        queue_config: 队列配置
        extractor_config: 特征提取配置
        output_dir: 输出目录
        stop_event: 停止事件
    """
    logger.info("Consumer worker started")
    
    # 初始化组件
    extractor = FeatureExtractor(**extractor_config)
    queue = TaskQueue(**queue_config)
    
    success_count = 0
    failed_count = 0
    
    try:
        while not stop_event.is_set():
            # 获取任务（非阻塞，定期检查停止事件）
            task = queue.get(block=True, timeout=1.0)
            
            if task is None:
                continue
            
            molecule_id = task["molecule_id"]
            fchk_path = task["fchk_path"]
            
            try:
                logger.info(f"[{molecule_id}] Extracting features...")
                
                # 提取特征
                features = extractor.extract_all_features(fchk_path, molecule_id)
                
                if features["success"]:
                    # 保存特征
                    output_path = Path(output_dir) / molecule_id
                    extractor.save_features(features, str(output_path))
                    
                    # 删除临时 fchk 文件
                    try:
                        Path(fchk_path).unlink()
                        logger.debug(f"[{molecule_id}] Removed temp file: {fchk_path}")
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
    
    # 启动进程
    processes = []
    
    # 生产者进程
    producer = Process(
        target=producer_worker,
        args=(smiles_list, queue_config, dft_config, stop_event),
    )
    producer.start()
    processes.append(producer)
    
    # 消费者进程
    for i in range(args.n_consumers):
        consumer = Process(
            target=consumer_worker,
            args=(queue_config, extractor_config, args.output, stop_event),
        )
        consumer.start()
        processes.append(consumer)
    
    logger.info(f"Pipeline started with 1 producer and {args.n_consumers} consumers")
    
    # 等待完成
    try:
        for p in processes:
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
    main()

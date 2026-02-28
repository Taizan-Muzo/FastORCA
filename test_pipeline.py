"""
测试脚本
验证流水线功能
"""

import os
import sys
import time
import tempfile
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from producer.dft_calculator import DFTCalculator
from consumer.feature_extractor import FeatureExtractor
from taskqueue.task_queue import TaskQueue


# 测试分子（100 个简单的有机分子）
TEST_SMILES = [
    # 烷烃
    "C", "CC", "CCC", "CCCC", "CCCCC",
    "CCCCCC", "CCCCCCC", "CCCCCCCC", "CCCCCCCCC", "CCCCCCCCCC",
    # 烯烃
    "C=C", "C=CC", "C=CCC", "CC=CC", "C=CCCC",
    "CC=C(C)C", "C1=CCCCC1", "C1=CC=CCC1", "C1=CC=CC=C1", "C1=CC=C(C)C=C1",
    # 醇
    "CO", "CCO", "CCCO", "CC(C)O", "CCCCO",
    "C1CCCCC1O", "OC1=CC=CC=C1", "C1=CC=C(O)C=C1", "CC(O)C", "CCC(C)O",
    # 醛酮
    "C=O", "CC=O", "CCC=O", "CC(C)=O", "CCCC=O",
    "C1CCCCC1=O", "CC(=O)C", "CC(=O)CC", "CCC(=O)CC", "C1=CC=CC=C1C=O",
    # 酸
    "C(=O)O", "CC(=O)O", "CCC(=O)O", "CC(C)C(=O)O", "CCCC(=O)O",
    "C1=CC=CC=C1C(=O)O", "C1=CC=C(C(=O)O)C=C1", "CC=CC(=O)O", "C=C(C)C(=O)O", "CC(C)CC(=O)O",
    # 胺
    "N", "CN", "CCN", "CCCN", "C1CCNCC1",
    "C1=CC=CC=C1N", "CN(C)C", "CCN(CC)CC", "C1CCN(C)CC1", "NC1=CC=CC=C1",
    # 醚
    "COC", "COCC", "CCOCC", "C1COCCO1", "C1CCOCC1",
    "C1=CC=C(OC)C=C1", "COC1=CC=CC=C1", "CC(C)OC(C)C", "CCCCOCCCC", "C1CCCCC1OC1CCCCC1",
    # 卤代烃
    "CCl", "CBr", "CI", "CF", "CCCl",
    "CCBr", "CCI", "CCF", "C1=CC=C(Cl)C=C1", "C1=CC=C(Br)C=C1",
    # 杂环
    "C1=CC=NC=C1", "C1=CN=CC=N1", "C1=COC=C1", "C1=CSC=C1", "C1=CC=NN=C1",
    "C1=CN=C2C=CC=CC2=C1", "C1=CC2=CC=CC=C2C=C1", "C1=CC=C2N=CC=CC2=C1", "C1=CC=C2C=NC=CC2=C1", "C1=CC=C2C(=C1)C=CC1=CC=CC=C12",
    # 其他
    "C#C", "CC#C", "C#CC", "CC#CC", "C1CC1",
    "C1CCC1", "C1CCCC1", "C1CCCCC1", "C12CC1C2", "C1C2CC12",
    "CS", "CSC", "CCSC", "C1=CC=C(S)C=C1", "CS(=O)(=O)O",
    "CCS", "CCCS", "CCCCS", "C1=CC=C2C=CC=CC2=C1S", "C1=CC=CC2=C1SC=C2",
    "CNC", "CN(C)C", "C1CN1", "C1CNC1", "C1CCNC1",
    "C1=CC=C(CN)C=C1", "NCC1=CC=CC=C1", "C1=CC=C(CCN)C=C1", "CNCC1=CC=CC=C1", "C1=CC=C(N(C)C)C=C1",
]


def test_dft_calculator():
    """测试 DFT 计算器"""
    logger.info("=" * 60)
    logger.info("Testing DFT Calculator")
    logger.info("=" * 60)
    
    calculator = DFTCalculator(
        functional="B3LYP",
        basis="def2-SVP",
        verbose=0,  # 减少输出
    )
    
    # 测试 SMILES 处理
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    
    for smiles in test_smiles:
        logger.info(f"Testing SMILES: {smiles}")
        try:
            mol = calculator.from_smiles(smiles)
            logger.success(f"  Created molecule with {mol.natm} atoms")
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    logger.info("DFT Calculator test completed")


def test_queue():
    """测试任务队列"""
    logger.info("=" * 60)
    logger.info("Testing Task Queue")
    logger.info("=" * 60)
    
    # 测试 multiprocessing 后端
    queue = TaskQueue(backend="mp")
    
    # 放入测试任务
    for i in range(5):
        task = {
            "molecule_id": f"test_{i}",
            "fchk_path": f"/tmp/test_{i}.fchk",
            "metadata": {"test": True},
        }
        queue.put(task)
    
    logger.info(f"Queue size: {queue.qsize()}")
    
    # 取出任务
    for i in range(5):
        task = queue.get(timeout=1.0)
        if task:
            logger.info(f"Got task: {task['molecule_id']}")
            queue.task_done()
    
    queue.close()
    logger.info("Task Queue test completed")


def test_pipeline_end_to_end():
    """测试端到端流水线"""
    logger.info("=" * 60)
    logger.info("Testing End-to-End Pipeline")
    logger.info("=" * 60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # 初始化组件
        calculator = DFTCalculator(
            functional="B3LYP",
            basis="def2-SVP",
            verbose=0,
        )
        extractor = FeatureExtractor(output_format="json")
        
        # 选择前 3 个分子进行测试
        test_molecules = TEST_SMILES[:3]
        
        results = []
        
        for i, smiles in enumerate(test_molecules):
            molecule_id = f"test_{i:03d}"
            logger.info(f"Processing {molecule_id}: {smiles}")
            
            try:
                # 1. DFT 计算
                mol = calculator.from_smiles(smiles)
                result = calculator.calculate_and_export(
                    molecule_id=molecule_id,
                    mol_obj=mol,
                    output_dir=str(temp_dir),
                )
                
                if not result["success"]:
                    logger.error(f"  DFT failed: {result['error']}")
                    continue
                
                logger.success(f"  DFT completed: E = {result['energy']:.6f}")
                
                # 2. 特征提取
                features = extractor.extract_all_features(
                    result["fchk_path"],
                    molecule_id,
                )
                
                if not features["success"]:
                    logger.error(f"  Feature extraction failed: {features['error']}")
                    continue
                
                # 3. 保存特征
                output_path = temp_dir / f"{molecule_id}_features"
                saved_path = extractor.save_features(features, str(output_path))
                
                logger.success(f"  Features saved to {saved_path}")
                
                # 4. 清理
                Path(result["fchk_path"]).unlink(missing_ok=True)
                
                results.append({
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "energy": result["energy"],
                    "success": True,
                })
                
            except Exception as e:
                logger.error(f"  Processing failed: {e}")
                results.append({
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "error": str(e),
                    "success": False,
                })
        
        # 输出统计
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"End-to-end test: {success_count}/{len(results)} succeeded")
        
        return success_count == len(results)


def test_throughput():
    """测试吞吐量（模拟 100 个分子）"""
    logger.info("=" * 60)
    logger.info("Testing Throughput (100 molecules)")
    logger.info("=" * 60)
    
    # 这里我们只是模拟，实际运行 100 个 DFT 计算会很耗时
    # 在实际测试中，可以使用更小的分子或更便宜的计算方法
    
    logger.info(f"Test set size: {len(TEST_SMILES)} molecules")
    logger.info("Note: Full throughput test requires GPU and takes significant time")
    logger.info("Run with: python main.py --input test_molecules.smi --output output/")
    
    # 保存测试分子到文件
    output_file = Path("test_molecules.smi")
    with open(output_file, 'w') as f:
        for smiles in TEST_SMILES:
            f.write(smiles + "\n")
    
    logger.info(f"Test molecules saved to {output_file}")
    logger.info("Throughput test setup completed")


def main():
    """运行所有测试"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    logger.info("Starting qcgem_gpu_pipeline tests")
    
    # 运行测试
    try:
        test_dft_calculator()
        test_queue()
        
        if test_pipeline_end_to_end():
            logger.success("All end-to-end tests passed!")
        else:
            logger.warning("Some tests failed")
        
        test_throughput()
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.success("All tests completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
测试脚本
验证 FastORCA 流水线功能
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


# 快速测试分子（少量分子用于快速验证）
TEST_SMILES_QUICK = [
    "CCO",      # 乙醇
    "c1ccccc1", # 苯
    "CC(=O)O",  # 乙酸
]

# 完整测试分子（100 个简单的有机分子）
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
        basis="sto-3g",  # 使用 GPU 友好基组加速测试
        verbose=0,  # 减少输出
        geometry_optimization=False,  # 禁用几何优化加速测试
    )
    
    # 测试 SMILES 处理
    test_smiles = TEST_SMILES_QUICK
    
    for smiles in test_smiles:
        logger.info(f"Testing SMILES: {smiles}")
        try:
            mol = calculator.from_smiles(smiles)
            logger.success(f"  Created molecule with {mol.natm} atoms")
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    logger.info("DFT Calculator test completed")


def test_geometry_optimization():
    """测试几何优化功能 (NEW)"""
    logger.info("=" * 60)
    logger.info("Testing Geometry Optimization")
    logger.info("=" * 60)
    
    from producer.dft_calculator import XTB_AVAILABLE, GEOMETRIC_AVAILABLE
    
    logger.info(f"xtb-python available: {XTB_AVAILABLE}")
    logger.info(f"geometric available: {GEOMETRIC_AVAILABLE}")
    
    # 测试禁用几何优化
    calc_no_opt = DFTCalculator(
        functional="B3LYP",
        basis="3-21g",  # 小基组加速测试
        verbose=0,
        geometry_optimization=False,
    )
    
    logger.info("Testing without geometry optimization...")
    mol_no_opt = calc_no_opt.from_smiles("CCO")
    logger.success(f"  Created molecule without opt: {mol_no_opt.natm} atoms")
    
    # 测试启用几何优化（如果可用）
    if XTB_AVAILABLE or GEOMETRIC_AVAILABLE:
        calc_with_opt = DFTCalculator(
            functional="B3LYP",
            basis="3-21g",
            verbose=0,
            geometry_optimization=True,
            geo_opt_method="xtb" if XTB_AVAILABLE else "pyscf",
        )
        
        logger.info(f"Testing with geometry optimization ({calc_with_opt.geo_opt_method})...")
        try:
            mol_with_opt = calc_with_opt.from_smiles("CCO")
            logger.success(f"  Created molecule with opt: {mol_with_opt.natm} atoms")
        except Exception as e:
            logger.warning(f"  Geometry optimization test failed: {e}")
    else:
        logger.warning("  No geometry optimization method available, skipping")
    
    logger.info("Geometry Optimization test completed")


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
            basis="sto-3g",  # 使用 GPU 友好基组
            verbose=0,
            geometry_optimization=False,  # 禁用以加速测试
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
                    result["pkl_path"],
                    molecule_id,
                    smiles=smiles,
                )
                
                if not features["success"]:
                    logger.error(f"  Feature extraction failed: {features['error']}")
                    continue
                
                # 3. 保存特征
                output_path = temp_dir / f"{molecule_id}_features"
                saved_path = extractor.save_features(features, str(output_path))
                
                logger.success(f"  Features saved to {saved_path}")
                
                # 4. 清理
                Path(result["pkl_path"]).unlink(missing_ok=True)
                
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


def test_new_features():
    """测试新功能：ELF、IAO 矩阵、全局特征 (NEW)"""
    logger.info("=" * 60)
    logger.info("Testing New Features (ELF, IAO Matrix, Global)")
    logger.info("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # 使用 GPU 友好小基组加速测试
        calculator = DFTCalculator(
            functional="B3LYP",
            basis="sto-3g",  # 最小基组用于快速测试
            verbose=0,
            geometry_optimization=False,  # 禁用以加速测试
        )
        extractor = FeatureExtractor(output_format="json")
        
        test_smiles = "CCO"  # 乙醇
        molecule_id = "test_new_features"
        
        try:
            # 1. DFT 计算
            logger.info("Running DFT calculation...")
            mol = calculator.from_smiles(test_smiles)
            result = calculator.calculate_and_export(
                molecule_id=molecule_id,
                mol_obj=mol,
                output_dir=str(temp_dir),
            )
            
            if not result["success"]:
                logger.error(f"  DFT failed: {result['error']}")
                return False
            
            logger.success(f"  DFT completed: E = {result['energy']:.6f}")
            
            # 2. 特征提取（启用 Fock 矩阵保存）
            logger.info("Extracting features with new methods...")
            features = extractor.extract_all_features(
                result["pkl_path"],
                molecule_id,
                smiles=test_smiles,  # 传递 SMILES
                save_fock_matrix=True,  # 测试 Fock 矩阵保存
            )
            
            if not features["success"]:
                logger.error(f"  Feature extraction failed: {features['error']}")
                return False
            
            feat = features["features"]
            
            # 3. 验证 ELF 特征
            logger.info("Checking ELF features...")
            if "elf" in feat:
                elf_data = feat["elf"]
                logger.info(f"  ELF at atoms: {len(elf_data.get('elf_at_atoms', []))} values")
                logger.info(f"  ELF bond midpoints: {len(elf_data.get('elf_bond_midpoints', []))} values")
                logger.info(f"  ELF mean: {elf_data.get('elf_mean', 'N/A')}")
                logger.success("  ELF features extracted successfully")
            else:
                logger.warning("  ELF features not found")
            
            # 4. 验证 IAO 矩阵
            logger.info("Checking IAO matrix features...")
            if "iao_fock_matrix" in feat:
                fock_shape = len(feat["iao_fock_matrix"])
                logger.info(f"  IAO Fock matrix shape: ({fock_shape}, {fock_shape})")
                logger.success("  IAO Fock matrix extracted successfully")
            else:
                logger.info("  IAO Fock matrix not saved (expected if save_fock_matrix=False)")
            
            if "iao_charges" in feat:
                logger.info(f"  IAO charges: {len(feat['iao_charges'])} atoms")
                logger.success("  IAO charges extracted successfully")
            
            # 5. 验证全局特征
            logger.info("Checking global features...")
            global_features = [
                "homo_energy", "lumo_energy", "homo_lumo_gap",
                "dipole_moment", "total_energy", "n_electrons"
            ]
            for gf in global_features:
                if gf in feat:
                    logger.success(f"  {gf}: {feat[gf]}")
                else:
                    logger.warning(f"  {gf} not found")
            
            # 6. 验证传统特征
            logger.info("Checking traditional features...")
            if "charge_iao" in feat:
                logger.success(f"  IAO charges: {len(feat['charge_iao'])} atoms")
            if "charge_cm5" in feat:
                logger.success(f"  CM5 charges: {len(feat['charge_cm5'])} atoms")
            if "mayer_bond_orders" in feat:
                logger.success(f"  Mayer bond orders: {len(feat['mayer_bond_orders'])}x{len(feat['mayer_bond_orders'][0])} matrix")
            
            # 7. 保存并清理
            output_path = temp_dir / f"{molecule_id}_features.json"
            with open(output_path, 'w') as f:
                json.dump(feat, f, indent=2)
            logger.success(f"  Features saved to {output_path}")
            
            Path(result["pkl_path"]).unlink(missing_ok=True)
            
            logger.success("New features test passed!")
            return True
            
        except Exception as e:
            logger.error(f"New features test failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False


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
    import argparse
    
    parser = argparse.ArgumentParser(description="FastORCA Test Suite")
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="仅运行快速测试（跳过耗时的 DFT 计算）"
    )
    parser.add_argument(
        "--feature",
        choices=["dft", "queue", "e2e", "geometry", "new", "all"],
        default="all",
        help="运行特定功能测试"
    )
    args = parser.parse_args()
    
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    logger.info("=" * 60)
    logger.info("Starting FastORCA Test Suite")
    logger.info("=" * 60)
    
    # 运行测试
    try:
        if args.feature in ["dft", "all"]:
            test_dft_calculator()
        
        if args.feature in ["queue", "all"]:
            test_queue()
        
        if args.feature in ["e2e", "all"] and not args.quick:
            if test_pipeline_end_to_end():
                logger.success("End-to-end tests passed!")
            else:
                logger.warning("Some end-to-end tests failed")
        
        if args.feature in ["geometry", "all"] and not args.quick:
            test_geometry_optimization()
        
        if args.feature in ["new", "all"] and not args.quick:
            if test_new_features():
                logger.success("New features tests passed!")
            else:
                logger.warning("New features tests failed")
        
        if args.feature == "all":
            test_throughput()
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("=" * 60)
    logger.success("All requested tests completed")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

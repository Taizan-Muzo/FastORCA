#!/usr/bin/env python
"""
FastORCA 环境检查脚本
快速诊断常见问题并提供解决方案
"""

import sys
from pathlib import Path

def check_dependencies():
    """检查关键依赖"""
    print("=" * 70)
    print("FastORCA 环境检查")
    print("=" * 70)
    
    # 1. 检查 loguru
    try:
        from loguru import logger
        print("✅ loguru - OK")
    except ImportError:
        print("❌ loguru - 未安装")
        print("   安装: pip install loguru")
        return False
    
    # 2. 检查 numpy, scipy
    try:
        import numpy
        import scipy
        print(f"✅ numpy - {numpy.__version__}")
        print(f"✅ scipy - {scipy.__version__}")
    except ImportError as e:
        print(f"❌ {e}")
        print("   安装: pip install numpy scipy")
        return False
    
    # 3. 检查 PySCF
    try:
        import pyscf
        print(f"✅ pyscf - {pyscf.__version__}")
    except ImportError:
        print("❌ pyscf - 未安装")
        print("   安装: pip install pyscf")
        return False
    
    # 4. 检查 RDKit
    try:
        from rdkit import Chem
        print("✅ rdkit - OK")
    except ImportError:
        print("❌ rdkit - 未安装")
        print("   安装: pip install rdkit")
        return False
    
    return True


def check_gpu():
    """检查 GPU 支持"""
    print("\n" + "=" * 70)
    print("GPU 支持检查")
    print("=" * 70)
    
    # 检查 CUDA
    try:
        import cupy
        cuda_available = True
        print(f"✅ CUDA 可用")
        print(f"   CUDA 版本: {cupy.cuda.runtime.getDeviceCount()} device(s)")
    except ImportError:
        cuda_available = False
        print("⚠️  CUDA 不可用 (cupy 未安装)")
    
    # 检查 gpu4pyscf
    try:
        from gpu4pyscf.dft import RKS as GPU_RKS
        print("✅ gpu4pyscf - OK")
        gpu_available = True
    except ImportError:
        print("❌ gpu4pyscf - 未安装")
        print("   GPU 计算将不可用，自动回退到 CPU")
        print("   安装: https://github.com/pyscf/gpu4pyscf")
        gpu_available = False
    
    if cuda_available and gpu_available:
        print("\n🚀 GPU 加速已启用！")
    else:
        print("\n⚠️  使用 CPU 模式（较慢）")
    
    return gpu_available


def check_geometry_optimization():
    """检查几何优化支持"""
    print("\n" + "=" * 70)
    print("几何优化检查 (qcGEM 必需)")
    print("=" * 70)
    
    # 检查 xtb
    try:
        from xtb.interface import Calculator
        print("✅ xtb-python - OK (推荐)")
        print("   几何优化速度: 快 (~0.5-5s/分子)")
        xtb_ok = True
    except ImportError:
        print("❌ xtb-python - 未安装")
        print("   几何优化将回退到 PySCF + geometric (慢 100-1000x)")
        print("   ⚠️  强烈建议安装 xTB！")
        print("   安装: conda install -c conda-forge xtb")
        print("      或: pip install xtb")
        xtb_ok = False
    
    # 检查 ase (用于 xTB 几何优化)
    try:
        import ase
        print(f"✅ ase - {ase.__version__} (用于 xTB 几何优化)")
        ase_ok = True
    except ImportError:
        print("❌ ase - 未安装")
        print("   安装: pip install ase")
        ase_ok = False
    
    # 检查 geometric
    try:
        from pyscf.geomopt.geometric_solver import optimize
        print("✅ geometric - OK (备用)")
        geometric_ok = True
    except ImportError:
        print("❌ geometric - 未安装")
        print("   安装: pip install geometric")
        geometric_ok = False
    
    if xtb_ok and ase_ok:
        print("\n🚀 推荐使用 xTB + ASE 进行几何优化")
    elif xtb_ok:
        print("\n⚠️  xTB 已安装但缺少 ASE，几何优化可能无法正常工作")
    elif geometric_ok:
        print("\n⚠️  只有 PySCF geometric 可用（较慢）")
    else:
        print("\n❌ 无几何优化支持！将无法生成最稳定构型")
    
    return (xtb_ok and ase_ok) or geometric_ok


def check_basis_compatibility():
    """检查基组兼容性"""
    print("\n" + "=" * 70)
    print("GPU 基组兼容性")
    print("=" * 70)
    
    gpu_friendly = ['sto-3g', '3-21g', '6-31g', '6-311g']
    gpu_limited = ['def2-svp', 'def2-tzvp', '6-31g*', 'cc-pvdz']
    
    print("✅ GPU 友好基组（推荐）:")
    for b in gpu_friendly:
        print(f"   - {b}")
    
    print("\n❌ GPU 不支持（含 d/f 函数，将回退 CPU）:")
    for b in gpu_limited:
        print(f"   - {b}")
    
    print("\n💡 建议: 使用 --basis 3-21g 获得最佳 GPU 性能")


def run_quick_test():
    """运行快速测试"""
    print("\n" + "=" * 70)
    print("快速功能测试")
    print("=" * 70)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from producer.dft_calculator import DFTCalculator, XTB_AVAILABLE
        from consumer.feature_extractor import FeatureExtractor
        
        print("测试 DFTCalculator 初始化...")
        calc = DFTCalculator(
            functional="B3LYP",
            basis="sto-3g",  # 最小基组，快速测试
            verbose=0,
            geometry_optimization=False,  # 禁用几何优化加速测试
        )
        print("✅ DFTCalculator 初始化成功")
        
        print("测试 SMILES 解析...")
        mol = calc.from_smiles("CCO")  # 乙醇
        print(f"✅ SMILES 解析成功: {mol.natm} 个原子")
        
        print("测试 FeatureExtractor 初始化...")
        ext = FeatureExtractor()
        print("✅ FeatureExtractor 初始化成功")
        
        print("\n🎉 所有快速测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("FastORCA 环境检查工具")
    print("=" * 70)
    print("\n此脚本检查 FastORCA 的依赖和配置")
    print("如果发现 ❌，请按照提示安装相应依赖\n")
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 基础依赖检查失败，请先安装 requirements.txt")
        print("   pip install -r requirements.txt")
        return 1
    
    # 检查 GPU
    gpu_ok = check_gpu()
    
    # 检查几何优化
    geo_ok = check_geometry_optimization()
    
    # 检查基组兼容性
    if gpu_ok:
        check_basis_compatibility()
    
    # 运行快速测试
    test_ok = run_quick_test()
    
    # 总结
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)
    
    if test_ok:
        print("✅ FastORCA 可以正常运行")
        
        if not geo_ok:
            print("\n⚠️  警告: 未安装 xTB，几何优化将非常慢")
            print("   建议运行: conda install -c conda-forge xtb")
        
        if not gpu_ok:
            print("\n⚠️  提示: 未检测到 GPU 支持，将使用 CPU 计算")
            print("   如需 GPU 加速，请安装 gpu4pyscf")
        
        print("\n🚀 可以开始运行测试:")
        print("   python test_pipeline.py --quick")
        print("\n   或完整流水线:")
        print("   python main.py --input test_molecules.smi --output output/ --basis 3-21g")
        
        return 0
    else:
        print("❌ 环境检查失败，请修复上述问题后再试")
        return 1


if __name__ == "__main__":
    sys.exit(main())

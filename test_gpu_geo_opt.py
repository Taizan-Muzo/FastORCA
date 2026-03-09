"""
测试 GPU 几何优化功能
"""
import sys
import time
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")

from producer.dft_calculator import DFTCalculator, GPU_AVAILABLE


def test_gpu_geometry_optimization():
    """测试 GPU 加速的几何优化"""
    
    print("=" * 70)
    print("测试 GPU 几何优化功能")
    print("=" * 70)
    print(f"GPU 可用: {GPU_AVAILABLE}")
    print()
    
    # 测试分子：水分子
    smiles = "O"
    
    # 配置 1: 使用 pyscf 方法（将触发 GPU 优化）
    print("测试 1: PySCF 几何优化 (GPU 优先)")
    print("-" * 50)
    
    calc = DFTCalculator(
        functional="B3LYP",
        basis="3-21g",  # 使用 GPU 友好的基组
        verbose=3,
        geometry_optimization=True,
        geo_opt_method="pyscf",  # 强制使用 PySCF 方法
        geo_opt_maxsteps=20,  # 限制步数以加快测试
    )
    
    try:
        start = time.time()
        mol = calc.from_smiles(smiles, charge=0, spin=0)
        elapsed = time.time() - start
        
        print(f"✅ 几何优化成功！耗时: {elapsed:.2f}s")
        print(f"   原子数: {mol.natm}")
        print(f"   基组: {mol.basis}")
        
        # 打印优化后的坐标
        print("   优化后的坐标:")
        for i in range(mol.natm):
            symbol = mol.atom_symbol(i)
            x, y, z = mol.atom_coord(i)
            print(f"     {symbol}: ({x:.4f}, {y:.4f}, {z:.4f})")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_gpu_with_dft_single_point():
    """测试 GPU 几何优化 + DFT 单点能计算"""
    
    print()
    print("=" * 70)
    print("测试 2: GPU 几何优化 + DFT 单点能")
    print("=" * 70)
    
    # 测试分子：甲醇
    smiles = "CO"
    
    calc = DFTCalculator(
        functional="B3LYP",
        basis="3-21g",
        verbose=3,
        geometry_optimization=True,
        geo_opt_method="pyscf",
        geo_opt_maxsteps=20,
    )
    
    try:
        # 几何优化
        print("步骤 1: 几何优化...")
        start = time.time()
        mol = calc.from_smiles(smiles, charge=0, spin=0)
        geo_time = time.time() - start
        print(f"   几何优化完成，耗时: {geo_time:.2f}s")
        
        # DFT 单点能计算
        print("步骤 2: DFT 单点能计算...")
        start = time.time()
        mf = calc.run_sp("test_methanol", mol)
        dft_time = time.time() - start
        
        print(f"   DFT 计算完成，耗时: {dft_time:.2f}s")
        print(f"   总能量: {mf.e_tot:.6f} Hartree")
        print(f"   SCF 收敛: {mf.converged}")
        
        print(f"\n✅ 完整流程测试成功！总耗时: {geo_time + dft_time:.2f}s")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_fallback_to_cpu():
    """测试 GPU 失败时的 CPU 回退"""
    
    print()
    print("=" * 70)
    print("测试 3: GPU → CPU 回退机制 (使用 def2-SVP 触发回退)")
    print("=" * 70)
    
    smiles = "O"
    
    # 使用 def2-SVP 基组（含 d 函数，可能触发 GPU 失败）
    calc = DFTCalculator(
        functional="B3LYP",
        basis="def2-SVP",  # 这个基组可能导致 GPU 失败
        verbose=3,
        geometry_optimization=True,
        geo_opt_method="pyscf",
        geo_opt_maxsteps=10,  # 限制步数
    )
    
    try:
        start = time.time()
        mol = calc.from_smiles(smiles, charge=0, spin=0)
        elapsed = time.time() - start
        
        print(f"✅ 几何优化完成（可能使用 CPU 回退）！耗时: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_comparison_xtb_vs_pyscf():
    """对比 xTB 和 PySCF (GPU) 优化速度"""
    
    print()
    print("=" * 70)
    print("测试 4: 对比 xTB vs PySCF (GPU) 优化速度")
    print("=" * 70)
    
    smiles = "CC(=O)O"  # 乙酸，稍复杂的分子
    
    # xTB 优化
    print("方法 1: xTB 优化")
    calc_xtb = DFTCalculator(
        functional="B3LYP",
        basis="3-21g",
        geometry_optimization=True,
        geo_opt_method="xtb",
        geo_opt_maxsteps=50,
    )
    
    try:
        start = time.time()
        mol_xtb = calc_xtb.from_smiles(smiles, charge=0, spin=0)
        xtb_time = time.time() - start
        print(f"   ✅ xTB 完成，耗时: {xtb_time:.2f}s")
    except Exception as e:
        print(f"   ❌ xTB 失败: {e}")
        xtb_time = None
    
    # PySCF (GPU) 优化
    print("\n方法 2: PySCF (GPU) 优化")
    calc_pyscf = DFTCalculator(
        functional="B3LYP",
        basis="3-21g",
        geometry_optimization=True,
        geo_opt_method="pyscf",
        geo_opt_maxsteps=20,
    )
    
    try:
        start = time.time()
        mol_pyscf = calc_pyscf.from_smiles(smiles, charge=0, spin=0)
        pyscf_time = time.time() - start
        print(f"   ✅ PySCF (GPU) 完成，耗时: {pyscf_time:.2f}s")
    except Exception as e:
        print(f"   ❌ PySCF (GPU) 失败: {e}")
        pyscf_time = None
    
    # 对比
    print(f"\n速度对比:")
    if xtb_time and pyscf_time:
        ratio = pyscf_time / xtb_time if xtb_time > 0 else 0
        print(f"   xTB: {xtb_time:.2f}s")
        print(f"   PySCF (GPU): {pyscf_time:.2f}s")
        print(f"   比例: PySCF 是 xTB 的 {ratio:.1f}x")


if __name__ == "__main__":
    # 运行所有测试
    test_gpu_geometry_optimization()
    test_gpu_with_dft_single_point()
    test_fallback_to_cpu()
    test_comparison_xtb_vs_pyscf()
    
    print()
    print("=" * 70)
    print("所有测试完成")
    print("=" * 70)

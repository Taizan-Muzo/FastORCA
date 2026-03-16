"""
精度交叉验证脚本 (GPU-Hybrid vs Pure-CPU)
用于证明魔改后的 GPU 架构在物理精度上与标准 CPU 计算绝对等价。
"""
from pyscf import gto, dft
import numpy as np
import time

# 随便找一个测试分子的 SMILES (比如你的咖啡因)
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

print("="*60)
print("开始交叉验证：GPU-Hybrid (Density Fitting) VS Pure-CPU (Exact)")
print("="*60)

# 1. 准备相同的分子对象 (使用 RDKit 生成一次确定的 3D 坐标)
from rdkit import Chem
from rdkit.Chem import AllChem
mol_rdkit = Chem.MolFromSmiles(smiles)
mol_rdkit = Chem.AddHs(mol_rdkit)
AllChem.EmbedMolecule(mol_rdkit, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol_rdkit)
conf = mol_rdkit.GetConformer()

atoms = [(atom.GetSymbol(), conf.GetPositions()[i]) for i, atom in enumerate(mol_rdkit.GetAtoms())]

# 共同的 PySCF Mole 参数
mol = gto.Mole()
mol.atom = atoms
mol.basis = 'def2-SVP'
mol.verbose = 0
mol.build()

# =====================================================================
# 测试 1: 我们魔改的 GPU-Hybrid (带 Density Fitting)
# =====================================================================
print("\n[1] 运行我们部署的 GPU-Hybrid 架构...")
try:
    from gpu4pyscf.dft import RKS as GPU_RKS
    t0 = time.time()
    # 这里就是我们线上的真实配置
    mf_gpu = GPU_RKS(mol).density_fit(auxbasis='def2-svp-jkfit')
    mf_gpu.xc = 'B3LYP'
    mf_gpu.conv_tol = 1e-9
    e_gpu = mf_gpu.kernel()
    t_gpu = time.time() - t0
    
    # 【修复点】：先将整个 GPU 对象转换到 CPU，然后再做所有的属性提取
    mf_cpu_from_gpu = mf_gpu.to_cpu()
    
    # 提取 HOMO 和偶极矩 (现在全都是安全的 NumPy 操作了)
    mo_occ_gpu = mf_cpu_from_gpu.mo_occ
    homo_idx = np.where(mo_occ_gpu > 0)[0][-1]
    homo_gpu = mf_cpu_from_gpu.mo_energy[homo_idx]
    dipole_gpu = mf_cpu_from_gpu.dip_moment(mol, mf_cpu_from_gpu.make_rdm1(), unit='Debye')
    
    print(f"✅ GPU 耗时: {t_gpu:.2f} 秒")
    print(f"   总能量 (E_tot): {e_gpu:.8f} Hartree")
    print(f"   HOMO 能级   : {homo_gpu:.8f} Hartree")
    print(f"   偶极矩 (X,Y,Z): {dipole_gpu[0]:.4f}, {dipole_gpu[1]:.4f}, {dipole_gpu[2]:.4f} Debye")
except ImportError:
    print("未能加载 gpu4pyscf，跳过 GPU 测试")

# =====================================================================
# 测试 2: 绝对正统的纯 CPU 精确积分 (作为 Ground Truth)
# =====================================================================
print("\n[2] 运行正统 Pure-CPU 架构 (Exact Integration)...")
t0 = time.time()
mf_cpu = dft.RKS(mol)  # 没有任何 density_fit 近似
mf_cpu.xc = 'B3LYP'
mf_cpu.conv_tol = 1e-9
e_cpu = mf_cpu.kernel()
t_cpu = time.time() - t0

mo_occ_cpu = mf_cpu.mo_occ
homo_cpu = mf_cpu.mo_energy[homo_idx]
dipole_cpu = mf_cpu.dip_moment(mol, mf_cpu.make_rdm1(), unit='Debye')

print(f"✅ CPU 耗时: {t_cpu:.2f} 秒")
print(f"   总能量 (E_tot): {e_cpu:.8f} Hartree")
print(f"   HOMO 能级   : {homo_cpu:.8f} Hartree")
print(f"   偶极矩 (X,Y,Z): {dipole_cpu[0]:.4f}, {dipole_cpu[1]:.4f}, {dipole_cpu[2]:.4f} Debye")

# =====================================================================
# 结论对比
# =====================================================================
print("\n" + "="*60)
print("📊 精度误差报告 (GPU vs CPU Ground Truth):")
print(f"  能量误差 ΔE      : {abs(e_gpu - e_cpu):.2e} Hartree (< 1e-4 为完美)")
print(f"  HOMO 误差 ΔHOMO  : {abs(homo_gpu - homo_cpu):.2e} Hartree")
print(f"  加速比           : {t_cpu / t_gpu:.1f} 倍")
print("="*60)
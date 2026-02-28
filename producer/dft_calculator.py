"""
GPU DFT Calculator Module
使用 gpu4pyscf 进行高速量子化学计算
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
import tempfile
import time

from loguru import logger
import numpy as np

# GPU PySCF imports
# 注意：gpu4pyscf 使用 pyscf 的 gto，但使用自己的 dft
from pyscf import gto, lib

try:
    from gpu4pyscf.dft import RKS as GPU_RKS
    GPU_AVAILABLE = True
    logger.info("gpu4pyscf loaded, GPU acceleration enabled")
except ImportError:
    logger.warning("gpu4pyscf not available, falling back to CPU PySCF")
    GPU_AVAILABLE = False

from pyscf import dft
from pyscf.dft import rks


# GPU 基组兼容性提示
def check_basis_gpu_compatibility(basis: str) -> tuple[bool, str]:
    """
    检查基组是否与 GPU 模式兼容
    
    Returns:
        (is_compatible, message)
    """
    basis_lower = basis.lower()
    
    # 已知完全支持 GPU 的基组（无 d/f 极化函数）
    gpu_friendly = ['sto-3g', '3-21g', '6-31g', '6-311g', 'mini', 'midi']
    
    # 已知可能有问题（含 d/f 函数）
    gpu_limited = ['def2-svp', 'def2-tzvp', 'def2-svpp', 'def2-tzvpp',
                   'cc-pvdz', 'cc-pvtz', 'aug-cc-pvdz', 'aug-cc-pvtz']
    
    for b in gpu_friendly:
        if b in basis_lower:
            return True, f"基组 {basis} 应该支持 GPU 计算"
    
    for b in gpu_limited:
        if b in basis_lower:
            return False, f"基组 {basis} 包含 d/f 极化函数，可能触发 GPU 回退到 CPU"
    
    # 默认警告
    if GPU_AVAILABLE:
        return True, f"基组 {basis} GPU 兼容性未知，将尝试 GPU 计算，失败时自动回退 CPU"
    else:
        return False, "GPU 不可用，使用 CPU 计算"

from pyscf import lib
import pickle

# RDKit for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available, SMILES processing disabled")


class DFTCalculator:
    """
    GPU 加速的 DFT 计算器
    
    使用 gpu4pyscf 在 GPU 上执行 DFT 计算，
    然后将结果转回 CPU 并导出波函数文件
    """
    
    def __init__(
        self,
        functional: str = "B3LYP",
        basis: str = "def2-SVP",
        verbose: int = 3,
        max_memory: int = 8000,  # MB
        scf_conv_tol: float = 1e-9,
    ):
        """
        初始化 DFT 计算器
        
        Args:
            functional: DFT 泛函，默认 B3LYP
            basis: 基组，默认 def2-SVP
            verbose: 输出详细程度 (0-9)
            max_memory: 最大内存使用 (MB)
            scf_conv_tol: SCF 收敛阈值
        """
        self.functional = functional
        self.basis = basis
        self.verbose = verbose
        self.max_memory = max_memory
        self.scf_conv_tol = scf_conv_tol
        
        # 配置日志
        logger.info(f"DFTCalculator initialized: {functional}/{basis}")
        logger.info(f"GPU available: {GPU_AVAILABLE}")
        
        # 检查 GPU 基组兼容性
        if GPU_AVAILABLE:
            compatible, msg = check_basis_gpu_compatibility(basis)
            if not compatible:
                logger.warning(msg)
            else:
                logger.debug(msg)
    
    def from_smiles(
        self,
        smiles: str,
        charge: int = 0,
        spin: int = 0,
        n_conformers: int = 1,
        random_seed: int = 42,
    ) -> gto.Mole:
        """
        从 SMILES 字符串生成分子对象
        
        Args:
            smiles: SMILES 字符串
            charge: 分子电荷
            spin: 自旋多重度 (2S, 0 for singlet)
            n_conformers: 构象数量
            random_seed: 随机种子
            
        Returns:
            PySCF Mole 对象
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES processing")
        
        try:
            # 解析 SMILES
            mol_rdkit = Chem.MolFromSmiles(smiles)
            if mol_rdkit is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # 添加氢原子
            mol_rdkit = Chem.AddHs(mol_rdkit)
            
            # 生成 3D 构象
            AllChem.EmbedMolecule(mol_rdkit, randomSeed=random_seed)
            
            # 力场优化
            try:
                AllChem.MMFFOptimizeMolecule(mol_rdkit)
            except:
                # 如果 MMFF 失败，尝试 UFF
                AllChem.UFFOptimizeMolecule(mol_rdkit)
            
            # 获取坐标
            conf = mol_rdkit.GetConformer()
            coords = conf.GetPositions()
            
            # 构建原子列表
            atoms = []
            for i, atom in enumerate(mol_rdkit.GetAtoms()):
                symbol = atom.GetSymbol()
                x, y, z = coords[i]
                atoms.append((symbol, (x, y, z)))
            
            # 创建 PySCF Mole 对象
            mol = gto.Mole()
            mol.atom = atoms
            mol.basis = self.basis
            mol.charge = charge
            mol.spin = spin
            mol.verbose = self.verbose
            mol.max_memory = self.max_memory
            mol.build()
            
            logger.info(f"Created Mole from SMILES: {smiles[:30]}...")
            logger.info(f"  Atoms: {len(atoms)}, Charge: {charge}, Spin: {spin}")
            
            return mol
            
        except Exception as e:
            logger.error(f"Failed to process SMILES '{smiles}': {e}")
            raise
    
    def from_xyz(
        self,
        xyz_file: str,
        charge: int = 0,
        spin: int = 0,
    ) -> gto.Mole:
        """
        从 XYZ 文件读取分子结构
        
        Args:
            xyz_file: XYZ 文件路径
            charge: 分子电荷
            spin: 自旋多重度
            
        Returns:
            PySCF Mole 对象
        """
        try:
            # 读取 XYZ 文件
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            # 解析 XYZ
            n_atoms = int(lines[0].strip())
            comment = lines[1].strip() if len(lines) > 1 else ""
            
            atoms = []
            for i in range(2, 2 + n_atoms):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append((symbol, (x, y, z)))
            
            # 创建 Mole 对象
            mol = gto.Mole()
            mol.atom = atoms
            mol.basis = self.basis
            mol.charge = charge
            mol.spin = spin
            mol.verbose = self.verbose
            mol.max_memory = self.max_memory
            mol.build()
            
            logger.info(f"Created Mole from XYZ: {xyz_file}")
            logger.info(f"  Atoms: {len(atoms)}, Charge: {charge}, Spin: {spin}")
            
            return mol
            
        except Exception as e:
            logger.error(f"Failed to read XYZ file '{xyz_file}': {e}")
            raise
    
    def run_sp(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
    ):
        """
        执行单点能计算
        
        Args:
            molecule_id: 分子唯一标识
            mol_obj: PySCF Mole 对象
            
        Returns:
            已收敛的 RKS 对象（CPU 端）
        """
        logger.info(f"[{molecule_id}] Starting DFT calculation...")
        start_time = time.time()
        
        try:
            # 如果使用 GPU，尝试使用 gpu4pyscf
            if GPU_AVAILABLE:
                logger.info(f"[{molecule_id}] Trying GPU acceleration...")
                try:
                    mf = GPU_RKS(mol_obj)
                    mf.xc = self.functional
                    mf.conv_tol = self.scf_conv_tol
                    mf.max_cycle = 100
                    
                    # 执行 SCF 计算
                    energy = mf.kernel()
                    
                    # 检查收敛
                    if not mf.converged:
                        logger.warning(f"[{molecule_id}] SCF did not converge!")
                        logger.info(f"[{molecule_id}] Trying second-order convergence...")
                        mf = mf.newton()
                        energy = mf.kernel()
                    
                    # GPU 计算结果转回 CPU（用于后续特征提取）
                    if hasattr(mf, 'to_cpu'):
                        mf = mf.to_cpu()
                        logger.debug(f"[{molecule_id}] Moved back to CPU")
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[{molecule_id}] GPU DFT completed in {elapsed:.2f}s")
                    logger.info(f"[{molecule_id}] Energy: {energy:.6f} Hartree")
                    return mf
                    
                except Exception as gpu_error:
                    # GPU 计算失败，回退到 CPU
                    logger.warning(f"[{molecule_id}] GPU calculation failed: {gpu_error}")
                    logger.info(f"[{molecule_id}] Falling back to CPU...")
            
            # CPU 计算（回退或原始 CPU 模式）
            # 对大分子使用密度拟合加速
            if mol_obj.natm > 6:  # 6原子以上启用密度拟合
                logger.info(f"[{molecule_id}] Using density fitting for CPU calculation")
                mf = dft.RKS(mol_obj).density_fit()
            else:
                mf = dft.RKS(mol_obj)
            mf.xc = self.functional
            # 添加 D3BJ 色散校正（Grimme D3 with Becke-Johnson damping）
            try:
                mf.disp = 'd3bj'
                logger.info(f"[{molecule_id}] D3BJ dispersion correction enabled")
            except Exception as disp_error:
                logger.warning(f"[{molecule_id}] Failed to enable D3BJ: {disp_error}")
            mf.conv_tol = self.scf_conv_tol
            mf.max_cycle = 100
            
            energy = mf.kernel()
            
            if not mf.converged:
                logger.warning(f"[{molecule_id}] SCF did not converge!")
                logger.info(f"[{molecule_id}] Trying second-order convergence...")
                mf = mf.newton()
                energy = mf.kernel()
            
            elapsed = time.time() - start_time
            logger.info(f"[{molecule_id}] CPU DFT completed in {elapsed:.2f}s")
            logger.info(f"[{molecule_id}] Energy: {energy:.6f} Hartree")
            
            return mf
            
        except Exception as e:
            logger.error(f"[{molecule_id}] DFT calculation failed: {e}")
            raise
    
    def export_wavefunction(
        self,
        mf: dft.rks.RKS,
        molecule_id: str,
        output_dir: str = "temp/",
    ) -> str:
        """
        导出波函数到 pickle 文件
        
        Args:
            mf: 已收敛的 RKS 对象（CPU 端）
            molecule_id: 分子唯一标识
            output_dir: 输出目录
            
        Returns:
            导出的文件路径
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            pkl_file = output_path / f"{molecule_id}.pkl"
            
            # 导出为 pickle（只保存需要的属性）
            data = {
                'mol': mf.mol,
                'mo_energy': mf.mo_energy,
                'mo_coeff': mf.mo_coeff,
                'mo_occ': mf.mo_occ,
                'e_tot': mf.e_tot,
                'converged': mf.converged,
            }
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"[{molecule_id}] Wavefunction exported to {pkl_file}")
            
            return str(pkl_file)
            
        except Exception as e:
            logger.error(f"[{molecule_id}] Failed to export wavefunction: {e}")
            raise
    
    def calculate_and_export(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
        output_dir: str = "temp/",
    ) -> dict:
        """
        执行完整流程：计算 + 导出
        
        Args:
            molecule_id: 分子唯一标识
            mol_obj: PySCF Mole 对象
            output_dir: 输出目录
            
        Returns:
            包含计算结果的字典
        """
        result = {
            "molecule_id": molecule_id,
            "success": False,
            "energy": None,
            "pkl_file": None,
            "error": None,
        }
        
        try:
            # 执行 DFT 计算
            mf = self.run_sp(molecule_id, mol_obj)
            result["energy"] = mf.e_tot
            result["converged"] = mf.converged
            
            # 导出波函数
            pkl_path = self.export_wavefunction(mf, molecule_id, output_dir)
            result["pkl_file"] = pkl_path
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{molecule_id}] Calculation pipeline failed: {e}")
        
        return result

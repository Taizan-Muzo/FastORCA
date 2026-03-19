"""
CPU 特征提取模块
从波函数文件中提取量子化学特征
"""

import os
import json
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from pyscf import symm
import numpy as np
import scipy
from loguru import logger
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
# PySCF imports for wavefunction analysis
from pyscf import gto, scf, dft
from pyscf.lo import orth
from pyscf import lib
import pickle
from rdkit import Chem
from rdkit.Chem import inchi as rdinchi

# Import unified output schema (Milestone 1)
from utils.output_schema import UnifiedOutputBuilder
from analysis.orbital_features import extract_orbital_features
from analysis.realspace_features import extract_realspace_features
from analysis.external.adapters.critic2_adapter import run_critic2_analysis, Critic2Adapter
from analysis.external.bridge_context import BridgeContext

# Optional: HDF5 support
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False


class FeatureExtractor:
    """
    量子化学特征提取器
    
    从 pickle 文件中提取以下特征：
    - 电荷：Mulliken、Hirshfeld
    - 键级：Mayer、Wiberg
    - NBO 分析（如果可用）
    """
    
    def __init__(
        self,
        use_multiwfn: bool = False,
        multiwfn_path: str = "Multiwfn",
        output_format: str = "json",
    ):
        """
        初始化特征提取器
        
        Args:
            use_multiwfn: 是否使用 Multiwfn 进行高级分析
            multiwfn_path: Multiwfn 可执行文件路径
            output_format: 输出格式 ("json" 或 "hdf5")
        """
        self.use_multiwfn = use_multiwfn
        self.multiwfn_path = multiwfn_path
        self.output_format = output_format.lower()
        
        if self.output_format == "hdf5" and not HDF5_AVAILABLE:
            logger.warning("HDF5 not available, falling back to JSON")
            self.output_format = "json"
        
        logger.info(f"FeatureExtractor initialized (Multiwfn: {use_multiwfn})")
    
    def extract_rdkit_global_features(self, mol_rdkit) -> dict:
        """
        提取 RDKit 全局理化特征 (对应 qcMol Supplementary Table 4)
        """
        
        return {
            "Molecular Size": mol_rdkit.GetNumAtoms(),
            "Molecular Weight": Descriptors.MolWt(mol_rdkit),
            "LogP": Descriptors.MolLogP(mol_rdkit),
            "TPSA": Descriptors.TPSA(mol_rdkit),
            "H Bond Donor": Lipinski.NumHDonors(mol_rdkit),
            "H Bond Acceptor": Lipinski.NumHAcceptors(mol_rdkit),
            "Rotatable Bonds": Lipinski.NumRotatableBonds(mol_rdkit),
            "Heavy Atom Count": mol_rdkit.GetNumHeavyAtoms()
        }

    def extract_rdkit_atom_features(self, mol_rdkit) -> list:
        """
        提取 RDKit 节点 (原子) 特征 (对应 qcMol Supplementary Table 5)
        """
        atom_features = []
        for atom in mol_rdkit.GetAtoms():
            atom_features.append({
                "Index": atom.GetIdx(),
                "Element Type": atom.GetSymbol(),
                "Atomic Number": atom.GetAtomicNum(),
                "Chiral Tag": str(atom.GetChiralTag()),
                "Degree": atom.GetDegree(),
                "Number of hydrogens": atom.GetTotalNumHs(),
                "Formal Charge": atom.GetFormalCharge(),
                "Radical Electrons": atom.GetNumRadicalElectrons(),
                "Aromatic": atom.GetIsAromatic(),
                "In Ring": atom.IsInRing(),
                "Hybridization Type": str(atom.GetHybridization())
            })
        return atom_features

    def extract_rdkit_bond_features(self, mol_rdkit) -> list:
        """
        提取 RDKit 边 (化学键) 特征 (对应 qcMol Supplementary Table 6)
        """
        bond_features = []
        for bond in mol_rdkit.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            bond_features.append({
                "Source Atom Index": begin_atom.GetIdx(),
                "Source Atom Symbol": begin_atom.GetSymbol(),
                "Target Atom Index": end_atom.GetIdx(),
                "Target Atom Symbol": end_atom.GetSymbol(),
                "Bond Type": str(bond.GetBondType()),
                "Stereo": str(bond.GetStereo()),
                "Conjugated": bond.GetIsConjugated(),
                "In Ring": bond.IsInRing(),
                "Aromatic": bond.GetIsAromatic()
            })
        return bond_features

    def extract_pyscf_global_features(self, mol) -> dict:
        """
        提取 PySCF 原生全局量子特征 (对应 qcMol Supplementary Table 4)
        """
        
        # 尝试检测分子点群 (Point Group)
        try:
            point_group = symm.geom.detect_symm(mol.atom)[0]
        except Exception:
            point_group = "C1"  # 如果检测失败，默认降级为 C1 点群 (无对称性)
            
        return {
            # PySCF 的 mol.spin 存储的是未成对电子数 (2S)。
            # 自旋多重度 (Multiplicity) = 2S + 1
            "Multiplicity": mol.spin + 1,
            "Point Group": point_group
        }
    
    def load_wavefunction(self, pkl_path: str) -> Tuple[gto.Mole, dft.rks.RKS]:
        """
        从 pickle 文件加载波函数
        
        Args:
            pkl_path: pickle 文件路径
            
        Returns:
            (mol, mf) 元组
        """
        logger.debug(f"Loading wavefunction from {pkl_path}")
        
        try:
            # 使用 pickle 读取
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            mol = data['mol']
            mol.verbose = 0  # M5: 静音
            
            # 重建 mf 对象
            mf = dft.RKS(mol)
            mf.verbose = 0  # M5: 静音
            mf.mo_energy = data['mo_energy']
            mf.mo_coeff = data['mo_coeff']
            mf.mo_occ = data['mo_occ']
            mf.e_tot = data['e_tot']
            mf.converged = data['converged']
            
            logger.debug(f"Loaded molecule with {mol.natm} atoms")
            return mol, mf
            
        except Exception as e:
            logger.error(f"Failed to load wavefunction: {e}")
            raise
    
    def extract_mulliken_charges(self, mol: gto.Mole, mf: dft.rks.RKS) -> np.ndarray:
        """
        计算 Mulliken 电荷
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            原子电荷数组 (natm,)
        """
        try:
            # 获取密度矩阵
            dm = mf.make_rdm1()
            
            # 计算 Mulliken 布居
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]  # 对于开壳层
            
            # 使用 PySCF 的 mulliken_meta 函数
            from pyscf.lo import orth
            
            # 获取重叠矩阵
            s = mol.intor_symmetric('int1e_ovlp')
            
            # 计算 Mulliken 布居
            pop = np.einsum('ij,ji->i', dm, s)
            
            # 获取原子序数
            charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
            
            # 计算每个原子的电子数
            ao_slices = mol.aoslice_by_atom()
            atomic_pop = np.zeros(mol.natm)
            for i in range(mol.natm):
                p0, p1 = ao_slices[i, 2:]
                atomic_pop[i] = pop[p0:p1].sum()
            
            # Mulliken 电荷 = 核电荷 - 电子布居
            mulliken_charges = charges - atomic_pop
            
            logger.debug(f"Mulliken charges calculated: {mulliken_charges}")
            return mulliken_charges
            
        except Exception as e:
            logger.error(f"Failed to calculate Mulliken charges: {e}")
            raise
    
    def extract_hirshfeld_charges(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> np.ndarray:
        """
        计算 Hirshfeld 电荷
        
        Hirshfeld 电荷定义：Q_A = Z_A - N_A
        其中 N_A = ∫ ρ(r) * [ρ_A^atom(r) / Σ_B ρ_B^atom(r)] dr
        
        使用分子网格进行数值积分
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            原子电荷数组 (natm,)
        """
        try:
            from pyscf import dft as dft_module
            
            # 获取分子密度矩阵
            dm = mf.make_rdm1()
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # 设置积分网格（使用分子已有的网格或创建新的）
            grids = mf.grids if hasattr(mf, 'grids') and mf.grids.coords is not None else dft_module.Grids(mol)
            if grids.coords is None:
                grids.build()
            
            # 计算 AO 值和分子密度
            ao = dft_module.numint.eval_ao(mol, grids.coords, deriv=0)
            rho = dft_module.numint.eval_rho(mol, ao, dm)
            
            # 为每个原子计算孤立原子密度
            natm = mol.natm
            atomic_densities = []
            
            for i in range(natm):
                # 获取原子符号和电荷
                symbol = mol.atom_symbol(i)
                charge = mol.atom_charge(i)
                
                # 创建孤立原子（单原子分子）
                atom_mol = gto.Mole()
                atom_mol.atom = f'{symbol} 0 0 0'
                # 使用与分子相同的基组
                atom_mol.basis = mol.basis
                atom_mol.spin = 0 if charge % 2 == 0 else 1  # 根据电子数设置自旋
                atom_mol.build()
                
                # 计算孤立原子的密度
                atom_mf = dft_module.RKS(atom_mol)
                atom_mf.xc = 'lda,vwn'  # 使用 LDA 快速计算
                atom_mf.kernel()
                
                if not atom_mf.converged:
                    # 如果 LDA 不收敛，尝试使用与母分子相同的泛函
                    atom_mf = dft_module.RKS(atom_mol)
                    atom_mf.xc = mf.xc if hasattr(mf, 'xc') else 'b3lyp'
                    atom_mf.kernel()
                
                # 计算原子密度在分子网格上的值
                atom_dm = atom_mf.make_rdm1()
                if isinstance(atom_dm, tuple):
                    atom_dm = atom_dm[0] + atom_dm[1]
                
                # 将原子位置移到分子中该原子的位置
                coords = grids.coords - mol.atom_coord(i)
                atom_ao = dft_module.numint.eval_ao(atom_mol, coords, deriv=0)
                atom_rho = dft_module.numint.eval_rho(atom_mol, atom_ao, atom_dm)
                atomic_densities.append(atom_rho)
                
                atom_mol = None
                atom_mf = None
            
            # 计算总的原子密度叠加
            atomic_densities = np.array(atomic_densities)
            total_atomic_rho = np.sum(atomic_densities, axis=0)
            
            # 避免除零
            total_atomic_rho = np.where(total_atomic_rho < 1e-30, 1e-30, total_atomic_rho)
            
            # 计算 Hirshfeld 权重和布居
            hirshfeld_charges = np.zeros(natm)
            weights = grids.weights
            
            for i in range(natm):
                # 权重函数 w_A = ρ_A^atom / Σ_B ρ_B^atom
                weight = atomic_densities[i] / total_atomic_rho
                # 布居 N_A = ∫ ρ(r) * w_A(r) dr
                population = np.sum(rho * weight * weights)
                # 电荷 Q_A = Z_A - N_A
                hirshfeld_charges[i] = mol.atom_charge(i) - population
            
            logger.debug(f"Hirshfeld charges calculated: {hirshfeld_charges}")
            return hirshfeld_charges
            
        except Exception as e:
            logger.warning(f"Hirshfeld charges calculation failed: {e}, returning zeros")
            import traceback
            logger.debug(traceback.format_exc())
            return np.zeros(mol.natm)
    
    def extract_mayer_bond_orders(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> np.ndarray:
        """
        计算 Mayer 键级
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            键级矩阵 (natm, natm)
        """
        try:
            # 获取密度矩阵和重叠矩阵
            dm = mf.make_rdm1()
            s = mol.intor_symmetric('int1e_ovlp')
            
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # 计算 Mayer 键级
            # B_ij = sum_{mu in i} sum_{nu in j} (PS)_{mu nu} (PS)_{nu mu}
            ps = dm @ s
            
            # 按原子分组
            ao_slices = mol.aoslice_by_atom()
            
            bond_orders = np.zeros((mol.natm, mol.natm))
            for i in range(mol.natm):
                i0, i1 = ao_slices[i, 2:]
                for j in range(mol.natm):
                    j0, j1 = ao_slices[j, 2:]
                    
                    # 计算 (PS)_{mu in i, nu in j} 块
                    ps_block = ps[i0:i1, j0:j1]
                    
                    # Mayer 键级
                    bond_orders[i, j] = np.trace(ps_block @ ps_block.T)
            
            logger.debug(f"Mayer bond orders calculated")
            return bond_orders
            
        except Exception as e:
            logger.error(f"Failed to calculate Mayer bond orders: {e}")
            raise
    
    def extract_wiberg_bond_orders(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> np.ndarray:
        """
        计算 Wiberg 键级（基于 Löwdin 正交化）
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            键级矩阵 (natm, natm)
        """
        try:
            # 获取密度矩阵
            dm = mf.make_rdm1()
            s = mol.intor_symmetric('int1e_ovlp')
            
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # Löwdin 正交化: S^(-1/2)
            s_sqrt = scipy.linalg.sqrtm(s)
            s_inv_sqrt = np.linalg.inv(s_sqrt)
            
            # 正交化密度矩阵
            dm_orth = s_inv_sqrt @ dm @ s_inv_sqrt
            
            # 按原子分组
            ao_slices = mol.aoslice_by_atom()
            
            bond_orders = np.zeros((mol.natm, mol.natm))
            for i in range(mol.natm):
                i0, i1 = ao_slices[i, 2:]
                for j in range(mol.natm):
                    j0, j1 = ao_slices[j, 2:]
                    
                    # Wiberg 键级
                    dm_block = dm_orth[i0:i1, j0:j1]
                    bond_orders[i, j] = np.sum(dm_block ** 2)
            
            logger.debug(f"Wiberg bond orders calculated")
            return bond_orders
            
        except Exception as e:
            logger.error(f"Failed to calculate Wiberg bond orders: {e}")
            raise
    
    def extract_iao_charges(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> np.ndarray:
        """
        计算 IAO (Intrinsic Atomic Orbital) 电荷
        
        IAO 电荷是 NPA (Natural Population Analysis) 的完美平替，
        物理意义清晰，在机器学习任务中表现稳定。
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            原子电荷数组 (natm,)
        """
        try:
            from pyscf.lo import iao
            from pyscf import lo
            
            # 构建 IAO 轨道（最小基组投影）
            iao_coeff = iao.iao(mol, mf.mo_coeff[:, mf.mo_occ > 0])
            
            # 使用 Löwdin 正交化
            s = mf.get_ovlp()
            iao_coeff = lo.vec_lowdin(iao_coeff, s)
            
            # 计算 IAO 布居数
            dm = mf.make_rdm1()
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # 投影密度矩阵到 IAO 基: P_IAO = C_IAO^T * S * D * S * C_IAO
            iao_dm = iao_coeff.T @ s @ dm @ s @ iao_coeff
            
            # 计算每个 IAO 的布居数
            iao_pops = np.diag(iao_dm).real
            
            # 将 IAO 布居数分配到原子
            # 使用参考最小基组的原子切片
            ref_mol = iao.reference_mol(mol)
            natm = mol.natm
            atomic_pops = np.zeros(natm)
            
            for i in range(natm):
                p0, p1 = ref_mol.aoslice_by_atom()[i, 2:]
                atomic_pops[i] = np.sum(iao_pops[p0:p1])
            
            # 计算电荷: Q_A = Z_A - N_A
            charges = np.array([mol.atom_charge(i) for i in range(natm)])
            iao_charges = charges - atomic_pops
            
            logger.debug(f"IAO charges calculated: {iao_charges}")
            return iao_charges
            
        except Exception as e:
            logger.warning(f"IAO charges calculation failed: {e}, returning zeros")
            return np.zeros(mol.natm)
    
    def extract_iao_matrix(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> dict:
        """
        提取 IAO 基组下的矩阵（用于描述轨道相互作用）
        
        qcGEM 论文提到的 "non-local internuclear communications" (非局域核间通讯)
        正是由这些矩阵元素描述的。这对于图神经网络理解共轭效应、电荷转移等至关重要。
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            包含以下键的字典:
            - iao_fock: Fock 矩阵在 IAO 基下的表示 (n_iao, n_iao)
            - iao_dm: 密度矩阵在 IAO 基下的表示 (n_iao, n_iao)
            - iao_charges: IAO 电荷 (natm,)
            - atomic_slices: 每个原子对应的 IAO 切片
        """
        try:
            from pyscf.lo import iao
            from pyscf import lo
            
            # 构建 IAO 轨道
            iao_coeff = iao.iao(mol, mf.mo_coeff[:, mf.mo_occ > 0])
            
            # 使用 Löwdin 正交化
            s = mf.get_ovlp()
            iao_coeff = lo.vec_lowdin(iao_coeff, s)
            
            # 获取密度矩阵
            dm = mf.make_rdm1()
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # 投影密度矩阵到 IAO 基
            iao_dm = iao_coeff.T @ s @ dm @ s @ iao_coeff
            
            # 计算 Fock 矩阵并在 IAO 基下投影
            fock = mf.get_fock()
            iao_fock = iao_coeff.T @ fock @ iao_coeff
            
            # 获取 IAO 布居数和电荷
            iao_pops = np.diag(iao_dm).real
            ref_mol = iao.reference_mol(mol)
            natm = mol.natm
            atomic_pops = np.zeros(natm)
            atomic_slices = []
            
            for i in range(natm):
                p0, p1 = ref_mol.aoslice_by_atom()[i, 2:]
                atomic_pops[i] = np.sum(iao_pops[p0:p1])
                atomic_slices.append((int(p0), int(p1)))
            
            charges = np.array([mol.atom_charge(i) for i in range(natm)])
            iao_charges = charges - atomic_pops
            
            result = {
                "iao_fock": iao_fock.real.tolist(),  # Fock 矩阵：轨道能量和相互作用
                "iao_dm": iao_dm.real.tolist(),      # 密度矩阵
                "iao_charges": iao_charges.tolist(),  # IAO 电荷
                "atomic_slices": atomic_slices,      # 原子切片信息
                "n_iao": iao_coeff.shape[1],         # IAO 数量
            }
            
            logger.debug(f"IAO matrix extracted: Fock ({iao_fock.shape}), DM ({iao_dm.shape})")
            return result
            
        except Exception as e:
            logger.warning(f"IAO matrix extraction failed: {e}")
            return {
                "iao_fock": None,
                "iao_dm": None,
                "iao_charges": np.zeros(mol.natm).tolist(),
                "atomic_slices": [],
                "n_iao": 0,
            }
    
    def extract_cm5_charges(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> np.ndarray:
        """
        计算 CM5 (Charge Model 5) 电荷
        
        CM5 是基于 Hirshfeld 电荷的工业级修正模型，
        通过原子间距离和电负性参数修正 Hirshfeld 电荷的低估问题。
        
        CM5 公式: Q_i^CM5 = Q_i^Hirshfeld + Σ_{j≠i} T_ij(B_j - B_i)
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            原子电荷数组 (natm,)
        """
        try:
            # 首先计算 Hirshfeld 电荷
            hirshfeld_charges = self.extract_hirshfeld_charges(mol, mf)
            
            # CM5 参数 (原子电负性参数 B)
            # 来自原始 CM5 论文的拟合参数
            cm5_params = {
                'H': 0.0056, 'He': 0.0000,
                'Li': -0.0269, 'Be': -0.0157, 'B': -0.0023, 'C': 0.0000,
                'N': 0.0056, 'O': 0.0084, 'F': 0.0102, 'Ne': 0.0000,
                'Na': -0.0281, 'Mg': -0.0183, 'Al': -0.0077, 'Si': -0.0043,
                'P': 0.0000, 'S': 0.0036, 'Cl': 0.0052, 'Ar': 0.0000,
            }
            
            natm = mol.natm
            cm5_charges = hirshfeld_charges.copy()
            
            # 计算原子间距离矩阵
            coords = mol.atom_coords()
            dist_matrix = np.zeros((natm, natm))
            for i in range(natm):
                for j in range(i+1, natm):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
            
            # 计算 CM5 修正
            for i in range(natm):
                symbol_i = mol.atom_symbol(i)
                B_i = cm5_params.get(symbol_i, 0.0)
                
                correction = 0.0
                for j in range(natm):
                    if i == j:
                        continue
                    symbol_j = mol.atom_symbol(j)
                    B_j = cm5_params.get(symbol_j, 0.0)
                    
                    # T_ij 是距离相关的权重函数
                    # CM5 使用指数衰减: T_ij = exp(-α * R_ij) / R_ij
                    alpha = 1.0  # 衰减常数 (Angstrom^-1)
                    R_ij = dist_matrix[i, j]
                    if R_ij > 0.1:  # 避免除零
                        T_ij = np.exp(-alpha * R_ij) / R_ij
                        correction += T_ij * (B_j - B_i)
                
                cm5_charges[i] += correction
            
            logger.debug(f"CM5 charges calculated: {cm5_charges}")
            return cm5_charges
            
        except Exception as e:
            logger.warning(f"CM5 charges calculation failed: {e}, returning Hirshfeld charges")
            return hirshfeld_charges if 'hirshfeld_charges' in locals() else np.zeros(mol.natm)
    
    def extract_elf_features(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
        grid_level: int = 2,
    ) -> dict:
        """
        计算 ELF (Electron Localization Function) 特征
        
        ELF 是描述电子在空间中定域化程度的关键描述符，qcGEM 论文明确使用。
        高 ELF 值表示电子定域（孤对电子、共价键），低值表示离域。
        
        由于 ELF 是 3D 空间函数，我们提取以下压缩特征：
        1. 原子处的 ELF 值（原子中心的电子定域程度）
        2. 键中点的 ELF 值（键区域的电子定域程度）
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            grid_level: 网格密度等级 (0-3, 越高越密)
            
        Returns:
            包含 ELF 特征的字典:
            - elf_at_atoms: 原子处的 ELF 值 (natm,)
            - elf_bond_midpoints: 键中点的 ELF 值 (n_bonds,)
            - elf_mean: 平均 ELF 值
            - elf_max: 最大 ELF 值
            - bond_pairs: 键原子对列表
        """
        try:
            from pyscf.dft import numint, gen_grid
            
            # 获取密度矩阵
            dm = mf.make_rdm1()
            if isinstance(dm, tuple):
                dm = dm[0] + dm[1]
            
            # 创建积分网格
            grids = gen_grid.Grids(mol)
            grids.level = grid_level
            grids.build()
            
            # 计算 ELF 在网格上的值
            # ELF = 1 / (1 + (D/D_h)^2)
            # 其中 D 是实际动能密度，D_h 是均匀电子气动能密度
            
            # 计算电子密度和梯度
            rho = numint.eval_rho(mol, numint.eval_ao(mol, grids.coords), dm, xctype='LDA')
            
            # 计算梯度 (用于动能密度)
            ao = numint.eval_ao(mol, grids.coords, deriv=1)
            rho_grad = numint.eval_rho(mol, ao, dm, xctype='GGA', with_lapl=False)
            
            # 计算 Weizsäcker 动能密度（近似 ELF 分子）
            # D = |∇ρ|^2 / (8 * ρ)
            grad_rho_sq = np.sum(rho_grad[1:4]**2, axis=0)
            
            # 避免除零
            rho_safe = np.where(rho < 1e-30, 1e-30, rho)
            D_weizsacker = grad_rho_sq / (8.0 * rho_safe)
            
            # Thomas-Fermi 动能密度（均匀电子气参考）
            # D_TF = (3/10) * (3π^2)^(2/3) * ρ^(5/3)
            D_TF = (3.0/10.0) * (3.0 * np.pi**2)**(2.0/3.0) * rho_safe**(5.0/3.0)
            
            # ELF
            D_ratio = D_weizsacker / D_TF
            elf_values = 1.0 / (1.0 + D_ratio**2)
            
            # 提取原子处的 ELF 值
            elf_at_atoms = []
            for i in range(mol.natm):
                coord = mol.atom_coord(i)
                # 找到最近的网格点
                distances = np.sum((grids.coords - coord)**2, axis=1)
                nearest_idx = np.argmin(distances)
                elf_at_atoms.append(float(elf_values[nearest_idx]))
            
            # 提取键中点的 ELF 值
            elf_bond_midpoints = []
            bond_pairs = []
            
            # 使用 Mayer 键级识别键
            mayer_bo = self.extract_mayer_bond_orders(mol, mf)
            for i in range(mol.natm):
                for j in range(i+1, mol.natm):
                    if mayer_bo[i, j] > 0.3:  # 键级阈值
                        # 计算中点
                        mid_point = (mol.atom_coord(i) + mol.atom_coord(j)) / 2.0
                        # 找到最近的网格点
                        distances = np.sum((grids.coords - mid_point)**2, axis=1)
                        nearest_idx = np.argmin(distances)
                        elf_bond_midpoints.append(float(elf_values[nearest_idx]))
                        bond_pairs.append((i, j))
            
            result = {
                "elf_at_atoms": elf_at_atoms,
                "elf_bond_midpoints": elf_bond_midpoints,
                "bond_pairs": bond_pairs,
                "elf_mean": float(np.mean(elf_values)),
                "elf_max": float(np.max(elf_values)),
                "elf_min": float(np.min(elf_values)),
            }
            
            logger.debug(f"ELF calculated: {len(elf_at_atoms)} atoms, {len(elf_bond_midpoints)} bonds")
            return result
            
        except Exception as e:
            logger.warning(f"ELF calculation failed: {e}")
            return {
                "elf_at_atoms": [],
                "elf_bond_midpoints": [],
                "bond_pairs": [],
                "elf_mean": 0.0,
                "elf_max": 0.0,
                "elf_min": 0.0,
            }
    
    def extract_global_features(
        self,
        mol: gto.Mole,
        mf: dft.rks.RKS,
    ) -> dict:
        """
        提取分子级全局特征
        
        这些特征描述整个分子的电子结构性质，对图神经网络的图级任务很重要。
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 RKS 对象
            
        Returns:
            包含全局特征的字典:
            - homo_energy: HOMO 能级
            - lumo_energy: LUMO 能级
            - homo_lumo_gap: HOMO-LUMO 能隙
            - dipole_moment: 偶极矩 (Debye)
            - quadrupole_moment: 四极矩 (可选)
            - total_energy: 总能量
        """
        try:
            features = {}
            
            # 1. HOMO/LUMO 能级
            mo_energy = mf.mo_energy
            mo_occ = mf.mo_occ
            
            # 找到 HOMO 和 LUMO
            occupied_idx = np.where(mo_occ > 0)[0]
            virtual_idx = np.where(mo_occ == 0)[0]
            
            if len(occupied_idx) > 0:
                homo_idx = occupied_idx[-1]
                features["homo_energy"] = float(mo_energy[homo_idx])
            else:
                features["homo_energy"] = None
            
            if len(virtual_idx) > 0:
                lumo_idx = virtual_idx[0]
                features["lumo_energy"] = float(mo_energy[lumo_idx])
            else:
                features["lumo_energy"] = None
            
            if features["homo_energy"] is not None and features["lumo_energy"] is not None:
                features["homo_lumo_gap"] = features["lumo_energy"] - features["homo_energy"]
            else:
                features["homo_lumo_gap"] = None
            
            # 2. 偶极矩
            try:
                dipole = mf.dip_moment()
                if dipole is not None:
                    features["dipole_moment"] = float(np.linalg.norm(dipole))
                    features["dipole_vector"] = dipole.tolist()
                else:
                    features["dipole_moment"] = 0.0
                    features["dipole_vector"] = [0.0, 0.0, 0.0]
            except Exception as dip_error:
                logger.debug(f"Dipole moment calculation failed: {dip_error}")
                features["dipole_moment"] = 0.0
                features["dipole_vector"] = [0.0, 0.0, 0.0]
            
            # 3. 总能量
            features["total_energy"] = float(mf.e_tot) if hasattr(mf, 'e_tot') else None
            
            # 4. 电子数
            features["n_electrons"] = int(mol.nelectron)
            
            # 5. SCF 收敛状态
            features["scf_converged"] = bool(mf.converged) if hasattr(mf, 'converged') else None
            
            logger.debug(f"Global features extracted: E_HOMO={features.get('homo_energy')}, gap={features.get('homo_lumo_gap')}")
            return features
            
        except Exception as e:
            logger.warning(f"Global features extraction failed: {e}")
            return {
                "homo_energy": None,
                "lumo_energy": None,
                "homo_lumo_gap": None,
                "dipole_moment": None,
                "dipole_vector": None,
                "total_energy": None,
                "n_electrons": mol.nelectron if hasattr(mol, 'nelectron') else None,
                "scf_converged": None,
            }
    
    def extract_all_features(
        self,
        pkl_path: str,
        molecule_id: str,
        smiles: str = None,  # 新增：需要 SMILES 来重建 RDKit 拓扑图
        save_fock_matrix: bool = False,
    ) -> Dict[str, Any]:
        """
        从 pickle 文件提取所有特征，并结合 RDKit 补充拓扑特征
        
        Args:
            pkl_path: pickle 文件路径
            molecule_id: 分子标识
            smiles: 原始 SMILES 字符串，用于提取图拓扑和理化特征
            save_fock_matrix: 是否保存 Fock 矩阵（数据量大，谨慎使用）
            
        Returns:
            包含所有特征的字典
        """
        import time
        from loguru import logger
        
        start_time = time.time()
        
        result = {
            "molecule_id": molecule_id,
            "pkl_path": pkl_path,
            "success": False,
            "features": {},
            "error": None,
        }
        
        try:
            # 加载波函数
            mol, mf = self.load_wavefunction(pkl_path)
            features = {}
            
            # ==========================================
            # 新增模块 1: RDKit 拓扑与理化特征
            # ==========================================
            if smiles:
                try:
                    from rdkit import Chem
                    mol_rdkit = Chem.MolFromSmiles(smiles)
                    if mol_rdkit is not None:
                        # 必须加氢以对齐 PySCF 的全原子系统，否则节点数量对不上
                        mol_rdkit = Chem.AddHs(mol_rdkit)
                        features["rdkit_global"] = self.extract_rdkit_global_features(mol_rdkit)
                        features["rdkit_atoms"] = self.extract_rdkit_atom_features(mol_rdkit)
                        features["rdkit_bonds"] = self.extract_rdkit_bond_features(mol_rdkit)
                    else:
                        logger.warning(f"[{molecule_id}] RDKit 无法解析 SMILES '{smiles}'，拓扑特征缺失")
                except ImportError:
                    logger.warning(f"[{molecule_id}] RDKit 未安装，跳过拓扑特征提取")
            else:
                logger.debug(f"[{molecule_id}] 未提供 SMILES，跳过 RDKit 拓扑特征提取")

            # ==========================================
            # 新增模块 2: PySCF 补充全局量子特征
            # ==========================================
            features["pyscf_global"] = self.extract_pyscf_global_features(mol)
            
            # ==========================================
            # 核心模块 3: 原生波函数特征 (保持不变)
            # ==========================================
            # 1. 电荷 (高精度方案)
            features["charge_iao"] = self.extract_iao_charges(mol, mf).tolist()  # IAO 电荷 (NPA 平替)
            features["charge_cm5"] = self.extract_cm5_charges(mol, mf).tolist()  # CM5 电荷 (Hirshfeld 修正)
            features["hirshfeld_charges"] = self.extract_hirshfeld_charges(mol, mf).tolist()  
            features["mulliken_charges"] = self.extract_mulliken_charges(mol, mf).tolist()  
            
            # 2. 键级
            features["mayer_bond_orders"] = self.extract_mayer_bond_orders(mol, mf).tolist()
            features["wiberg_bond_orders"] = self.extract_wiberg_bond_orders(mol, mf).tolist()
            
            # 3. ELF 电子定域化函数 (qcGEM 关键特征)
            elf_features = self.extract_elf_features(mol, mf)
            features["elf"] = elf_features
            
            # 4. IAO 矩阵 (轨道相互作用信息)
            iao_matrix = self.extract_iao_matrix(mol, mf)
            features["iao_charges_matrix"] = iao_matrix["iao_charges"] # 为避免与外层电荷命名冲突略作区分
            if save_fock_matrix and iao_matrix["iao_fock"] is not None:
                features["iao_fock_matrix"] = iao_matrix["iao_fock"]  # 大矩阵，可选保存
            features["iao_atomic_slices"] = iao_matrix["atomic_slices"]
            
            # 5. 全局分子特征 (HOMO, LUMO, Gap, Dipole)
            global_features = self.extract_global_features(mol, mf)
            features.update(global_features)
            
            # 6. 分子基本信息
            features["natm"] = mol.natm
            features["atom_symbols"] = [mol.atom_symbol(i) for i in range(mol.natm)]
            features["atom_coords"] = mol.atom_coords().tolist()
            
            # 7. 色散校正信息
            features["dispersion_correction"] = hasattr(mf, 'disp') and mf.disp is not None
            
            result["features"] = features
            result["success"] = True
            
            elapsed = time.time() - start_time
            logger.info(f"[{molecule_id}] Feature extraction completed in {elapsed:.2f}s")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{molecule_id}] Feature extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return result

    def save_features(
        self,
        features: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        保存特征到文件
        
        Args:
            features: 特征字典
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "hdf5":
            # HDF5 格式
            output_file = output_path.with_suffix(".h5")
            with h5py.File(output_file, 'w') as f:
                self._dict_to_hdf5(f, features)
        else:
            # JSON 格式
            output_file = output_path.with_suffix(".json")
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=2)
        
        logger.info(f"Features saved to {output_file}")
        return str(output_file)
    
    def _dict_to_hdf5(self, group: Any, data: Dict[str, Any]):
        """递归地将字典写入 HDF5"""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._dict_to_hdf5(subgroup, value)
            elif isinstance(value, (list, np.ndarray)):
                group.create_dataset(key, data=np.array(value))
            elif isinstance(value, (int, float, str)):
                group.attrs[key] = value
            else:
                group.attrs[key] = str(value)

    # ============ Milestone 1: Unified Schema 接口 ============

    def extract_unified(
        self,
        pkl_path: str,
        molecule_id: str,
        smiles: str = None,
        save_fock_matrix: bool = False,
        dft_config: Dict[str, Any] = None,
        plugin_plan: Dict[str, Any] = None,
        run_mode: str = "full",
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """
        使用统一 Schema 提取特征（Milestone 1 新接口）
        
        不改原有 extract_all_features()，新增此方法作为统一出口。
        
        Args:
            pkl_path: pickle 文件路径
            molecule_id: 分子标识
            smiles: 原始 SMILES 字符串
            save_fock_matrix: 是否保存 Fock 矩阵
            dft_config: DFT 配置信息（用于 provenance）
            plugin_plan: 插件执行计划（由 PluginRegistry 生成）
            run_mode: 运行模式 ("fast" 或 "full")
            output_dir: 输出目录（用于保存 cube 文件等）
            
        Returns:
            符合统一 Schema 的字典
        """
        import time
        from loguru import logger
        
        start_time = time.time()
        
        # 创建 unified builder
        builder = UnifiedOutputBuilder(molecule_id, smiles or "")
        
        # 设置 provenance 中的 DFT 配置
        if dft_config:
            builder.set_provenance(
                dft_functional=dft_config.get("functional"),
                basis_set=dft_config.get("basis"),
                geometry_optimization_method=dft_config.get("geo_opt_method"),
                gpu_acceleration_used=dft_config.get("gpu_used"),
            )
        
        # 检查插件可用性
        try:
            from rdkit import Chem
            builder.set_plugin_status("rdkit", available=True)
        except ImportError:
            builder.set_plugin_status("rdkit", available=False)
        
        try:
            from xtb.interface import Calculator
            builder.set_plugin_status("xtb", available=True)
        except ImportError:
            builder.set_plugin_status("xtb", available=False)
        
        builder.set_plugin_status("pyscf", available=True)  # PySCF 是必须依赖
        
        # ============ 步骤 1: RDKit 解析 ============
        mol_rdkit = None
        if smiles:
            builder.set_plugin_status("rdkit", used=True)
            try:
                from rdkit import Chem
                mol_rdkit = Chem.MolFromSmiles(smiles)
                if mol_rdkit is not None:
                    mol_rdkit = Chem.AddHs(mol_rdkit)
                    builder.set_status(rdkit_parse_success=True)
                    builder.set_plugin_status("rdkit", success=True)
                    
                    # 提取 RDKit 全局特征
                    rdkit_global = self.extract_rdkit_global_features(mol_rdkit)
                    builder.set_global_rdkit(
                        molecular_weight=rdkit_global.get("Molecular Weight"),
                        logp=rdkit_global.get("LogP"),
                        tpsa=rdkit_global.get("TPSA"),
                        h_bond_donors=rdkit_global.get("H Bond Donor"),
                        h_bond_acceptors=rdkit_global.get("H Bond Acceptor"),
                        rotatable_bonds=rdkit_global.get("Rotatable Bonds"),
                        heavy_atom_count=rdkit_global.get("Heavy Atom Count")
                    )
                    
                    # 提取分子式
                    from rdkit.Chem import rdMolDescriptors
                    formula = rdMolDescriptors.CalcMolFormula(mol_rdkit)
                    builder.set_molecule_info(formula=formula)
                    
                    # 1) InChI / InChIKey (exact standard generation)
                    try:
                        inchi = rdinchi.MolToInchi(mol_rdkit)
                        inchikey = rdinchi.MolToInchiKey(mol_rdkit)
                        builder.set_molecule_info(inchi=inchi, inchikey=inchikey)
                    except Exception as e:
                        logger.warning(f"[{molecule_id}] InChI generation failed: {e}")

                    # 2) SMARTS / SMART (proxy, pending exact qcMol naming freeze)
                    try:
                        smarts_mol = Chem.RemoveHs(mol_rdkit)
                        smarts = Chem.MolToSmarts(smarts_mol, isomericSmiles=True)
                        builder.set_molecule_info(smarts=smarts)
                        builder.set_molecule_representation_metadata(
                            "smarts",
                            available=bool(smarts),
                            source="rdkit_canonical_smarts_remove_hs",
                            is_proxy=True,
                            proxy_note="Proxy representation; waiting for exact qcMol SMART naming/definition freeze.",
                        )
                    except Exception as e:
                        logger.warning(f"[{molecule_id}] SMARTS generation failed: {e}")
                        builder.set_molecule_representation_metadata(
                            "smarts",
                            available=False,
                            source="rdkit_canonical_smarts_remove_hs",
                            is_proxy=True,
                            proxy_note=f"SMARTS generation failed: {e}",
                        )
                    
                else:
                    builder.set_status(rdkit_parse_success=False, invalid_input=True)
                    builder.add_error(f"RDKit failed to parse SMILES: '{smiles}'")
                    builder.set_plugin_status("rdkit", success=False, errors=["Failed to parse SMILES"])
                    return builder.build()
            except Exception as e:
                builder.set_status(rdkit_parse_success=False)
                builder.add_error(f"RDKit error: {e}")
                builder.set_plugin_status("rdkit", success=False, errors=[str(e)])
        
        # ============ 步骤 2: 加载波函数 ============
        mol = None
        mf = None
        builder.set_artifacts_wavefunction(pkl_path=pkl_path)
        
        try:
            mol, mf = self.load_wavefunction(pkl_path)
            builder.set_status(wavefunction_load_success=True)
            builder.set_artifacts_wavefunction(loaded_successfully=True)
            builder.set_plugin_status("pyscf", used=True, success=True)
            
            # 设置分子基本信息
            builder.set_molecule_info(
                natm=mol.natm,
                charge=mol.charge,
                spin=mol.spin,  # N_alpha - N_beta
                multiplicity=mol.spin + 1  # 2S + 1
            )
            
            # 设置几何信息
            builder.set_geometry(
                atom_symbols=[mol.atom_symbol(i) for i in range(mol.natm)],
                atom_coords_angstrom=mol.atom_coords(unit='A').tolist(),
                point_group=self._detect_point_group(mol)
            )
            builder.set_structural_features(
                optimized_3d_geometry={
                    "available": True,
                    "source": "wavefunction_geometry",
                    "is_proxy": False,
                },
                most_stable_conformation={
                    "available": False,
                    "source": "external_conformer_workflow_required",
                    "is_proxy": True,
                    "proxy_note": "Not computed in current pipeline.",
                },
            )
            
            # 设置 SCF 状态和几何优化状态（从 dft_config 显式设置）
            scf_converged = getattr(mf, 'converged', False)
            # 几何优化语义：
            # - geometry_optimization=False 表示策略性不执行，不应判定 failed_geometry
            # - geometry_optimization=True 但无显式失败标志时，默认视为成功
            if dft_config:
                geo_opt_enabled = bool(dft_config.get("geometry_optimization", True))
                if not geo_opt_enabled:
                    geo_opt_success = True
                else:
                    geo_opt_success = bool(dft_config.get("geometry_optimization_success", True))
            else:
                geo_opt_success = True

            builder.set_status(
                scf_convergence_success=scf_converged,
                geometry_optimization_success=geo_opt_success
            )
            builder.set_global_dft(scf_converged=scf_converged)
            
            if not scf_converged:
                builder.add_error("SCF did not converge")
                builder.set_plugin_status("pyscf", success=False, errors=["SCF not converged"])
            
        except Exception as e:
            builder.set_status(wavefunction_load_success=False)
            builder.set_artifacts_wavefunction(loaded_successfully=False)
            builder.add_error(f"Failed to load wavefunction: {e}")
            builder.set_plugin_status("pyscf", used=True, success=False, errors=[str(e)])
            return builder.build()
        
        # ============ 步骤 3: 提取全局 DFT 特征 ============
        if mol is not None and mf is not None:
            try:
                global_features = self.extract_global_features(mol, mf)
                builder.set_global_dft(
                    total_energy_hartree=global_features.get("total_energy"),
                    homo_energy_hartree=global_features.get("homo_energy"),
                    lumo_energy_hartree=global_features.get("lumo_energy"),
                    homo_lumo_gap_hartree=global_features.get("homo_lumo_gap"),
                    dipole_moment_debye=global_features.get("dipole_moment"),
                    dipole_vector_debye=global_features.get("dipole_vector"),
                    dispersion_correction=getattr(mf, 'disp', None) is not None
                )
            except Exception as e:
                logger.warning(f"[{molecule_id}] Global features extraction failed: {e}")
                builder.add_error(f"Global features failed: {e}")
        
        # ============ 步骤 4: 提取 RDKit 原子特征 ============
        # RDKit 与 PySCF 原子索引对齐保护
        rdkit_pyscf_aligned = False
        if mol_rdkit is not None and mol is not None:
            try:
                # 检查1: 原子数量匹配
                if mol_rdkit.GetNumAtoms() != mol.natm:
                    logger.warning(
                        f"[{molecule_id}] RDKit/PySCF atom count mismatch: "
                        f"RDKit={mol_rdkit.GetNumAtoms()}, PySCF={mol.natm}. "
                        f"Skipping RDKit atom/bond mapping."
                    )
                else:
                    # 检查2: 原子序数匹配（逐原子验证）
                    rdkit_atoms = list(mol_rdkit.GetAtoms())
                    atomic_nums_match = True
                    mismatched_atoms = []
                    
                    for i, (rdkit_atom, pyscf_symbol) in enumerate(zip(rdkit_atoms, 
                                                                        [mol.atom_symbol(j) for j in range(mol.natm)])):
                        rdkit_z = rdkit_atom.GetAtomicNum()
                        # PySCF 元素符号转原子序数
                        pyscf_z = self._symbol_to_atomic_num(pyscf_symbol)
                        if rdkit_z != pyscf_z:
                            atomic_nums_match = False
                            mismatched_atoms.append(f"{i}(R:{rdkit_z}!=P:{pyscf_z})")
                            if len(mismatched_atoms) >= 3:  # 最多显示3个
                                break
                    
                    if atomic_nums_match:
                        atom_features = self.extract_rdkit_atom_features(mol_rdkit)
                        builder.set_atom_features(
                            atomic_number=[a["Atomic Number"] for a in atom_features],
                            rdkit_degree=[a["Degree"] for a in atom_features],
                            rdkit_hybridization=[a["Hybridization Type"] for a in atom_features],
                            rdkit_aromatic=[a["Aromatic"] for a in atom_features]
                        )
                        rdkit_pyscf_aligned = True
                        logger.debug(f"[{molecule_id}] RDKit/PySCF atomic index alignment verified")
                    else:
                        logger.warning(
                            f"[{molecule_id}] RDKit/PySCF atomic number mismatch at atoms: "
                            f"{', '.join(mismatched_atoms)}. Skipping RDKit atom/bond mapping."
                        )
            except Exception as e:
                logger.warning(f"[{molecule_id}] RDKit atom features failed: {e}")
        
        # ============ 步骤 5: 提取波函数原子特征（电荷） ============
        if mol is not None and mf is not None:
            try:
                mulliken = self.extract_mulliken_charges(mol, mf).tolist()
                builder.set_atom_features(charge_mulliken=mulliken)
            except Exception as e:
                logger.warning(f"[{molecule_id}] Mulliken charges failed: {e}")
            
            try:
                hirshfeld = self.extract_hirshfeld_charges(mol, mf).tolist()
                builder.set_atom_features(charge_hirshfeld=hirshfeld)
            except Exception as e:
                logger.warning(f"[{molecule_id}] Hirshfeld charges failed: {e}")
            
            try:
                iao = self.extract_iao_charges(mol, mf).tolist()
                builder.set_atom_features(charge_iao=iao)
            except Exception as e:
                logger.warning(f"[{molecule_id}] IAO charges failed: {e}")
        
        # ============ 步骤 6: 提取键特征 ============
        # 初始化空键数组（空分子也使用 [] 而不是 None）
        bond_indices: List[List[int]] = []
        bond_types: List[Optional[str]] = []
        bond_stereo_info: List[str] = []
        mayer_values: List[float] = []
        wiberg_values: List[float] = []
        
        # 只有在 RDKit/PySCF 原子索引对齐时才使用 RDKit 键拓扑
        if mol_rdkit is not None and rdkit_pyscf_aligned:
            try:
                for bond in mol_rdkit.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()
                    # 验证索引有效性
                    if 0 <= i < mol.natm and 0 <= j < mol.natm:
                        bond_indices.append([i, j])
                        bond_types.append(str(bond.GetBondType()))
                        bond_stereo_info.append(self._normalize_bond_stereo_enum(str(bond.GetStereo())))
                    else:
                        logger.warning(f"[{molecule_id}] RDKit bond index out of range: {i}-{j}")
            except Exception as e:
                logger.warning(f"[{molecule_id}] RDKit bond extraction failed: {e}")
        
        if mol is not None and mf is not None:
            try:
                mayer_matrix = self.extract_mayer_bond_orders(mol, mf)
                wiberg_matrix = self.extract_wiberg_bond_orders(mol, mf)
                
                # 如果有 RDKit 键拓扑（原子已对齐），按拓扑提取键级
                if bond_indices:
                    for i, j in bond_indices:
                        mayer_values.append(float(mayer_matrix[i, j]))
                        wiberg_values.append(float(wiberg_matrix[i, j]))
                else:
                    # 无 RDKit 拓扑时，使用阈值检测键
                    threshold = 0.3
                    for i in range(mol.natm):
                        for j in range(i+1, mol.natm):
                            if mayer_matrix[i, j] > threshold:
                                bond_indices.append([i, j])
                                bond_types.append(None)  # 无 RDKit 类型信息
                                bond_stereo_info.append("unknown")
                                mayer_values.append(float(mayer_matrix[i, j]))
                                wiberg_values.append(float(wiberg_matrix[i, j]))
                
                # 统一使用空数组 [] 而不是 None（即使是空分子）
                builder.set_bond_features(
                    bond_indices=bond_indices,
                    bond_types_rdkit=bond_types if bond_types else [],  # 空键用 []
                    bond_stereo_info=bond_stereo_info if bond_stereo_info else [],
                    bond_orders_mayer=mayer_values if mayer_values else [],
                    bond_orders_wiberg=wiberg_values if wiberg_values else []
                )
                builder.set_bond_metadata(
                    "bond_stereo_info",
                    available=bool(bond_stereo_info),
                    source="rdkit_bond_stereo_perception" if rdkit_pyscf_aligned else "inferred_without_rdkit_stereo",
                    is_proxy=True,
                    proxy_note=(
                        "Proxy stereo field from RDKit perception; exact qcMol stereo semantics may differ."
                        if rdkit_pyscf_aligned
                        else "Proxy stereo unavailable without aligned RDKit topology; filled as 'unknown'."
                    ),
                )
            except Exception as e:
                logger.warning(f"[{molecule_id}] Bond order extraction failed: {e}")
                # 失败时仍设置空数组
                builder.set_bond_features(
                    bond_indices=[],
                    bond_types_rdkit=[],
                    bond_stereo_info=[],
                    bond_orders_mayer=[],
                    bond_orders_wiberg=[]
                )
                builder.set_bond_metadata(
                    "bond_stereo_info",
                    available=False,
                    source="rdkit_bond_stereo_perception",
                    is_proxy=True,
                    proxy_note=f"Bond stereo extraction failed: {e}",
                )
        
        # ============ 步骤 7: 提取 ELF 特征（扩展特征） ============
        elf_bond_midpoints: List[float] = []  # 默认空数组
        elf_alignment_stats = {
            "raw_count": 0,
            "aligned_count": 0,
            "dropped_count": 0,
        }
        elf_alignment_partial = False
        
        if mol is not None and mf is not None:
            try:
                elf = self.extract_elf_features(mol, mf)
                builder.set_atom_features(elf_value=elf.get("elf_at_atoms"))
                
                # M5.5: 对齐 ELF bond midpoints 到 bond_indices
                elf_bond_midpoints, elf_alignment_stats = self._align_elf_to_bond_indices(
                    elf.get("elf_bond_midpoints", []),
                    elf.get("bond_pairs", []),
                    bond_indices,
                    molecule_id
                )
            except Exception as e:
                logger.warning(f"[{molecule_id}] ELF extraction failed: {e}")
                # 保持与 bond_indices 严格对齐，避免长度 mismatch
                elf_bond_midpoints = [0.0] * len(bond_indices)
                elf_alignment_stats = {
                    "raw_count": 0,
                    "aligned_count": 0,
                    "dropped_count": 0,
                }

        builder.set_bond_features(elf_bond_midpoint=elf_bond_midpoints)
        # 记录对齐统计到 runtime_metadata
        builder.data["runtime_metadata"]["elf_bond_midpoints_raw_count"] = elf_alignment_stats["raw_count"]
        builder.data["runtime_metadata"]["elf_bond_midpoints_aligned_count"] = elf_alignment_stats["aligned_count"]
        builder.data["runtime_metadata"]["elf_bond_midpoints_dropped_count"] = elf_alignment_stats["dropped_count"]
        # 向后兼容：保留旧的内部统计字段
        builder.data["_elf_alignment_stats"] = elf_alignment_stats

        # 对齐不完整时，标记 soft-fail reason（保持状态诚实）
        # 仅当 bond_indices 中存在未对齐键时才记为 partial；
        # ELF 侧多出的键（dropped）只记录统计，不单独降级状态。
        if bond_indices and elf_alignment_stats["aligned_count"] < len(bond_indices):
            elf_alignment_partial = True
        
        # ============ 步骤 8: 提取 CM5 电荷（扩展特征） ============
        if mol is not None and mf is not None:
            try:
                cm5 = self.extract_cm5_charges(mol, mf).tolist()
                builder.set_atom_features(charge_cm5=cm5)
            except Exception as e:
                logger.warning(f"[{molecule_id}] CM5 charges failed: {e}")
        
        # ============ 步骤 9: 提取 Fock 矩阵（可选） ============
        if save_fock_matrix and mol is not None and mf is not None:
            try:
                iao_matrix = self.extract_iao_matrix(mol, mf)
                builder.set_artifacts_fock(
                    iao_matrix=iao_matrix.get("iao_fock"),
                    atomic_slices=iao_matrix.get("atomic_slices"),
                    available=iao_matrix.get("iao_fock") is not None
                )
            except Exception as e:
                logger.warning(f"[{molecule_id}] Fock matrix extraction failed: {e}")
        
        # ============ 步骤 10: 提取局域轨道特征（Milestone 2） ============
        # 检查 plugin_plan
        should_run_orbital = True
        orbital_skip_reason = None
        if plugin_plan and "orbital_features" in plugin_plan:
            plan = plugin_plan["orbital_features"]
            should_run_orbital = plan.get("should_execute", True)
            orbital_skip_reason = plan.get("skip_reason")
        
        if should_run_orbital and mol is not None and mf is not None:
            try:
                logger.info(f"[{molecule_id}] Extracting orbital features (IBO)...")
                orbital_features = extract_orbital_features(mol, mf)
                
                # 将结果填入 builder
                builder.set_orbital_features(
                    local_orbital_method=orbital_features.get("local_orbital_method"),
                    ibo_count=orbital_features.get("ibo_count"),
                    ibo_occupancies=orbital_features.get("ibo_occupancies"),
                    ibo_centers_angstrom=orbital_features.get("ibo_centers_angstrom"),
                    ibo_atom_contributions=orbital_features.get("ibo_atom_contributions"),
                    ibo_class_heuristic=orbital_features.get("ibo_class_heuristic"),
                    orbital_locality_score=orbital_features.get("orbital_locality_score"),
                    iao_atom_mapping=orbital_features.get("iao_atom_mapping"),
                )
                
                # 设置 metadata
                metadata = orbital_features.get("metadata", {})
                builder.set_orbital_metadata(**metadata)
                
                status = metadata.get("extraction_status", "unknown")
                if status == "success":
                    logger.success(f"[{molecule_id}] Orbital features extracted: {orbital_features.get('ibo_count')} IBOs")
                else:
                    reason = metadata.get("failure_reason", "unknown")
                    logger.warning(f"[{molecule_id}] Orbital features soft-fail: {reason}")
                    
            except Exception as e:
                logger.error(f"[{molecule_id}] Orbital features extraction error: {e}")
                logger.debug(f"[{molecule_id}] Orbital features traceback: {traceback.format_exc()}")
                builder.set_orbital_metadata(
                    extraction_status="failed",
                    failure_reason="plugin_execution_error",
                    exception_detail=str(e)  # 保留原始异常在 metadata 中
                )
        elif orbital_skip_reason:
            # 策略性跳过
            logger.info(f"[{molecule_id}] Skipping orbital features: {orbital_skip_reason}")
            builder.set_orbital_metadata(
                extraction_status="skipped",
                failure_reason=orbital_skip_reason
            )
        
        # ============ 步骤 11: 提取实空间特征（Milestone 3） ============
        # 检查 plugin_plan
        should_run_realspace = True
        realspace_skip_reason = None
        if plugin_plan and "realspace_features" in plugin_plan:
            plan = plugin_plan["realspace_features"]
            should_run_realspace = plan.get("should_execute", True)
            realspace_skip_reason = plan.get("skip_reason")
        
        if should_run_realspace and mol is not None and mf is not None:
            try:
                logger.info(f"[{molecule_id}] Extracting realspace features...")
                
                # 创建 cube 输出目录
                cube_output_dir = str(Path(output_dir) / "cubes") if output_dir else "./cubes"
                
                # M5: 获取 timeout 配置
                timeout_seconds = None
                realspace_runtime_config = None
                if plugin_plan and "realspace_features" in plugin_plan:
                    realspace_plan = plugin_plan["realspace_features"]
                    timeout_seconds = realspace_plan.get("effective_timeout")
                    realspace_runtime_config = realspace_plan.get("runtime_config")
                    if isinstance(realspace_runtime_config, dict):
                        # 兼容 policy 层命名 -> realspace extractor 命名
                        normalized_cfg = dict(realspace_runtime_config)
                        if "max_atoms" in normalized_cfg and "max_atoms_for_cube" not in normalized_cfg:
                            normalized_cfg["max_atoms_for_cube"] = normalized_cfg["max_atoms"]
                        realspace_runtime_config = normalized_cfg
                
                realspace_result = extract_realspace_features(
                    mol, mf,
                    molecule_id=molecule_id,
                    output_dir=cube_output_dir,
                    config=realspace_runtime_config,
                    timeout_seconds=timeout_seconds
                )
                
                # 填入 realspace_features
                builder.set_realspace_features(
                    density_isosurface_volume=realspace_result.get("density_isosurface_volume"),
                    density_isosurface_area=realspace_result.get("density_isosurface_area"),
                    density_sphericity_like=realspace_result.get("density_sphericity_like"),
                    esp_extrema_summary=realspace_result.get("esp_extrema_summary"),
                    orbital_extent_homo=realspace_result.get("orbital_extent_homo"),
                    orbital_extent_lumo=realspace_result.get("orbital_extent_lumo"),
                )
                
                # 填入 cube_files 到 artifacts
                cube_files = realspace_result.get("artifacts", {}).get("cube_files", {})
                for cube_type, info in cube_files.items():
                    if info:
                        builder.set_cube_file(cube_type, info)
                
                # 设置 metadata
                metadata = realspace_result.get("metadata", {})
                builder.set_realspace_metadata(**metadata)
                if metadata.get("realspace_definition_version") is not None:
                    builder.set_provenance(
                        feature_definition_version=metadata.get("realspace_definition_version"),
                        realspace_definition_version=metadata.get("realspace_definition_version"),
                    )
                
                status = metadata.get("extraction_status", "unknown")
                if status == "success":
                    logger.success(f"[{molecule_id}] Realspace features extracted")
                elif status == "skipped":
                    reason = metadata.get("failure_reason", "unknown")
                    logger.warning(f"[{molecule_id}] Realspace features skipped: {reason}")
                else:
                    reason = metadata.get("failure_reason", "unknown")
                    logger.warning(f"[{molecule_id}] Realspace features soft-fail: {reason}")
                    
            except Exception as e:
                logger.error(f"[{molecule_id}] Realspace features extraction error: {e}")
                logger.debug(f"[{molecule_id}] Realspace features traceback: {traceback.format_exc()}")
                # M5: 使用正式的 reason code，原始异常保留在 debug log 中
                error_msg = str(e)
                if "unexpected keyword argument" in error_msg:
                    failure_reason = "plugin_api_mismatch"
                elif "subprocess" in error_msg.lower():
                    failure_reason = "plugin_subprocess_error"
                else:
                    failure_reason = "plugin_execution_error"
                
                builder.set_realspace_metadata(
                    extraction_status="failed",
                    failure_reason=failure_reason,
                    exception_detail=error_msg
                )
        elif realspace_skip_reason:
            # 策略性跳过
            logger.info(f"[{molecule_id}] Skipping realspace features: {realspace_skip_reason}")
            builder.set_realspace_metadata(
                extraction_status="skipped",
                failure_reason=realspace_skip_reason
            )
        
        # ============ 步骤 12: External Bridge - Critic2 (Milestone 4) ============
        # 检查 plugin_plan
        should_run_critic2 = False  # 默认关闭
        critic2_skip_reason = None
        if plugin_plan and "critic2_bridge" in plugin_plan:
            plan = plugin_plan["critic2_bridge"]
            should_run_critic2 = plan.get("should_execute", False)
            critic2_skip_reason = plan.get("skip_reason")
        
        if should_run_critic2 and mol is not None and mf is not None:
            try:
                # 检查是否有 density cube
                cube_files = builder.data.get("artifacts", {}).get("cube_files", {})
                density_cube = cube_files.get("density", {}).get("path")
                
                if density_cube and Path(density_cube).exists():
                    logger.info(f"[{molecule_id}] Running critic2 analysis...")
                    
                    # 创建 BridgeContext
                    bridge_context = BridgeContext(
                        molecule_id=molecule_id,
                        atom_symbols=builder.data["geometry"]["atom_symbols"] or [mol.atom_symbol(i) for i in range(mol.natm)],
                        atom_coords_angstrom=mol.atom_coords(unit='A').tolist(),
                        natm=mol.natm,
                        charge=mol.charge,
                        spin=mol.spin,
                        multiplicity=mol.spin + 1,
                        density_cube_path=density_cube,
                        geometry_coordinate_unit="angstrom",
                        cube_native_unit="bohr",
                        cube_output_unit="angstrom",
                    )
                    
                    # 运行 critic2
                    critic2_result = run_critic2_analysis(bridge_context)
                    
                    # 填入 external_bridge
                    # New contract first: metadata + artifact_refs
                    builder.set_external_bridge(
                        "critic2",
                        execution_status=critic2_result.execution_status,
                        failure_reason=critic2_result.failure_reason,
                        metadata={
                            "tool_version": critic2_result.bridge_tool_version,
                            "execution_time_seconds": critic2_result.bridge_execution_time_seconds,
                            "command": None,
                            "parser_version": "critic2_adapter_v1",
                            "environment": None,
                        },
                        artifact_refs={
                            "input_file": critic2_result.bridge_input_file,
                            "output_file": critic2_result.bridge_output_file,
                            "stdout_file": None,
                            "stderr_file": None,
                        },
                        warnings=[],
                        # Deprecated compatibility fields (legacy readers only)
                        input_file=critic2_result.bridge_input_file,   # deprecated
                        output_file=critic2_result.bridge_output_file,  # deprecated
                        execution_time_seconds=critic2_result.bridge_execution_time_seconds,  # deprecated
                        critic2_version=critic2_result.bridge_tool_version,  # deprecated
                    )
                    
                    # 填入 external_features
                    if critic2_result.success:
                        features = critic2_result.features
                        if "qtaim" in features:
                            builder.set_external_features("critic2", features)
                            logger.success(f"[{molecule_id}] Critic2 analysis completed: {features['qtaim'].get('n_bader_volumes', 0)} Bader volumes")
                    else:
                        logger.warning(f"[{molecule_id}] Critic2 soft-fail: {critic2_result.failure_reason}")
                else:
                    logger.warning(f"[{molecule_id}] Skipping critic2: no density cube available")
                    builder.set_external_bridge(
                        "critic2",
                        execution_status="skipped",
                        failure_reason="no_density_cube_available",
                        warnings=["density_cube_missing"],
                    )
                    
            except Exception as e:
                logger.error(f"[{molecule_id}] Critic2 analysis error: {e}")
                builder.set_external_bridge(
                    "critic2",
                    execution_status="failed",
                    failure_reason=f"external_execution_failed: {e}",
                    warnings=[str(e)],
                )
        elif critic2_skip_reason:
            # 策略性跳过
            logger.info(f"[{molecule_id}] Skipping critic2: {critic2_skip_reason}")
            builder.set_external_bridge(
                "critic2",
                execution_status="disabled",
                failure_reason=critic2_skip_reason
            )
        elif plugin_plan and "critic2_bridge" in plugin_plan and not should_run_critic2:
            # 明确标记为 disabled（未给 skip_reason 的配置性关闭）
            builder.set_external_bridge(
                "critic2",
                execution_status="disabled",
                failure_reason="disabled_by_plan",
            )
        
        # 长度校验 (M5: 记录 reason codes)
        valid, mismatch_reasons = self._validate_feature_lengths(builder, molecule_id)
        validation_reasons = list(mismatch_reasons)
        if elf_alignment_partial and "elf_alignment_partial" not in validation_reasons:
            validation_reasons.append("elf_alignment_partial")

        if validation_reasons:
            # 将 length mismatch 记录到 error_messages 和 metadata
            for reason in validation_reasons:
                builder.add_error(f"Feature validation: {reason}")
            # 存储到 builder 的特殊字段供 StatusDeterminer 使用
            builder.data["_validation_errors"] = validation_reasons
        
        # 记录耗时到 runtime_metadata（正式 schema）
        elapsed = time.time() - start_time
        builder.data["runtime_metadata"]["unified_extraction_time_seconds"] = elapsed
        builder.data["runtime_metadata"]["extraction_timestamp"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        
        # 最终化（所有状态判定在此执行）
        result = builder.build()
        
        logger.info(
            f"[{molecule_id}] Unified extraction completed: "
            f"status={result['calculation_status']['overall_status']}, "
            f"time={elapsed:.3f}s"
        )
        
        return result
    
    def _detect_point_group(self, mol) -> Optional[str]:
        """检测分子点群"""
        try:
            from pyscf import symm
            point_group = symm.geom.detect_symm(mol.atom)[0]
            return point_group
        except Exception:
            return "C1"
    
    def _symbol_to_atomic_num(self, symbol: str) -> int:
        """元素符号转原子序数"""
        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
            'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
            'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
            'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
            'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
            'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
            'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
            'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
        }
        return periodic_table.get(symbol.capitalize(), 0)

    def _normalize_bond_stereo_enum(self, rdkit_stereo: str) -> str:
        """
        统一输出键立体化学枚举值，避免暴露 RDKit 内部对象/数字。
        Allowed: none | any | cis | trans | e | z | unknown
        """
        s = (rdkit_stereo or "").strip().upper()
        mapping = {
            "STEREONONE": "none",
            "STEREOANY": "any",
            "STEREOCIS": "cis",
            "STEREOTRANS": "trans",
            "STEREOE": "e",
            "STEREOZ": "z",
        }
        return mapping.get(s, "unknown")

    def save_unified_features(
        self,
        unified_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        保存统一 Schema 特征到文件
        
        Args:
            unified_data: 统一 Schema 字典
            output_path: 输出路径（不含扩展名）
            
        Returns:
            保存的文件路径
        """
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用 numpy 安全的 JSON 序列化
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        if self.output_format == "hdf5":
            output_file = output_path.with_suffix(".h5")
            with h5py.File(output_file, 'w') as f:
                self._dict_to_hdf5(f, unified_data)
        else:
            output_file = output_path.with_suffix(".unified.json")
            with open(output_file, 'w') as f:
                json.dump(unified_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Unified features saved to {output_file}")
        return str(output_file)
    
    def _validate_feature_lengths(self, builder, molecule_id: str) -> tuple[bool, list]:
        """
        校验 atom_features 和 bond_features 的长度一致性和有效性
        
        M5: 将所有 mismatch 记录为 reason codes，影响最终状态判定
        
        Returns:
            (是否通过校验, reason_codes 列表)
        """
        data = builder.data
        natm = data["molecule_info"]["natm"]
        reason_codes = []
        
        if natm is None:
            return True, reason_codes  # 尚无原子数信息，跳过校验
        
        atom_feat = data["atom_features"]
        
        # 1. 检查所有原子级特征数组长度是否等于 natm
        atom_arrays = [
            ("atomic_number", atom_feat.get("atomic_number")),
            ("charge_mulliken", atom_feat.get("charge_mulliken")),
            ("charge_hirshfeld", atom_feat.get("charge_hirshfeld")),
            ("charge_cm5", atom_feat.get("charge_cm5")),
            ("charge_iao", atom_feat.get("charge_iao")),
            ("elf_value", atom_feat.get("elf_value")),
        ]
        
        for name, arr in atom_arrays:
            if arr is not None and len(arr) != natm:
                logger.warning(
                    f"[{molecule_id}] Length mismatch: {name} has {len(arr)} elements, "
                    f"expected {natm} (natm)"
                )
                reason_codes.append(f"atom_feature_length_mismatch_{name}")
        
        # 检查键特征
        bond_feat = data["bond_features"]
        bond_indices = bond_feat.get("bond_indices")
        
        if bond_indices is None:
            # 空键分子，跳过后续键检查
            return len(reason_codes) == 0, reason_codes
        
        if not isinstance(bond_indices, list):
            logger.warning(f"[{molecule_id}] bond_indices is not a list")
            reason_codes.append("bond_indices_invalid_format")
            return False, reason_codes
        
        n_bonds = len(bond_indices)
        
        # 2. 检查键特征数组长度是否一致
        bond_arrays = [
            ("bond_orders_mayer", bond_feat.get("bond_orders_mayer")),
            ("bond_orders_wiberg", bond_feat.get("bond_orders_wiberg")),
            ("elf_bond_midpoint", bond_feat.get("elf_bond_midpoint")),
            ("bond_stereo_info", bond_feat.get("bond_stereo_info")),
        ]
        
        for name, arr in bond_arrays:
            if arr is not None and len(arr) != n_bonds:
                logger.warning(
                    f"[{molecule_id}] Length mismatch: {name} has {len(arr)} elements, "
                    f"expected {n_bonds} (n_bonds)"
                )
                reason_codes.append(f"bond_feature_length_mismatch_{name}")
        
        # 3. 检查 bond_types_rdkit 长度
        bond_types = bond_feat.get("bond_types_rdkit")
        if bond_types is not None and len(bond_types) != n_bonds:
            logger.warning(
                f"[{molecule_id}] Length mismatch: bond_types_rdkit has {len(bond_types)} elements, "
                f"expected {n_bonds} (n_bonds)"
            )
            reason_codes.append("bond_feature_length_mismatch_bond_types_rdkit")

        # 3.1 检查 bond_stereo_info 枚举值
        stereo_values = bond_feat.get("bond_stereo_info")
        allowed_stereo = {"none", "any", "cis", "trans", "e", "z", "unknown"}
        if isinstance(stereo_values, list):
            for idx, val in enumerate(stereo_values):
                if val not in allowed_stereo:
                    logger.warning(
                        f"[{molecule_id}] Invalid bond_stereo_info enum at index {idx}: {val}"
                    )
                    reason_codes.append("bond_feature_invalid_enum_bond_stereo_info")
        
        # 4-6. 检查 bond_indices 有效性
        seen_bonds = set()
        
        for idx, bond in enumerate(bond_indices):
            if not isinstance(bond, (list, tuple)) or len(bond) != 2:
                logger.warning(f"[{molecule_id}] Invalid bond format at index {idx}: {bond}")
                reason_codes.append("bond_indices_invalid_format")
                continue
            
            i, j = bond
            
            # 4. 检查越界
            if not (0 <= i < natm and 0 <= j < natm):
                logger.warning(
                    f"[{molecule_id}] Bond index out of range at bond {idx}: "
                    f"[{i}, {j}], natm={natm}"
                )
                reason_codes.append("bond_indices_out_of_range")
            
            # 5. 检查 self-loop
            if i == j:
                logger.warning(f"[{molecule_id}] Self-loop detected at bond {idx}: [{i}, {j}]")
                reason_codes.append("bond_indices_self_loop")
            
            # 6. 检查无向重复边
            canonical = (min(i, j), max(i, j))
            if canonical in seen_bonds:
                logger.warning(
                    f"[{molecule_id}] Duplicate bond detected at index {idx}: "
                    f"[{i}, {j}] (canonical: {canonical})"
                )
                reason_codes.append("bond_indices_duplicate")
            seen_bonds.add(canonical)
        
        return len(reason_codes) == 0, list(set(reason_codes))  # 去重
    
    def _align_elf_to_bond_indices(
        self,
        elf_midpoints: List[float],
        elf_bond_pairs: List[tuple],
        bond_indices: List[List[int]],
        molecule_id: str
    ) -> tuple[List[float], dict]:
        """
        对齐 ELF bond midpoints 到 bond_indices
        
        M5.5: 使用无向排序键 (min, max) 作为 join key
        
        Args:
            elf_midpoints: ELF 计算的键中点值
            elf_bond_pairs: ELF 计算的键原子对列表
            bond_indices: 目标 bond_indices（来自 RDKit 或 Mayer 阈值）
            molecule_id: 分子 ID
            
        Returns:
            (对齐后的 midpoints, 对齐统计信息)
        """
        stats = {
            "raw_count": len(elf_midpoints),
            "aligned_count": 0,
            "dropped_count": 0,
        }
        
        if not bond_indices:
            # 无 bond_indices，返回空
            stats["dropped_count"] = len(elf_midpoints)
            return [], stats
        
        if not elf_midpoints or not elf_bond_pairs:
            # 无 ELF 数据，返回零填充
            stats["dropped_count"] = len(elf_midpoints)
            return [0.0] * len(bond_indices), stats

        # 构建 ELF 数据字典：key = 无向排序键, value = midpoint
        elf_dict = {}
        invalid_or_unusable_elf_rows = 0
        for pair, midpoint in zip(elf_bond_pairs, elf_midpoints):
            try:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    invalid_or_unusable_elf_rows += 1
                    logger.warning(f"[{molecule_id}] Invalid ELF bond pair format: {pair}")
                    continue
                i, j = int(pair[0]), int(pair[1])
                canonical = (min(i, j), max(i, j))
                if canonical in elf_dict:
                    logger.warning(f"[{molecule_id}] Duplicate ELF bond: {canonical}")
                elf_dict[canonical] = float(midpoint)
            except Exception:
                invalid_or_unusable_elf_rows += 1
                logger.warning(f"[{molecule_id}] Failed to parse ELF bond pair: {pair}")

        # elf_midpoints 和 elf_bond_pairs 长度不一致时，超出的 midpoint 视为 dropped
        if len(elf_midpoints) > len(elf_bond_pairs):
            invalid_or_unusable_elf_rows += len(elf_midpoints) - len(elf_bond_pairs)

        # 按 bond_indices 顺序重建 midpoints
        aligned_midpoints = []
        for bond in bond_indices:
            if not isinstance(bond, (list, tuple)) or len(bond) != 2:
                aligned_midpoints.append(0.0)
                continue
            canonical = (min(bond[0], bond[1]), max(bond[0], bond[1]))
            if canonical in elf_dict:
                aligned_midpoints.append(elf_dict[canonical])
                stats["aligned_count"] += 1
            else:
                # bond_indices 中有，但 ELF 中没有 -> 填充 0.0
                aligned_midpoints.append(0.0)
                logger.debug(f"[{molecule_id}] Bond {bond} not found in ELF, filling 0.0")
        
        # 计算 dropped（ELF 中有，但 bond_indices 中没有）
        bond_indices_set = {
            (min(b[0], b[1]), max(b[0], b[1])) for b in bond_indices
        }
        for canonical in elf_dict:
            if canonical not in bond_indices_set:
                stats["dropped_count"] += 1

        stats["dropped_count"] += invalid_or_unusable_elf_rows

        if stats["dropped_count"] > 0 or stats["aligned_count"] < len(bond_indices):
            logger.info(
                f"[{molecule_id}] ELF alignment: "
                f"raw={stats['raw_count']}, "
                f"aligned={stats['aligned_count']}, "
                f"dropped={stats['dropped_count']}, "
                f"target={len(bond_indices)}"
            )
        
        return aligned_midpoints, stats

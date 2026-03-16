"""
CPU 特征提取模块
从波函数文件中提取量子化学特征
"""

import os
import json
import subprocess
import tempfile
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
            
            # 重建 mf 对象
            mf = dft.RKS(mol)
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

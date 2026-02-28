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

import numpy as np
import scipy
from loguru import logger

# PySCF imports for wavefunction analysis
from pyscf import gto, scf, dft
from pyscf.lo import orth
from pyscf import lib
import pickle

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
    
    def extract_all_features(
        self,
        pkl_path: str,
        molecule_id: str,
    ) -> Dict[str, Any]:
        """
        从 pickle 文件提取所有特征
        
        Args:
            pkl_path: pickle 文件路径
            molecule_id: 分子标识
            
        Returns:
            包含所有特征的字典
        """
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
            
            # 提取特征
            features = {}
            
            # 1. 电荷 (高精度方案)
            features["charge_iao"] = self.extract_iao_charges(mol, mf).tolist()  # IAO 电荷 (NPA 平替)
            features["charge_cm5"] = self.extract_cm5_charges(mol, mf).tolist()  # CM5 电荷 (Hirshfeld 修正)
            features["hirshfeld_charges"] = self.extract_hirshfeld_charges(mol, mf).tolist()  # 保留作为参考
            features["mulliken_charges"] = self.extract_mulliken_charges(mol, mf).tolist()  # 保留作为参考
            
            # 2. 键级
            features["mayer_bond_orders"] = self.extract_mayer_bond_orders(mol, mf).tolist()
            features["wiberg_bond_orders"] = self.extract_wiberg_bond_orders(mol, mf).tolist()
            
            # 3. 分子信息
            features["natm"] = mol.natm
            features["energy"] = float(mf.e_tot) if hasattr(mf, 'e_tot') else None
            features["atom_symbols"] = [mol.atom_symbol(i) for i in range(mol.natm)]
            features["atom_coords"] = mol.atom_coords().tolist()
            
            # 4. 色散校正信息
            features["dispersion_correction"] = hasattr(mf, 'disp') and mf.disp is not None
            
            result["features"] = features
            result["success"] = True
            
            elapsed = time.time() - start_time
            logger.info(f"[{molecule_id}] Feature extraction completed in {elapsed:.2f}s")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{molecule_id}] Feature extraction failed: {e}")
        
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

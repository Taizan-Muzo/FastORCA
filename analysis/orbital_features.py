"""
Milestone 2: Orbital Features Module
基于 PySCF IBO/IAO 的局域轨道特征提取

设计原则:
- 仅使用 PySCF 内置功能，不依赖外部程序
- 诚实命名：使用 ibo_/iao_ 前缀，metadata 中明确说明方法
- 软失败策略：开壳层或不支持的情况优雅降级
- 所有输出可序列化到 unified JSON
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from loguru import logger

# PySCF imports
from pyscf import gto, scf, dft
from pyscf.lo import iao, ibo as ibo_module
from pyscf.lo import orth


class OrbitalFeatureExtractor:
    """
    局域轨道特征提取器
    
    基于 IBO (Intrinsic Bonding Orbitals) 和 IAO (Intrinsic Atomic Orbitals)
    提取分子的局域轨道特征，用于描述化学键、孤对电子等。
    
    限制:
    - 仅支持闭壳层（Restricted RHF/RKS, spin=0, 所有 occupied MO occ=2.0）
    - 开壳层会 soft-fail
    """
    
    def __init__(self, verbose: int = 3):
        self.verbose = verbose
    
    def extract_orbital_features(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
    ) -> Dict[str, Any]:
        """
        提取局域轨道特征
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 SCF 对象（RHF/RKS）
            
        Returns:
            orbital_features 字典，符合 unified schema
            失败时返回部分字段为 null，并在 metadata 中记录 reason
        """
        start_time = time.time()
        
        result = self._init_orbital_features_skeleton()
        
        # Step 1: 闭壳层检查
        is_closed_shell, reason = self._check_closed_shell(mol, mf)
        if not is_closed_shell:
            result["metadata"]["extraction_status"] = "unavailable"
            result["metadata"]["failure_reason"] = reason
            logger.warning(f"Orbital features unavailable: {reason}")
            return result
        
        try:
            # Step 2: 构建 IAO
            logger.debug("Building IAO basis...")
            mo_occ = mf.mo_coeff[:, mf.mo_occ > 0]
            iao_coeff = iao.iao(mol, mo_occ)
            n_iao = iao_coeff.shape[1]
            
            # 获取 IAO 原子映射
            ref_mol = iao.reference_mol(mol)
            iao_atom_mapping = self._build_iao_atom_mapping(ref_mol, n_iao)
            result["iao_atom_mapping"] = iao_atom_mapping
            
            # Step 3: 构建 IBO
            logger.debug("Building IBOs...")
            s = mol.intor_symmetric('int1e_ovlp')
            ibo_coeff = ibo_module.ibo(
                mol, 
                mo_occ, 
                locmethod='IBO',
                iaos=iao_coeff,
                s=s,
                verbose=0  # 减少输出
            )
            
            n_ibo = ibo_coeff.shape[1]
            result["ibo_count"] = n_ibo
            result["local_orbital_method"] = "IBO"
            
            # Step 4: 计算 Occupancy（闭壳层近似）
            occupancies = self._compute_occupancies(n_ibo, closed_shell=True)
            result["ibo_occupancies"] = occupancies
            
            # Step 5: 计算 Atom Contributions（基于 IAO basis）
            logger.debug("Computing atom contributions...")
            contributions = self._compute_atom_contributions(
                ibo_coeff, iao_coeff, iao_atom_mapping, mol.natm
            )
            result["ibo_atom_contributions"] = contributions
            
            # Step 6: 计算 Orbital Centers
            logger.debug("Computing orbital centers...")
            centers = self._compute_orbital_centers(
                contributions, mol.atom_coords(unit='A')
            )
            result["ibo_centers_angstrom"] = centers
            
            # Step 7: 计算 Locality Score
            logger.debug("Computing locality scores...")
            locality_scores = self._compute_locality_scores(contributions)
            result["orbital_locality_score"] = locality_scores
            
            # Step 8: Heuristic 分类
            logger.debug("Classifying orbitals...")
            classifications = self._classify_orbitals_heuristic(contributions)
            result["ibo_class_heuristic"] = classifications
            
            # 验证
            if not self._validate_orbital_features(result, mol.natm):
                result["metadata"]["extraction_status"] = "unavailable"
                result["metadata"]["failure_reason"] = "validation_failed"
                logger.warning("Orbital features validation failed")
            else:
                result["metadata"]["extraction_status"] = "success"
                elapsed = time.time() - start_time
                result["metadata"]["extraction_time_seconds"] = elapsed
                logger.info(f"Orbital features extracted: {n_ibo} IBOs in {elapsed:.3f}s")
            
        except Exception as e:
            result["metadata"]["extraction_status"] = "unavailable"
            result["metadata"]["failure_reason"] = f"exception: {str(e)}"
            logger.error(f"Orbital features extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return result
    
    def _init_orbital_features_skeleton(self) -> Dict[str, Any]:
        """初始化 orbital_features 骨架"""
        return {
            "local_orbital_method": None,
            "ibo_count": None,
            "ibo_occupancies": None,
            "ibo_centers_angstrom": None,
            "ibo_atom_contributions": None,
            "ibo_class_heuristic": None,
            "orbital_locality_score": None,
            "iao_atom_mapping": None,
            "metadata": {
                "coefficient_basis": "IAO",
                "coefficient_metric": "IAO basis representation with Löwdin orthogonalization; atom contribution defined as sum of squared coefficients over atomic IAO subsets",
                "center_definition": "weighted average of atomic coordinates using normalized per-atom IAO orbital contributions",
                "contribution_definition": "sum of squared coefficients over atomic basis functions in IAO representation",
                "occupancy_definition": "2.0 for closed shell (alpha+beta), closed_shell_assumption",
                "locality_score_formula": "1 / participation_ratio, where PR = 1 / sum(p_A^2)",
                "classification_is_heuristic": True,
                "heuristic_classification_rules": "lone_pair_like: p1>0.75 && p2<0.20; bond_like: p1+p2>0.80 && p1>0.25 && p2>0.25; delocalized_like: otherwise",
                "extraction_status": "not_attempted",
                "skip_reason": None,
                "failure_reason": None,
                "extraction_time_seconds": None,
                "pyscf_version": self._get_pyscf_version(),
            }
        }
    
    def _get_pyscf_version(self) -> str:
        """获取 PySCF 版本"""
        try:
            import pyscf
            return pyscf.__version__
        except:
            return "unknown"
    
    def _check_closed_shell(self, mol: gto.Mole, mf: scf.hf.SCF) -> Tuple[bool, str]:
        """
        检查是否为闭壳层
        
        判断标准:
        1. mf 是 RHF/RKS 类型（不是 UHF/UKS/ROHF）
        2. mol.spin == 0
        3. 占据模式是闭壳层（所有 occupied MO 的 occ == 2）
        
        Returns:
            (is_closed_shell, reason_if_not)
        """
        # 检查 1: 必须是 Restricted 类型
        is_restricted = isinstance(mf, (scf.hf.RHF, dft.rks.RKS))
        if not is_restricted:
            return False, "orbital_features_not_supported_for_open_shell_yet: not_restricted_calculation"
        
        # 检查 2: spin == 0
        if mol.spin != 0:
            return False, "orbital_features_not_supported_for_open_shell_yet: spin_not_zero"
        
        # 检查 3: 闭壳层占据模式
        occ = mf.mo_occ
        occ_mask = occ > 0
        if not np.any(occ_mask):
            return False, "no_occupied_orbitals"
        
        occupied_values = occ[occ_mask]
        if not np.allclose(occupied_values, 2.0):
            return False, "orbital_features_not_supported_for_open_shell_yet: non_closed_shell_occupation"
        
        return True, ""
    
    def _build_iao_atom_mapping(self, ref_mol: gto.Mole, n_iao: int) -> List[int]:
        """
        构建 IAO 到原子的映射
        
        Args:
            ref_mol: IAO reference molecule
            n_iao: IAO 数量
            
        Returns:
            iao_atom_mapping: iao_index -> atom_index
        """
        ao_slices = ref_mol.aoslice_by_atom()
        mapping = [-1] * n_iao  # 初始化
        
        for atom_idx in range(ref_mol.natm):
            p0, p1 = ao_slices[atom_idx, 2:]
            for iao_idx in range(p0, p1):
                mapping[iao_idx] = atom_idx
        
        # 验证
        if -1 in mapping:
            logger.warning("Some IAOs not mapped to any atom")
        
        return mapping
    
    def _compute_occupancies(self, n_ibo: int, closed_shell: bool = True) -> List[float]:
        """
        计算局域轨道占据数
        
        闭壳层假设：每个 occupied orbital 的 occupancy = 2.0
        """
        if closed_shell:
            return [2.0] * n_ibo
        else:
            # 开壳层暂不支持，返回 null
            return None
    
    def _compute_atom_contributions(
        self,
        ibo_coeff: np.ndarray,
        iao_coeff: np.ndarray,
        iao_atom_mapping: List[int],
        natm: int
    ) -> List[List[float]]:
        """
        计算每个 IBO 在每个原子上的贡献
        
        方法：求解线性方程组 IBO = IAO @ X，得到 X 是 IBO 在 IAO basis 下的系数
        然后按原子分组求系数平方和
        """
        nao, nibo = ibo_coeff.shape
        niao = iao_coeff.shape[1]
        
        # 求解最小二乘问题：iao_coeff @ X = ibo_coeff
        # X = (IAO^T @ IAO)^{-1} @ IAO^T @ IBO
        iao_iao = iao_coeff.T @ iao_coeff
        iao_ibo = iao_coeff.T @ ibo_coeff
        
        try:
            X = np.linalg.solve(iao_iao, iao_ibo)
        except np.linalg.LinAlgError:
            X = np.linalg.lstsq(iao_coeff, ibo_coeff, rcond=None)[0]
        
        # 现在 X[mu, k] 是第 k 个 IBO 在第 mu 个 IAO 上的系数
        contributions = []
        for k in range(nibo):
            coeffs_k = X[:, k]
            
            # 按原子分组求平方和
            atom_weights = np.zeros(natm)
            for iao_idx, atom_idx in enumerate(iao_atom_mapping):
                if atom_idx >= 0:
                    atom_weights[atom_idx] += np.abs(coeffs_k[iao_idx]) ** 2
            
            # 归一化
            total_weight = np.sum(atom_weights)
            if total_weight > 1e-10:
                atom_weights = atom_weights / total_weight
            
            contributions.append(atom_weights.tolist())
        
        return contributions
    
    def _compute_orbital_centers(
        self,
        contributions: List[List[float]],
        atom_coords: np.ndarray
    ) -> List[List[float]]:
        """
        计算每个局域轨道的中心位置
        
        center_k = sum_A w_{kA} * R_A
        """
        centers = []
        for contrib in contributions:
            center = np.dot(contrib, atom_coords)
            centers.append(center.tolist())
        return centers
    
    def _compute_locality_scores(self, contributions: List[List[float]]) -> List[float]:
        """
        计算局域性分数
        
        locality_score = 1 / participation_ratio = sum(p_A^2)
        """
        scores = []
        for contrib in contributions:
            p = np.array(contrib)
            if np.sum(p**2) < 1e-10:
                scores.append(0.0)
            else:
                pr = 1.0 / np.sum(p**2)
                locality = 1.0 / pr
                scores.append(float(locality))
        return scores
    
    def _classify_orbitals_heuristic(
        self,
        contributions: List[List[float]]
    ) -> List[Optional[str]]:
        """
        Heuristic 分类局域轨道
        
        规则（按优先级）:
        1. lone_pair_like: 主原子贡献 > 0.75 且次原子 < 0.20
        2. bond_like: 前两大原子贡献和 > 0.80 且各自 > 0.25
        3. delocalized_like: 以上都不满足
        """
        classifications = []
        for contrib in contributions:
            sorted_contrib = sorted(contrib, reverse=True)
            
            if len(sorted_contrib) < 2:
                classifications.append("delocalized_like")
                continue
            
            p1, p2 = sorted_contrib[0], sorted_contrib[1]
            
            if p1 > 0.75 and p2 < 0.20:
                classifications.append("lone_pair_like")
            elif p1 + p2 > 0.80 and p1 > 0.25 and p2 > 0.25:
                classifications.append("bond_like")
            else:
                classifications.append("delocalized_like")
        
        return classifications
    
    def _validate_orbital_features(self, data: Dict[str, Any], natm: int) -> bool:
        """
        验证 orbital_features 的 shape 和数值正确性
        """
        try:
            n_ibo = data.get("ibo_count")
            if n_ibo is None or n_ibo <= 0:
                logger.warning("Invalid ibo_count")
                return False
            
            # 检查 occupancies
            occ = data.get("ibo_occupancies")
            if occ is not None and len(occ) != n_ibo:
                logger.warning(f"Occupancies length mismatch: {len(occ)} vs {n_ibo}")
                return False
            
            # 检查 centers
            centers = data.get("ibo_centers_angstrom")
            if centers is not None:
                if len(centers) != n_ibo:
                    logger.warning(f"Centers length mismatch: {len(centers)} vs {n_ibo}")
                    return False
                for i, c in enumerate(centers):
                    if len(c) != 3:
                        logger.warning(f"Center {i} dimension != 3: {len(c)}")
                        return False
            
            # 检查 contributions
            contribs = data.get("ibo_atom_contributions")
            if contribs is not None:
                if len(contribs) != n_ibo:
                    logger.warning(f"Contributions length mismatch: {len(contribs)} vs {n_ibo}")
                    return False
                for i, c in enumerate(contribs):
                    if len(c) != natm:
                        logger.warning(f"Contribution {i} length != natm: {len(c)} vs {natm}")
                        return False
                    total = sum(c)
                    if abs(total - 1.0) > 1e-4:
                        logger.warning(f"Contribution {i} not normalized: sum={total}")
                        return False
            
            # 检查 locality_score
            scores = data.get("orbital_locality_score")
            if scores is not None and len(scores) != n_ibo:
                logger.warning(f"Locality scores length mismatch: {len(scores)} vs {n_ibo}")
                return False
            
            # 检查 classification
            classes = data.get("ibo_class_heuristic")
            if classes is not None and len(classes) != n_ibo:
                logger.warning(f"Classifications length mismatch: {len(classes)} vs {n_ibo}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False


def extract_orbital_features(mol: gto.Mole, mf: scf.hf.SCF) -> Dict[str, Any]:
    """
    便捷函数：提取局域轨道特征
    
    Args:
        mol: PySCF Mole 对象
        mf: 已收敛的 SCF 对象
        
    Returns:
        orbital_features 字典
    """
    extractor = OrbitalFeatureExtractor()
    return extractor.extract_orbital_features(mol, mf)

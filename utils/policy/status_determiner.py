"""
状态判定器

根据 unified schema 数据判定 overall_status，严格按优先级顺序。
"""

from typing import Dict, Any, List, Tuple, Optional
from .reason_codes import REASON_REGISTRY
from .field_tiers import FieldTierChecker


class StatusDeterminer:
    """状态判定器"""
    
    # 状态优先级（降序）
    STATUS_PRIORITY = [
        "invalid_input",
        "failed_geometry", 
        "failed_scf",
        "failed_core_features",
        "core_success_partial_features",
        "fully_success"
    ]
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.reason_codes: List[str] = []
    
    def determine(self) -> str:
        """
        判定 overall_status（严格优先级顺序）
        
        Returns:
            overall_status 字符串
        """
        self.reason_codes = []
        
        # === Level 1: invalid_input ===
        status = self._check_invalid_input()
        if status:
            return status
        
        # === Level 2: failed_geometry ===
        status = self._check_failed_geometry()
        if status:
            return status
        
        # === Level 3: failed_scf ===
        status = self._check_failed_scf()
        if status:
            return status
        
        # === Level 4: failed_core_features ===
        status = self._check_failed_core_features()
        if status:
            return status
        
        # === Level 5 & 6: partial vs fully_success ===
        return self._check_partial_or_full()
    
    def get_reason_codes(self) -> List[str]:
        """获取判定过程中收集的 reason codes"""
        return self.reason_codes.copy()
    
    def _get(self, path: str) -> Any:
        """按点分隔路径获取值"""
        keys = path.split(".")
        value = self.data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _check_invalid_input(self) -> Optional[str]:
        """检查 Level 1: invalid_input"""
        calc_status = self._get("calculation_status")
        if not calc_status:
            self.reason_codes.append("invalid_smiles")
            return "invalid_input"
        
        if calc_status.get("invalid_input", False):
            self.reason_codes.append("invalid_smiles")
            return "invalid_input"
        
        # 检查基本字段
        smiles = self._get("molecule_info.smiles")
        if not smiles:
            self.reason_codes.append("invalid_smiles")
            return "invalid_input"
        
        natm = self._get("molecule_info.natm")
        if natm is None or natm == 0:
            self.reason_codes.append("zero_atom_molecule")
            return "invalid_input"
        
        if natm > 1000:  # 硬限制
            self.reason_codes.append("atom_limit_exceeded_input")
            return "invalid_input"
        
        return None
    
    def _check_failed_geometry(self) -> Optional[str]:
        """检查 Level 2: failed_geometry"""
        geo_success = self._get("calculation_status.geometry_optimization_success")
        
        if geo_success is False:
            self.reason_codes.append("geometry_optimization_failed")
            return "failed_geometry"
        
        return None
    
    def _check_failed_scf(self) -> Optional[str]:
        """检查 Level 3: failed_scf"""
        calc_status = self._get("calculation_status")
        dft = self._get("global_features.dft") or {}
        
        wf_loaded = calc_status.get("wavefunction_load_success", False)
        scf_converged = dft.get("scf_converged", False)
        
        if not wf_loaded:
            self.reason_codes.append("wavefunction_corrupted")
            return "failed_scf"
        
        if not scf_converged:
            self.reason_codes.append("scf_not_converged")
            return "failed_scf"
        
        return None
    
    def _check_failed_core_features(self) -> Optional[str]:
        """检查 Level 4: failed_core_features"""
        checker = FieldTierChecker()
        
        # 基本能量信息
        total_energy = self._get("global_features.dft.total_energy_hartree")
        if total_energy is None:
            self.reason_codes.append("missing_total_energy")
            return "failed_core_features"
        
        # 使用 FieldTierChecker 检查核心字段
        core_ok, failed_field = checker.check_core_required(self.data)
        if not core_ok:
            if "charge" in failed_field:
                self.reason_codes.append("no_valid_atomic_charges")
            elif "bond" in failed_field:
                self.reason_codes.append("no_valid_bond_orders")
            else:
                # 通用 core 失败
                self.reason_codes.append("no_valid_atomic_charges")
            return "failed_core_features"
        
        return None
    
    def _check_partial_or_full(self) -> str:
        """检查 Level 5 & 6: partial vs fully_success"""
        checker = FieldTierChecker()
        
        # 检查 strongly_recommended 规则
        violation_count, violations = checker.check_strongly_recommended(self.data)
        
        # 检查插件失败（只统计应执行但失败的）
        plugin_failures = self._count_plugin_failures()
        
        # M5: 检查 validation errors (length/shape mismatch)
        validation_errors = self.data.get("_validation_errors", [])
        for error in validation_errors:
            if error not in self.reason_codes:
                self.reason_codes.append(error)
        
        if violation_count > 0 or plugin_failures > 0 or len(validation_errors) > 0:
            # 收集具体的 reason codes
            self._collect_strongly_recommended_reasons(violations)
            return "core_success_partial_features"
        
        return "fully_success"
    
    def _count_plugin_failures(self) -> int:
        """
        统计"应执行但失败"的插件数量
        
        注意：这里只检查 extraction_status，实际应在调用时结合 config 判断
        此处假设：metadata.extraction_status == "failed" 表示已尝试但失败
        """
        count = 0
        
        plugins = ["orbital_features", "realspace_features"]
        for plugin in plugins:
            metadata = self._get(f"{plugin}.metadata") or {}
            status = metadata.get("extraction_status")
            
            if status in ("failed", "timeout", "error"):
                failure_reason = metadata.get("failure_reason")
                if failure_reason and failure_reason not in self.reason_codes:
                    self.reason_codes.append(failure_reason)
                count += 1
        
        # external_bridge
        bridge = self._get("external_bridge.critic2") or {}
        bridge_status = bridge.get("execution_status")
        if bridge_status in ("failed", "timeout", "error"):
            failure_reason = bridge.get("failure_reason")
            if failure_reason and failure_reason not in self.reason_codes:
                self.reason_codes.append(failure_reason)
            count += 1
        
        return count
    
    def _collect_strongly_recommended_reasons(self, violations: List[str]):
        """根据 strongly_recommended 违规收集 reason codes"""
        violation_to_reason = {
            "frontier_orbital_complete": None,  # 这是推荐但不触发特定 reason
            "extended_charges_coverage": None,
            "extended_bond_orders": None,
            "elf_atomic_coverage": None,
            "dipole_moment": None,
        }
        
        # 这些违规不会直接导致特定的 reason code，
        # 它们只是影响 overall_status 判定为 partial
        # 实际 plugin 的 failure_reason 已经在 _count_plugin_failures 中收集
        pass


class MoleculeResult:
    """单分子处理结果"""
    
    def __init__(
        self,
        molecule_id: str,
        overall_status: str,
        reason_codes: List[str],
        data: Dict[str, Any],
        wall_time_seconds: float = 0.0
    ):
        self.molecule_id = molecule_id
        self.overall_status = overall_status
        self.reason_codes = reason_codes
        self.data = data
        self.wall_time_seconds = wall_time_seconds
    
    def has_hard_fail(self) -> bool:
        """是否包含 hard_fail"""
        from .reason_codes import is_hard_fail
        return any(is_hard_fail(code) for code in self.reason_codes)
    
    def has_soft_fail(self) -> bool:
        """是否包含 soft_fail（但不包含 hard_fail）"""
        from .reason_codes import is_soft_fail, is_hard_fail
        has_soft = any(is_soft_fail(code) for code in self.reason_codes)
        has_hard = any(is_hard_fail(code) for code in self.reason_codes)
        return has_soft and not has_hard
    
    def has_only_skip(self) -> bool:
        """是否只有 skip/info（无 fail）"""
        from .reason_codes import is_hard_fail, is_soft_fail
        return not any(is_hard_fail(code) or is_soft_fail(code) 
                      for code in self.reason_codes)
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "molecule_id": self.molecule_id,
            "overall_status": self.overall_status,
            "reason_codes": self.reason_codes,
            "wall_time_seconds": self.wall_time_seconds,
            "has_hard_fail": self.has_hard_fail(),
            "has_soft_fail": self.has_soft_fail(),
        }

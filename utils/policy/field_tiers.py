"""
字段分级检查器

将 unified schema 字段分为 core_required / strongly_recommended / optional 三级。
"""

from typing import Dict, Any, List, Callable, Optional


# === Core Required 字段列表 ===
# 缺失任一即判定为 failed_core_features
CORE_REQUIRED_FIELDS = [
    "molecule_info.molecule_id",
    "molecule_info.natm",
    "calculation_status.wavefunction_load_success",
    "global_features.dft.scf_converged",
    "geometry.atom_symbols",
    "geometry.atom_coords_angstrom",
    "global_features.dft.total_energy_hartree",
]

# Core 电荷字段：至少一个成功且长度匹配即满足
CORE_CHARGE_FIELDS = [
    "atom_features.charge_mulliken",
    "atom_features.charge_hirshfeld",
    "atom_features.charge_iao",
    "atom_features.charge_cm5",
]

# Core 键级字段：多原子分子至少一个成功
CORE_BOND_ORDER_FIELDS = [
    "bond_features.bond_orders_wiberg",
    "bond_features.bond_orders_mayer",
]


# === Strongly Recommended 规则组 ===
# 按规则组检查，非简单计数

class StronglyRecommendedRule:
    """强推荐规则定义"""
    
    def __init__(
        self,
        name: str,
        check_fn: Callable[[Dict[str, Any]], bool],
        description: str
    ):
        self.name = name
        self.check_fn = check_fn
        self.description = description
    
    def check(self, data: Dict[str, Any]) -> bool:
        """返回 True 表示规则满足，False 表示违规"""
        try:
            return self.check_fn(data)
        except Exception:
            return False


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """按点分隔路径获取嵌套字典值"""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def _check_field_exists(data: Dict[str, Any], path: str) -> bool:
    """检查字段是否存在且非 None"""
    value = _get_nested_value(data, path)
    return value is not None


def _check_array_length(data: Dict[str, Any], path: str, expected_path: str) -> bool:
    """检查数组字段长度是否匹配预期"""
    arr = _get_nested_value(data, path)
    if arr is None:
        return False
    if not isinstance(arr, (list, tuple)):
        return False
    
    expected_len = _get_nested_value(data, expected_path)
    if expected_len is None:
        return False
    
    return len(arr) == expected_len


# 定义 Strongly Recommended 规则组
STRONGLY_RECOMMENDED_RULES: List[StronglyRecommendedRule] = [
    # Rule 1: Frontier Orbital 完整性
    # 要求：HOMO 或 LUMO 至少一个存在
    StronglyRecommendedRule(
        name="frontier_orbital_complete",
        check_fn=lambda d: (
            _check_field_exists(d, "global_features.dft.homo_energy_hartree") or
            _check_field_exists(d, "global_features.dft.lumo_energy_hartree")
        ),
        description="HOMO 或 LUMO 能量至少一个存在"
    ),
    
    # Rule 2: 扩展电荷覆盖
    # 要求：除 core 的 Mulliken/IAO 外，CM5 或 Hirshfeld 至少一个成功
    StronglyRecommendedRule(
        name="extended_charges_coverage",
        check_fn=lambda d: (
            _check_array_length(d, "atom_features.charge_cm5", "molecule_info.natm") or
            _check_array_length(d, "atom_features.charge_hirshfeld", "molecule_info.natm")
        ),
        description="CM5 或 Hirshfeld 电荷至少一个成功"
    ),
    
    # Rule 3: 扩展键级覆盖
    # 要求：Mayer 成功（Wiberg 已在 core 要求，多原子时检查）
    StronglyRecommendedRule(
        name="extended_bond_orders",
        check_fn=lambda d: (
            # 单原子跳过
            (_get_nested_value(d, "molecule_info.natm") or 0) <= 1 or
            _check_field_exists(d, "bond_features.bond_orders_mayer")
        ),
        description="多原子分子有 Mayer 键级"
    ),
    
    # Rule 4: ELF 特征
    # 要求：原子级 ELF 值存在
    StronglyRecommendedRule(
        name="elf_atomic_coverage",
        check_fn=lambda d: _check_array_length(
            d, "atom_features.elf_value", "molecule_info.natm"
        ),
        description="原子级 ELF 值成功计算"
    ),
    
    # Rule 5: 偶极矩信息
    StronglyRecommendedRule(
        name="dipole_moment",
        check_fn=lambda d: _check_field_exists(
            d, "global_features.dft.dipole_moment_debye"
        ),
        description="偶极矩计算成功"
    ),
]


class FieldTierChecker:
    """字段分级检查器"""
    
    @staticmethod
    def check_core_required(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        检查 Core Required 字段
        
        Returns:
            (是否全部满足, 第一个缺失的字段路径)
        """
        for field_path in CORE_REQUIRED_FIELDS:
            if not _check_field_exists(data, field_path):
                return False, field_path
        
        # 检查原子电荷：至少一个成功且长度匹配
        natm = _get_nested_value(data, "molecule_info.natm")
        if natm is None or natm == 0:
            return False, "molecule_info.natm"
        
        valid_charge = False
        for charge_path in CORE_CHARGE_FIELDS:
            if _check_array_length(data, charge_path, "molecule_info.natm"):
                valid_charge = True
                break
        
        if not valid_charge:
            return False, "atom_features.charge_* (all failed)"
        
        # 检查键级：多原子分子至少一个成功
        if natm > 1:
            valid_bond = False
            for bond_path in CORE_BOND_ORDER_FIELDS:
                bond_data = _get_nested_value(data, bond_path)
                if bond_data is not None and len(bond_data) > 0:
                    valid_bond = True
                    break
            
            if not valid_bond:
                return False, "bond_features.bond_orders_* (all failed)"
        
        return True, None
    
    @staticmethod
    def check_strongly_recommended(data: Dict[str, Any]) -> tuple[int, List[str]]:
        """
        检查 Strongly Recommended 规则组
        
        Returns:
            (违规规则数, 违规规则名称列表)
        """
        violations = []
        for rule in STRONGLY_RECOMMENDED_RULES:
            if not rule.check(data):
                violations.append(rule.name)
        
        return len(violations), violations
    
    @staticmethod
    def get_core_status_detail(data: Dict[str, Any]) -> Dict[str, Any]:
        """获取 Core 检查详细状态"""
        natm = _get_nested_value(data, "molecule_info.natm") or 0
        
        # 检查各字段
        field_status = {}
        for field_path in CORE_REQUIRED_FIELDS:
            field_status[field_path] = _check_field_exists(data, field_path)
        
        # 检查电荷
        charge_status = {}
        valid_charge_count = 0
        for charge_path in CORE_CHARGE_FIELDS:
            valid = _check_array_length(data, charge_path, "molecule_info.natm")
            charge_status[charge_path] = valid
            if valid:
                valid_charge_count += 1
        
        # 检查键级
        bond_status = {}
        valid_bond_count = 0
        for bond_path in CORE_BOND_ORDER_FIELDS:
            bond_data = _get_nested_value(data, bond_path)
            valid = bond_data is not None and len(bond_data) > 0
            bond_status[bond_path] = valid
            if valid:
                valid_bond_count += 1
        
        return {
            "required_fields": field_status,
            "charge_fields": charge_status,
            "valid_charge_count": valid_charge_count,
            "bond_order_fields": bond_status,
            "valid_bond_count": valid_bond_count,
            "natm": natm,
            "is_single_atom": natm == 1
        }

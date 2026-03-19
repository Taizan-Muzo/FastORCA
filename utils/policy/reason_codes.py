"""
统一 Reason Code 注册表

所有失败/跳过/partial 必须使用本 registry 中的 machine-readable code。
"""

from typing import Dict, Any, Optional

REASON_REGISTRY: Dict[str, Dict[str, Any]] = {
    # === Input/Geometry Phase (hard_fail) ===
    "invalid_smiles": {
        "scope": "input",
        "severity": "hard_fail",
        "description": "RDKit 无法解析输入 SMILES"
    },
    "zero_atom_molecule": {
        "scope": "input",
        "severity": "hard_fail",
        "description": "解析后分子不含任何原子"
    },
    "atom_limit_exceeded_input": {
        "scope": "input",
        "severity": "hard_fail",
        "description": "原子数超过硬限制（默认1000）"
    },
    "geometry_optimization_failed": {
        "scope": "geometry",
        "severity": "hard_fail",
        "description": "xTB 或 PySCF 几何优化不收敛或报错"
    },
    "conformer_generation_failed": {
        "scope": "geometry",
        "severity": "hard_fail",
        "description": "RDKit ETKDG 构象生成失败"
    },
    
    # === SCF Phase (hard_fail) ===
    "scf_not_converged": {
        "scope": "scf",
        "severity": "hard_fail",
        "description": "自洽场迭代未收敛"
    },
    "wavefunction_corrupted": {
        "scope": "scf",
        "severity": "hard_fail",
        "description": "PKL 文件损坏或无法加载"
    },
    "gpu_kernel_failed": {
        "scope": "scf",
        "severity": "hard_fail",
        "description": "GPU CUDA kernel 执行失败，回退 CPU 也失败"
    },
    
    # === Core Features Phase (hard_fail) ===
    "missing_total_energy": {
        "scope": "core_features",
        "severity": "hard_fail",
        "description": "无法获取 DFT 总能量"
    },
    "no_valid_atomic_charges": {
        "scope": "core_features",
        "severity": "hard_fail",
        "description": "所有原子电荷方法均失败或长度不匹配"
    },
    "no_valid_bond_orders": {
        "scope": "core_features",
        "severity": "hard_fail",
        "description": "多原子分子无有效键级数据"
    },
    
    # === Plugin: Orbital Features (soft_fail) ===
    "orbital_features_timeout": {
        "scope": "orbital_features",
        "severity": "soft_fail",
        "description": "轨道特征计算超时"
    },
    "orbital_features_atom_limit": {
        "scope": "orbital_features",
        "severity": "soft_fail",
        "description": "原子数超过 orbital_features.max_atoms 限制"
    },
    "open_shell_not_supported": {
        "scope": "orbital_features",
        "severity": "soft_fail",
        "description": "开壳层分子暂不支持 IBO 计算"
    },
    "iao_orthogonalization_failed": {
        "scope": "orbital_features",
        "severity": "soft_fail",
        "description": "IAO 正交化过程数值不稳定"
    },
    "orbital_features_memory_exceeded": {
        "scope": "orbital_features",
        "severity": "soft_fail",
        "description": "轨道特征计算内存不足"
    },
    
    # === Plugin: Realspace Features (soft_fail) ===
    "realspace_timeout": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "实空间特征计算超时"
    },
    "realspace_atom_limit": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "原子数超过 realspace_features.max_atoms 限制"
    },
    "realspace_grid_limit": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "预估网格点数超过 max_total_grid_points"
    },
    "realspace_memory_exceeded": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "Cube 生成内存不足"
    },
    "realspace_core_failed": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "realspace core 层计算失败"
    },
    "realspace_extended_failed": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "realspace extended 层计算失败"
    },
    "cube_file_corrupted": {
        "scope": "realspace_features",
        "severity": "soft_fail",
        "description": "生成的 cube 文件无法读取"
    },
    
    # === Plugin: External Bridge (soft_fail) ===
    "external_tool_not_found": {
        "scope": "external_bridge",
        "severity": "soft_fail",
        "description": "外部工具可执行文件未找到"
    },
    "external_tool_timeout": {
        "scope": "external_bridge",
        "severity": "soft_fail",
        "description": "外部工具执行超时"
    },
    "external_tool_nonzero_exit": {
        "scope": "external_bridge",
        "severity": "soft_fail",
        "description": "外部工具返回非零退出码"
    },
    "external_output_parse_failed": {
        "scope": "external_bridge",
        "severity": "soft_fail",
        "description": "外部工具输出解析失败"
    },
    
    # === Skip/Info (skip / info) ===
    "fast_mode_skip": {
        "scope": "policy",
        "severity": "skip",
        "description": "Fast 模式下策略性跳过非必要计算"
    },
    "plugin_disabled_by_config": {
        "scope": "policy",
        "severity": "skip",
        "description": "配置中显式禁用该插件"
    },
    "plugin_disabled_by_mode": {
        "scope": "policy",
        "severity": "skip",
        "description": "当前运行模式禁用了该插件"
    },
    "atom_limit_skip": {
        "scope": "policy",
        "severity": "skip",
        "description": "原子数超过插件限制，策略性跳过"
    },
    "artifact_cleaned_by_policy": {
        "scope": "policy",
        "severity": "info",
        "description": "根据 artifact 保留策略清理文件"
    },
    
    # === Feature Validation Errors ===
    "bond_feature_length_mismatch_elf_bond_midpoint": {
        "scope": "validation",
        "severity": "soft_fail",
        "description": "ELF bond midpoint 数组长度与 bond_indices 不匹配"
    },
    "atom_feature_length_mismatch": {
        "scope": "validation",
        "severity": "soft_fail",
        "description": "原子特征数组长度与 natm 不匹配"
    },
    "bond_feature_length_mismatch": {
        "scope": "validation",
        "severity": "soft_fail",
        "description": "键特征数组长度与 bond_indices 不匹配"
    },
    "elf_alignment_partial": {
        "scope": "validation",
        "severity": "soft_fail",
        "description": "ELF bond midpoints 与 bond_indices 部分对齐，部分键缺失"
    },
    
    # === Plugin Interface Errors ===
    "plugin_api_mismatch": {
        "scope": "plugin",
        "severity": "soft_fail",
        "description": "插件接口参数不匹配"
    },
    "plugin_execution_error": {
        "scope": "plugin",
        "severity": "soft_fail",
        "description": "插件执行过程中发生异常"
    },
    "plugin_subprocess_error": {
        "scope": "plugin",
        "severity": "soft_fail",
        "description": "插件子进程通信失败"
    },
}


def get_reason_info(code: str) -> Optional[Dict[str, Any]]:
    """获取 reason code 的详细信息"""
    return REASON_REGISTRY.get(code)


def is_hard_fail(code: str) -> bool:
    """检查是否为 hard_fail 级别"""
    info = get_reason_info(code)
    return info is not None and info.get("severity") == "hard_fail"


def is_soft_fail(code: str) -> bool:
    """检查是否为 soft_fail 级别"""
    info = get_reason_info(code)
    return info is not None and info.get("severity") == "soft_fail"


def is_skip(code: str) -> bool:
    """检查是否为 skip 级别"""
    info = get_reason_info(code)
    return info is not None and info.get("severity") in ("skip", "info")


def validate_reason_code(code: str) -> bool:
    """验证 reason code 是否存在于 registry 中"""
    return code in REASON_REGISTRY


def get_all_reason_codes_by_severity(severity: str) -> list:
    """获取指定 severity 的所有 reason codes"""
    return [
        code for code, info in REASON_REGISTRY.items()
        if info.get("severity") == severity
    ]


def get_all_reason_codes_by_scope(scope: str) -> list:
    """获取指定 scope 的所有 reason codes"""
    return [
        code for code, info in REASON_REGISTRY.items()
        if info.get("scope") == scope
    ]

"""
Policy 层模块包

包含字段分级、状态判定、Reason Code 管理、Artifact 生命周期等策略逻辑。
"""

from .reason_codes import REASON_REGISTRY, get_reason_info, is_hard_fail, is_soft_fail, is_skip
from .field_tiers import FieldTierChecker, CORE_REQUIRED_FIELDS, STRONGLY_RECOMMENDED_RULES
from .status_determiner import StatusDeterminer, MoleculeResult
from .artifact_manager import ArtifactManager, ArtifactPolicy
from .plugin_config import PluginConfig, PluginRegistry, should_execute_plugin

__all__ = [
    # Reason codes
    "REASON_REGISTRY",
    "get_reason_info",
    "is_hard_fail",
    "is_soft_fail", 
    "is_skip",
    # Field tiers
    "FieldTierChecker",
    "CORE_REQUIRED_FIELDS",
    "STRONGLY_RECOMMENDED_RULES",
    # Status
    "StatusDeterminer",
    # Artifacts
    "ArtifactManager",
    "ArtifactPolicy",
    # Plugin config
    "PluginConfig",
    "should_execute_plugin",
]

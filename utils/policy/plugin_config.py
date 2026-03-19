"""
插件配置管理

处理插件的 enabled/disabled、timeout、resource limit、fast/full 模式覆盖等配置。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PluginConfig:
    """单个插件的配置"""
    
    name: str
    enabled: bool = True
    timeout_seconds: int = 60
    max_atoms: Optional[int] = None
    max_total_grid_points: Optional[int] = None  # 仅 realspace
    max_memory_mb: Optional[int] = None
    runtime_config: Optional[Dict[str, Any]] = None
    
    # 模式控制
    run_in_fast_mode: bool = False
    run_in_full_mode: bool = True
    
    # Fast 模式覆盖策略
    fast_mode_override: str = "skip"  # "skip", "fail_fast", "timeout_reduced"
    
    # Full 模式超时乘数
    full_mode_timeout_factor: float = 1.0
    
    # 外部工具特有
    executable_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, name: str, config: Dict[str, Any]) -> "PluginConfig":
        """从字典创建配置"""
        return cls(
            name=name,
            enabled=config.get("enabled", True),
            timeout_seconds=config.get("timeout_seconds", 60),
            max_atoms=config.get("max_atoms"),
            max_total_grid_points=config.get("max_total_grid_points"),
            max_memory_mb=config.get("max_memory_mb"),
            runtime_config=config.get("runtime_config"),
            run_in_fast_mode=config.get("run_in_fast_mode", False),
            run_in_full_mode=config.get("run_in_full_mode", True),
            fast_mode_override=config.get("fast_mode_override", "skip"),
            full_mode_timeout_factor=config.get("full_mode_timeout_factor", 1.0),
            executable_path=config.get("executable_path"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "max_atoms": self.max_atoms,
            "max_total_grid_points": self.max_total_grid_points,
            "max_memory_mb": self.max_memory_mb,
            "runtime_config": self.runtime_config,
            "run_in_fast_mode": self.run_in_fast_mode,
            "run_in_full_mode": self.run_in_full_mode,
            "fast_mode_override": self.fast_mode_override,
            "full_mode_timeout_factor": self.full_mode_timeout_factor,
            "executable_path": self.executable_path,
        }
    
    def get_effective_timeout(self, mode: str) -> int:
        """获取有效超时时间"""
        base_timeout = self.timeout_seconds
        
        if mode == "fast":
            if self.fast_mode_override == "timeout_reduced":
                return min(30, base_timeout)
            elif self.fast_mode_override == "fail_fast":
                return min(10, base_timeout)
            # skip 模式不会执行到这里
        
        if mode == "full":
            return int(base_timeout * self.full_mode_timeout_factor)
        
        return base_timeout


def should_execute_plugin(
    plugin_config: PluginConfig,
    mode: str,
    natm: int
) -> tuple[bool, Optional[str]]:
    """
    判定插件是否应该执行
    
    Args:
        plugin_config: 插件配置
        mode: "fast" 或 "full"
        natm: 分子原子数
        
    Returns:
        (是否应执行, skip reason code 或 None)
    """
    cfg = plugin_config
    
    # 1. 检查 enabled
    if not cfg.enabled:
        return False, "plugin_disabled_by_config"
    
    # 2. 检查原子数限制
    if cfg.max_atoms is not None and natm > cfg.max_atoms:
        return False, "atom_limit_skip"
    
    # 3. 模式检查
    if mode == "fast":
        if cfg.run_in_fast_mode:
            return True, None
        
        # 检查 override 策略
        override = cfg.fast_mode_override
        if override == "skip":
            return False, "fast_mode_skip"
        elif override in ("fail_fast", "timeout_reduced"):
            return True, None
        else:
            return False, "fast_mode_skip"
    
    if mode == "full":
        if cfg.run_in_full_mode:
            return True, None
        else:
            return False, "plugin_disabled_by_mode"
    
    return False, None


class PluginRegistry:
    """插件注册表，管理多个插件配置"""
    
    DEFAULT_CONFIGS = {
        "orbital_features": {
            "enabled": True,
            "timeout_seconds": 60,
            "max_atoms": 100,
            "max_memory_mb": 2048,
            "run_in_fast_mode": False,
            "run_in_full_mode": True,
            "fast_mode_override": "skip",
            "full_mode_timeout_factor": 1.0,
        },
        "realspace_features": {
            "enabled": True,
            "timeout_seconds": 120,
            "max_atoms": 50,
            "max_total_grid_points": 1_000_000,
            "max_memory_mb": 4096,
            "run_in_fast_mode": False,
            "run_in_full_mode": True,
            "fast_mode_override": "skip",
            "full_mode_timeout_factor": 1.0,
        },
        "critic2_bridge": {
            "enabled": False,
            "timeout_seconds": 300,
            "max_atoms": 100,
            "run_in_fast_mode": False,
            "run_in_full_mode": False,  # 默认关闭
            "fast_mode_override": "skip",
            "full_mode_timeout_factor": 1.0,
            "executable_path": "critic2",
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        初始化插件注册表
        
        Args:
            config: 自定义配置，会合并到默认配置上
        """
        self.configs: Dict[str, PluginConfig] = {}
        
        # 加载默认配置
        for name, default_cfg in self.DEFAULT_CONFIGS.items():
            merged = default_cfg.copy()
            if config and name in config:
                merged.update(config[name])
            self.configs[name] = PluginConfig.from_dict(name, merged)
    
    def get(self, name: str) -> Optional[PluginConfig]:
        """获取插件配置"""
        return self.configs.get(name)
    
    def should_execute(self, name: str, mode: str, natm: int) -> tuple[bool, Optional[str]]:
        """判定某插件是否应该执行"""
        cfg = self.get(name)
        if cfg is None:
            return False, "plugin_disabled_by_config"
        return should_execute_plugin(cfg, mode, natm)
    
    def get_all_plugin_status(self, mode: str, natm: int) -> Dict[str, Dict[str, Any]]:
        """获取所有插件的执行状态"""
        result = {}
        for name, cfg in self.configs.items():
            should_run, reason = should_execute_plugin(cfg, mode, natm)
            result[name] = {
                "enabled": cfg.enabled,
                "should_execute": should_run,
                "skip_reason": reason,
                "effective_timeout": cfg.get_effective_timeout(mode) if should_run else None,
                "runtime_config": cfg.runtime_config if should_run else None,
            }
        return result
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """导出为字典"""
        return {
            name: cfg.to_dict()
            for name, cfg in self.configs.items()
        }


def create_default_plugin_registry(
    run_mode: str = "full",
    overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> PluginRegistry:
    """
    创建默认插件注册表
    
    Args:
        run_mode: "fast" 或 "full"
        overrides: 配置覆盖
    """
    config = overrides or {}
    
    # 根据 run_mode 调整默认行为
    if run_mode == "fast":
        # Fast 模式下默认禁用所有插件
        for plugin_name in ["orbital_features", "realspace_features", "critic2_bridge"]:
            if plugin_name not in config:
                config[plugin_name] = {}
            config[plugin_name]["run_in_fast_mode"] = False
    
    return PluginRegistry(config)

"""
Artifact 生命周期管理器

管理各类中间文件和输出文件的保留与清理。
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from enum import Enum


class ArtifactPolicy(str, Enum):
    """Artifact 保留策略"""
    KEEP_NONE = "keep_none"
    KEEP_FAILED_ONLY = "keep_failed_only"
    KEEP_CORE_ONLY = "keep_core_only"
    KEEP_ALL_DEBUG = "keep_all_debug"


class ArtifactCategory:
    """Artifact 分类"""
    
    FINAL_OUTPUT = "final_output"        # .unified.json, _legacy.json
    RESTART = "restart"                   # .pkl
    REALSPACE = "realspace"               # .cube files
    EXTERNAL_BRIDGE = "external_bridge"   # critic2 in/out
    DEBUG = "debug"                       # logs, dumps


# 文件模式到分类的映射
ARTIFACT_PATTERNS = {
    ArtifactCategory.FINAL_OUTPUT: [
        "*.unified.json",
        "*_legacy.json",
    ],
    ArtifactCategory.RESTART: [
        "*.pkl",
    ],
    ArtifactCategory.REALSPACE: [
        "*_density.cube",
        "*_homo.cube",
        "*_lumo.cube",
        "*_esp.cube",
        "*_elf.cube",
    ],
    ArtifactCategory.EXTERNAL_BRIDGE: [
        "*.critic2.in",
        "*.critic2.out",
        "*.critic2.sum",
        "*.horton.*",
        "*.psi4.*",
    ],
    ArtifactCategory.DEBUG: [
        "*.log",
        "*.dump",
        "*_debug.*",
    ],
}


class ArtifactManager:
    """Artifact 生命周期管理器"""
    
    # 策略到保留分类的映射
    POLICY_RETENTION = {
        ArtifactPolicy.KEEP_NONE: {
            ArtifactCategory.FINAL_OUTPUT: True,
            ArtifactCategory.RESTART: False,
            ArtifactCategory.REALSPACE: False,
            ArtifactCategory.EXTERNAL_BRIDGE: False,
            ArtifactCategory.DEBUG: False,
        },
        ArtifactPolicy.KEEP_FAILED_ONLY: {
            # 动态判断：失败时全保留，成功时只保留 FINAL
            ArtifactCategory.FINAL_OUTPUT: True,
            ArtifactCategory.RESTART: "conditional",  # 失败时保留
            ArtifactCategory.REALSPACE: "conditional",
            ArtifactCategory.EXTERNAL_BRIDGE: "conditional",
            ArtifactCategory.DEBUG: "conditional",
        },
        ArtifactPolicy.KEEP_CORE_ONLY: {
            ArtifactCategory.FINAL_OUTPUT: True,
            ArtifactCategory.RESTART: True,
            ArtifactCategory.REALSPACE: False,
            ArtifactCategory.EXTERNAL_BRIDGE: False,
            ArtifactCategory.DEBUG: False,
        },
        ArtifactPolicy.KEEP_ALL_DEBUG: {
            ArtifactCategory.FINAL_OUTPUT: True,
            ArtifactCategory.RESTART: True,
            ArtifactCategory.REALSPACE: True,
            ArtifactCategory.EXTERNAL_BRIDGE: True,
            ArtifactCategory.DEBUG: True,
        },
    }
    
    # 判定为"失败"的状态列表
    FAILED_STATUSES = {
        "invalid_input",
        "failed_geometry",
        "failed_scf",
        "failed_core_features",
    }
    
    def __init__(self, policy: ArtifactPolicy, output_dir: Path):
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.cleaned_count = 0
        self.retained_count = 0
    
    def cleanup_molecule(self, molecule_id: str, overall_status: str) -> Dict[str, Any]:
        """
        清理单个分子的 artifacts
        
        Args:
            molecule_id: 分子 ID
            overall_status: 分子的 overall_status
            
        Returns:
            清理统计信息
        """
        if self.policy == ArtifactPolicy.KEEP_ALL_DEBUG:
            # 全保留，无需清理
            return {
                "molecule_id": molecule_id,
                "policy": self.policy.value,
                "cleaned": 0,
                "retained": "all",
                "skipped": True
            }
        
        # 构建文件列表
        molecule_files = self._find_molecule_files(molecule_id)
        
        cleaned = []
        retained = []
        
        for category, files in molecule_files.items():
            should_keep = self._should_keep_category(
                category, overall_status
            )
            
            for filepath in files:
                if should_keep:
                    retained.append(str(filepath))
                    self.retained_count += 1
                else:
                    # 删除文件
                    try:
                        if filepath.exists():
                            filepath.unlink()
                            cleaned.append(str(filepath))
                            self.cleaned_count += 1
                    except Exception:
                        pass
        
        return {
            "molecule_id": molecule_id,
            "policy": self.policy.value,
            "overall_status": overall_status,
            "is_failed": overall_status in self.FAILED_STATUSES,
            "cleaned_files": cleaned,
            "retained_files": retained,
            "cleaned_count": len(cleaned),
            "retained_count": len(retained)
        }
    
    def _find_molecule_files(self, molecule_id: str) -> Dict[str, List[Path]]:
        """查找与分子相关的所有文件，按分类组织"""
        result = {cat: [] for cat in ArtifactCategory.__dict__.values() 
                 if isinstance(cat, str) and not cat.startswith("_")}
        
        if not self.output_dir.exists():
            return result
        
        # 遍历所有文件，匹配模式
        for filepath in self.output_dir.iterdir():
            if not filepath.is_file():
                continue
            
            # 检查是否属于该分子
            if molecule_id not in filepath.name:
                continue
            
            # 判断分类
            category = self._classify_file(filepath.name)
            if category:
                result.setdefault(category, []).append(filepath)
        
        return result
    
    def _classify_file(self, filename: str) -> Optional[str]:
        """根据文件名判断 artifact 分类"""
        import fnmatch
        
        for category, patterns in ARTIFACT_PATTERNS.items():
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return category
        
        # 特殊判断
        if filename.endswith(".pkl"):
            return ArtifactCategory.RESTART
        if "critic2" in filename or "horton" in filename or "psi4" in filename:
            return ArtifactCategory.EXTERNAL_BRIDGE
        if ".log" in filename or ".dump" in filename:
            return ArtifactCategory.DEBUG
        
        return None
    
    def _should_keep_category(self, category: str, overall_status: str) -> bool:
        """判断某分类是否应该保留"""
        retention = self.POLICY_RETENTION[self.policy].get(category)
        
        if retention is True:
            return True
        if retention is False:
            return False
        if retention == "conditional":
            # 条件保留：失败时保留
            return overall_status in self.FAILED_STATUSES
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """获取清理汇总统计"""
        return {
            "policy": self.policy.value,
            "output_dir": str(self.output_dir),
            "total_cleaned": self.cleaned_count,
            "total_retained": self.retained_count
        }


def get_artifact_files_for_molecule(
    molecule_id: str, 
    output_dir: Path
) -> Dict[str, List[str]]:
    """
    获取某分子的所有 artifact 文件列表（不移除）
    
    Returns:
        按分类组织的文件路径列表
    """
    manager = ArtifactManager(ArtifactPolicy.KEEP_ALL_DEBUG, output_dir)
    files = manager._find_molecule_files(molecule_id)
    
    return {
        cat: [str(p) for p in paths]
        for cat, paths in files.items()
    }

"""
单分子处理器 (M5)

负责单分子的完整处理流程：
1. 调用特征提取
2. 状态判定
3. Artifact 清理
4. 结果打包
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

# 导入 policy 层
from utils.policy.status_determiner import StatusDeterminer, MoleculeResult
from utils.policy.artifact_manager import ArtifactManager, ArtifactPolicy
from utils.policy.plugin_config import PluginRegistry


class MoleculeProcessor:
    """单分子处理器"""
    
    def __init__(
        self,
        feature_extractor,  # UnifiedFeatureExtractor 实例
        plugin_registry: PluginRegistry,
        artifact_manager: ArtifactManager,
        run_mode: str = "full"
    ):
        self.feature_extractor = feature_extractor
        self.plugin_registry = plugin_registry
        self.artifact_manager = artifact_manager
        self.run_mode = run_mode
    
    def process_one(
        self,
        pkl_path: Path,
        molecule_id: str,
        smiles: Optional[str] = None,
        dft_config: Optional[Dict[str, Any]] = None
    ) -> MoleculeResult:
        """
        处理单个分子
        
        Args:
            pkl_path: 波函数文件路径
            molecule_id: 分子 ID
            smiles: SMILES 字符串
            dft_config: DFT 配置
            
        Returns:
            MoleculeResult 包含处理结果和状态
        """
        start_time = time.time()
        
        try:
            # 1. 获取插件执行计划
            # 需要先加载分子获取 natm
            natm = self._get_natm_from_pkl(pkl_path)
            plugin_plan = self.plugin_registry.get_all_plugin_status(
                self.run_mode, natm
            )
            
            # 2. 调用特征提取
            # 传入插件计划，让 extractor 知道哪些该执行
            unified_data = self.feature_extractor.extract_unified(
                pkl_path=str(pkl_path),
                molecule_id=molecule_id,
                smiles=smiles,
                dft_config=dft_config,
                plugin_plan=plugin_plan,
                run_mode=self.run_mode
            )
            
            # 3. 状态判定
            determiner = StatusDeterminer(unified_data)
            overall_status = determiner.determine()
            reason_codes = determiner.get_reason_codes()
            
            # 4. Artifact 清理
            cleanup_info = self.artifact_manager.cleanup_molecule(
                molecule_id, overall_status
            )
            
            # 5. 保存 unified json 文件
            output_path = self.artifact_manager.output_dir / f"{molecule_id}"
            self.feature_extractor.save_unified_features(unified_data, str(output_path))
            
            # 6. 记录 cleanup 信息到数据
            unified_data["_cleanup_info"] = cleanup_info
            
        except Exception as e:
            # 处理异常，构建错误结果
            overall_status = "failed_core_features"
            reason_codes = ["wavefunction_corrupted"]
            unified_data = self._build_error_data(molecule_id, smiles, str(e))
        
        wall_time = time.time() - start_time
        
        return MoleculeResult(
            molecule_id=molecule_id,
            overall_status=overall_status,
            reason_codes=reason_codes,
            data=unified_data,
            wall_time_seconds=wall_time
        )
    
    def _get_natm_from_pkl(self, pkl_path: Path) -> int:
        """从 pkl 文件快速获取原子数（不加载全部数据）"""
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'mol' in data:
                    return data['mol'].natm
                elif hasattr(data, 'natm'):
                    return data.natm
                elif hasattr(data, 'mol'):
                    return data.mol.natm
        except Exception:
            pass
        return 999  # 默认值，让后续检查决定
    
    def _build_error_data(
        self,
        molecule_id: str,
        smiles: Optional[str],
        error_msg: str
    ) -> Dict[str, Any]:
        """构建错误状态的数据结构"""
        from utils.output_schema import UnifiedOutputBuilder
        
        builder = UnifiedOutputBuilder(molecule_id, smiles or "")
        builder.set_status(
            wavefunction_load_success=False,
            error_messages=[error_msg]
        )
        
        return builder.build()


class MoleculeProcessorConfig:
    """处理器配置"""
    
    def __init__(
        self,
        run_mode: str = "full",
        artifact_policy: str = "keep_core_only",
        plugin_config: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self.run_mode = run_mode
        self.artifact_policy = ArtifactPolicy(artifact_policy)
        self.plugin_config = plugin_config or {}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MoleculeProcessorConfig":
        """从字典创建配置"""
        return cls(
            run_mode=config.get("run_mode", "full"),
            artifact_policy=config.get("artifact_policy", "keep_core_only"),
            plugin_config=config.get("plugins")
        )
    
    def create_processor(
        self,
        feature_extractor,
        output_dir: Path
    ) -> MoleculeProcessor:
        """创建处理器实例"""
        plugin_registry = PluginRegistry(self.plugin_config)
        artifact_manager = ArtifactManager(self.artifact_policy, output_dir)
        
        return MoleculeProcessor(
            feature_extractor=feature_extractor,
            plugin_registry=plugin_registry,
            artifact_manager=artifact_manager,
            run_mode=self.run_mode
        )

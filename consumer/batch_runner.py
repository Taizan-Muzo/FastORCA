"""
批处理运行器 (M5)

管理多分子的批处理流程，生成汇总统计。
"""

import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

from utils.batch_summary import BatchSummaryBuilder
from utils.policy.status_determiner import MoleculeResult
from .molecule_processor import MoleculeProcessor, MoleculeProcessorConfig


class BatchRunner:
    """批处理运行器"""
    
    def __init__(
        self,
        processor: MoleculeProcessor,
        config: MoleculeProcessorConfig,
        output_dir: Path,
        n_workers: int = 1
    ):
        self.processor = processor
        self.config = config
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        
        self.summary_builder = BatchSummaryBuilder(run_mode=config.run_mode)
    
    def run(
        self,
        molecule_list: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        运行批处理
        
        Args:
            molecule_list: 分子列表，每个元素为 {"pkl_path": ..., "molecule_id": ..., "smiles": ...}
            progress_callback: 进度回调函数 (current, total, molecule_id)
            
        Returns:
            Batch summary 字典
        """
        self.summary_builder.start()
        
        total = len(molecule_list)
        logger.info(f"Starting batch processing: {total} molecules, mode={self.config.run_mode}")
        
        if self.n_workers == 1:
            # 单线程模式
            for i, mol_info in enumerate(molecule_list):
                result = self._process_one_molecule(mol_info)
                self.summary_builder.add_molecule_result(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, mol_info.get("molecule_id", ""))
        else:
            # 多进程模式
            # 注意：这里简化处理，实际需要考虑进程间通信
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_mol = {
                    executor.submit(self._process_one_molecule, mol_info): mol_info
                    for mol_info in molecule_list
                }
                
                completed = 0
                for future in as_completed(future_to_mol):
                    mol_info = future_to_mol[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        # 构建错误结果
                        result = self._build_error_result(mol_info, str(e))
                    
                    self.summary_builder.add_molecule_result(result)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total, mol_info.get("molecule_id", ""))
        
        # 完成并生成报告
        self.summary_builder.finish()
        summary = self.summary_builder.build()
        
        # 保存报告
        summary_path = self.output_dir / f"batch_summary_{self.summary_builder.batch_id}.json"
        self.summary_builder.save(summary_path)
        
        logger.info(f"Batch completed. Summary saved to {summary_path}")
        self.summary_builder.print_summary()
        
        return summary
    
    def _process_one_molecule(self, mol_info: Dict[str, Any]) -> MoleculeResult:
        """处理单个分子（包装器）"""
        pkl_path = Path(mol_info["pkl_path"])
        molecule_id = mol_info["molecule_id"]
        smiles = mol_info.get("smiles")
        dft_config = mol_info.get("dft_config")
        
        return self.processor.process_one(
            pkl_path=pkl_path,
            molecule_id=molecule_id,
            smiles=smiles,
            dft_config=dft_config
        )
    
    def _build_error_result(
        self,
        mol_info: Dict[str, Any],
        error_msg: str
    ) -> MoleculeResult:
        """构建错误结果"""
        return MoleculeResult(
            molecule_id=mol_info.get("molecule_id", "unknown"),
            overall_status="failed_core_features",
            reason_codes=["wavefunction_corrupted"],
            data={"error": error_msg},
            wall_time_seconds=0.0
        )


def create_batch_runner(
    feature_extractor,
    config: Dict[str, Any],
    output_dir: Path,
    n_workers: int = 1
) -> BatchRunner:
    """
    创建批处理运行器的工厂函数
    
    Args:
        feature_extractor: UnifiedFeatureExtractor 实例
        config: 配置字典，包含 run_mode, artifact_policy, plugins 等
        output_dir: 输出目录
        n_workers: 并行工作数
        
    Returns:
        BatchRunner 实例
    """
    processor_config = MoleculeProcessorConfig.from_dict(config)
    processor = processor_config.create_processor(feature_extractor, output_dir)
    
    # 设置 config hash
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    runner = BatchRunner(processor, processor_config, output_dir, n_workers)
    runner.summary_builder.set_config_hash(config_hash)
    
    return runner


# === 便捷函数 ===

def run_batch(
    feature_extractor,
    molecule_list: List[Dict[str, Any]],
    output_dir: Path,
    run_mode: str = "full",
    artifact_policy: str = "keep_core_only",
    plugin_config: Optional[Dict[str, Any]] = None,
    n_workers: int = 1
) -> Dict[str, Any]:
    """
    便捷函数：一键运行批处理
    
    Example:
        results = run_batch(
            extractor,
            [
                {"pkl_path": "mol_001.pkl", "molecule_id": "mol_001", "smiles": "CCO"},
                ...
            ],
            output_dir=Path("./output"),
            run_mode="full"
        )
    """
    config = {
        "run_mode": run_mode,
        "artifact_policy": artifact_policy,
        "plugins": plugin_config or {}
    }
    
    runner = create_batch_runner(feature_extractor, config, output_dir, n_workers)
    return runner.run(molecule_list)


# if __name__ == "__main__":
#     results = run_batch(
#         extractor, molecules, output_dir,
#         run_mode="fast", 
#         artifact_policy="keep_none"
#     )

#     # Full 模式：完整特征
#     results = run_batch(
#         extractor, molecules, output_dir,
#         run_mode="full",
#         artifact_policy="keep_failed_only"
#     )
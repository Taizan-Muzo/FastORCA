"""
批处理统计汇总

收集多分子处理结果，生成 batch summary 报告。
"""

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .policy.reason_codes import is_hard_fail, is_soft_fail, get_reason_info
from .policy.status_determiner import MoleculeResult


@dataclass
class PluginStatusAccumulator:
    """插件状态累加器"""
    attempted: int = 0
    success: int = 0
    skipped: int = 0
    soft_fail: int = 0
    hard_fail: int = 0
    by_reason: Dict[str, int] = field(default_factory=dict)


class BatchSummaryBuilder:
    """批处理汇总构建器"""
    
    def __init__(self, batch_id: Optional[str] = None, run_mode: str = "full"):
        self.batch_id = batch_id or self._generate_batch_id()
        self.run_mode = run_mode
        self.config_hash: Optional[str] = None
        
        self.timestamp_start: Optional[str] = None
        self.timestamp_end: Optional[str] = None
        
        # 分子结果列表
        self.molecule_results: List[MoleculeResult] = []
        
        # 统计缓存
        self._finalized = False
        self._summary: Optional[Dict[str, Any]] = None
    
    def _generate_batch_id(self) -> str:
        """生成 batch ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def start(self):
        """开始记录"""
        self.timestamp_start = datetime.utcnow().isoformat() + "Z"
    
    def finish(self):
        """结束记录"""
        self.timestamp_end = datetime.utcnow().isoformat() + "Z"
    
    def add_molecule_result(self, result: MoleculeResult):
        """添加单分子结果"""
        self.molecule_results.append(result)
        self._finalized = False
    
    def set_config_hash(self, config_hash: str):
        """设置配置哈希"""
        self.config_hash = config_hash
    
    def build(self) -> Dict[str, Any]:
        """构建汇总报告"""
        if self._finalized and self._summary:
            return self._summary
        
        if not self.timestamp_end:
            self.finish()
        
        summary = {
            "batch_id": self.batch_id,
            "run_mode": self.run_mode,
            "config_hash": self.config_hash,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "molecule_counts": self._build_molecule_counts(),
            "plugin_status_counts": self._build_plugin_counts(),
            "reason_code_histogram": self._build_reason_histogram(),
            "timing_stats": self._build_timing_stats(),
        }
        
        self._summary = summary
        self._finalized = True
        return summary
    
    def _build_molecule_counts(self) -> Dict[str, Any]:
        """构建分子计数统计"""
        total = len(self.molecule_results)
        
        # by_final_status
        by_status: Dict[str, int] = {}
        for r in self.molecule_results:
            status = r.overall_status
            by_status[status] = by_status.get(status, 0) + 1
        
        # hard/soft/skip 统计（分子级）
        molecules_with_hard_fail = sum(
            1 for r in self.molecule_results if r.has_hard_fail()
        )
        molecules_with_soft_fail = sum(
            1 for r in self.molecule_results if r.has_soft_fail()
        )
        molecules_with_only_skip = sum(
            1 for r in self.molecule_results if r.has_only_skip()
        )
        
        return {
            "total": total,
            "by_final_status": by_status,
            "molecules_with_hard_fail": molecules_with_hard_fail,
            "molecules_with_soft_fail": molecules_with_soft_fail,
            "molecules_with_only_skip": molecules_with_only_skip,
        }
    
    def _build_plugin_counts(self) -> Dict[str, Any]:
        """构建插件状态统计"""
        # 从 molecule data 中提取插件状态
        accumulators = {
            "orbital_features": PluginStatusAccumulator(),
            "realspace_features": PluginStatusAccumulator(),
            "critic2_bridge": PluginStatusAccumulator(),
        }
        
        for result in self.molecule_results:
            data = result.data
            
            for plugin_name in accumulators.keys():
                acc = accumulators[plugin_name]
                
                # 获取插件元数据
                if plugin_name == "critic2_bridge":
                    meta = data.get("external_bridge", {}).get("critic2", {})
                    status = meta.get("execution_status")
                else:
                    meta = data.get(plugin_name, {}).get("metadata", {})
                    status = meta.get("extraction_status")
                
                if status is None:
                    continue
                
                acc.attempted += 1
                
                if status == "success":
                    acc.success += 1
                elif status == "skipped":
                    acc.skipped += 1
                elif status in ("failed", "timeout", "error"):
                    # 判断 severity
                    failure_reason = meta.get("failure_reason", "")
                    if is_hard_fail(failure_reason):
                        acc.hard_fail += 1
                    else:
                        acc.soft_fail += 1
                    
                    # 记录 reason
                    if failure_reason:
                        acc.by_reason[failure_reason] = acc.by_reason.get(failure_reason, 0) + 1
        
        # 转为输出格式
        return {
            name: {
                "attempted": acc.attempted,
                "success": acc.success,
                "skipped": acc.skipped,
                "soft_fail": acc.soft_fail,
                "hard_fail": acc.hard_fail,
                "by_reason": acc.by_reason,
            }
            for name, acc in accumulators.items()
        }
    
    def _build_reason_histogram(self) -> Dict[str, Any]:
        """构建 reason code 直方图（事件级）"""
        histogram: Dict[str, Dict[str, Any]] = {}
        
        for result in self.molecule_results:
            for code in result.reason_codes:
                if code not in histogram:
                    info = get_reason_info(code)
                    histogram[code] = {
                        "count": 0,
                        "severity": info.get("severity", "unknown") if info else "unknown",
                        "scope": info.get("scope", "unknown") if info else "unknown",
                    }
                histogram[code]["count"] += 1
        
        return histogram
    
    def _build_timing_stats(self) -> Dict[str, Any]:
        """构建时间统计"""
        times = [r.wall_time_seconds for r in self.molecule_results]
        
        if not times:
            return {
                "wall_time_total_sec": 0,
                "wall_time_per_molecule_sec": {},
            }
        
        # 基本统计
        total = sum(times)
        mean_time = statistics.mean(times)
        
        # 百分位
        sorted_times = sorted(times)
        p50 = self._percentile(sorted_times, 50)
        p90 = self._percentile(sorted_times, 90)
        p99 = self._percentile(sorted_times, 99)
        
        # 按状态分组
        by_status: Dict[str, List[float]] = {}
        for r in self.molecule_results:
            status = r.overall_status
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(r.wall_time_seconds)
        
        by_status_stats = {
            status: {"mean": statistics.mean(ts), "count": len(ts)}
            for status, ts in by_status.items()
        }
        
        return {
            "wall_time_total_sec": round(total, 2),
            "wall_time_per_molecule_sec": {
                "mean": round(mean_time, 2),
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p99": round(p99, 2),
                "max": round(max(times), 2),
                "by_status": by_status_stats,
            }
        }
    
    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        """计算百分位数"""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def save(self, output_path: Path):
        """保存汇总报告到文件"""
        summary = self.build()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def print_summary(self):
        """打印汇总到控制台"""
        summary = self.build()
        
        print("\n" + "=" * 60)
        print(f"Batch Summary: {summary['batch_id']}")
        print("=" * 60)
        
        counts = summary["molecule_counts"]
        print(f"\nTotal Molecules: {counts['total']}")
        print(f"  - Fully Success: {counts['by_final_status'].get('fully_success', 0)}")
        print(f"  - Partial Features: {counts['by_final_status'].get('core_success_partial_features', 0)}")
        print(f"  - Failed (any): {counts['molecules_with_hard_fail']}")
        print(f"  - Soft Fail Only: {counts['molecules_with_soft_fail']}")
        
        print("\nPlugin Status:")
        for plugin, stats in summary["plugin_status_counts"].items():
            if stats["attempted"] > 0:
                success_rate = stats["success"] / stats["attempted"] * 100
                print(f"  {plugin}: {stats['success']}/{stats['attempted']} ({success_rate:.1f}%)")
        
        timing = summary["timing_stats"]
        print(f"\nTiming:")
        print(f"  Total: {timing['wall_time_total_sec']:.1f}s")
        print(f"  Per Molecule: mean={timing['wall_time_per_molecule_sec']['mean']:.1f}s, "
              f"p90={timing['wall_time_per_molecule_sec']['p90']:.1f}s")
        
        print("\n" + "=" * 60)


def load_batch_summary(path: Path) -> Dict[str, Any]:
    """加载 batch summary 文件"""
    with open(path) as f:
        return json.load(f)

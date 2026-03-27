"""
Operator-oriented health check for qcMol substitute default runs.

This script converts multiple runtime artifacts into a single pass/fail decision:
- batch summary
- alignment closure validation
- alignment master table snapshot
- volume-partial tail explanation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (  # noqa: E402
    QCMOL_ALIGNMENT_MASTER_TABLE,
    summarize_alignment_status_counts,
)


UPSTREAM_VOLUME_REASON_ALLOWLIST = {
    "bader_volume_column_truly_missing",
    "bader_volume_column_not_reported_in_critic2_output",
    "bader_volume_column_present_but_non_numeric",
    "bader_volume_column_present_but_all_null",
    "bader_volume_column_present_but_partially_null",
}


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _find_batch_summary(unified_dir: Path) -> Path:
    files = sorted(unified_dir.glob("batch_summary_*.json"))
    if not files:
        raise FileNotFoundError(f"No batch_summary_*.json found in {unified_dir}")
    return files[-1]


def _detect_unified_dir(run_output_dir: Path) -> Path:
    if (run_output_dir / "A_main").exists():
        return run_output_dir / "A_main"
    return run_output_dir


def _plugin_success_rates(batch_summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    plugins = batch_summary.get("plugin_status_counts") or {}
    for plugin_name, payload in plugins.items():
        attempted = float(payload.get("attempted", 0) or 0)
        success = float(payload.get("success", 0) or 0)
        out[plugin_name] = (success / attempted) if attempted > 0 else 0.0
    return out


def _volume_reason_summary(unified_files: Iterable[Path]) -> Dict[str, Any]:
    status_hist: Counter[str] = Counter()
    reason_hist: Counter[str] = Counter()
    for fp in unified_files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        status = _get(data, "atom_features.metadata.atomic_density_partition_volume_proxy.bader_status")
        if status is None:
            status = _get(data, "atom_features.metadata.atomic_density_partition_charge_proxy.bader_volume_status")
        reason = _get(data, "atom_features.metadata.atomic_density_partition_volume_proxy.bader_status_reason")
        if reason is None:
            reason = _get(data, "atom_features.metadata.atomic_density_partition_charge_proxy.bader_volume_status_reason")
        if status is not None:
            status_hist[str(status)] += 1
        if reason is not None:
            reason_hist[str(reason)] += 1
    reasons = set(reason_hist.keys())
    upstream_limited = bool(reasons) and reasons.issubset(UPSTREAM_VOLUME_REASON_ALLOWLIST)
    return {
        "status_histogram": dict(status_hist),
        "reason_histogram": dict(reason_hist),
        "upstream_limited_evidence": upstream_limited,
        "unknown_reasons": sorted(reasons - UPSTREAM_VOLUME_REASON_ALLOWLIST),
    }


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="qcMol substitute operational health check")
    p.add_argument("--run-output-dir", required=True, help="Run output root; usually contains A_main/")
    p.add_argument("--closure-validation", default="", help="Closure validation JSON path (optional)")
    p.add_argument("--master-table", default="", help="Master table report JSON path (optional)")
    p.add_argument("--output-json", default="", help="Optional output JSON path")
    p.add_argument("--output-md", default="", help="Optional output markdown path")
    p.add_argument("--min-fully-success-ratio", type=float, default=0.95)
    p.add_argument("--min-plugin-success-rate", type=float, default=0.95)
    p.add_argument("--expected-implemented-exact", type=int, default=14)
    p.add_argument("--expected-implemented-proxy", type=int, default=17)
    p.add_argument("--expected-partial", type=int, default=1)
    p.add_argument("--expected-missing", type=int, default=2)
    p.add_argument("--expected-rejected-as-exact", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_output_dir = Path(args.run_output_dir).resolve()
    if not run_output_dir.exists():
        raise FileNotFoundError(f"run-output-dir not found: {run_output_dir}")

    unified_dir = _detect_unified_dir(run_output_dir)
    batch_summary_path = _find_batch_summary(unified_dir)
    batch = json.loads(batch_summary_path.read_text(encoding="utf-8"))
    closure_path = (
        Path(args.closure_validation).resolve()
        if args.closure_validation
        else (run_output_dir / "qcmol_alignment_closure_validation.json").resolve()
    )
    master_path = (
        Path(args.master_table).resolve()
        if args.master_table
        else (run_output_dir / "qcmol_alignment_master_table.json").resolve()
    )
    closure = _load_json_if_exists(closure_path)
    master = _load_json_if_exists(master_path)

    counts = batch.get("molecule_counts") or {}
    total = int(counts.get("total", 0) or 0)
    fully_success = int((counts.get("by_final_status") or {}).get("fully_success", 0) or 0)
    fully_success_ratio = (fully_success / total) if total > 0 else 0.0
    plugin_rates = _plugin_success_rates(batch)

    expected_counts = {
        "implemented_exact": int(args.expected_implemented_exact),
        "implemented_proxy": int(args.expected_implemented_proxy),
        "partial": int(args.expected_partial),
        "missing": int(args.expected_missing),
        "rejected_as_exact": int(args.expected_rejected_as_exact),
    }
    master_counts = master.get("status_counts") or summarize_alignment_status_counts(
        [{"status": row.get("current_status")} for row in QCMOL_ALIGNMENT_MASTER_TABLE]
    )
    master_counts_match = all(int(master_counts.get(k, -1)) == v for k, v in expected_counts.items())

    volume_summary = _volume_reason_summary(sorted(unified_dir.glob("*.unified.json")))
    volume_tail_ok = volume_summary["upstream_limited_evidence"]

    pass_flags = {
        "fully_success_ratio_ok": fully_success_ratio >= float(args.min_fully_success_ratio),
        "plugin_success_rate_ok": all(v >= float(args.min_plugin_success_rate) for v in plugin_rates.values()),
        "master_table_counts_ok": master_counts_match,
        "volume_partial_tail_explained": volume_tail_ok,
    }
    overall_pass = all(pass_flags.values())

    payload = {
        "run_output_dir": str(run_output_dir),
        "unified_dir": str(unified_dir),
        "batch_summary_path": str(batch_summary_path),
        "closure_validation_path": str(closure_path) if closure_path.exists() else None,
        "master_table_path": str(master_path) if master_path.exists() else None,
        "metrics": {
            "total_molecules": total,
            "fully_success": fully_success,
            "fully_success_ratio": fully_success_ratio,
            "plugin_success_rates": plugin_rates,
            "master_table_counts": master_counts,
            "expected_master_table_counts": expected_counts,
            "remaining_gap_count": master.get("remaining_gap_count"),
        },
        "volume_partial_reason_summary": volume_summary,
        "gates": pass_flags,
        "overall_pass": overall_pass,
    }

    if args.output_json:
        out_json = Path(args.output_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        out_md = Path(args.output_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# qcMol Operational Health Check",
            "",
            f"- overall_pass: {overall_pass}",
            f"- fully_success_ratio: {fully_success_ratio}",
            f"- plugin_success_rates: {plugin_rates}",
            f"- master_table_counts: {master_counts}",
            f"- expected_master_table_counts: {expected_counts}",
            f"- volume_partial_upstream_limited: {volume_tail_ok}",
            f"- volume_reason_histogram: {volume_summary['reason_histogram']}",
        ]
        out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


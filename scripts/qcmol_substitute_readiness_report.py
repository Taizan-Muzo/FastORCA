"""
Build qcMol substitute readiness report from validation artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (
    CANONICAL_SURFACE_ITEMS,
    DISCOURAGED_DEFAULT_FIELDS,
    PROFILE_CONFIG_PATH,
    QCMOL_ALIGNMENT_MASTER_TABLE,
    QCMOL_ALIGNMENT_ITEMS,
    load_qcmol_substitute_default_profile,
    remaining_alignment_gaps,
    summarize_alignment_status_counts,
    summarize_alignment_next_actions,
)


def _safe_load_json(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_main_validation_metrics(validation: Dict[str, Any]) -> Dict[str, Any]:
    overall = validation.get("A_overall_status_distribution") or {}
    total = sum(int(v) for v in overall.values()) if overall else 0
    fully_success = int(overall.get("fully_success", 0))
    partial = int(overall.get("core_success_partial_features", 0))
    failed = total - fully_success - partial
    return {
        "overall_status_distribution": overall,
        "total": total,
        "fully_success": fully_success,
        "partial": partial,
        "failed": max(0, failed),
        "fully_success_ratio": (fully_success / total) if total else None,
        "partial_ratio": (partial / total) if total else None,
        "failed_ratio": (max(0, failed) / total) if total else None,
        "edge_cases": validation.get("C_edge_cases") or [],
        "top_reason_codes": validation.get("A_top_reason_codes") or [],
    }


def _extract_bader_metrics(finalmile: Dict[str, Any], uplift_baseline: Dict[str, Any]) -> Dict[str, Any]:
    cov = finalmile.get("coverage") or {}
    baseline_cov = uplift_baseline.get("coverage") or {}
    bader_rate = cov.get("bader_charge_validated_writeback_rate")
    baseline_rate = baseline_cov.get("bader_charge_validated_writeback_rate")
    if baseline_rate is None:
        baseline_rate = 0.6666666666666666
    uplift_pp = None
    if isinstance(bader_rate, (int, float)) and isinstance(baseline_rate, (int, float)):
        uplift_pp = (float(bader_rate) - float(baseline_rate)) * 100.0
    return {
        "critic2_execution_success_rate": cov.get("critic2_execution_success_rate"),
        "bader_charge_writeback_rate": bader_rate,
        "bader_volume_writeback_rate": cov.get("bader_volume_writeback_rate"),
        "bader_rate_uplift_pp_vs_uplift_baseline": uplift_pp,
        "unavailable_reason_histogram": (finalmile.get("unavailable_reason_histogram") or {}).get("category") or {},
        "rescue_retry": finalmile.get("rescue_retry") or {},
        "timing": finalmile.get("timing") or {},
    }


def _split_alignment_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    deliverable = []
    roadmap = []
    for row in QCMOL_ALIGNMENT_MASTER_TABLE:
        status = row["current_status"]
        action = row["next_action"]
        if status in {"missing", "rejected_as_exact"} or action in {"roadmap_only", "reject"}:
            roadmap.append(row)
        elif status in {"implemented_exact", "implemented_proxy"} and action == "keep":
            deliverable.append(row)
        else:
            roadmap.append(row)
    return deliverable, roadmap


def _canonical_surface_summary() -> Dict[str, Any]:
    group_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in CANONICAL_SURFACE_ITEMS:
        group_map.setdefault(row["group"], []).append(row)
    return {
        "recommended_groups": group_map,
        "discouraged_default_fields": DISCOURAGED_DEFAULT_FIELDS,
    }


def _build_readiness_judgement(main_metrics: Dict[str, Any], bader_metrics: Dict[str, Any]) -> Dict[str, Any]:
    fully_success_ratio = main_metrics.get("fully_success_ratio")
    bader_rate = bader_metrics.get("bader_charge_writeback_rate")
    critic2_rate = bader_metrics.get("critic2_execution_success_rate")
    ready = (
        isinstance(fully_success_ratio, (int, float))
        and isinstance(bader_rate, (int, float))
        and isinstance(critic2_rate, (int, float))
        and fully_success_ratio >= 0.95
        and critic2_rate >= 0.95
        and bader_rate >= 0.80
    )
    risks = [
        "Bader volume remains partial/unavailable in default runs; keep optional.",
        "Open-shell orbital path is still unavailable-by-design for IBO proxies.",
        "Exact qcMol roadmap fields (NAO/ADCH/NBO/LBO/DI exact) are still missing placeholders.",
    ]
    fit_scenarios = [
        "Open-source molecular descriptor production for closed-shell organic molecules.",
        "Batch ranking/filtering pipelines that accept exact+proxy mixed surfaces with explicit metadata.",
        "Downstream ML feature generation that can tolerate roadmap placeholders remaining null.",
    ]
    return {
        "is_open_source_qcmol_substitute_mainline_ready": ready,
        "criteria": {
            "fully_success_ratio_gte_0_95": fully_success_ratio,
            "critic2_success_rate_gte_0_95": critic2_rate,
            "bader_charge_writeback_rate_gte_0_80": bader_rate,
        },
        "primary_risks": risks,
        "best_fit_scenarios": fit_scenarios,
    }


def build_report(validation: Dict[str, Any], finalmile: Dict[str, Any], uplift_baseline: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    status_counts = summarize_alignment_status_counts(QCMOL_ALIGNMENT_ITEMS)
    next_action_counts = summarize_alignment_next_actions(QCMOL_ALIGNMENT_MASTER_TABLE)
    gaps = remaining_alignment_gaps(QCMOL_ALIGNMENT_MASTER_TABLE)
    deliverable_rows, roadmap_rows = _split_alignment_rows()
    main_metrics = _extract_main_validation_metrics(validation)
    bader_metrics = _extract_bader_metrics(finalmile, uplift_baseline)
    judgement = _build_readiness_judgement(main_metrics, bader_metrics)
    return {
        "profile": {
            "profile_id": profile.get("profile_id"),
            "profile_version": profile.get("profile_version"),
            "goal": profile.get("goal"),
            "config_path": str(PROFILE_CONFIG_PATH),
        },
        "coverage_snapshot": {
            "alignment_status_counts": status_counts,
            "alignment_next_action_counts": next_action_counts,
            "remaining_gap_count": len(gaps),
            "main_validation": main_metrics,
            "bader_finalmile": bader_metrics,
        },
        "alignment_master_table": QCMOL_ALIGNMENT_MASTER_TABLE,
        "default_deliverable_alignment_items": deliverable_rows,
        "roadmap_placeholder_alignment_items": roadmap_rows,
        "remaining_alignment_gaps": gaps,
        "canonical_surface": _canonical_surface_summary(),
        "readiness_judgement": judgement,
    }


def write_markdown(report: Dict[str, Any], out_md: Path) -> None:
    cov = report["coverage_snapshot"]
    counts = cov["alignment_status_counts"]
    main_v = cov["main_validation"]
    bader = cov["bader_finalmile"]
    judgement = report["readiness_judgement"]

    lines = [
        "# qcMol Substitute Readiness Report",
        "",
        "## Coverage Snapshot",
        f"- alignment counts: exact={counts['implemented_exact']}, proxy={counts['implemented_proxy']}, partial={counts['partial']}, missing={counts['missing']}, rejected_as_exact={counts['rejected_as_exact']}",
        f"- main batch fully_success_ratio: {main_v.get('fully_success_ratio')}",
        f"- critic2 execution success rate: {bader.get('critic2_execution_success_rate')}",
        f"- bader charge validated writeback rate: {bader.get('bader_charge_writeback_rate')}",
        f"- bader volume writeback rate: {bader.get('bader_volume_writeback_rate')}",
        "",
        "## Delivery Judgement",
        f"- open-source qcMol substitute mainline ready: {judgement['is_open_source_qcmol_substitute_mainline_ready']}",
        "- primary risks:",
    ]
    for r in judgement["primary_risks"]:
        lines.append(f"  - {r}")
    lines.extend(
        [
            "",
            "## Canonical Surface",
            "- recommended groups: basic, global, atom, bond, structural, realspace, external",
            "- discouraged default fields are all roadmap placeholders under external_bridge_roadmap.*",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build qcMol substitute readiness report")
    p.add_argument("--validation-report", default="", help="validation_round_report.json path")
    p.add_argument("--bader-finalmile-report", default="", help="bader_coverage_after_finalmile.json path")
    p.add_argument("--bader-uplift-baseline-report", default="", help="bader_coverage_after.json path")
    p.add_argument("--profile-path", default=str(PROFILE_CONFIG_PATH), help="qcmol_substitute_default.json path")
    p.add_argument("--output-json", required=True, help="output readiness JSON path")
    p.add_argument("--output-md", required=True, help="output readiness markdown path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    validation = _safe_load_json(Path(args.validation_report).resolve()) if args.validation_report else {}
    finalmile = _safe_load_json(Path(args.bader_finalmile_report).resolve()) if args.bader_finalmile_report else {}
    uplift_baseline = _safe_load_json(Path(args.bader_uplift_baseline_report).resolve()) if args.bader_uplift_baseline_report else {}
    profile = load_qcmol_substitute_default_profile(Path(args.profile_path).resolve())

    report = build_report(validation, finalmile, uplift_baseline, profile)
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out_md)

    print(
        json.dumps(
            {
                "output_json": str(out_json),
                "output_md": str(out_md),
                "is_ready": report["readiness_judgement"]["is_open_source_qcmol_substitute_mainline_ready"],
                "bader_rate": report["coverage_snapshot"]["bader_finalmile"]["bader_charge_writeback_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

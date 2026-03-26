"""
Bader coverage uplift summary

Reads unified JSON outputs and reports:
- critic2 execution success rate
- validated bader writeback rate
- bader volume writeback rate
- unavailable reason histogram (raw + categorized)
- stratified view (size/family/grid bins)
- timing summary (realspace/critic2)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_THRESHOLD_POLICIES = {
    "current": {"abs_tol_e": 0.50, "rel_tol": 0.02},
    "candidate_A": {"abs_tol_e": 0.60, "rel_tol": 0.025},
    "candidate_B": {"abs_tol_e": 0.75, "rel_tol": 0.03},
}


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _classify_bader_reason(reason: Optional[str], parser_note: Optional[str] = None) -> str:
    if not reason:
        return "other_or_unknown"
    r = str(reason).lower()
    p = str(parser_note).lower() if parser_note else ""
    if "sum_mismatch" in r:
        return "bader_population_sum_mismatch"
    if "volume_column" in r or "bader_volumes" in r or "all_null" in r:
        return "volume_column_missing_or_all_null"
    if "pop_column_missing" in r or "column_ambiguity" in r or "header" in r:
        return "parser_column_ambiguity"
    if "density_cube" in r or "grid" in r or "fft" in r or "cube" in r:
        return "cube_grid_related_issues"
    if "length_mismatch" in r or "natm" in r or "mapping" in r or "atomic_number" in r:
        return "atom_mapping_or_length_mismatch"
    if "integrated_header_found_but_pop_column_missing" in p:
        return "parser_column_ambiguity"
    if "without_volume_column" in p:
        return "volume_column_missing_or_all_null"
    return "other_or_unknown"


def _p90(values: List[float]) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    k = (len(arr) - 1) * 0.9
    f = int(math.floor(k))
    c = min(len(arr) - 1, f + 1)
    return float(arr[f] + (k - f) * (arr[c] - arr[f]))


def _size_bin(natm: Optional[int]) -> str:
    if not isinstance(natm, int):
        return "unknown"
    if natm <= 15:
        return "small_<=15"
    if natm <= 30:
        return "medium_16_30"
    return "large_>30"


def _family_label(smiles: Optional[str], atom_symbols: Optional[List[str]], rotatable_bonds: Optional[int]) -> str:
    smi = smiles or ""
    syms = atom_symbols or []
    aromatic = any(ch in smi for ch in ["c", "n", "o", "s", "p"])
    contains_on = any(sym in ("O", "N") for sym in syms)
    flexible = isinstance(rotatable_bonds, int) and rotatable_bonds >= 4
    if aromatic and flexible:
        return "aromatic_flexible"
    if aromatic and contains_on:
        return "aromatic_hetero"
    if aromatic:
        return "aromatic"
    if flexible:
        return "flexible_aliphatic"
    if contains_on:
        return "hetero_aliphatic"
    return "aliphatic_compact"


def _spacing_bin(spacing: Optional[float]) -> str:
    if spacing is None:
        return "unknown"
    if spacing <= 0.16:
        return "<=0.16A"
    if spacing <= 0.20:
        return "(0.16,0.20]A"
    if spacing <= 0.25:
        return "(0.20,0.25]A"
    return ">0.25A"

def _mismatch_abs_bin(abs_diff: Optional[float]) -> str:
    if abs_diff is None:
        return "unknown"
    if abs_diff <= 0.50:
        return "<=0.50e"
    if abs_diff <= 1.00:
        return "(0.50,1.00]e"
    if abs_diff <= 2.00:
        return "(1.00,2.00]e"
    return ">2.00e"


def _mismatch_rel_bin(rel_diff: Optional[float]) -> str:
    if rel_diff is None:
        return "unknown"
    if rel_diff <= 0.02:
        return "<=2%"
    if rel_diff <= 0.03:
        return "(2%,3%]"
    if rel_diff <= 0.05:
        return "(3%,5%]"
    return ">5%"


def _bool_key(v: bool) -> str:
    return "yes" if v else "no"


def _is_polar_on_molecule(atom_symbols: Optional[List[str]]) -> bool:
    syms = atom_symbols or []
    return any(sym in ("O", "N") for sym in syms)


def _natm_bin(natm: Optional[int]) -> str:
    if not isinstance(natm, int):
        return "unknown"
    if natm <= 12:
        return "<=12"
    if natm <= 24:
        return "13-24"
    if natm <= 36:
        return "25-36"
    return ">=37"


def build_report(
    unified_dir: Path,
    baseline_report: Optional[Path] = None,
    baseline_summary: Optional[Path] = None,
) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    n_total = len(files)

    overall_status = Counter()
    critic2_status = Counter()
    bader_status = Counter()
    bader_volume_status = Counter()
    bader_reason_raw = Counter()
    bader_reason_category = Counter()
    timing_realspace = []
    timing_critic2 = []

    strat_size = defaultdict(lambda: {"total": 0, "bader_success": 0, "critic2_success": 0})
    strat_family = defaultdict(lambda: {"total": 0, "bader_success": 0, "critic2_success": 0})
    strat_spacing = defaultdict(lambda: {"total": 0, "bader_success": 0, "critic2_success": 0})
    mismatch_abs_hist = Counter()
    mismatch_rel_hist = Counter()
    mismatch_natm_hist = Counter()
    mismatch_family_hist = Counter()
    mismatch_spacing_hist = Counter()
    mismatch_polar_on_hist = Counter()
    mismatch_rescue_triggered_hist = Counter()
    mismatch_validation_stage_hist = Counter()
    mismatch_current_refined_hist = Counter()
    mismatch_examples: List[Dict[str, Any]] = []
    rescue_stats = {
        "triggered_cases": 0,
        "recovered_cases": 0,
        "failed_cases": 0,
        "attempt_count_histogram": Counter(),
        "attempt_label_histogram": Counter(),
        "final_stage_histogram": Counter(),
    }
    threshold_scan_candidates: List[Dict[str, Any]] = []

    unavailable_examples: List[Dict[str, Any]] = []

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id", fp.stem)
        natm = _get(data, "molecule_info.natm")
        smiles = _get(data, "molecule_info.smiles")
        atom_symbols = _get(data, "geometry.atom_symbols")
        rotatable_bonds = _get(data, "global_features.rdkit.rotatable_bonds")
        spacing_arr = _get(data, "artifacts.cube_files.density.spacing_angstrom")
        spacing = float(spacing_arr[0]) if isinstance(spacing_arr, list) and spacing_arr else None

        st = _get(data, "calculation_status.overall_status", "unknown")
        c2 = _get(data, "external_bridge.critic2.execution_status", "unknown")
        bmeta = _get(data, "atom_features.metadata.atomic_density_partition_charge_proxy", {}) or {}
        bs = bmeta.get("bader_status", "missing")
        bsr = bmeta.get("bader_status_reason")
        bcat = bmeta.get("bader_status_category") or _classify_bader_reason(
            bsr,
            parser_note=bmeta.get("bader_population_parser_note"),
        )
        bvs = bmeta.get("bader_volume_status", "missing")
        bader = _get(data, "atom_features.atomic_density_partition_charge_proxy.bader")
        bvol = _get(data, "atom_features.atomic_density_partition_volume_proxy.bader")

        bader_non_null = isinstance(bader, list) and (not isinstance(natm, int) or len(bader) == natm)
        bvol_non_null = isinstance(bvol, list) and any(v is not None for v in bvol)
        expected_e = bmeta.get("bader_population_expected_electrons")
        parsed_sum = bmeta.get("bader_population_sum")
        stage = bmeta.get("bader_validation_stage")
        rescue_triggered = bool(bmeta.get("bader_rescue_triggered"))
        retry_attempts = bmeta.get("bader_rescue_attempts")
        is_current_refined = stage in ("refined_density_retry", "rescue_density_retry")

        if isinstance(retry_attempts, list):
            rescue_stats["attempt_count_histogram"][str(len(retry_attempts))] += 1
            for a in retry_attempts:
                if isinstance(a, dict):
                    rescue_stats["attempt_label_histogram"][str(a.get("label", "unknown"))] += 1

        overall_status[st] += 1
        critic2_status[c2] += 1
        bader_status[bs] += 1
        bader_volume_status[bvs] += 1
        if bs == "unavailable":
            bader_reason_raw[str(bsr)] += 1
            bader_reason_category[bcat] += 1
            unavailable_examples.append(
                {
                    "molecule_id": mid,
                    "natm": natm,
                    "smiles": smiles,
                    "bader_status_reason": bsr,
                    "bader_status_category": bcat,
                    "population_sum": bmeta.get("bader_population_sum"),
                    "expected_electrons": bmeta.get("bader_population_expected_electrons"),
                    "sum_tolerance": bmeta.get("bader_population_sum_tolerance"),
                    "validation_stage": bmeta.get("bader_validation_stage"),
                }
            )
            if bcat == "bader_population_sum_mismatch":
                abs_diff: Optional[float] = None
                rel_diff: Optional[float] = None
                if isinstance(expected_e, (int, float)) and isinstance(parsed_sum, (int, float)):
                    abs_diff = abs(float(parsed_sum) - float(expected_e))
                    rel_diff = abs_diff / max(1.0, abs(float(expected_e)))
                mismatch_abs_hist[_mismatch_abs_bin(abs_diff)] += 1
                mismatch_rel_hist[_mismatch_rel_bin(rel_diff)] += 1
                mismatch_natm_hist[_natm_bin(natm)] += 1
                mismatch_family_hist[_family_label(smiles, atom_symbols, rotatable_bonds)] += 1
                mismatch_spacing_hist[_spacing_bin(spacing)] += 1
                mismatch_polar_on_hist[_bool_key(_is_polar_on_molecule(atom_symbols))] += 1
                mismatch_rescue_triggered_hist[_bool_key(rescue_triggered)] += 1
                mismatch_validation_stage_hist[str(stage or "unknown")] += 1
                mismatch_current_refined_hist[_bool_key(is_current_refined)] += 1
                mismatch_examples.append(
                    {
                        "molecule_id": mid,
                        "natm": natm,
                        "smiles": smiles,
                        "expected_electrons": expected_e,
                        "parsed_population_sum": parsed_sum,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                        "validation_stage": stage,
                        "rescue_triggered": rescue_triggered,
                        "retry_attempts": retry_attempts,
                        "spacing_angstrom": spacing,
                    }
                )

        rt = _get(data, "realspace_features.metadata.extraction_time_seconds")
        if isinstance(rt, (int, float)):
            timing_realspace.append(float(rt))
        ct = _get(data, "external_bridge.critic2.execution_time_seconds")
        if isinstance(ct, (int, float)):
            timing_critic2.append(float(ct))

        size_key = _size_bin(natm)
        fam_key = _family_label(smiles, atom_symbols, rotatable_bonds)
        spacing_key = _spacing_bin(spacing)
        for key, box in (
            (size_key, strat_size),
            (fam_key, strat_family),
            (spacing_key, strat_spacing),
        ):
            box[key]["total"] += 1
            if c2 == "success":
                box[key]["critic2_success"] += 1
            if bader_non_null:
                box[key]["bader_success"] += 1

        if rescue_triggered:
            rescue_stats["triggered_cases"] += 1
            rescue_stats["final_stage_histogram"][str(stage or "unknown")] += 1
            if bader_non_null:
                rescue_stats["recovered_cases"] += 1
            else:
                rescue_stats["failed_cases"] += 1

        # offline threshold scan candidate pool: critic2 success + bader unavailable + sum mismatch + valid sums.
        if (
            c2 == "success"
            and bs == "unavailable"
            and bcat == "bader_population_sum_mismatch"
            and isinstance(expected_e, (int, float))
            and isinstance(parsed_sum, (int, float))
        ):
            threshold_scan_candidates.append(
                {
                    "molecule_id": mid,
                    "natm": natm,
                    "expected_electrons": float(expected_e),
                    "parsed_population_sum": float(parsed_sum),
                    "abs_diff": abs(float(parsed_sum) - float(expected_e)),
                    "rel_diff": abs(float(parsed_sum) - float(expected_e)) / max(1.0, abs(float(expected_e))),
                    "validation_stage": stage,
                }
            )

    def enrich_rate_rows(rows: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, vals in rows.items():
            total = max(1, vals["total"])
            out[key] = {
                **vals,
                "critic2_success_rate": vals["critic2_success"] / total,
                "bader_writeback_rate": vals["bader_success"] / total,
            }
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    baseline_ratio = None
    baseline_reason_category_hist: Optional[Dict[str, int]] = None
    baseline_reason_raw_hist: Optional[Dict[str, int]] = None
    if baseline_summary and baseline_summary.exists():
        bs = json.loads(baseline_summary.read_text(encoding="utf-8"))
        baseline_ratio = _get(bs, "coverage.bader_charge_validated_writeback_rate")
        baseline_reason_category_hist = _get(bs, "unavailable_reason_histogram.category")
        baseline_reason_raw_hist = _get(bs, "unavailable_reason_histogram.raw_reason")
    if baseline_report and baseline_report.exists():
        b = json.loads(baseline_report.read_text(encoding="utf-8"))
        if baseline_ratio is None:
            baseline_ratio = _get(b, "A_bader_charge_success_rate.non_null_ratio_overall")

    bader_writeback_count = sum(1 for _ in files if False)  # placeholder for shape; overwritten below
    bader_writeback_count = 0
    bader_volume_writeback_count = 0
    critic2_success_count = 0
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        natm = _get(data, "molecule_info.natm")
        bader = _get(data, "atom_features.atomic_density_partition_charge_proxy.bader")
        bvol = _get(data, "atom_features.atomic_density_partition_volume_proxy.bader")
        if _get(data, "external_bridge.critic2.execution_status") == "success":
            critic2_success_count += 1
        if isinstance(bader, list) and (not isinstance(natm, int) or len(bader) == natm):
            bader_writeback_count += 1
        if isinstance(bvol, list) and any(v is not None for v in bvol):
            bader_volume_writeback_count += 1

    current_ratio = bader_writeback_count / max(1, n_total)
    uplift_pp = None if baseline_ratio is None else (current_ratio - float(baseline_ratio)) * 100.0

    report = {
        "input": {
            "unified_dir": str(unified_dir),
            "n_files": n_total,
            "baseline_report": str(baseline_report) if baseline_report else None,
        },
        "coverage": {
            "overall_status_distribution": dict(overall_status),
            "critic2_execution_success_rate": critic2_success_count / max(1, n_total),
            "bader_charge_validated_writeback_rate": current_ratio,
            "bader_volume_writeback_rate": bader_volume_writeback_count / max(1, n_total),
            "counts": {
                "total": n_total,
                "critic2_success": critic2_success_count,
                "bader_charge_writeback": bader_writeback_count,
                "bader_volume_writeback": bader_volume_writeback_count,
            },
            "baseline_bader_charge_writeback_rate": baseline_ratio,
            "bader_charge_uplift_percentage_points": uplift_pp,
        },
        "status_histogram": {
            "critic2_status": dict(critic2_status),
            "bader_status": dict(bader_status),
            "bader_volume_status": dict(bader_volume_status),
        },
        "unavailable_reason_histogram": {
            "raw_reason": dict(bader_reason_raw),
            "category": dict(bader_reason_category),
            "baseline_raw_reason": baseline_reason_raw_hist,
            "baseline_category": baseline_reason_category_hist,
        },
        "stratified": {
            "by_size_bin": enrich_rate_rows(strat_size),
            "by_family": enrich_rate_rows(strat_family),
            "by_grid_spacing_bin": enrich_rate_rows(strat_spacing),
        },
        "timing": {
            "realspace_mean_seconds": statistics.mean(timing_realspace) if timing_realspace else None,
            "realspace_p90_seconds": _p90(timing_realspace),
            "critic2_mean_seconds": statistics.mean(timing_critic2) if timing_critic2 else None,
            "critic2_p90_seconds": _p90(timing_critic2),
        },
        "top_unavailable_examples": unavailable_examples[:20],
        "mismatch_grouping": {
            "count": int(sum(mismatch_abs_hist.values())),
            "abs_diff_bin_histogram": dict(mismatch_abs_hist),
            "rel_diff_bin_histogram": dict(mismatch_rel_hist),
            "natm_bin_histogram": dict(mismatch_natm_hist),
            "family_histogram": dict(mismatch_family_hist),
            "spacing_bin_histogram": dict(mismatch_spacing_hist),
            "polar_on_histogram": dict(mismatch_polar_on_hist),
            "rescue_triggered_histogram": dict(mismatch_rescue_triggered_hist),
            "validation_stage_histogram": dict(mismatch_validation_stage_hist),
            "current_refined_config_histogram": dict(mismatch_current_refined_hist),
            "examples": mismatch_examples[:30],
        },
        "rescue_retry": {
            "triggered_cases": rescue_stats["triggered_cases"],
            "recovered_cases": rescue_stats["recovered_cases"],
            "failed_cases": rescue_stats["failed_cases"],
            "recovery_rate_when_triggered": (
                rescue_stats["recovered_cases"] / max(1, rescue_stats["triggered_cases"])
            ),
            "attempt_count_histogram": dict(rescue_stats["attempt_count_histogram"]),
            "attempt_label_histogram": dict(rescue_stats["attempt_label_histogram"]),
            "final_stage_histogram": dict(rescue_stats["final_stage_histogram"]),
        },
    }
    if isinstance(baseline_reason_category_hist, dict):
        delta = {}
        keys = set(baseline_reason_category_hist.keys()) | set(report["unavailable_reason_histogram"]["category"].keys())
        for k in sorted(keys):
            now_v = int(report["unavailable_reason_histogram"]["category"].get(k, 0))
            base_v = int(baseline_reason_category_hist.get(k, 0))
            delta[k] = now_v - base_v
        report["unavailable_reason_histogram"]["category_delta_vs_baseline"] = delta

    # Offline threshold sensitivity scan (analysis only, no logic change).
    scan_rows: Dict[str, Any] = {}
    baseline_pass = len(threshold_scan_candidates)  # current status for this pool: all failed
    for policy_name, policy in DEFAULT_THRESHOLD_POLICIES.items():
        abs_tol = float(policy["abs_tol_e"])
        rel_tol = float(policy["rel_tol"])
        rescued = []
        for row in threshold_scan_candidates:
            tol = max(abs_tol, rel_tol * max(1.0, abs(row["expected_electrons"])))
            if row["abs_diff"] <= tol:
                rescued.append(
                    {
                        "molecule_id": row["molecule_id"],
                        "natm": row["natm"],
                        "abs_diff": row["abs_diff"],
                        "rel_diff": row["rel_diff"],
                        "tol": tol,
                        "validation_stage": row["validation_stage"],
                    }
                )
        scan_rows[policy_name] = {
            "policy": policy,
            "candidate_pool_size": len(threshold_scan_candidates),
            "additional_pass_cases": len(rescued),
            "additional_pass_ratio_in_candidate_pool": (
                len(rescued) / max(1, len(threshold_scan_candidates))
            ),
            "additional_pass_examples": rescued[:30],
        }
    report["offline_threshold_scan"] = {
        "note": "analysis only; canonical writeback logic unchanged",
        "baseline_candidate_pool_size": baseline_pass,
        "policies": scan_rows,
    }

    return report


def write_outputs(report: Dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cov = report["coverage"]
    lines = [
        "# Bader Coverage Uplift Summary",
        f"- total molecules: {cov['counts']['total']}",
        f"- critic2 execution success rate: {cov['critic2_execution_success_rate']:.3f}",
        f"- bader charge validated writeback rate: {cov['bader_charge_validated_writeback_rate']:.3f}",
        f"- bader volume writeback rate: {cov['bader_volume_writeback_rate']:.3f}",
        f"- baseline bader rate: {cov['baseline_bader_charge_writeback_rate']}",
        f"- bader uplift (pp): {cov['bader_charge_uplift_percentage_points']}",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize bader coverage and unavailable reasons from unified outputs")
    p.add_argument("--unified-dir", required=True, help="Directory containing *.unified.json")
    p.add_argument("--output-json", default="", help="Output JSON path")
    p.add_argument("--output-md", default="", help="Output Markdown path")
    p.add_argument("--baseline-report", default="", help="Previous validation_round_report.json for uplift comparison")
    p.add_argument("--baseline-summary", default="", help="Previous bader_coverage_uplift_summary.json for reason histogram diff")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    unified_dir = Path(args.unified_dir).resolve()
    if not unified_dir.exists():
        raise FileNotFoundError(f"unified-dir not found: {unified_dir}")

    output_json = Path(args.output_json).resolve() if args.output_json else unified_dir / "bader_coverage_uplift_summary.json"
    output_md = Path(args.output_md).resolve() if args.output_md else unified_dir / "bader_coverage_uplift_summary.md"
    baseline_report = Path(args.baseline_report).resolve() if args.baseline_report else None
    baseline_summary = Path(args.baseline_summary).resolve() if args.baseline_summary else None
    report = build_report(unified_dir, baseline_report=baseline_report, baseline_summary=baseline_summary)
    write_outputs(report, output_json, output_md)
    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "bader_charge_rate": report["coverage"]["bader_charge_validated_writeback_rate"],
        "bader_uplift_pp": report["coverage"]["bader_charge_uplift_percentage_points"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

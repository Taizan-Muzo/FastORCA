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
    }
    if isinstance(baseline_reason_category_hist, dict):
        delta = {}
        keys = set(baseline_reason_category_hist.keys()) | set(report["unavailable_reason_histogram"]["category"].keys())
        for k in sorted(keys):
            now_v = int(report["unavailable_reason_histogram"]["category"].get(k, 0))
            base_v = int(baseline_reason_category_hist.get(k, 0))
            delta[k] = now_v - base_v
        report["unavailable_reason_histogram"]["category_delta_vs_baseline"] = delta

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

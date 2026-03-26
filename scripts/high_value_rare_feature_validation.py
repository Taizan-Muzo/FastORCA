"""
High-Value / Rare-Feature augmentation validation summary.

Reads unified outputs and reports availability / sample values for:
- critic2 integrated high-value layer
- conformer-aware candidate-set statistics
- proxy-family global aggregate summaries
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_non_null_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and any(x is not None for x in v)


def _non_null(v: Any) -> bool:
    return v is not None


def build_report(unified_dir: Path, max_samples: int = 8) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    n = len(files)

    module_a = {
        "laplacian_canonical_available": 0,
        "laplacian_external_available": 0,
        "candidate_status_hist": Counter(),
        "laplacian_unavailable_reason_hist": Counter(),
        "samples": [],
    }
    module_b = {
        "candidate_stats_available": 0,
        "energy_span_available": 0,
        "size_variability_available": 0,
        "compactness_available": 0,
        "samples": [],
    }
    module_c = {
        "global_proxy_summary_available": 0,
        "atom_charge_dispersion_available": 0,
        "hetero_charge_extrema_available": 0,
        "lone_pair_count_available": 0,
        "bond_deloc_extrema_available": 0,
        "high_deloc_bond_count_available": 0,
        "status_reason_hist": Counter(),
        "samples": [],
    }

    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id", fp.stem)

        # Module A
        lap_can = _get(data, "atom_features.atomic_density_partition_laplacian_proxy_v1.bader")
        lap_ext = _get(
            data,
            "external_features.critic2.qtaim.stable_atomic_integrated_properties_v1.laplacian_integral",
        )
        lap_reason = _get(data, "atom_features.metadata.atomic_density_partition_laplacian_proxy_v1.bader_status_reason")
        if _is_non_null_list(lap_can):
            module_a["laplacian_canonical_available"] += 1
        else:
            module_a["laplacian_unavailable_reason_hist"][str(lap_reason)] += 1
        if _is_non_null_list(lap_ext):
            module_a["laplacian_external_available"] += 1
        cand = _get(data, "external_features.critic2.qtaim.atomic_integrated_property_candidate_assessment_v1", {}) or {}
        if isinstance(cand, dict):
            for k, v in cand.items():
                status = None
                if isinstance(v, dict):
                    status = v.get("status")
                module_a["candidate_status_hist"][f"{k}:{status}"] += 1
        if len(module_a["samples"]) < max_samples:
            module_a["samples"].append(
                {
                    "molecule_id": mid,
                    "lap_canonical_preview": None if not isinstance(lap_can, list) else lap_can[:3],
                    "lap_external_preview": None if not isinstance(lap_ext, list) else lap_ext[:3],
                    "lap_status_reason": lap_reason,
                    "candidate_assessment": cand,
                }
            )

        # Module B
        conf_stats = _get(data, "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1")
        if isinstance(conf_stats, dict) and conf_stats.get("available") is True:
            module_b["candidate_stats_available"] += 1
        energy_span = _get(data, "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1.conformer_energy_span_proxy")
        size_var = _get(data, "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1.geometry_size_variability_proxy")
        compact = _get(data, "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1.conformer_compactness_proxy_v1")
        if _non_null(energy_span):
            module_b["energy_span_available"] += 1
        if _non_null(size_var):
            module_b["size_variability_available"] += 1
        if _non_null(compact):
            module_b["compactness_available"] += 1
        if len(module_b["samples"]) < max_samples:
            module_b["samples"].append(
                {
                    "molecule_id": mid,
                    "conformer_count_ranked": _get(data, "structural_features.most_stable_conformation.n_conformers_ranked"),
                    "energy_span": energy_span,
                    "size_variability": size_var,
                    "compactness": compact,
                }
            )

        # Module C
        proxy_sum = _get(data, "global_features.proxy_family_summary_v1")
        if isinstance(proxy_sum, dict) and proxy_sum.get("available") is True:
            module_c["global_proxy_summary_available"] += 1
        module_c["status_reason_hist"][str(_get(data, "global_features.proxy_family_summary_v1.metadata.status_reason"))] += 1
        if _non_null(_get(data, "global_features.proxy_family_summary_v1.atom_charge_dispersion_proxy")):
            module_c["atom_charge_dispersion_available"] += 1
        if _non_null(_get(data, "global_features.proxy_family_summary_v1.hetero_atom_charge_extrema_proxy")):
            module_c["hetero_charge_extrema_available"] += 1
        if _non_null(_get(data, "global_features.proxy_family_summary_v1.lone_pair_rich_atom_count_proxy")):
            module_c["lone_pair_count_available"] += 1
        if _non_null(_get(data, "global_features.proxy_family_summary_v1.bond_delocalization_extrema_proxy")):
            module_c["bond_deloc_extrema_available"] += 1
        if _non_null(_get(data, "global_features.proxy_family_summary_v1.high_delocalization_bond_count_proxy")):
            module_c["high_deloc_bond_count_available"] += 1
        if len(module_c["samples"]) < max_samples:
            module_c["samples"].append(
                {
                    "molecule_id": mid,
                    "summary_status": _get(data, "global_features.proxy_family_summary_v1.metadata.status"),
                    "status_reason": _get(data, "global_features.proxy_family_summary_v1.metadata.status_reason"),
                    "atom_charge_dispersion_proxy": _get(data, "global_features.proxy_family_summary_v1.atom_charge_dispersion_proxy"),
                    "lone_pair_rich_atom_count_proxy": _get(data, "global_features.proxy_family_summary_v1.lone_pair_rich_atom_count_proxy"),
                    "bond_delocalization_extrema_proxy": _get(data, "global_features.proxy_family_summary_v1.bond_delocalization_extrema_proxy"),
                }
            )

    def _rate(v: int) -> float:
        return float(v / max(1, n))

    return {
        "input": {"unified_dir": str(unified_dir), "n_files": n},
        "module_a_critic2_integrated": {
            "laplacian_canonical_available_rate": _rate(module_a["laplacian_canonical_available"]),
            "laplacian_external_available_rate": _rate(module_a["laplacian_external_available"]),
            "candidate_status_hist": dict(module_a["candidate_status_hist"]),
            "laplacian_unavailable_reason_hist": dict(module_a["laplacian_unavailable_reason_hist"]),
            "samples": module_a["samples"],
        },
        "module_b_conformer_aware": {
            "candidate_stats_available_rate": _rate(module_b["candidate_stats_available"]),
            "energy_span_available_rate": _rate(module_b["energy_span_available"]),
            "size_variability_available_rate": _rate(module_b["size_variability_available"]),
            "compactness_available_rate": _rate(module_b["compactness_available"]),
            "samples": module_b["samples"],
        },
        "module_c_proxy_aggregates": {
            "global_proxy_summary_available_rate": _rate(module_c["global_proxy_summary_available"]),
            "atom_charge_dispersion_available_rate": _rate(module_c["atom_charge_dispersion_available"]),
            "hetero_charge_extrema_available_rate": _rate(module_c["hetero_charge_extrema_available"]),
            "lone_pair_count_available_rate": _rate(module_c["lone_pair_count_available"]),
            "bond_deloc_extrema_available_rate": _rate(module_c["bond_deloc_extrema_available"]),
            "high_deloc_bond_count_available_rate": _rate(module_c["high_deloc_bond_count_available"]),
            "status_reason_hist": dict(module_c["status_reason_hist"]),
            "samples": module_c["samples"],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate high-value rare-feature augmentation on unified outputs")
    p.add_argument("--unified-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--max-samples", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    unified_dir = Path(args.unified_dir).resolve()
    if not unified_dir.exists():
        raise FileNotFoundError(f"unified-dir not found: {unified_dir}")
    report = build_report(unified_dir, max_samples=int(args.max_samples))
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# High-Value Rare-Feature Validation",
        f"- total files: {report['input']['n_files']}",
        f"- module A laplacian canonical available rate: {report['module_a_critic2_integrated']['laplacian_canonical_available_rate']}",
        f"- module B candidate stats available rate: {report['module_b_conformer_aware']['candidate_stats_available_rate']}",
        f"- module C proxy summary available rate: {report['module_c_proxy_aggregates']['global_proxy_summary_available_rate']}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(out_json),
                "output_md": str(out_md),
                "n_files": report["input"]["n_files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

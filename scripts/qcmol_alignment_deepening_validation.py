"""
qcMol Feature Alignment Deepening validation summary.

This script validates the deepening sprint additions:
- Module A: critic2/basin family deepening
- Module B: atom/bond proxy alignment deepening
- Module C: molecule-level scarce summary deepening
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_non_null_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and any(x is not None for x in v)


def _is_non_null(v: Any) -> bool:
    return v is not None


def _rate(x: int, n: int) -> float:
    return float(x / max(1, n))


def _judgement_from_rate(rate: float) -> str:
    if rate >= 0.80:
        return "implemented"
    if rate > 0.0:
        return "partial"
    return "rejected"


def _molecule_categories(data: Dict[str, Any]) -> Dict[str, bool]:
    heavy = _get(data, "global_features.rdkit.heavy_atom_count")
    aromatic = _get(data, "atom_features.rdkit_aromatic")
    rot = _get(data, "global_features.rdkit.rotatable_bonds")
    symbols = _get(data, "geometry.atom_symbols") or []
    has_on = any(str(s).upper() in {"O", "N"} for s in symbols)
    return {
        "small": isinstance(heavy, int) and heavy <= 6,
        "aromatic": isinstance(aromatic, list) and any(bool(x) for x in aromatic),
        "flexible": isinstance(rot, int) and rot >= 3,
        "contains_O_N": has_on,
    }


def _pick_category_samples(rows: List[Tuple[str, Dict[str, Any]]], max_per_category: int = 2) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "small": [],
        "aromatic": [],
        "flexible": [],
        "contains_O_N": [],
    }
    for mid, data in rows:
        cats = _molecule_categories(data)
        for c, ok in cats.items():
            if ok and len(out[c]) < max_per_category:
                out[c].append(mid)
    return out


def _collect_sample_payload(
    rows: List[Tuple[str, Dict[str, Any]]],
    sample_ids: List[str],
) -> List[Dict[str, Any]]:
    wanted = set(sample_ids)
    out: List[Dict[str, Any]] = []
    for mid, data in rows:
        if mid not in wanted:
            continue
        out.append(
            {
                "molecule_id": mid,
                "categories": _molecule_categories(data),
                "module_a_basin": _get(data, "global_features.basin_proxy_summary_v1"),
                "module_b_atom": {
                    "atomic_charge_laplacian_coupling_proxy_v1": _get(data, "atom_features.atomic_charge_laplacian_coupling_proxy_v1"),
                    "atomic_local_reactivity_proxy_v1": _get(data, "atom_features.atomic_local_reactivity_proxy_v1"),
                    "lone_pair_environment_proxy_v1": _get(data, "atom_features.lone_pair_environment_proxy_v1"),
                },
                "module_b_bond": {
                    "bond_covalency_polarity_proxy_v1": _get(data, "bond_features.bond_covalency_polarity_proxy_v1"),
                    "bond_delocalization_localization_balance_proxy_v1": _get(data, "bond_features.bond_delocalization_localization_balance_proxy_v1"),
                    "bond_elf_deloc_coupling_proxy_v1": _get(data, "bond_features.bond_elf_deloc_coupling_proxy_v1"),
                    "bond_strength_pattern_proxy_v1": _get(data, "bond_features.bond_strength_pattern_proxy_v1"),
                },
                "module_c_summary": {
                    "polarity_heterogeneity_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.polarity_heterogeneity_proxy_v1"),
                    "basin_charge_asymmetry_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.basin_charge_asymmetry_proxy_v1"),
                    "localized_vs_delocalized_balance_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.localized_vs_delocalized_balance_proxy_v1"),
                    "conformer_sensitivity_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.conformer_sensitivity_proxy_v1"),
                    "electronic_compactness_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.electronic_compactness_proxy_v1"),
                    "lone_pair_driven_polarity_proxy_v1": _get(data, "global_features.proxy_family_summary_v1.lone_pair_driven_polarity_proxy_v1"),
                },
            }
        )
    return out


def build_report(unified_dir: Path, max_samples: int = 8) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    n = len(files)
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id", fp.stem)
        rows.append((mid, data))

    # ---------------- Module A ----------------
    module_a = {
        "basin_summary_available": 0,
        "external_basin_companion_available": 0,
        "feature_available": Counter(),
        "candidate_status_hist": Counter(),
        "status_reason_hist": Counter(),
        "candidate_reason_hist": Counter(),
    }
    a_features = [
        "bader_population_dispersion_proxy",
        "hetero_bader_charge_extrema_proxy",
        "bader_laplacian_extrema_proxy",
        "bader_laplacian_dispersion_proxy",
        "atomwise_basin_companion_summary_proxy_v1",
    ]
    for _, data in rows:
        basin = _get(data, "global_features.basin_proxy_summary_v1") or {}
        if isinstance(basin, dict) and basin.get("available") is True:
            module_a["basin_summary_available"] += 1
        if _get(data, "external_features.critic2.qtaim.basin_companion_summary_v1.available") is True:
            module_a["external_basin_companion_available"] += 1
        module_a["status_reason_hist"][str(_get(data, "global_features.basin_proxy_summary_v1.metadata.status_reason"))] += 1
        for key in a_features:
            if _is_non_null(basin.get(key)):
                module_a["feature_available"][key] += 1
        cand = _get(data, "global_features.basin_proxy_summary_v1.metadata.candidate_assessment_v1") or {}
        if isinstance(cand, dict):
            for k, v in cand.items():
                status = v.get("status") if isinstance(v, dict) else None
                reason = v.get("reason") if isinstance(v, dict) else None
                module_a["candidate_status_hist"][f"{k}:{status}"] += 1
                if reason:
                    module_a["candidate_reason_hist"][str(reason)] += 1

    # ---------------- Module B ----------------
    module_b = {
        "atom_feature_available": Counter(),
        "bond_feature_available": Counter(),
        "atom_status_reason_hist": defaultdict(Counter),
        "bond_status_reason_hist": defaultdict(Counter),
    }
    b_atom_features = [
        "atomic_charge_laplacian_coupling_proxy_v1",
        "atomic_local_reactivity_proxy_v1",
        "lone_pair_environment_proxy_v1",
    ]
    b_bond_features = [
        "bond_covalency_polarity_proxy_v1",
        "bond_delocalization_localization_balance_proxy_v1",
        "bond_elf_deloc_coupling_proxy_v1",
        "bond_strength_pattern_proxy_v1",
    ]
    for _, data in rows:
        for key in b_atom_features:
            val = _get(data, f"atom_features.{key}")
            if _is_non_null_list(val):
                module_b["atom_feature_available"][key] += 1
            reason = _get(data, f"atom_features.metadata.{key}.status_reason")
            module_b["atom_status_reason_hist"][key][str(reason)] += 1
        for key in b_bond_features:
            val = _get(data, f"bond_features.{key}")
            if _is_non_null_list(val):
                module_b["bond_feature_available"][key] += 1
            reason = _get(data, f"bond_features.metadata.{key}.status_reason")
            module_b["bond_status_reason_hist"][key][str(reason)] += 1

    # ---------------- Module C ----------------
    module_c = {
        "summary_available": 0,
        "feature_available": Counter(),
        "status_reason_hist": Counter(),
    }
    c_features = [
        "polarity_heterogeneity_proxy_v1",
        "basin_charge_asymmetry_proxy_v1",
        "localized_vs_delocalized_balance_proxy_v1",
        "conformer_sensitivity_proxy_v1",
        "electronic_compactness_proxy_v1",
        "lone_pair_driven_polarity_proxy_v1",
    ]
    for _, data in rows:
        summary = _get(data, "global_features.proxy_family_summary_v1") or {}
        if isinstance(summary, dict) and summary.get("available") is True:
            module_c["summary_available"] += 1
        module_c["status_reason_hist"][str(_get(data, "global_features.proxy_family_summary_v1.metadata.status_reason"))] += 1
        for key in c_features:
            if _is_non_null(summary.get(key)):
                module_c["feature_available"][key] += 1

    # Samples by category
    category_samples = _pick_category_samples(rows, max_per_category=2)
    ordered_ids: List[str] = []
    for c in ("small", "aromatic", "flexible", "contains_O_N"):
        for mid in category_samples[c]:
            if mid not in ordered_ids:
                ordered_ids.append(mid)
    ordered_ids = ordered_ids[: max(1, max_samples)]
    sample_payload = _collect_sample_payload(rows, ordered_ids)

    module_a_feature_rates = {k: _rate(module_a["feature_available"][k], n) for k in a_features}
    module_b_atom_rates = {k: _rate(module_b["atom_feature_available"][k], n) for k in b_atom_features}
    module_b_bond_rates = {k: _rate(module_b["bond_feature_available"][k], n) for k in b_bond_features}
    module_c_feature_rates = {k: _rate(module_c["feature_available"][k], n) for k in c_features}

    judgement = {
        "module_a": {k: _judgement_from_rate(v) for k, v in module_a_feature_rates.items()},
        "module_b_atom": {k: _judgement_from_rate(v) for k, v in module_b_atom_rates.items()},
        "module_b_bond": {k: _judgement_from_rate(v) for k, v in module_b_bond_rates.items()},
        "module_c": {k: _judgement_from_rate(v) for k, v in module_c_feature_rates.items()},
    }

    return {
        "input": {
            "unified_dir": str(unified_dir),
            "n_files": n,
        },
        "module_a_critic2_basin_deepening": {
            "basin_summary_available_rate": _rate(module_a["basin_summary_available"], n),
            "external_basin_companion_available_rate": _rate(module_a["external_basin_companion_available"], n),
            "feature_available_rates": module_a_feature_rates,
            "candidate_status_hist": dict(module_a["candidate_status_hist"]),
            "status_reason_hist": dict(module_a["status_reason_hist"]),
            "candidate_reason_hist_top10": dict(module_a["candidate_reason_hist"].most_common(10)),
        },
        "module_b_atom_bond_proxy_deepening": {
            "atom_feature_available_rates": module_b_atom_rates,
            "bond_feature_available_rates": module_b_bond_rates,
            "atom_status_reason_hist": {k: dict(v) for k, v in module_b["atom_status_reason_hist"].items()},
            "bond_status_reason_hist": {k: dict(v) for k, v in module_b["bond_status_reason_hist"].items()},
        },
        "module_c_molecule_level_summary_deepening": {
            "summary_available_rate": _rate(module_c["summary_available"], n),
            "feature_available_rates": module_c_feature_rates,
            "status_reason_hist": dict(module_c["status_reason_hist"]),
        },
        "implemented_partial_rejected_judgement": judgement,
        "sample_ids": ordered_ids,
        "samples": sample_payload,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate qcMol alignment deepening sprint features")
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
        "# qcMol Alignment Deepening Validation",
        f"- total files: {report['input']['n_files']}",
        f"- module A basin summary available rate: {report['module_a_critic2_basin_deepening']['basin_summary_available_rate']}",
        f"- module B atom deepening median available rate: {sum(report['module_b_atom_bond_proxy_deepening']['atom_feature_available_rates'].values())/max(1,len(report['module_b_atom_bond_proxy_deepening']['atom_feature_available_rates']))}",
        f"- module B bond deepening median available rate: {sum(report['module_b_atom_bond_proxy_deepening']['bond_feature_available_rates'].values())/max(1,len(report['module_b_atom_bond_proxy_deepening']['bond_feature_available_rates']))}",
        f"- module C summary available rate: {report['module_c_molecule_level_summary_deepening']['summary_available_rate']}",
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

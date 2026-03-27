"""
qcMol Alignment Closure Sprint validation summary.

Validates closure-sprint enhancements and reports:
- availability/judgement for new high-value closure features
- alignment master-table status snapshots and status deltas
- remaining hard gaps under current open-source constraints
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (  # noqa: E402
    EXACT_ONLY_ARCHIVED_LIST,
    QCMOL_ALIGNMENT_MASTER_TABLE,
    remaining_alignment_gaps,
    summarize_alignment_next_actions,
    summarize_alignment_status_counts,
)


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_available(v: Any) -> bool:
    if isinstance(v, list):
        return len(v) > 0 and any(x is not None for x in v)
    if isinstance(v, dict):
        return len(v) > 0
    return v is not None


def _rate(x: int, n: int) -> float:
    return float(x / max(1, n))


def _judgement(rate: float) -> str:
    if rate >= 0.80:
        return "implemented"
    if rate > 0.0:
        return "partial"
    return "rejected"


def _category_flags(data: Dict[str, Any]) -> Dict[str, bool]:
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


def _pick_sample_ids(rows: List[Tuple[str, Dict[str, Any]]], max_per_category: int = 2) -> List[str]:
    buckets: Dict[str, List[str]] = {"small": [], "aromatic": [], "flexible": [], "contains_O_N": []}
    for mid, data in rows:
        flags = _category_flags(data)
        for k in buckets:
            if flags[k] and len(buckets[k]) < max_per_category:
                buckets[k].append(mid)
    ordered: List[str] = []
    for k in ("small", "aromatic", "flexible", "contains_O_N"):
        for mid in buckets[k]:
            if mid not in ordered:
                ordered.append(mid)
    return ordered


def _module_feature_report(
    rows: List[Tuple[str, Dict[str, Any]]],
    specs: List[Dict[str, str]],
) -> Dict[str, Any]:
    n = len(rows)
    available_counts: Counter[str] = Counter()
    reason_hist: Dict[str, Counter[str]] = defaultdict(Counter)
    samples: List[Dict[str, Any]] = []

    sample_ids = set(_pick_sample_ids(rows))
    for mid, data in rows:
        sample_row: Dict[str, Any] = {"molecule_id": mid, "categories": _category_flags(data)}
        for spec in specs:
            feature = spec["name"]
            v = _get(data, spec["value_path"])
            if _is_available(v):
                available_counts[feature] += 1
            reason = _get(data, spec["reason_path"])
            if reason is not None:
                reason_hist[feature][str(reason)] += 1
            sample_row[feature] = v
        if mid in sample_ids:
            samples.append(sample_row)

    rates = {spec["name"]: _rate(available_counts[spec["name"]], n) for spec in specs}
    judgements = {k: _judgement(v) for k, v in rates.items()}
    return {
        "feature_available_rates": rates,
        "implemented_partial_rejected": judgements,
        "reason_hist_top10": {k: dict(v.most_common(10)) for k, v in reason_hist.items()},
        "samples": samples[:8],
    }


def _master_table_delta() -> Dict[str, Any]:
    # Previous sprint snapshot did not have "rejected_as_exact", those rows were treated as missing.
    prev_status: Dict[str, str] = {}
    for row in QCMOL_ALIGNMENT_MASTER_TABLE:
        cur = str(row.get("current_status"))
        prev_status[row["qcMol_item_name"]] = "missing" if cur == "rejected_as_exact" else cur

    changed: List[Dict[str, Any]] = []
    for row in QCMOL_ALIGNMENT_MASTER_TABLE:
        item = row["qcMol_item_name"]
        prev = prev_status.get(item)
        cur = row["current_status"]
        if prev != cur:
            changed.append(
                {
                    "qcMol_item_name": item,
                    "previous_status": prev,
                    "current_status": cur,
                    "next_action": row.get("next_action"),
                }
            )

    hard_gaps = [
        {
            "qcMol_item_name": row["qcMol_item_name"],
            "current_status": row["current_status"],
            "next_action": row["next_action"],
            "mapped_path": row["mapped_path"],
            "notes": row.get("notes"),
        }
        for row in remaining_alignment_gaps(QCMOL_ALIGNMENT_MASTER_TABLE)
        if row.get("current_status") in {"missing", "partial", "rejected_as_exact"}
    ]
    roadmap_or_rejected = [
        {
            "qcMol_item_name": row["qcMol_item_name"],
            "current_status": row["current_status"],
            "next_action": row["next_action"],
            "notes": row.get("notes"),
        }
        for row in QCMOL_ALIGNMENT_MASTER_TABLE
        if row.get("next_action") in {"roadmap_only", "reject"} or row.get("current_status") == "rejected_as_exact"
    ]

    return {
        "status_counts": summarize_alignment_status_counts(
            [{"status": row.get("current_status")} for row in QCMOL_ALIGNMENT_MASTER_TABLE]
        ),
        "next_action_counts": summarize_alignment_next_actions(QCMOL_ALIGNMENT_MASTER_TABLE),
        "status_changes_vs_previous_snapshot": changed,
        "remaining_hard_gaps": hard_gaps,
        "roadmap_only_or_rejected_as_exact": roadmap_or_rejected,
        "exact_only_archived_list": EXACT_ONLY_ARCHIVED_LIST,
    }


def _final_sprint_focus_report(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    bader_volume_available = 0
    mayer_ok = 0
    optimized_ref_ok = 0
    smart_proxy_ok = 0
    molecule_size_proxy_ok = 0
    ionization_proxy_ok = 0
    di_proxy_frozen_ok = 0

    volume_status_hist: Counter[str] = Counter()
    volume_reason_hist: Counter[str] = Counter()
    mayer_reason_hist: Counter[str] = Counter()
    ionization_reason_hist: Counter[str] = Counter()

    samples: List[Dict[str, Any]] = []
    for idx, (mid, data) in enumerate(rows):
        natm = _get(data, "molecule_info.natm")
        bader_volume = _get(data, "atom_features.atomic_density_partition_volume_proxy.bader")
        if isinstance(bader_volume, list) and len(bader_volume) > 0 and all(v is not None for v in bader_volume):
            bader_volume_available += 1

        vol_status = _get(data, "atom_features.metadata.atomic_density_partition_volume_proxy.bader_status")
        if vol_status is None:
            vol_status = _get(data, "atom_features.metadata.atomic_density_partition_charge_proxy.bader_volume_status")
        vol_reason = _get(data, "atom_features.metadata.atomic_density_partition_volume_proxy.bader_status_reason")
        if vol_reason is None:
            vol_reason = _get(data, "atom_features.metadata.atomic_density_partition_charge_proxy.bader_volume_status_reason")
        if vol_status is not None:
            volume_status_hist[str(vol_status)] += 1
        if vol_reason is not None:
            volume_reason_hist[str(vol_reason)] += 1

        bond_indices = _get(data, "bond_features.bond_indices") or []
        mayer = _get(data, "bond_features.bond_orders_mayer")
        mayer_status = _get(data, "bond_features.metadata.bond_orders_mayer.availability_status")
        mayer_reason = _get(data, "bond_features.metadata.bond_orders_mayer.status_reason")
        if isinstance(mayer, list) and isinstance(bond_indices, list) and len(mayer) == len(bond_indices):
            mayer_ok += 1
        if mayer_reason:
            mayer_reason_hist[str(mayer_reason)] += 1
        elif mayer_status:
            mayer_reason_hist[str(mayer_status)] += 1

        optimized = _get(data, "structural_features.optimized_3d_geometry") or {}
        if (
            isinstance(optimized, dict)
            and optimized.get("coordinate_ref") == "geometry.atom_coords_angstrom"
            and optimized.get("coordinate_source_of_truth") == "geometry.atom_coords_angstrom"
            and optimized.get("coordinate_embedding") == "reference_only"
            and optimized.get("natm_reference") == natm
            and isinstance(optimized.get("geometry_fingerprint_sha256"), str)
            and len(optimized.get("geometry_fingerprint_sha256")) == 64
        ):
            optimized_ref_ok += 1

        if (
            _get(data, "molecule_info.representation_metadata.smarts.is_proxy") is True
            and _get(data, "molecule_info.representation_metadata.smarts.source") is not None
        ):
            smart_proxy_ok += 1

        if (
            _get(data, "global_features.metadata.molecule_size_bounding_box_diagonal_angstrom.implementation_status")
            == "implemented_proxy"
        ):
            molecule_size_proxy_ok += 1

        ion_proxy = _get(data, "global_features.dft.ionization_related_proxy_v1") or {}
        if isinstance(ion_proxy, dict) and ion_proxy.get("available") is True:
            ionization_proxy_ok += 1
        ion_reason = ion_proxy.get("status_reason")
        if ion_reason is not None:
            ionization_reason_hist[str(ion_reason)] += 1

        if (
            _get(data, "bond_features.metadata.bond_delocalization_index_proxy_v1.substitute_scope")
            == "qcMol_DI_values_or_matrix_substitute_only"
            and _get(data, "bond_features.metadata.bond_delocalization_index_proxy_v1.exact_di_matrix_available") is False
        ):
            di_proxy_frozen_ok += 1

        if idx < 8:
            samples.append(
                {
                    "molecule_id": mid,
                    "bader_volume_status": vol_status,
                    "bader_volume_reason": vol_reason,
                    "mayer_status": mayer_status,
                    "mayer_reason": mayer_reason,
                    "optimized_geometry_ref_ok": bool(
                        isinstance(optimized, dict)
                        and optimized.get("coordinate_ref") == "geometry.atom_coords_angstrom"
                    ),
                    "ionization_related_proxy_status": ion_proxy.get("status"),
                    "ionization_related_proxy_value_hartree": ion_proxy.get("koopmans_ip_proxy_hartree"),
                }
            )

    feature_rates = {
        "density_partition_volume_proxy_bader": _rate(bader_volume_available, n),
        "mayer_bl_substitute_alignment": _rate(mayer_ok, n),
        "optimized_3d_geometry_reference_contract": _rate(optimized_ref_ok, n),
        "smart_substitute_semantics_frozen": _rate(smart_proxy_ok, n),
        "molecule_size_substitute_semantics_frozen": _rate(molecule_size_proxy_ok, n),
        "ionization_related_proxy_v1": _rate(ionization_proxy_ok, n),
        "di_substitute_semantics_frozen": _rate(di_proxy_frozen_ok, n),
    }
    judgements = {k: _judgement(v) for k, v in feature_rates.items()}

    return {
        "feature_available_rates": feature_rates,
        "implemented_partial_rejected": judgements,
        "volume_status_hist": dict(volume_status_hist),
        "volume_reason_hist_top10": dict(volume_reason_hist.most_common(10)),
        "mayer_reason_hist_top10": dict(mayer_reason_hist.most_common(10)),
        "ionization_status_reason_hist": dict(ionization_reason_hist),
        "samples": samples,
    }


def build_report(unified_dir: Path) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id", fp.stem)
        rows.append((mid, data))

    module_a_specs = [
        {
            "name": "bader_population_entropy_proxy_v1",
            "value_path": "global_features.basin_proxy_summary_v1.bader_population_entropy_proxy_v1",
            "reason_path": "global_features.basin_proxy_summary_v1.metadata.candidate_assessment_v1.bader_population_entropy_proxy_v1.reason",
        },
        {
            "name": "hetero_basin_population_share_proxy_v1",
            "value_path": "global_features.basin_proxy_summary_v1.hetero_basin_population_share_proxy_v1",
            "reason_path": "global_features.basin_proxy_summary_v1.metadata.candidate_assessment_v1.hetero_basin_population_share_proxy_v1.reason",
        },
        {
            "name": "bader_laplacian_sign_balance_proxy_v1",
            "value_path": "global_features.basin_proxy_summary_v1.bader_laplacian_sign_balance_proxy_v1",
            "reason_path": "global_features.basin_proxy_summary_v1.metadata.candidate_assessment_v1.bader_laplacian_sign_balance_proxy_v1.reason",
        },
        {
            "name": "bader_charge_laplacian_correlation_proxy_v1",
            "value_path": "global_features.basin_proxy_summary_v1.bader_charge_laplacian_correlation_proxy_v1",
            "reason_path": "global_features.basin_proxy_summary_v1.metadata.candidate_assessment_v1.bader_charge_laplacian_correlation_proxy_v1.reason",
        },
    ]
    module_b_specs = [
        {
            "name": "atomic_local_reactivity_refined_proxy_v1",
            "value_path": "atom_features.atomic_local_reactivity_refined_proxy_v1",
            "reason_path": "atom_features.metadata.atomic_local_reactivity_refined_proxy_v1.status_reason",
        },
        {
            "name": "lone_pair_polarization_proxy_v1",
            "value_path": "atom_features.lone_pair_polarization_proxy_v1",
            "reason_path": "atom_features.metadata.lone_pair_polarization_proxy_v1.status_reason",
        },
        {
            "name": "bond_localization_tension_proxy_v1",
            "value_path": "bond_features.bond_localization_tension_proxy_v1",
            "reason_path": "bond_features.metadata.bond_localization_tension_proxy_v1.status_reason",
        },
        {
            "name": "bond_polarized_delocalization_proxy_v1",
            "value_path": "bond_features.bond_polarized_delocalization_proxy_v1",
            "reason_path": "bond_features.metadata.bond_polarized_delocalization_proxy_v1.status_reason",
        },
    ]
    module_c_specs = [
        {
            "name": "reactivity_concentration_proxy_v1",
            "value_path": "global_features.proxy_family_summary_v1.reactivity_concentration_proxy_v1",
            "reason_path": "global_features.proxy_family_summary_v1.metadata.status_reason",
        },
        {
            "name": "bond_pattern_heterogeneity_proxy_v1",
            "value_path": "global_features.proxy_family_summary_v1.bond_pattern_heterogeneity_proxy_v1",
            "reason_path": "global_features.proxy_family_summary_v1.metadata.status_reason",
        },
        {
            "name": "lp_environment_polarization_proxy_v1",
            "value_path": "global_features.proxy_family_summary_v1.lp_environment_polarization_proxy_v1",
            "reason_path": "global_features.proxy_family_summary_v1.metadata.status_reason",
        },
    ]

    module_a = _module_feature_report(rows, module_a_specs)
    module_b = _module_feature_report(rows, module_b_specs)
    module_c = _module_feature_report(rows, module_c_specs)
    final_focus = _final_sprint_focus_report(rows)

    return {
        "input": {"unified_dir": str(unified_dir), "n_files": len(rows)},
        "module_a_basin_closure": module_a,
        "module_b_atom_bond_closure": module_b,
        "module_c_molecule_summary_closure": module_c,
        "final_sprint_focus": final_focus,
        "alignment_master_table_summary": _master_table_delta(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate qcMol alignment closure sprint features")
    p.add_argument("--unified-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    unified_dir = Path(args.unified_dir).resolve()
    if not unified_dir.exists():
        raise FileNotFoundError(f"unified-dir not found: {unified_dir}")

    report = build_report(unified_dir)
    out_json = Path(args.output_json).resolve()
    out_md = Path(args.output_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    status_counts = report["alignment_master_table_summary"]["status_counts"]
    lines = [
        "# qcMol Alignment Closure Validation",
        f"- total files: {report['input']['n_files']}",
        f"- module A mean available rate: {sum(report['module_a_basin_closure']['feature_available_rates'].values())/max(1,len(report['module_a_basin_closure']['feature_available_rates']))}",
        f"- module B mean available rate: {sum(report['module_b_atom_bond_closure']['feature_available_rates'].values())/max(1,len(report['module_b_atom_bond_closure']['feature_available_rates']))}",
        f"- module C mean available rate: {sum(report['module_c_molecule_summary_closure']['feature_available_rates'].values())/max(1,len(report['module_c_molecule_summary_closure']['feature_available_rates']))}",
        f"- final focus bader volume rate: {report['final_sprint_focus']['feature_available_rates']['density_partition_volume_proxy_bader']}",
        f"- final focus ionization proxy rate: {report['final_sprint_focus']['feature_available_rates']['ionization_related_proxy_v1']}",
        f"- master status counts: exact={status_counts['implemented_exact']}, proxy={status_counts['implemented_proxy']}, partial={status_counts['partial']}, missing={status_counts['missing']}, rejected_as_exact={status_counts['rejected_as_exact']}",
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

"""
Validation for qcMol substitute alignment enhancement:
1) molecule_size family
2) density shape descriptor family v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (  # noqa: E402
    QCMOL_ALIGNMENT_MASTER_TABLE,
    remaining_alignment_gaps,
    summarize_alignment_status_counts,
)


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _rate(hit: int, total: int) -> float:
    return float(hit / max(1, total))


def _category_flags(data: Dict[str, Any]) -> Dict[str, bool]:
    heavy = _get(data, "global_features.rdkit.heavy_atom_count")
    aromatic = _get(data, "atom_features.rdkit_aromatic")
    rot = _get(data, "global_features.rdkit.rotatable_bonds")
    symbols = _get(data, "geometry.atom_symbols") or []
    return {
        "small": isinstance(heavy, int) and heavy <= 6,
        "aromatic": isinstance(aromatic, list) and any(bool(x) for x in aromatic),
        "flexible": isinstance(rot, int) and rot >= 3,
        "contains_O_N": any(str(s).upper() in {"O", "N"} for s in symbols),
    }


def _pick_samples(rows: List[Tuple[str, Dict[str, Any]]], limit_per_cat: int = 2) -> List[str]:
    buckets: Dict[str, List[str]] = {"small": [], "aromatic": [], "flexible": [], "contains_O_N": []}
    for mid, data in rows:
        flags = _category_flags(data)
        for key in buckets:
            if flags[key] and len(buckets[key]) < limit_per_cat:
                buckets[key].append(mid)
    out: List[str] = []
    for key in ("small", "aromatic", "flexible", "contains_O_N"):
        for mid in buckets[key]:
            if mid not in out:
                out.append(mid)
    return out


def _is_valid_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _validate_molecule_size(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    available = Counter()
    checks = Counter()
    sample_ids = set(_pick_samples(rows))
    samples: List[Dict[str, Any]] = []

    for mid, data in rows:
        gsize = _get(data, "global_features.geometry_size", {}) or {}
        bbox = gsize.get("bounding_box_diagonal_angstrom")
        heavy = gsize.get("heavy_atom_count_proxy")
        total_atoms = gsize.get("total_atom_count_proxy")
        num_bonds = gsize.get("num_bonds_proxy")
        num_rings = gsize.get("num_rings_proxy")
        bond_indices = _get(data, "bond_features.bond_indices")

        if _is_valid_number(bbox):
            available["bounding_box_diagonal_angstrom"] += 1
            if float(bbox) > 0.0:
                checks["bbox_positive"] += 1
        if isinstance(heavy, int):
            available["heavy_atom_count_proxy"] += 1
        if isinstance(total_atoms, int):
            available["total_atom_count_proxy"] += 1
        if isinstance(num_bonds, int):
            available["num_bonds_proxy"] += 1
        if isinstance(num_rings, int):
            available["num_rings_proxy"] += 1

        if isinstance(total_atoms, int) and isinstance(heavy, int) and total_atoms >= heavy:
            checks["total_atom_count_gte_heavy_atom_count"] += 1
        if isinstance(num_bonds, int) and isinstance(bond_indices, list) and num_bonds == len(bond_indices):
            checks["num_bonds_matches_bond_indices_len"] += 1

        if mid in sample_ids:
            samples.append(
                {
                    "molecule_id": mid,
                    "categories": _category_flags(data),
                    "bounding_box_diagonal_angstrom": bbox,
                    "heavy_atom_count_proxy": heavy,
                    "total_atom_count_proxy": total_atoms,
                    "num_bonds_proxy": num_bonds,
                    "num_rings_proxy": num_rings,
                    "bond_indices_len": len(bond_indices) if isinstance(bond_indices, list) else None,
                }
            )

    return {
        "available_rate": {
            "bounding_box_diagonal_angstrom": _rate(available["bounding_box_diagonal_angstrom"], n),
            "heavy_atom_count_proxy": _rate(available["heavy_atom_count_proxy"], n),
            "total_atom_count_proxy": _rate(available["total_atom_count_proxy"], n),
            "num_bonds_proxy": _rate(available["num_bonds_proxy"], n),
            "num_rings_proxy": _rate(available["num_rings_proxy"], n),
        },
        "checks": {
            "bbox_positive_rate": _rate(checks["bbox_positive"], max(1, available["bounding_box_diagonal_angstrom"])),
            "total_atom_count_gte_heavy_atom_count_rate": _rate(checks["total_atom_count_gte_heavy_atom_count"], n),
            "num_bonds_matches_bond_indices_len_rate": _rate(checks["num_bonds_matches_bond_indices_len"], n),
        },
        "samples": samples[:8],
    }


def _validate_density_shape(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    available = Counter()
    checks = Counter()
    reason_hist = Counter()
    samples: List[Dict[str, Any]] = []

    sample_ids = set(_pick_samples(rows))
    for mid, data in rows:
        fam = _get(data, "realspace_features.density_shape_descriptor_family_v1") or {}
        meta = _get(data, "realspace_features.metadata.density_shape_descriptor_family_v1") or {}
        status = str(fam.get("status") or meta.get("status") or "unavailable")
        reason = str(fam.get("status_reason") or meta.get("status_reason") or "unknown")
        reason_hist[reason] += 1

        success = status == "success"
        if success:
            available["family"] += 1

        for key in ("sphericity", "asphericity", "anisotropy", "elongation", "planarity"):
            v = fam.get(key)
            if _is_valid_number(v):
                available[key] += 1

        raw = fam.get("eigenvalues_raw")
        norm = fam.get("eigenvalues_normalized")
        if isinstance(raw, list) and len(raw) == 3 and all(_is_valid_number(x) for x in raw):
            available["eigenvalues_raw"] += 1
            if float(raw[0]) + 1e-12 >= float(raw[1]) and float(raw[1]) + 1e-12 >= float(raw[2]):
                checks["eigenvalues_raw_sorted_desc"] += 1
        if isinstance(norm, list) and len(norm) == 3 and all(_is_valid_number(x) for x in norm):
            available["eigenvalues_normalized"] += 1
            s = float(norm[0]) + float(norm[1]) + float(norm[2])
            if abs(s - 1.0) <= 1e-6:
                checks["eigenvalues_normalized_sum_one"] += 1
        # nan/inf check over descriptor scalars
        vals = [fam.get(k) for k in ("sphericity", "asphericity", "anisotropy", "elongation", "planarity")]
        if all((v is None) or _is_valid_number(v) for v in vals):
            checks["descriptor_not_nan_inf"] += 1

        # heuristics for representative samples
        elongated = _is_valid_number(fam.get("elongation")) and float(fam.get("elongation")) >= 0.5
        compact = _is_valid_number(fam.get("sphericity")) and float(fam.get("sphericity")) >= 0.6
        is_aromatic = _category_flags(data)["aromatic"]
        is_flexible = _category_flags(data)["flexible"]
        contains_on = _category_flags(data)["contains_O_N"]

        if mid in sample_ids or elongated or (compact and is_aromatic) or is_flexible or contains_on:
            samples.append(
                {
                    "molecule_id": mid,
                    "status": status,
                    "status_reason": reason,
                    "categories": _category_flags(data),
                    "sphericity": fam.get("sphericity"),
                    "asphericity": fam.get("asphericity"),
                    "anisotropy": fam.get("anisotropy"),
                    "elongation": fam.get("elongation"),
                    "planarity": fam.get("planarity"),
                    "eigenvalues_raw": raw,
                    "eigenvalues_normalized": norm,
                    "selected_density_fraction": fam.get("selected_density_fraction"),
                }
            )

    return {
        "family_available_rate": _rate(available["family"], n),
        "descriptor_available_rate": {
            "sphericity": _rate(available["sphericity"], n),
            "asphericity": _rate(available["asphericity"], n),
            "anisotropy": _rate(available["anisotropy"], n),
            "elongation": _rate(available["elongation"], n),
            "planarity": _rate(available["planarity"], n),
        },
        "eigenvalues_available_rate": {
            "eigenvalues_raw": _rate(available["eigenvalues_raw"], n),
            "eigenvalues_normalized": _rate(available["eigenvalues_normalized"], n),
        },
        "checks": {
            "eigenvalues_raw_sorted_desc_rate": _rate(checks["eigenvalues_raw_sorted_desc"], max(1, available["eigenvalues_raw"])),
            "eigenvalues_normalized_sum_one_rate": _rate(checks["eigenvalues_normalized_sum_one"], max(1, available["eigenvalues_normalized"])),
            "descriptor_not_nan_inf_rate": _rate(checks["descriptor_not_nan_inf"], n),
        },
        "status_reason_histogram": dict(reason_hist.most_common(20)),
        "samples": samples[:12],
    }


def _master_table_sync() -> Dict[str, Any]:
    counts = summarize_alignment_status_counts(
        [{"status": row.get("current_status")} for row in QCMOL_ALIGNMENT_MASTER_TABLE]
    )
    molecule_size_row = next(
        (row for row in QCMOL_ALIGNMENT_MASTER_TABLE if row.get("qcMol_item_name") == "molecule_size"),
        None,
    )
    sphericity_row = next(
        (row for row in QCMOL_ALIGNMENT_MASTER_TABLE if row.get("qcMol_item_name") == "sphericity_parameters"),
        None,
    )
    gaps = remaining_alignment_gaps(QCMOL_ALIGNMENT_MASTER_TABLE)
    return {
        "status_counts": counts,
        "remaining_gap_count": len(gaps),
        "molecule_size_alignment_row": molecule_size_row,
        "sphericity_parameters_alignment_row": sphericity_row,
    }


def build_report(unified_dir: Path) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id") or fp.stem.replace(".unified", "")
        rows.append((str(mid), data))

    return {
        "input": {
            "unified_dir": str(unified_dir),
            "n_files": len(rows),
        },
        "molecule_size_family_validation": _validate_molecule_size(rows),
        "density_shape_family_validation": _validate_density_shape(rows),
        "master_table_sync": _master_table_sync(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate molecule_size and density shape family enhancement")
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

    size_rates = report["molecule_size_family_validation"]["available_rate"]
    shape = report["density_shape_family_validation"]
    status_counts = report["master_table_sync"]["status_counts"]
    lines = [
        "# qcMol Shape/Size Enhancement Validation",
        f"- total files: {report['input']['n_files']}",
        f"- molecule_size availability: bbox={size_rates['bounding_box_diagonal_angstrom']}, heavy={size_rates['heavy_atom_count_proxy']}, total={size_rates['total_atom_count_proxy']}, bonds={size_rates['num_bonds_proxy']}, rings={size_rates['num_rings_proxy']}",
        f"- density_shape family available rate: {shape['family_available_rate']}",
        f"- density_shape status reasons: {shape['status_reason_histogram']}",
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


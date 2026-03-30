"""
Validation for descriptor-system enhancement sprint:
1) radius_of_gyration companion in molecule_size family
2) multiscale density shape family (0.50 / 0.90 / 0.95)
3) relative anisotropy kappa2
4) density/grid metadata completeness
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _rate(hit: int, total: int) -> float:
    return float(hit / max(1, total))


def _is_valid_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


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


def _validate_radius_of_gyration(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    available_rg = 0
    available_bbox = 0
    positive_rg = 0
    sample_ids = set(_pick_samples(rows))
    samples: List[Dict[str, Any]] = []
    pairs: List[Tuple[float, float]] = []

    for mid, data in rows:
        gsize = _get(data, "global_features.geometry_size", {}) or {}
        rg = gsize.get("radius_of_gyration_angstrom")
        bbox = gsize.get("bounding_box_diagonal_angstrom")

        if _is_valid_number(rg):
            available_rg += 1
            if float(rg) > 0.0:
                positive_rg += 1
        if _is_valid_number(bbox):
            available_bbox += 1

        if _is_valid_number(rg) and _is_valid_number(bbox):
            pairs.append((float(bbox), float(rg)))

        if mid in sample_ids:
            samples.append(
                {
                    "molecule_id": mid,
                    "categories": _category_flags(data),
                    "radius_of_gyration_angstrom": rg,
                    "bounding_box_diagonal_angstrom": bbox,
                }
            )

    corr = None
    if len(pairs) >= 2:
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        if float(np.std(x)) > 1e-12 and float(np.std(y)) > 1e-12:
            corr = float(np.corrcoef(x, y)[0, 1])

    return {
        "available_rate": _rate(available_rg, n),
        "bbox_available_rate": _rate(available_bbox, n),
        "positive_rate_when_available": _rate(positive_rg, max(1, available_rg)),
        "bbox_rg_correlation": corr,
        "samples": samples[:10],
    }


def _bin_kappa2(v: float) -> str:
    if v < 0.2:
        return "[0.0,0.2)"
    if v < 0.4:
        return "[0.2,0.4)"
    if v < 0.6:
        return "[0.4,0.6)"
    if v < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def _validate_multiscale(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    scale_keys = ["0.50", "0.90", "0.95"]
    scale_available = Counter()
    descriptor_available = {k: Counter() for k in scale_keys}
    reason_hist = Counter()
    checks = Counter()
    kappa_hist = Counter()
    samples: List[Dict[str, Any]] = []
    sample_ids = set(_pick_samples(rows))

    for mid, data in rows:
        fam = _get(data, "realspace_features.density_shape_multiscale_family_v1") or {}
        meta = _get(data, "realspace_features.metadata.density_shape_multiscale_family_v1") or {}
        scales = fam.get("scales") if isinstance(fam.get("scales"), dict) else {}

        status = str(fam.get("status") or meta.get("status") or "unavailable")
        reason = str(fam.get("status_reason") or meta.get("status_reason") or "unknown")
        reason_hist[reason] += 1

        points: Dict[str, int] = {}
        frac: Dict[str, float] = {}
        row_preview: Dict[str, Any] = {"molecule_id": mid, "status": status, "status_reason": reason, "categories": _category_flags(data), "scales": {}}

        for scale in scale_keys:
            node = scales.get(scale) if isinstance(scales, dict) else None
            if not isinstance(node, dict):
                continue

            node_status = str(node.get("status") or "unavailable")
            if node_status == "success":
                scale_available[scale] += 1
            s_reason = str(node.get("status_reason") or "unknown")
            reason_hist[f"{scale}:{s_reason}"] += 1

            for key in ("sphericity", "asphericity", "anisotropy", "elongation", "planarity", "relative_anisotropy_kappa2"):
                if _is_valid_number(node.get(key)):
                    descriptor_available[scale][key] += 1

            raw = node.get("eigenvalues_raw")
            norm = node.get("eigenvalues_normalized")
            if isinstance(raw, list) and len(raw) == 3 and all(_is_valid_number(x) for x in raw):
                if float(raw[0]) + 1e-12 >= float(raw[1]) and float(raw[1]) + 1e-12 >= float(raw[2]):
                    checks[f"{scale}:eig_order"] += 1
            if isinstance(norm, list) and len(norm) == 3 and all(_is_valid_number(x) for x in norm):
                if abs((float(norm[0]) + float(norm[1]) + float(norm[2])) - 1.0) <= 1e-6:
                    checks[f"{scale}:norm_sum"] += 1

            np_sel = node.get("n_points_selected")
            sfrac = node.get("selected_density_fraction")
            cutoff = node.get("mass_cutoff")
            if isinstance(np_sel, int):
                points[scale] = int(np_sel)
            if _is_valid_number(sfrac):
                frac[scale] = float(sfrac)
            if _is_valid_number(sfrac) and _is_valid_number(cutoff):
                if float(sfrac) + 1e-9 >= float(cutoff):
                    checks[f"{scale}:fraction_meets_cutoff"] += 1

            kappa2 = node.get("relative_anisotropy_kappa2")
            if _is_valid_number(kappa2):
                kappa_hist[_bin_kappa2(float(kappa2))] += 1
                if 0.0 <= float(kappa2) <= 1.0 + 1e-9:
                    checks[f"{scale}:kappa2_in_range"] += 1

            scalar_vals = [node.get(k) for k in ("sphericity", "asphericity", "anisotropy", "elongation", "planarity", "relative_anisotropy_kappa2")]
            if all((v is None) or _is_valid_number(v) for v in scalar_vals):
                checks[f"{scale}:descriptor_not_nan_inf"] += 1

            row_preview["scales"][scale] = {
                "status": node_status,
                "selected_density_fraction": sfrac,
                "n_points_selected": np_sel,
                "sphericity": node.get("sphericity"),
                "relative_anisotropy_kappa2": node.get("relative_anisotropy_kappa2"),
            }

        if all(k in points for k in scale_keys):
            if points["0.50"] <= points["0.90"] <= points["0.95"]:
                checks["points_monotonic_0.50_0.90_0.95"] += 1
        if all(k in frac for k in scale_keys):
            if frac["0.50"] <= frac["0.90"] <= frac["0.95"] + 1e-9:
                checks["fraction_monotonic_0.50_0.90_0.95"] += 1

        if mid in sample_ids:
            samples.append(row_preview)

    scale_rates = {s: _rate(scale_available[s], n) for s in scale_keys}
    desc_rates: Dict[str, Dict[str, float]] = {}
    for s in scale_keys:
        desc_rates[s] = {
            "sphericity": _rate(descriptor_available[s]["sphericity"], n),
            "asphericity": _rate(descriptor_available[s]["asphericity"], n),
            "anisotropy": _rate(descriptor_available[s]["anisotropy"], n),
            "elongation": _rate(descriptor_available[s]["elongation"], n),
            "planarity": _rate(descriptor_available[s]["planarity"], n),
            "relative_anisotropy_kappa2": _rate(descriptor_available[s]["relative_anisotropy_kappa2"], n),
        }

    return {
        "scale_available_rate": scale_rates,
        "descriptor_available_rate_by_scale": desc_rates,
        "status_reason_histogram": dict(reason_hist.most_common(25)),
        "kappa2_histogram": dict(kappa_hist),
        "checks": {
            "points_monotonic_rate": _rate(checks["points_monotonic_0.50_0.90_0.95"], n),
            "fraction_monotonic_rate": _rate(checks["fraction_monotonic_0.50_0.90_0.95"], n),
            "eigenvalue_order_rate_by_scale": {s: _rate(checks[f"{s}:eig_order"], max(1, scale_available[s])) for s in scale_keys},
            "normalized_sum_rate_by_scale": {s: _rate(checks[f"{s}:norm_sum"], max(1, scale_available[s])) for s in scale_keys},
            "fraction_meets_cutoff_rate_by_scale": {s: _rate(checks[f"{s}:fraction_meets_cutoff"], max(1, scale_available[s])) for s in scale_keys},
            "kappa2_in_range_rate_by_scale": {s: _rate(checks[f"{s}:kappa2_in_range"], max(1, scale_available[s])) for s in scale_keys},
            "descriptor_not_nan_inf_rate_by_scale": {s: _rate(checks[f"{s}:descriptor_not_nan_inf"], max(1, scale_available[s])) for s in scale_keys},
        },
        "samples": samples[:12],
    }


def _validate_metadata_completeness(rows: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
    n = len(rows)
    checks = Counter()
    required_top = [
        "density_source_method",
        "cube_spacing_angstrom",
        "density_grid_resolution_angstrom",
        "margin_angstrom",
    ]
    required_single_meta = [
        "normalization_rule",
        "mass_cutoff_default",
        "formula_relative_anisotropy_kappa2",
    ]
    required_multi_meta = [
        "normalization_rule",
        "scale_semantics",
        "canonical_single_scale_view",
    ]

    for _, data in rows:
        meta = _get(data, "realspace_features.metadata", {}) or {}
        single_meta = meta.get("density_shape_descriptor_family_v1") if isinstance(meta.get("density_shape_descriptor_family_v1"), dict) else {}
        multi_meta = meta.get("density_shape_multiscale_family_v1") if isinstance(meta.get("density_shape_multiscale_family_v1"), dict) else {}

        for key in required_top:
            if meta.get(key) is not None:
                checks[f"top:{key}"] += 1
        for key in required_single_meta:
            if single_meta.get(key) is not None:
                checks[f"single:{key}"] += 1
        for key in required_multi_meta:
            if multi_meta.get(key) is not None:
                checks[f"multi:{key}"] += 1

    return {
        "top_level_rate": {k: _rate(checks[f"top:{k}"], n) for k in required_top},
        "single_scale_meta_rate": {k: _rate(checks[f"single:{k}"], n) for k in required_single_meta},
        "multiscale_meta_rate": {k: _rate(checks[f"multi:{k}"], n) for k in required_multi_meta},
    }


def build_report(unified_dir: Path) -> Dict[str, Any]:
    files = sorted(unified_dir.glob("*.unified.json"))
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        mid = _get(data, "molecule_info.molecule_id") or fp.stem.replace(".unified", "")
        rows.append((str(mid), data))

    return {
        "input": {"unified_dir": str(unified_dir), "n_files": len(rows)},
        "radius_of_gyration_validation": _validate_radius_of_gyration(rows),
        "multiscale_density_shape_validation": _validate_multiscale(rows),
        "metadata_completeness_validation": _validate_metadata_completeness(rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate density-shape enhancement descriptors")
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

    rg = report["radius_of_gyration_validation"]
    ms = report["multiscale_density_shape_validation"]
    lines = [
        "# Density Shape Enhancement Validation",
        f"- total files: {report['input']['n_files']}",
        f"- radius_of_gyration available rate: {rg['available_rate']}",
        f"- bbox-rg correlation: {rg['bbox_rg_correlation']}",
        f"- multiscale available rate: {ms['scale_available_rate']}",
        f"- multiscale status reasons(top): {ms['status_reason_histogram']}",
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

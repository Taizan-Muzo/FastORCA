"""
M5.5 Task D: lightweight validation runner.

Outputs:
- by_final_status
- plugin_status_counts
- reason_code_histogram
- timing stats
- partial source breakdown
and a short conclusion block.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consumer.batch_runner import run_batch
from consumer.feature_extractor import FeatureExtractor
from producer.dft_calculator import DFTCalculator


FAST_MOLECULES: List[Tuple[str, str]] = [
    ("f01_ethanol", "CCO"),
    ("f02_benzene", "c1ccccc1"),
    ("f03_acetic_acid", "CC(=O)O"),
    ("f04_pyridine", "c1ccncc1"),
    ("f05_triethylamine", "CCN(CC)CC"),
    ("f06_acetone", "CC(=O)C"),
    ("f07_formamide", "NC=O"),
    ("f08_ethyl_acetate", "CCOC(=O)C"),
    ("f09_aniline", "c1ccccc1N"),
    ("f10_toluene", "Cc1ccccc1"),
    ("f11_phenol", "Oc1ccccc1"),
    ("f12_methylamine", "CN"),
    ("f13_propionamide", "CCC(=O)N"),
    ("f14_acetonitrile", "CC#N"),
    ("f15_isopropanol", "CC(O)C"),
    ("f16_piperidine", "N1CCCCC1"),
    ("f17_furan", "c1ccoc1"),
    ("f18_thiophene", "c1ccsc1"),
    ("f19_imidazole", "c1ncc[nH]1"),
    ("f20_morpholine", "O1CCNCC1"),
]

FULL_MOLECULES: List[Tuple[str, str]] = [
    ("g01_ethanol", "CCO"),
    ("g02_benzene", "c1ccccc1"),
    ("g03_acetic_acid", "CC(=O)O"),
    ("g04_pyridine", "c1ccncc1"),
    ("g05_tea_like", "CCN(CC)CCO"),
]

FULL_HARD_MOLECULES: List[Tuple[str, str]] = [
    ("h01_caffeine", "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"),
    ("h02_aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("h03_lidocaine_like", "CCN(CC)C(=O)N(c1ccccc1)C(C)C"),
    ("h04_nitro_aniline", "Nc1ccc(cc1)[N+](=O)[O-]"),
    ("h05_quinoline", "c1ccc2ncccc2c1"),
]


def get_production_profile(benchmark_path: Path | None) -> Dict[str, Any]:
    # default fallback == P0
    fallback = {
        "name": "P0",
        "timeout_seconds": 30,
        "max_atoms": 25,
        "max_total_grid_points": 500_000,
        "grid_resolution_angstrom": 0.30,
        "margin_angstrom": 3.0,
    }
    if benchmark_path is None or not benchmark_path.exists():
        return fallback

    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rec = data.get("recommendations", {}).get("production_full_default", "P0")
    profile = (data.get("profiles") or {}).get(rec)
    if not isinstance(profile, dict):
        return fallback
    return {"name": rec, **profile}


def prepare_wavefunctions(
    calculator: DFTCalculator,
    molecules: List[Tuple[str, str]],
    pkl_dir: Path,
) -> List[Dict[str, Any]]:
    pkl_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for molecule_id, smiles in molecules:
        mol = calculator.from_smiles(smiles)
        res = calculator.calculate_and_export(molecule_id=molecule_id, mol_obj=mol, output_dir=str(pkl_dir))
        if not res.get("success"):
            raise RuntimeError(f"DFT failed for {molecule_id}: {res.get('error')}")
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "pkl_path": res["pkl_path"],
            }
        )
    return rows


def partial_source_breakdown(summary: Dict[str, Any]) -> Dict[str, int]:
    hist = summary.get("reason_code_histogram", {}) or {}
    return {k: int(v.get("count", 0)) for k, v in hist.items() if int(v.get("count", 0)) > 0}


def collect_partial_molecules(unified_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not unified_dir.exists():
        return rows
    for fp in sorted(unified_dir.glob("*.unified.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            status = (data.get("calculation_status") or {}).get("overall_status")
            if status != "core_success_partial_features":
                continue
            rows.append(
                {
                    "molecule_id": (data.get("molecule_info") or {}).get("molecule_id"),
                    "reason_codes": data.get("_validation_errors", []),
                    "elf_alignment_stats": data.get("_elf_alignment_stats"),
                }
            )
        except Exception:
            continue
    return rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(args.benchmark_json).resolve() if args.benchmark_json else None
    production = get_production_profile(benchmark_path)

    extractor = FeatureExtractor()
    calc = DFTCalculator(
        functional=args.functional,
        basis=args.basis,
        verbose=0,
        geometry_optimization=False,
    )

    fast_list = prepare_wavefunctions(calc, FAST_MOLECULES[: args.fast_n], out_root / "pkl_fast")
    full_pool = FULL_MOLECULES if args.full_set == "core" else FULL_HARD_MOLECULES
    full_list = prepare_wavefunctions(calc, full_pool[: args.full_n], out_root / "pkl_full")

    fast_summary = run_batch(
        feature_extractor=extractor,
        molecule_list=fast_list,
        output_dir=out_root / "fast",
        run_mode="fast",
        artifact_policy="keep_none",
        plugin_config={
            "critic2_bridge": {"enabled": False},
        },
        n_workers=1,
    )

    full_summary = run_batch(
        feature_extractor=extractor,
        molecule_list=full_list,
        output_dir=out_root / "full",
        run_mode="full",
        artifact_policy="keep_failed_only",
        plugin_config={
            "critic2_bridge": {"enabled": False, "run_in_full_mode": False},
            "realspace_features": {
                "enabled": True,
                "run_in_full_mode": True,
                "timeout_seconds": production["timeout_seconds"],
                "max_atoms": production["max_atoms"],
                "max_total_grid_points": production["max_total_grid_points"],
                "runtime_config": {
                    "max_atoms": production["max_atoms"],
                    "max_total_grid_points": production["max_total_grid_points"],
                    "grid_resolution_angstrom": production["grid_resolution_angstrom"],
                    "margin_angstrom": production["margin_angstrom"],
                },
            },
        },
        n_workers=1,
    )

    fast_by_status = ((fast_summary.get("molecule_counts") or {}).get("by_final_status") or {})
    full_by_status = ((full_summary.get("molecule_counts") or {}).get("by_final_status") or {})
    full_reason_hist = full_summary.get("reason_code_histogram") or {}
    fast_partial_rows = collect_partial_molecules(out_root / "fast")

    full_partial = int(full_by_status.get("core_success_partial_features", 0))
    full_total = int((full_summary.get("molecule_counts") or {}).get("total", 0))

    conclusion = {
        "fast_mode_local_default_recommended": bool(
            fast_by_status.get("failed_core_features", 0) == 0
            and fast_by_status.get("failed_scf", 0) == 0
            and fast_by_status.get("failed_geometry", 0) == 0
        ),
        "full_mode_main_partial_sources": {
            k: int(v.get("count", 0))
            for k, v in sorted(
                full_reason_hist.items(),
                key=lambda kv: int(kv[1].get("count", 0)),
                reverse=True,
            )[:5]
        },
        "realspace_params_conservative_enough": bool(full_partial <= max(1, full_total // 3)),
        "top_next_fix_candidates": [],
    }

    if conclusion["full_mode_main_partial_sources"]:
        conclusion["top_next_fix_candidates"] = list(conclusion["full_mode_main_partial_sources"].keys())[:2]
    else:
        conclusion["top_next_fix_candidates"] = ["no_obvious_partial_source_in_sample"]

    report = {
        "config": {
            "functional": args.functional,
            "basis": args.basis,
            "fast_n": args.fast_n,
            "full_n": args.full_n,
            "full_set": args.full_set,
            "production_full_default": production,
        },
        "fast_mode": {
            "by_final_status": fast_by_status,
            "plugin_status_counts": fast_summary.get("plugin_status_counts"),
            "reason_code_histogram": fast_summary.get("reason_code_histogram"),
            "timing_stats": fast_summary.get("timing_stats"),
            "partial_source_breakdown": partial_source_breakdown(fast_summary),
            "partial_molecules": fast_partial_rows,
        },
        "full_mode": {
            "by_final_status": full_by_status,
            "plugin_status_counts": full_summary.get("plugin_status_counts"),
            "reason_code_histogram": full_summary.get("reason_code_histogram"),
            "timing_stats": full_summary.get("timing_stats"),
            "partial_source_breakdown": partial_source_breakdown(full_summary),
        },
        "conclusion": conclusion,
    }

    out_path = out_root / "m55_validation_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation report saved: {out_path}")
    print(json.dumps(report["conclusion"], ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M5.5 lightweight validation report runner")
    p.add_argument("--output-dir", default="test_output_m55_validation_report")
    p.add_argument("--benchmark-json", default="", help="Path to realspace benchmark summary JSON")
    p.add_argument("--functional", default="B3LYP")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--fast-n", type=int, default=20)
    p.add_argument("--full-n", type=int, default=5)
    p.add_argument("--full-set", choices=["core", "hard"], default="core")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

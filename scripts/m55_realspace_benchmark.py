"""
M5.5 Task C: lightweight realspace benchmark runner.

Runs 4-6 representative molecules across multiple conservative realspace
parameter profiles and writes a compact comparison report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from consumer.feature_extractor import FeatureExtractor
from producer.dft_calculator import DFTCalculator
from utils.policy.status_determiner import StatusDeterminer


DEFAULT_MOLECULES: List[Tuple[str, str]] = [
    ("m_ethanol", "CCO"),  # small
    ("m_benzene", "c1ccccc1"),  # aromatic
    ("m_acetic_acid", "CC(=O)O"),  # hetero
    ("m_pyridine", "c1ccncc1"),  # aromatic hetero
    ("m_tea_like", "CCN(CC)CCO"),  # slightly larger
]

DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "P0": {
        "timeout_seconds": 30,
        "max_atoms": 25,
        "max_total_grid_points": 500_000,
        "grid_resolution_angstrom": 0.30,
        "margin_angstrom": 3.0,
    },
    "P1": {
        "timeout_seconds": 60,
        "max_atoms": 30,
        "max_total_grid_points": 750_000,
        "grid_resolution_angstrom": 0.30,
        "margin_angstrom": 3.0,
    },
    "P2": {
        "timeout_seconds": 90,
        "max_atoms": 40,
        "max_total_grid_points": 1_000_000,
        "grid_resolution_angstrom": 0.25,
        "margin_angstrom": 3.5,
    },
}


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (k - f) * (xs[c] - xs[f])


def total_cube_bytes(data: Dict[str, Any]) -> int:
    total = 0
    cube_files = data.get("artifacts", {}).get("cube_files", {}) or {}
    for _, info in cube_files.items():
        if not isinstance(info, dict):
            continue
        p = info.get("path")
        if not p:
            continue
        fp = Path(p)
        if fp.exists():
            total += fp.stat().st_size
    return total


def build_plugin_plan(profile_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "orbital_features": {
            "should_execute": True,
            "skip_reason": None,
            "effective_timeout": 60,
        },
        "realspace_features": {
            "should_execute": True,
            "skip_reason": None,
            "effective_timeout": profile_cfg["timeout_seconds"],
            "runtime_config": {
                "max_atoms": profile_cfg["max_atoms"],
                "max_total_grid_points": profile_cfg["max_total_grid_points"],
                "grid_resolution_angstrom": profile_cfg["grid_resolution_angstrom"],
                "margin_angstrom": profile_cfg["margin_angstrom"],
            },
        },
        "critic2_bridge": {
            "should_execute": False,
            "skip_reason": "benchmark_skip_external_bridge",
            "effective_timeout": None,
        },
    }


def aggregate_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    rs_status_counter = Counter(r["realspace_status"] for r in rows)
    overall_counter = Counter(r["overall_status"] for r in rows)
    partial_source_counter = Counter()
    reason_counter = Counter()
    rs_failure_reasons = Counter()

    times = []
    artifact_sizes = []
    for r in rows:
        if r["realspace_time_seconds"] is not None:
            times.append(float(r["realspace_time_seconds"]))
        artifact_sizes.append(int(r["artifact_bytes"]))
        reason_counter.update(r["reason_codes"])
        if r["realspace_failure_reason"]:
            rs_failure_reasons.update([r["realspace_failure_reason"]])
        if r["overall_status"] == "core_success_partial_features":
            partial_source_counter.update(r["reason_codes"])

    success = rs_status_counter.get("success", 0)
    timeout = rs_status_counter.get("timeout", 0)

    return {
        "n_molecules": n,
        "success_rate": round(success / n, 4) if n else 0.0,
        "timeout_rate": round(timeout / n, 4) if n else 0.0,
        "realspace_status_counts": dict(rs_status_counter),
        "overall_status_counts": dict(overall_counter),
        "mean_wall_time_seconds": round(statistics.mean(times), 3) if times else 0.0,
        "p90_wall_time_seconds": round(percentile(times, 90), 3) if times else 0.0,
        "mean_artifact_bytes": int(statistics.mean(artifact_sizes)) if artifact_sizes else 0,
        "total_artifact_bytes": int(sum(artifact_sizes)),
        "reason_code_histogram": dict(reason_counter),
        "realspace_failure_reason_histogram": dict(rs_failure_reasons),
        "partial_source_breakdown": dict(partial_source_counter),
    }


def choose_recommendations(profile_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    def key_prod(item):
        name, s = item
        return (-s["success_rate"], s["timeout_rate"], s["mean_wall_time_seconds"], s["mean_artifact_bytes"], name)

    production = sorted(profile_summaries.items(), key=key_prod)[0][0]

    # research: bias toward finer grid if success is still acceptable
    acceptable = [
        (name, s) for name, s in profile_summaries.items() if s["success_rate"] >= max(0.0, profile_summaries[production]["success_rate"] - 0.1)
    ]
    if not acceptable:
        acceptable = list(profile_summaries.items())
    research = sorted(acceptable, key=lambda it: (it[1]["mean_artifact_bytes"], -it[1]["success_rate"]), reverse=True)[0][0]

    return {
        "production_full_default": production,
        "research_full_default": research,
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir).resolve()
    pkl_dir = out_root / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor()
    calculator = DFTCalculator(
        functional=args.functional,
        basis=args.basis,
        verbose=0,
        geometry_optimization=False,
    )

    # Step 1: generate wavefunctions once
    pkl_map: Dict[str, str] = {}
    for mol_id, smiles in DEFAULT_MOLECULES:
        mol = calculator.from_smiles(smiles)
        res = calculator.calculate_and_export(molecule_id=mol_id, mol_obj=mol, output_dir=str(pkl_dir))
        if not res.get("success"):
            raise RuntimeError(f"DFT failed for {mol_id}: {res.get('error')}")
        pkl_map[mol_id] = res["pkl_path"]

    profile_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Step 2: benchmark profiles
    for profile_name in args.profiles:
        if profile_name not in DEFAULT_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        profile_cfg = DEFAULT_PROFILES[profile_name]
        plugin_plan = build_plugin_plan(profile_cfg)
        profile_out = out_root / profile_name
        profile_out.mkdir(parents=True, exist_ok=True)

        for mol_id, smiles in DEFAULT_MOLECULES:
            t0 = time.time()
            data = extractor.extract_unified(
                pkl_path=pkl_map[mol_id],
                molecule_id=f"{mol_id}_{profile_name}",
                smiles=smiles,
                dft_config={
                    "functional": args.functional,
                    "basis": args.basis,
                    "geometry_optimization": False,
                    "geo_opt_method": "none",
                    "gpu_used": None,
                },
                plugin_plan=plugin_plan,
                run_mode="full",
                output_dir=str(profile_out),
            )
            total_wall = time.time() - t0
            save_path = profile_out / f"{mol_id}_{profile_name}"
            extractor.save_unified_features(data, str(save_path))

            determiner = StatusDeterminer(data)
            _ = determiner.determine()
            reason_codes = determiner.get_reason_codes()

            real_meta = data.get("realspace_features", {}).get("metadata", {}) or {}
            rs_status = real_meta.get("extraction_status")
            rs_reason = real_meta.get("failure_reason")
            rs_time = real_meta.get("extraction_time_seconds")
            artifact_bytes = total_cube_bytes(data)
            profile_rows[profile_name].append(
                {
                    "molecule_id": mol_id,
                    "smiles": smiles,
                    "overall_status": data.get("calculation_status", {}).get("overall_status"),
                    "reason_codes": reason_codes,
                    "realspace_status": rs_status,
                    "realspace_failure_reason": rs_reason,
                    "realspace_time_seconds": rs_time if rs_time is not None else total_wall,
                    "total_wall_time_seconds": total_wall,
                    "artifact_bytes": artifact_bytes,
                }
            )

    profile_summaries = {
        name: aggregate_profile(rows)
        for name, rows in profile_rows.items()
    }
    recommendations = choose_recommendations(profile_summaries)

    result = {
        "profiles": {name: DEFAULT_PROFILES[name] for name in args.profiles},
        "molecules": [{"molecule_id": m[0], "smiles": m[1]} for m in DEFAULT_MOLECULES],
        "profile_summaries": profile_summaries,
        "rows": profile_rows,
        "recommendations": recommendations,
    }

    report_path = out_root / "realspace_benchmark_summary.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark report saved: {report_path}")
    for name in args.profiles:
        s = profile_summaries[name]
        print(
            f"{name}: success_rate={s['success_rate']:.2%}, timeout_rate={s['timeout_rate']:.2%}, "
            f"mean={s['mean_wall_time_seconds']:.2f}s, p90={s['p90_wall_time_seconds']:.2f}s, "
            f"mean_artifact={s['mean_artifact_bytes']/1024:.1f}KB"
        )
    print("Recommended:")
    print(json.dumps(recommendations, ensure_ascii=False, indent=2))

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M5.5 realspace lightweight benchmark runner")
    parser.add_argument("--output-dir", default="test_output_m55_realspace_benchmark")
    parser.add_argument("--functional", default="B3LYP")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--profiles", nargs="+", default=["P0", "P1", "P2"])
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())

"""
FastORCA end-to-end performance benchmark harness.

Goals:
- keep full-mode feature semantics unchanged
- benchmark real def2-TZVP throughput under comparable settings
- compare serial baseline vs parallel consumer vs GPU/CPU pipeline overlap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (
    apply_bader_validation_profile,
    build_batch_kwargs_from_profile,
    load_qcmol_substitute_default_profile,
)


DEFAULT_COMPARE_PATHS = [
    "calculation_status.overall_status",
    "global_features.dft.total_energy_hartree",
    "global_features.dft.homo_energy_hartree",
    "global_features.dft.lumo_energy_hartree",
    "global_features.dft.homo_lumo_gap_hartree",
    "global_features.dft.dipole_moment_debye",
    "atom_features.charge_iao",
    "atom_features.atomic_density_partition_charge_proxy.bader",
    "bond_features.bond_orders_mayer",
    "bond_features.elf_bond_midpoint",
    "structural_features.most_stable_conformation",
    "realspace_features.density_isosurface_volume",
]


def _capture_feature_extractor_state(feature_extractor: Any) -> Dict[str, Any]:
    state = {
        "constructor_kwargs": {
            "use_multiwfn": getattr(feature_extractor, "use_multiwfn", False),
            "multiwfn_path": getattr(feature_extractor, "multiwfn_path", "Multiwfn"),
            "output_format": getattr(feature_extractor, "output_format", "json"),
        },
        "runtime_overrides": {},
    }
    for key in [
        "BADER_POPULATION_SUM_ABS_TOL_E",
        "BADER_POPULATION_SUM_REL_TOL",
        "BADER_REFINED_RETRY_ENABLED",
        "BADER_REFINED_GRID_RES_ANGSTROM",
        "BADER_REFINED_MARGIN_ANGSTROM",
        "BADER_REFINED_MAX_POINTS_PER_DIM",
        "BADER_REFINED_MAX_TOTAL_GRID_POINTS",
        "BADER_RESCUE_RETRY_ENABLED",
        "BADER_RESCUE_GRID_RES_ANGSTROM",
        "BADER_RESCUE_MARGIN_ANGSTROM",
        "BADER_RESCUE_MAX_POINTS_PER_DIM",
        "BADER_RESCUE_MAX_TOTAL_GRID_POINTS",
    ]:
        if hasattr(feature_extractor, key):
            state["runtime_overrides"][key] = getattr(feature_extractor, key)
    return state


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_molecules(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input-json must be a JSON list")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"input-json row #{idx} must be object")
        smiles = row.get("smiles")
        if not smiles:
            raise ValueError(f"input-json row #{idx} missing smiles")
        molecule_id = row.get("molecule_id") or f"mol_{idx+1:04d}"
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "charge": int(row.get("charge", 0)),
                "spin": int(row.get("spin", 0)),
            }
        )
    if not rows:
        raise ValueError("No molecules loaded from input-json")
    return rows


def _detect_gpu_count() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return len(lines)
    except Exception:
        return 0


class ResourceMonitor:
    def __init__(self, interval_seconds: float = 1.0):
        self.interval_seconds = max(0.2, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cpu_samples: List[float] = []
        self.gpu_samples: List[float] = []
        self.gpu_mem_samples_mb: List[float] = []
        self.cpu_reason: Optional[str] = None
        self.gpu_reason: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return {
            "cpu": self._summarize(self.cpu_samples, self.cpu_reason),
            "gpu": self._summarize(self.gpu_samples, self.gpu_reason, extra_key="memory_used_mb", extra_values=self.gpu_mem_samples_mb),
        }

    def _run(self) -> None:
        psutil = None
        try:
            import psutil as _psutil  # type: ignore

            psutil = _psutil
            psutil.cpu_percent(interval=None)
        except Exception as exc:
            self.cpu_reason = f"psutil_unavailable:{exc}"

        has_nvidia_smi = shutil.which("nvidia-smi") is not None
        if not has_nvidia_smi:
            self.gpu_reason = "nvidia_smi_unavailable"

        while not self._stop.is_set():
            if psutil is not None:
                try:
                    self.cpu_samples.append(float(psutil.cpu_percent(interval=None)))
                except Exception as exc:
                    if self.cpu_reason is None:
                        self.cpu_reason = f"psutil_sampling_failed:{exc}"
                    psutil = None

            if has_nvidia_smi:
                try:
                    proc = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    line = proc.stdout.strip().splitlines()[0]
                    util_s, mem_s = [item.strip() for item in line.split(",")[:2]]
                    self.gpu_samples.append(float(util_s))
                    self.gpu_mem_samples_mb.append(float(mem_s))
                except Exception as exc:
                    if self.gpu_reason is None:
                        self.gpu_reason = f"nvidia_smi_sampling_failed:{exc}"
                    has_nvidia_smi = False

            self._stop.wait(self.interval_seconds)

    @staticmethod
    def _summarize(
        values: List[float],
        unavailable_reason: Optional[str],
        extra_key: Optional[str] = None,
        extra_values: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        if not values:
            summary = {
                "available": False,
                "reason": unavailable_reason or "no_samples_collected",
            }
            if extra_key:
                summary[extra_key] = None
            return summary
        summary = {
            "available": True,
            "mean": round(statistics.mean(values), 3),
            "p50": round(percentile(values, 50), 3),
            "p90": round(percentile(values, 90), 3),
            "max": round(max(values), 3),
            "n_samples": len(values),
        }
        if extra_key:
            ev = extra_values or []
            summary[extra_key] = {
                "mean": round(statistics.mean(ev), 3) if ev else None,
                "p90": round(percentile(ev, 90), 3) if ev else None,
                "max": round(max(ev), 3) if ev else None,
            }
        return summary


def _collect_unified_outputs(unified_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(unified_dir.glob("*.unified.json")):
        data = _read_json(fp)
        molecule_id = _get(data, "molecule_info.molecule_id") or fp.stem.replace(".unified", "")
        rows[molecule_id] = data
    return rows


def _stage_stats_from_unified(outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    stage_values: Dict[str, List[float]] = defaultdict(list)
    for data in outputs.values():
        stage_map = _get(data, "runtime_metadata.stage_timing_seconds") or {}
        if not isinstance(stage_map, dict):
            continue
        for key, value in stage_map.items():
            try:
                if value is not None:
                    stage_values[key].append(float(value))
            except Exception:
                continue
    summary: Dict[str, Any] = {}
    for key, values in stage_values.items():
        if not values:
            continue
        summary[key] = {
            "mean_seconds": round(statistics.mean(values), 6),
            "p50_seconds": round(percentile(values, 50), 6),
            "p90_seconds": round(percentile(values, 90), 6),
            "total_seconds": round(sum(values), 6),
        }
    return summary


def _compare_values(a: Any, b: Any, tol: float = 1e-10) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
        except Exception:
            return False
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_compare_values(x, y, tol=tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare_values(a[k], b[k], tol=tol) for k in a.keys())
    return a == b


def compare_unified_dirs(dir_a: Path, dir_b: Path, compare_paths: Iterable[str]) -> Dict[str, Any]:
    rows_a = _collect_unified_outputs(dir_a)
    rows_b = _collect_unified_outputs(dir_b)
    common_ids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    mismatches: List[Dict[str, Any]] = []

    for molecule_id in common_ids:
        data_a = rows_a[molecule_id]
        data_b = rows_b[molecule_id]
        for path in compare_paths:
            va = _get(data_a, path)
            vb = _get(data_b, path)
            if not _compare_values(va, vb):
                mismatches.append(
                    {
                        "molecule_id": molecule_id,
                        "path": path,
                        "value_a": va,
                        "value_b": vb,
                    }
                )

    return {
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "n_common_files": len(common_ids),
        "mismatch_count": len(mismatches),
        "matches": len(mismatches) == 0,
        "mismatch_preview": mismatches[:20],
    }


def _aggregate_result_rows(
    mode: str,
    total_wall_seconds: float,
    rows: List[Dict[str, Any]],
    resource_summary: Dict[str, Any],
    unified_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    wall_times = [float(r["wall_time_seconds"]) for r in rows if r.get("wall_time_seconds") is not None]
    dft_times = [
        float((r.get("dft_timing_seconds") or {}).get("calculate_and_export_seconds"))
        for r in rows
        if (r.get("dft_timing_seconds") or {}).get("calculate_and_export_seconds") is not None
    ]
    producer_build_times = [
        float(r["molecule_build_seconds"])
        for r in rows
        if r.get("molecule_build_seconds") is not None
    ]
    statuses = Counter(r.get("overall_status") or "unknown" for r in rows)
    failures = [r for r in rows if (r.get("overall_status") or "").startswith("failed") or r.get("error")]

    result: Dict[str, Any] = {
        "mode": mode,
        "molecule_count": len(rows),
        "wall_time_total_seconds": round(total_wall_seconds, 3),
        "throughput_molecules_per_hour": round((len(rows) / total_wall_seconds) * 3600.0, 3) if total_wall_seconds > 1e-12 else 0.0,
        "overall_status_counts": dict(statuses),
        "fail_rate": round(len(failures) / len(rows), 6) if rows else 0.0,
        "per_molecule_wall_time_seconds": {
            "mean": round(statistics.mean(wall_times), 6) if wall_times else None,
            "p50": round(percentile(wall_times, 50), 6) if wall_times else None,
            "p90": round(percentile(wall_times, 90), 6) if wall_times else None,
            "max": round(max(wall_times), 6) if wall_times else None,
        },
        "dft_stage_seconds": {
            "mean": round(statistics.mean(dft_times), 6) if dft_times else None,
            "p50": round(percentile(dft_times, 50), 6) if dft_times else None,
            "p90": round(percentile(dft_times, 90), 6) if dft_times else None,
        },
        "molecule_build_seconds": {
            "mean": round(statistics.mean(producer_build_times), 6) if producer_build_times else None,
            "p50": round(percentile(producer_build_times, 50), 6) if producer_build_times else None,
            "p90": round(percentile(producer_build_times, 90), 6) if producer_build_times else None,
        },
        "resource_utilization": resource_summary,
        "result_preview": rows[:10],
    }
    if unified_outputs is not None:
        result["unified_stage_timing_seconds"] = _stage_stats_from_unified(unified_outputs)
    return result


def _prepare_runtime(profile_path: Path):
    from consumer.feature_extractor import FeatureExtractor

    profile = load_qcmol_substitute_default_profile(profile_path)
    batch_kwargs = build_batch_kwargs_from_profile(profile)
    extractor = FeatureExtractor()
    apply_bader_validation_profile(extractor, profile)
    return profile, batch_kwargs, extractor


def _build_dft_config(functional: str, basis: str, geometry_optimization: bool, geo_opt_method: str) -> Dict[str, Any]:
    return {
        "calculator_kwargs": {
            "functional": functional,
            "basis": basis,
            "verbose": 3,
            "geometry_optimization": geometry_optimization,
            "geo_opt_method": geo_opt_method,
        },
        "extraction_context": {
            "functional": functional,
            "basis": basis,
            "geometry_optimization": geometry_optimization,
            "geo_opt_method": geo_opt_method,
            "geometry_optimization_success": geometry_optimization,
            "gpu_used": None,
        },
    }


def run_serial_benchmark(
    molecules: List[Dict[str, Any]],
    output_dir: Path,
    profile_path: Path,
    dft_config: Dict[str, Any],
) -> Dict[str, Any]:
    from consumer.molecule_processor import MoleculeProcessorConfig
    from producer.dft_calculator import DFTCalculator

    profile, batch_kwargs, extractor = _prepare_runtime(profile_path)
    calculator_kwargs = dict(dft_config.get("calculator_kwargs") or {})
    extraction_dft_context = dict(dft_config.get("extraction_context") or {})
    processor = MoleculeProcessorConfig.from_dict(
        {
            "run_mode": batch_kwargs["run_mode"],
            "artifact_policy": batch_kwargs["artifact_policy"],
            "plugins": batch_kwargs["plugin_config"],
        }
    ).create_processor(extractor, output_dir)
    calculator = DFTCalculator(**calculator_kwargs)
    pkl_dir = output_dir / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    monitor = ResourceMonitor()
    monitor.start()
    total_start = time.time()

    for item in molecules:
        molecule_id = item["molecule_id"]
        smiles = item["smiles"]
        build_start = time.time()
        mol = calculator.from_smiles(
            smiles,
            charge=int(item.get("charge", 0)),
            spin=int(item.get("spin", 0)),
        )
        molecule_build_seconds = time.time() - build_start
        dft_result = calculator.calculate_and_export(
            molecule_id=molecule_id,
            mol_obj=mol,
            output_dir=str(pkl_dir),
        )
        if not dft_result.get("success"):
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "overall_status": "failed_scf",
                    "wall_time_seconds": None,
                    "dft_timing_seconds": dft_result.get("timing_seconds"),
                    "molecule_build_seconds": molecule_build_seconds,
                    "error": dft_result.get("error"),
                }
            )
            continue

        result = processor.process_one(
            pkl_path=Path(dft_result["pkl_path"]),
            molecule_id=molecule_id,
            smiles=smiles,
            dft_config=extraction_dft_context,
            return_data_mode="summary",
        )
        rows.append(
            {
                "molecule_id": molecule_id,
                "overall_status": result.overall_status,
                "reason_codes": result.reason_codes,
                "wall_time_seconds": result.wall_time_seconds,
                "metrics": result.metrics,
                "dft_timing_seconds": dft_result.get("timing_seconds"),
                "execution_backend": dft_result.get("execution_backend"),
                "molecule_build_seconds": molecule_build_seconds,
            }
        )

    total_wall = time.time() - total_start
    resource_summary = monitor.stop()
    outputs = _collect_unified_outputs(output_dir)
    return _aggregate_result_rows("serial_baseline", total_wall, rows, resource_summary, outputs)


def run_parallel_consumer_benchmark(
    molecules: List[Dict[str, Any]],
    output_dir: Path,
    profile_path: Path,
    dft_config: Dict[str, Any],
    n_workers: int,
) -> Dict[str, Any]:
    from consumer.batch_runner import run_batch
    from producer.dft_calculator import DFTCalculator

    profile, batch_kwargs, extractor = _prepare_runtime(profile_path)
    calculator_kwargs = dict(dft_config.get("calculator_kwargs") or {})
    extraction_dft_context = dict(dft_config.get("extraction_context") or {})
    calculator = DFTCalculator(**calculator_kwargs)
    pkl_dir = output_dir / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    dft_rows: Dict[str, Dict[str, Any]] = {}

    monitor = ResourceMonitor()
    monitor.start()
    total_start = time.time()

    molecule_list: List[Dict[str, Any]] = []
    for item in molecules:
        molecule_id = item["molecule_id"]
        smiles = item["smiles"]
        build_start = time.time()
        mol = calculator.from_smiles(
            smiles,
            charge=int(item.get("charge", 0)),
            spin=int(item.get("spin", 0)),
        )
        molecule_build_seconds = time.time() - build_start
        dft_result = calculator.calculate_and_export(
            molecule_id=molecule_id,
            mol_obj=mol,
            output_dir=str(pkl_dir),
        )
        dft_rows[molecule_id] = {
            "molecule_build_seconds": molecule_build_seconds,
            "dft_timing_seconds": dft_result.get("timing_seconds"),
            "execution_backend": dft_result.get("execution_backend"),
            "error": dft_result.get("error"),
        }
        if dft_result.get("success"):
            molecule_list.append(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "pkl_path": dft_result["pkl_path"],
                    "dft_config": extraction_dft_context,
                }
            )

    batch_summary = run_batch(
        feature_extractor=extractor,
        molecule_list=molecule_list,
        output_dir=output_dir,
        run_mode=batch_kwargs["run_mode"],
        artifact_policy=batch_kwargs["artifact_policy"],
        plugin_config=batch_kwargs["plugin_config"],
        n_workers=n_workers,
        include_molecule_results=True,
    )

    rows: List[Dict[str, Any]] = []
    batch_result_map = {
        row["molecule_id"]: row for row in batch_summary.get("molecule_results", [])
    }
    for item in molecules:
        molecule_id = item["molecule_id"]
        dft_meta = dft_rows.get(molecule_id, {})
        row = batch_result_map.get(molecule_id)
        if row is None:
            rows.append(
                {
                    "molecule_id": molecule_id,
                    "overall_status": "failed_scf",
                    "wall_time_seconds": None,
                    "dft_timing_seconds": dft_meta.get("dft_timing_seconds"),
                    "molecule_build_seconds": dft_meta.get("molecule_build_seconds"),
                    "execution_backend": dft_meta.get("execution_backend"),
                    "error": dft_meta.get("error"),
                }
            )
            continue
        rows.append(
            {
                "molecule_id": molecule_id,
                "overall_status": row["overall_status"],
                "reason_codes": row.get("reason_codes"),
                "wall_time_seconds": row.get("wall_time_seconds"),
                "metrics": row.get("metrics"),
                "dft_timing_seconds": dft_meta.get("dft_timing_seconds"),
                "molecule_build_seconds": dft_meta.get("molecule_build_seconds"),
                "execution_backend": dft_meta.get("execution_backend"),
            }
        )

    total_wall = time.time() - total_start
    resource_summary = monitor.stop()
    outputs = _collect_unified_outputs(output_dir)
    result = _aggregate_result_rows("parallel_consumer", total_wall, rows, resource_summary, outputs)
    result["batch_summary"] = batch_summary
    return result


def _pipeline_consumer_worker(
    task_queue: Any,
    result_queue: Any,
    extractor_state: Dict[str, Any],
    processor_config: Dict[str, Any],
    output_dir: str,
) -> None:
    from consumer.batch_runner import _build_feature_extractor_from_state
    from consumer.molecule_processor import MoleculeProcessorConfig

    extractor = _build_feature_extractor_from_state(extractor_state)
    config = MoleculeProcessorConfig.from_dict(processor_config)
    processor = config.create_processor(extractor, Path(output_dir))

    while True:
        task = task_queue.get()
        if task is None:
            break
        molecule_id = task["molecule_id"]
        try:
            result = processor.process_one(
                pkl_path=Path(task["pkl_path"]),
                molecule_id=molecule_id,
                smiles=task.get("smiles"),
                dft_config=task.get("dft_config"),
                return_data_mode="summary",
            )
            result_queue.put(
                {
                    "molecule_id": molecule_id,
                    "overall_status": result.overall_status,
                    "reason_codes": result.reason_codes,
                    "wall_time_seconds": result.wall_time_seconds,
                    "metrics": result.metrics,
                    "dft_timing_seconds": task.get("dft_timing_seconds"),
                    "execution_backend": task.get("execution_backend"),
                    "molecule_build_seconds": task.get("molecule_build_seconds"),
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "molecule_id": molecule_id,
                    "overall_status": "failed_core_features",
                    "reason_codes": ["wavefunction_corrupted"],
                    "wall_time_seconds": None,
                    "metrics": {"error": str(exc)},
                    "dft_timing_seconds": task.get("dft_timing_seconds"),
                    "execution_backend": task.get("execution_backend"),
                    "molecule_build_seconds": task.get("molecule_build_seconds"),
                    "error": str(exc),
                }
            )


def _pipeline_producer_worker(
    molecules: List[Dict[str, Any]],
    dft_config: Dict[str, Any],
    pkl_dir: str,
    task_queue: Any,
    result_queue: Any,
    gpu_device_id: Optional[int] = None,
) -> None:
    from producer.dft_calculator import DFTCalculator

    calculator_kwargs = dict(dft_config.get("calculator_kwargs") or {})
    if gpu_device_id is not None:
        calculator_kwargs["gpu_device_id"] = int(gpu_device_id)
    extraction_dft_context = dict(dft_config.get("extraction_context") or {})
    extraction_dft_context["gpu_used"] = gpu_device_id
    print(
        f"[benchmark] pipeline producer start pid={os.getpid()} gpu_device_id={gpu_device_id} molecules={len(molecules)}",
        flush=True,
    )
    calculator = DFTCalculator(**calculator_kwargs)
    for item in molecules:
        molecule_id = item["molecule_id"]
        smiles = item["smiles"]
        try:
            build_start = time.time()
            mol = calculator.from_smiles(
                smiles,
                charge=int(item.get("charge", 0)),
                spin=int(item.get("spin", 0)),
            )
            molecule_build_seconds = time.time() - build_start
            dft_result = calculator.calculate_and_export(
                molecule_id=molecule_id,
                mol_obj=mol,
                output_dir=pkl_dir,
            )
            if not dft_result.get("success"):
                result_queue.put(
                    {
                        "molecule_id": molecule_id,
                        "overall_status": "failed_scf",
                        "reason_codes": ["wavefunction_corrupted"],
                        "wall_time_seconds": None,
                        "dft_timing_seconds": dft_result.get("timing_seconds"),
                        "execution_backend": dft_result.get("execution_backend"),
                        "molecule_build_seconds": molecule_build_seconds,
                        "error": dft_result.get("error"),
                    }
                )
                continue

            task_queue.put(
                {
                    "molecule_id": molecule_id,
                    "smiles": smiles,
                    "pkl_path": dft_result["pkl_path"],
                    "dft_config": extraction_dft_context,
                    "dft_timing_seconds": dft_result.get("timing_seconds"),
                    "execution_backend": dft_result.get("execution_backend"),
                    "molecule_build_seconds": molecule_build_seconds,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "molecule_id": molecule_id,
                    "overall_status": "failed_scf",
                    "reason_codes": ["wavefunction_corrupted"],
                    "wall_time_seconds": None,
                    "dft_timing_seconds": None,
                    "execution_backend": None,
                    "molecule_build_seconds": None,
                    "error": str(exc),
                }
            )


def run_pipeline_benchmark(
    molecules: List[Dict[str, Any]],
    output_dir: Path,
    profile_path: Path,
    dft_config: Dict[str, Any],
    n_consumers: int,
    queue_size: int,
    n_gpu_producers: int,
) -> Dict[str, Any]:
    import multiprocessing as mp

    profile, batch_kwargs, extractor = _prepare_runtime(profile_path)
    processor_config = {
        "run_mode": batch_kwargs["run_mode"],
        "artifact_policy": batch_kwargs["artifact_policy"],
        "plugins": batch_kwargs["plugin_config"],
    }
    extractor_state = _capture_feature_extractor_state(extractor)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(1, int(queue_size)))
    result_queue = ctx.Queue()
    pkl_dir = output_dir / "pkl"
    pkl_dir.mkdir(parents=True, exist_ok=True)

    gpu_count = _detect_gpu_count()
    n_producers = max(1, int(n_gpu_producers))
    # Allow multiple producers per GPU. Small/medium molecules often under-utilize
    # a single A800 stream, so benchmark mode should support oversubscription.
    n_producers = min(n_producers, len(molecules))

    chunk_size = max(1, math.ceil(len(molecules) / max(1, n_producers)))
    producers = []
    for idx in range(n_producers):
        start = idx * chunk_size
        end = min(len(molecules), (idx + 1) * chunk_size)
        chunk = molecules[start:end]
        if not chunk:
            continue
        gpu_device_id = idx % gpu_count if gpu_count > 0 else None
        producers.append(
            ctx.Process(
                target=_pipeline_producer_worker,
                args=(chunk, dft_config, str(pkl_dir), task_queue, result_queue, gpu_device_id),
                name=f"FastORCA-GPU-Producer-{idx}",
            )
        )
    consumers = [
        ctx.Process(
            target=_pipeline_consumer_worker,
            args=(task_queue, result_queue, extractor_state, processor_config, str(output_dir)),
            name=f"FastORCA-CPU-Consumer-{idx}",
        )
        for idx in range(n_consumers)
    ]

    monitor = ResourceMonitor()
    monitor.start()
    total_start = time.time()
    for producer in producers:
        producer.start()
    for proc in consumers:
        proc.start()

    rows: List[Dict[str, Any]] = []
    while len(rows) < len(molecules):
        rows.append(result_queue.get())

    for producer in producers:
        producer.join()
    for _ in range(n_consumers):
        task_queue.put(None)
    for proc in consumers:
        proc.join()

    total_wall = time.time() - total_start
    resource_summary = monitor.stop()
    outputs = _collect_unified_outputs(output_dir)
    result = _aggregate_result_rows("pipeline_overlap", total_wall, rows, resource_summary, outputs)
    result["pipeline_scheduler"] = {
        "n_gpu_producers": len(producers),
        "n_cpu_consumers": n_consumers,
        "detected_gpu_count": gpu_count,
        "producers_per_gpu": (
            round(len(producers) / max(1, gpu_count), 3) if gpu_count > 0 else None
        ),
        "queue_size": int(queue_size),
        "molecule_chunk_size": int(chunk_size),
    }
    return result


def write_report(report: Dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# FastORCA Performance Benchmark",
        f"- basis: {report['config']['basis']}",
        f"- functional: {report['config']['functional']}",
        f"- run_mode: {report['config']['run_mode']}",
        f"- molecule_count: {report['config']['molecule_count']}",
        "",
    ]
    for mode in report["benchmarks"]:
        lines.extend(
            [
                f"## {mode['mode']}",
                f"- wall_time_total_seconds: {mode['wall_time_total_seconds']}",
                f"- throughput_molecules_per_hour: {mode['throughput_molecules_per_hour']}",
                f"- overall_status_counts: {mode['overall_status_counts']}",
                f"- fail_rate: {mode['fail_rate']}",
                f"- per_molecule_mean_seconds: {mode['per_molecule_wall_time_seconds']['mean']}",
                f"- per_molecule_p90_seconds: {mode['per_molecule_wall_time_seconds']['p90']}",
                "",
            ]
        )

    if report.get("correctness_comparisons"):
        lines.append("## Correctness")
        for item in report["correctness_comparisons"]:
            lines.append(
                f"- {item['label']}: matches={item['result']['matches']} mismatch_count={item['result']['mismatch_count']}"
            )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    detected_gpu_count = _detect_gpu_count()
    p = argparse.ArgumentParser(description="FastORCA full-mode throughput benchmark")
    p.add_argument("--input-json", required=True, help="JSON list with molecule_id/smiles[/charge/spin]")
    p.add_argument("--output-dir", required=True, help="Benchmark output root")
    p.add_argument("--profile-path", default=str(REPO_ROOT / "configs" / "qcmol_substitute_default.json"))
    p.add_argument("--functional", default="B3LYP")
    p.add_argument("--basis", default="def2-TZVP")
    p.add_argument("--geo-opt-method", default="xtb", choices=["xtb", "pyscf", "none"])
    p.add_argument("--disable-geometry-optimization", action="store_true")
    p.add_argument("--n-consumer-workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    p.add_argument(
        "--n-gpu-producers",
        type=int,
        default=max(1, (detected_gpu_count or 1) * 2),
        help="Number of GPU-side producer processes for pipeline mode; may exceed GPU count for oversubscription",
    )
    p.add_argument("--queue-size", type=int, default=8)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["serial", "parallel-consumer", "pipeline"],
        choices=["serial", "parallel-consumer", "pipeline"],
        help="Benchmark modes to execute",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile_path = Path(args.profile_path).resolve()

    molecules = _load_molecules(input_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "input_json": str(input_json),
        "profile_path": str(profile_path),
        "functional": args.functional,
        "basis": args.basis,
        "run_mode": "full",
        "molecule_count": len(molecules),
        "n_consumer_workers": args.n_consumer_workers,
        "n_gpu_producers": args.n_gpu_producers,
        "detected_gpu_count": _detect_gpu_count(),
        "queue_size": args.queue_size,
        "modes": args.modes,
    }
    (output_dir / "benchmark_config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dft_config = _build_dft_config(
        functional=args.functional,
        basis=args.basis,
        geometry_optimization=not args.disable_geometry_optimization,
        geo_opt_method=args.geo_opt_method,
    )

    results: Dict[str, Dict[str, Any]] = {}

    if "serial" in args.modes:
        print("[benchmark] starting mode=serial_baseline", flush=True)
        results["serial"] = run_serial_benchmark(
            molecules=molecules,
            output_dir=output_dir / "serial",
            profile_path=profile_path,
            dft_config=dft_config,
        )

    if "parallel-consumer" in args.modes:
        print("[benchmark] starting mode=parallel_consumer", flush=True)
        results["parallel-consumer"] = run_parallel_consumer_benchmark(
            molecules=molecules,
            output_dir=output_dir / "parallel_consumer",
            profile_path=profile_path,
            dft_config=dft_config,
            n_workers=args.n_consumer_workers,
        )

    if "pipeline" in args.modes:
        print(
            f"[benchmark] starting mode=pipeline_overlap n_gpu_producers={args.n_gpu_producers} n_cpu_consumers={args.n_consumer_workers}",
            flush=True,
        )
        results["pipeline"] = run_pipeline_benchmark(
            molecules=molecules,
            output_dir=output_dir / "pipeline",
            profile_path=profile_path,
            dft_config=dft_config,
            n_consumers=args.n_consumer_workers,
            queue_size=args.queue_size,
            n_gpu_producers=args.n_gpu_producers,
        )

    comparisons: List[Dict[str, Any]] = []
    if "serial" in results and "parallel-consumer" in results:
        comparisons.append(
            {
                "label": "serial_vs_parallel_consumer",
                "result": compare_unified_dirs(
                    output_dir / "serial",
                    output_dir / "parallel_consumer",
                    DEFAULT_COMPARE_PATHS,
                ),
            }
        )
    if "serial" in results and "pipeline" in results:
        comparisons.append(
            {
                "label": "serial_vs_pipeline",
                "result": compare_unified_dirs(
                    output_dir / "serial",
                    output_dir / "pipeline",
                    DEFAULT_COMPARE_PATHS,
                ),
            }
        )

    benchmark_rows = [results[key] for key in ["serial", "parallel-consumer", "pipeline"] if key in results]
    report = {
        "config": config_snapshot,
        "benchmarks": benchmark_rows,
        "correctness_comparisons": comparisons,
    }

    output_json = output_dir / "fastorca_performance_benchmark.json"
    output_md = output_dir / "fastorca_performance_benchmark.md"
    write_report(report, output_json, output_md)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "modes": [row["mode"] for row in benchmark_rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

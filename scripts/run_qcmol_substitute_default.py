"""
Run batch extraction with frozen qcMol substitute default profile.

This script is the canonical entrypoint for "default delivery" runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.qcmol_substitute_profile import (
    PROFILE_CONFIG_PATH,
    apply_bader_validation_profile,
    build_batch_kwargs_from_profile,
    load_qcmol_substitute_default_profile,
)


def _load_molecules_from_pkl_dir(input_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(input_path.glob("*.pkl")):
        rows.append(
            {
                "molecule_id": fp.stem,
                "pkl_path": str(fp),
            }
        )
    return rows


def _load_molecules_from_unified_dir(input_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fp in sorted(input_path.glob("*.unified.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        molecule_id = (data.get("molecule_info") or {}).get("molecule_id") or fp.stem.replace(".unified", "")
        smiles = (data.get("molecule_info") or {}).get("smiles")
        pkl_path = (((data.get("artifacts") or {}).get("wavefunction") or {}).get("pkl_path"))
        if not pkl_path:
            continue
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "pkl_path": pkl_path,
            }
        )
    return rows


def _load_molecules_from_json(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("molecule-json must be a JSON list")
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"molecule-json item #{idx} must be object")
        if "pkl_path" not in row:
            raise ValueError(f"molecule-json item #{idx} missing pkl_path")
        molecule_id = row.get("molecule_id") or Path(str(row["pkl_path"])).stem
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": row.get("smiles"),
                "pkl_path": row["pkl_path"],
                "dft_config": row.get("dft_config"),
            }
        )
    return rows


def _load_molecule_list(input_mode: str, input_path: Path) -> List[Dict[str, Any]]:
    if input_mode == "pkl-dir":
        return _load_molecules_from_pkl_dir(input_path)
    if input_mode == "unified-dir":
        return _load_molecules_from_unified_dir(input_path)
    if input_mode == "molecule-json":
        return _load_molecules_from_json(input_path)
    raise ValueError(f"Unsupported input mode: {input_mode}")


def _validate_molecules(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No molecules loaded from input")
    missing = []
    for row in rows:
        pkl = Path(str(row.get("pkl_path", "")))
        if not pkl.exists():
            missing.append(str(pkl))
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"Missing pkl files: {preview} (and {max(0, len(missing)-5)} more)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run qcMol substitute default profile in one command")
    p.add_argument(
        "--input-mode",
        required=True,
        choices=["pkl-dir", "unified-dir", "molecule-json"],
        help="How to build the molecule list",
    )
    p.add_argument("--input-path", required=True, help="Input directory/file according to input-mode")
    p.add_argument("--output-dir", required=True, help="Output directory for unified json files and batch summary")
    p.add_argument("--profile-path", default=str(PROFILE_CONFIG_PATH), help="Profile JSON path")
    p.add_argument("--n-workers", type=int, default=-1, help="Override profile n_workers; -1 means keep profile")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Delay heavy imports so --help works even without runtime deps.
    from consumer.batch_runner import run_batch
    from consumer.feature_extractor import FeatureExtractor

    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    profile_path = Path(args.profile_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input-path not found: {input_path}")
    if not profile_path.exists():
        raise FileNotFoundError(f"profile-path not found: {profile_path}")

    profile = load_qcmol_substitute_default_profile(profile_path)
    molecules = _load_molecule_list(args.input_mode, input_path)
    _validate_molecules(molecules)

    extractor = FeatureExtractor()
    apply_bader_validation_profile(extractor, profile)

    batch_kwargs = build_batch_kwargs_from_profile(profile)
    if args.n_workers >= 1:
        batch_kwargs["n_workers"] = int(args.n_workers)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qcmol_substitute_profile_snapshot.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = run_batch(
        feature_extractor=extractor,
        molecule_list=molecules,
        output_dir=output_dir,
        run_mode=batch_kwargs["run_mode"],
        artifact_policy=batch_kwargs["artifact_policy"],
        plugin_config=batch_kwargs["plugin_config"],
        n_workers=batch_kwargs["n_workers"],
    )

    print(
        json.dumps(
            {
                "profile_id": profile.get("profile_id"),
                "profile_path": str(profile_path),
                "molecule_count": len(molecules),
                "output_dir": str(output_dir),
                "batch_summary": summary,
                "effective_batch_kwargs": batch_kwargs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

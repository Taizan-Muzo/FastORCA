# qcMol Substitute Default Profile

This document freezes the default-delivery profile for FastORCA qcMol substitute runs.

## Profile ID

- `qcmol_substitute_default`
- config file: `configs/qcmol_substitute_default.json`
- target: stability-first open-source substitute mainline

## Frozen Execution Policy

- `run_mode = full`
- `artifact_policy = keep_failed_only`
- `n_workers = 1` (can be overridden by CLI)

## Frozen Plugin Policy

- `orbital_features`: enabled in full mode
- `realspace_features`: enabled in full mode with production-like runtime config
  - grid resolution: `0.2 A`
  - margin: `4.0 A`
  - timeout: `120s`
  - core + extended realspace both enabled
  - required artifacts: `density`
  - optional artifacts: `homo`, `lumo`
- `critic2_bridge`: enabled in full mode
  - timeout: `300s`
  - executable: `critic2`

## Frozen Bader Validation and Retry Policy

- population consistency gate:
  - `abs_tol_e = 0.50`
  - `rel_tol = 0.02`
- refined retry:
  - enabled, spacing `0.16 A`, margin `5.5 A`
- rescue retry:
  - enabled, spacing `0.14 A`, margin `6.0 A`

## Single Canonical Entry

Use script:

- `scripts/run_qcmol_substitute_default.py`

Supported inputs:

- `--input-mode pkl-dir`
- `--input-mode unified-dir`
- `--input-mode molecule-json`

Example:

```bash
python scripts/run_qcmol_substitute_default.py \
  --input-mode unified-dir \
  --input-path /home/sulixian/FastORCA/test_output_stage_validation/A_main \
  --output-dir /home/sulixian/FastORCA/test_output_qcmol_default
```

The script writes:

- unified outputs + batch summary in output dir
- `qcmol_substitute_profile_snapshot.json` for reproducible delivery config

# qcMol Substitute Default Profile (Blessed Entry)

This file freezes the **single recommended delivery path** for FastORCA qcMol substitute runs.

## Blessed Identity

- profile id: `qcmol_substitute_default`
- profile config: `configs/qcmol_substitute_default.json`
- blessed entrypoint: `scripts/run_qcmol_substitute_default.py`
- target: stability-first open-source qcMol substitute mainline

## Frozen Runtime Contract

- `run_mode = full`
- `artifact_policy = keep_failed_only`
- `n_workers = 1` (CLI override allowed)
- orbital/realspace/critic2 are all enabled in full mode
- realspace runtime baseline:
  - `grid_resolution_angstrom = 0.2`
  - `margin_angstrom = 4.0`
  - `timeout_seconds = 120`
- critic2 timeout: `300s`
- Bader validation guard:
  - `abs_tol_e = 0.50`
  - `rel_tol = 0.02`
- Bader retries:
  - refined retry: `0.16 A / 5.5 A`
  - rescue retry: `0.14 A / 6.0 A`

## Single Blessed CLI Path

Always invoke:

```bash
python scripts/run_qcmol_substitute_default.py ...
```

### Input modes and when to use them

1. `--input-mode pkl-dir`
- Use when you already have wavefunction PKLs.
- Required: directory with `*.pkl`.

2. `--input-mode unified-dir`
- Use when you want to re-consume previous unified outputs that contain artifact PKL refs.
- Required: directory with `*.unified.json` and valid `artifacts.wavefunction.pkl_path`.

3. `--input-mode molecule-json`
- Use when you need explicit per-molecule metadata control.
- Required: JSON list with `pkl_path`; optional `molecule_id`, `smiles`, `dft_config`.

### Raw-input note (important)

`run_qcmol_substitute_default.py` is a **consumer-stage blessed entrypoint**.  
Raw SMILES ingestion belongs to upstream producer workflows; convert raw inputs to PKLs first, then use `pkl-dir` here.

## Minimal command examples

### A) unified-dir consume mode (recommended for reproducible reruns)

```bash
python scripts/run_qcmol_substitute_default.py \
  --input-mode unified-dir \
  --input-path /home/sulixian/FastORCA/test_output_stage_validation/A_main \
  --output-dir /home/sulixian/FastORCA/test_output_qcmol_default
```

### B) pkl-dir consume mode

```bash
python scripts/run_qcmol_substitute_default.py \
  --input-mode pkl-dir \
  --input-path /home/sulixian/FastORCA/test_output_stage_validation/A_main/pkl \
  --output-dir /home/sulixian/FastORCA/test_output_qcmol_default_from_pkl
```

### C) molecule-json consume mode

```bash
python scripts/run_qcmol_substitute_default.py \
  --input-mode molecule-json \
  --input-path /home/sulixian/FastORCA/molecules_for_consume.json \
  --output-dir /home/sulixian/FastORCA/test_output_qcmol_default_from_json
```

## Deterministic outputs

Each run writes:

- `A_main/*.unified.json`
- `A_main/batch_summary_<id>.json`
- `qcmol_substitute_profile_snapshot.json` (frozen effective profile snapshot)

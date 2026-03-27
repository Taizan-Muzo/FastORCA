# qcMol Operational Checklist

This checklist is the operator-facing way to confirm the qcMol substitute mainline is healthy.

## Inputs

- frozen profile: `configs/qcmol_substitute_default.json`
- blessed runner: `scripts/run_qcmol_substitute_default.py`
- run output root: `<OUT>` (contains `A_main/`)

## One-Pass Command Set

```bash
OUT=/home/sulixian/FastORCA/test_output_qcmol_operational_$(date +%Y%m%d_%H%M%S)

python scripts/run_qcmol_substitute_default.py \
  --input-mode unified-dir \
  --input-path /home/sulixian/FastORCA/test_output_stage_validation/A_main \
  --output-dir "$OUT/A_main"

python scripts/qcmol_alignment_closure_validation.py \
  --unified-dir "$OUT/A_main" \
  --output-json "$OUT/qcmol_alignment_closure_validation.json" \
  --output-md "$OUT/qcmol_alignment_closure_validation.md"

python scripts/qcmol_alignment_master_table_report.py \
  --output-json "$OUT/qcmol_alignment_master_table.json" \
  --output-md "$OUT/qcmol_alignment_master_table.md"

python scripts/qcmol_substitute_readiness_report.py \
  --output-json "$OUT/qcmol_substitute_readiness.json" \
  --output-md "$OUT/qcmol_substitute_readiness.md"

python scripts/qcmol_operational_health_check.py \
  --run-output-dir "$OUT" \
  --output-json "$OUT/qcmol_operational_health_check.json" \
  --output-md "$OUT/qcmol_operational_health_check.md"
```

## Pass/Fail Gates

The run is considered **PASS** only if all gates are true:

1. `fully_success_ratio >= 0.95`
2. each plugin success rate (`orbital_features`, `realspace_features`, `critic2_bridge`) `>= 0.95`
3. master table status counts exactly match frozen closure snapshot:
   - `implemented_exact = 14`
   - `implemented_proxy = 17`
   - `partial = 1`
   - `missing = 2`
   - `rejected_as_exact = 6`
4. volume partial tail is explained by known upstream critic2 reasons:
   - e.g. `bader_volume_column_truly_missing`

## Operator Interpretation

- If gate 1/2 fails: runtime pipeline health regression.
- If gate 3 fails: alignment contract drift (must investigate before release).
- If gate 4 fails: new volume-tail reason appeared (likely parser/upstream format drift).
- If only volume remains partial with known upstream reason and all other gates pass: release remains acceptable.

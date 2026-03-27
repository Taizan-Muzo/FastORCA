# qcMol Substitute Readiness (Finalized)

## What This Is

FastORCA delivers an **open-source qcMol substitute**.  
It is intentionally **not** an exact replica of closed/proprietary exact-only families.

## Aligned Coverage Snapshot (Frozen)

- `implemented_exact = 14`
- `implemented_proxy = 17`
- `partial = 1`
- `missing = 2`
- `rejected_as_exact = 6`

The single non-blocking partial tail is:
- `atom_features.atomic_density_partition_volume_proxy.bader`

with evidence pointing to upstream critic2 volume-column availability limits.

## Out of Scope (Archived Exact-Only Families)

These are explicitly archived as `roadmap_only / rejected_as_exact`:

- `NAO_descriptors`
- `LI_values`
- `ADCH_charges`
- `NPA_exact`
- `NBO_BD`
- `LBO`

This is an active boundary choice under open-source substitute constraints, not an omission.

## Readiness Report Generator

```bash
python scripts/qcmol_substitute_readiness_report.py \
  --output-json /home/sulixian/FastORCA/test_output_readiness/qcmol_readiness_report.json \
  --output-md /home/sulixian/FastORCA/test_output_readiness/qcmol_readiness_report.md
```

## Health Confirmation (Operator Shortcut)

Use `docs/qcmol_operational_checklist.md` and:

```bash
python scripts/qcmol_operational_health_check.py \
  --run-output-dir /home/sulixian/FastORCA/test_output_qcmol_default \
  --output-json /home/sulixian/FastORCA/test_output_qcmol_default/qcmol_operational_health_check.json \
  --output-md /home/sulixian/FastORCA/test_output_qcmol_default/qcmol_operational_health_check.md
```

If all gates pass, the qcMol substitute mainline is considered release-ready.

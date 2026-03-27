# qcMol Alignment Master Table

This document defines the closure-sprint source of truth for qcMol gap closure.

## Source Of Truth

- Runtime table: `utils/qcmol_substitute_profile.py`
- Constant name: `QCMOL_ALIGNMENT_MASTER_TABLE`
- Every row includes:
  - `qcMol_item_name`
  - `mapped_path`
  - `current_status`
  - `next_action`
  - `completion_criterion`
  - `notes`

## Allowed Enums

- `current_status`
  - `implemented_exact`
  - `implemented_proxy`
  - `partial`
  - `missing`
  - `rejected_as_exact`
- `next_action`
  - `keep`
  - `upgrade`
  - `redefine`
  - `roadmap_only`
  - `reject`

## Why This Exists

- Prevents ad-hoc “gap guessing”.
- Keeps roadmap-only and rejected-as-exact families explicit under open-source constraints.
- Gives a stable target for closure validation scripts and readiness reporting.
- Freezes substitute-only semantics for redefine items (SMART / molecule_size / ionization-related / DI).
- Treats exact-only families (NAO/LI/ADCH/NPA_exact/NBO_BD/LBO) as explicitly archived roadmap-only scope.

## Snapshot Command

```bash
python scripts/qcmol_alignment_master_table_report.py \
  --output-json test_output_alignment_master/alignment_master_table.json \
  --output-md test_output_alignment_master/alignment_master_table.md
```

## Closure Validation Command

```bash
python scripts/qcmol_alignment_closure_validation.py \
  --unified-dir <unified_output_dir> \
  --output-json <report.json> \
  --output-md <report.md>
```

## Productization Note

For operational pass/fail gating, combine this report with:

- `scripts/qcmol_operational_health_check.py`
- `docs/qcmol_operational_checklist.md`

This keeps runtime health checks, master-table counts, and partial-tail explanations in one operator path.

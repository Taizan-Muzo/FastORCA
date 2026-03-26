# qcMol Substitute Readiness

This document defines readiness evaluation for default delivery and records the latest consolidation judgement.

## Report Generator

Use:

- `scripts/qcmol_substitute_readiness_report.py`

Example:

```bash
python scripts/qcmol_substitute_readiness_report.py \
  --validation-report /home/sulixian/FastORCA/test_output_stage_validation/validation_round_report.json \
  --bader-finalmile-report /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/bader_coverage_after_finalmile.json \
  --bader-uplift-baseline-report /home/sulixian/FastORCA/test_output_stage_validation_bader_uplift/bader_coverage_after.json \
  --output-json /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/qcmol_readiness_report.json \
  --output-md /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/qcmol_readiness_report.md
```

## Latest Snapshot (2026-03-26)

Based on validation + Bader final-mile outputs provided in the sprint thread:

- main batch: `fully_success = 30/30`
- critic2 execution success rate: `100%`
- Bader validated writeback rate: `90%`
- Bader volume writeback rate: `0%` (known partial)
- mismatch reason reduced to small hard-case set
- rescue retry is effective and bounded

## Readiness Judgement

Current state is ready for **default open-source qcMol substitute mainline** under the frozen profile:

- stable core pipeline
- strong Bader charge coverage with validation guard
- honest proxy/partial semantics preserved
- roadmap placeholders remain explicit and not promoted to exact fields

## Remaining Risks (Expected)

- Bader volume remains optional/partial and should not be a hard dependency.
- Open-shell orbital path still unavailable for IBO-derived proxies.
- exact qcMol external roadmap items (NAO/ADCH/NBO/LBO/DI exact) are still future work.

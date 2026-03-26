# High-Value / Rare-Feature Augmentation

This sprint focuses on high information-density features that are feasible on the current open-source stack.

## Module A: critic2 Integrated Properties (Deepened)

### Curated stable layer

- `external_features.critic2.qtaim.stable_atomic_integrated_properties_v1`
  - `population_e`
  - `laplacian_integral`
  - `volume`
- `external_features.critic2.qtaim.stable_atomic_integrated_property_summary_v1`
  - per-column stats: `count/sum/mean/std/min/max/abs_sum`
- `external_features.critic2.qtaim.atomic_integrated_property_candidate_assessment_v1`
  - per candidate status:
    - `implemented`
    - `partial`
    - `rejected`

### Candidate decisions

- `population_e`: implemented (high reliability)
- `laplacian_integral`: implemented when Lap-like column exists; otherwise partial
- `volume`: partial (column often missing / all-null in current critic2 outputs)
- `rho_integral`: rejected (not stable enough and often redundant with population for this route)

### Canonical companion

- `atom_features.atomic_density_partition_laplacian_proxy_v1.bader`
  - availability semantics: `success | unavailable | not_attempted`
  - source key persisted in metadata
  - only written when numeric + natm-aligned

## Module B: Conformer-Aware Geometry/Shape Enhancement

Added companion block:

- `structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1`
  - `conformer_count_ranked`
  - `conformer_energy_span_proxy`
  - `conformer_energy_std_proxy`
  - `geometry_size_variability_proxy` (bbox diagonal stats across ranked candidate set)
  - `conformer_compactness_proxy_v1` (radius-of-gyration stats across ranked candidate set)

All are explicitly candidate-set statistics, not global conformer-space truth.

## Module C: Proxy Family High-Value Aggregates

Added molecule-level companion summary:

- `global_features.proxy_family_summary_v1`
  - `atom_charge_dispersion_proxy`
  - `hetero_atom_charge_extrema_proxy`
  - `lone_pair_rich_atom_count_proxy`
  - `bond_delocalization_extrema_proxy`
  - `high_delocalization_bond_count_proxy`

Metadata includes formulas, scope, and availability reason.

## Validation Command

Run on a medium subset/full validation output directory:

```bash
python scripts/high_value_rare_feature_validation.py \
  --unified-dir /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/A_main \
  --output-json /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/high_value_augmentation_report.json \
  --output-md /home/sulixian/FastORCA/test_output_stage_validation_bader_finalmile/high_value_augmentation_report.md
```

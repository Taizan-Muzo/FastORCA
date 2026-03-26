# qcMol Feature Alignment Deepening Sprint

This sprint deepens qcMol-aligned high-value features on the existing open-source route.

## Scope Constraints

- No NBO / Multiwfn dependency.
- No exact ADCH / LI / DI claim.
- No major unified schema restructuring.
- New fields remain honestly named as `*_proxy`, `*_heuristic_proxy`, `*_v1`.

## Module A: Critic2 / Basin Family Deepening

### Added canonical basin companion block

- `global_features.basin_proxy_summary_v1`
  - `bader_population_dispersion_proxy`
  - `hetero_bader_charge_extrema_proxy`
  - `bader_laplacian_extrema_proxy`
  - `bader_laplacian_dispersion_proxy`
  - `atomwise_basin_companion_summary_proxy_v1`
  - `metadata.candidate_assessment_v1` with `implemented|partial|rejected`

### Added external stable summary companion

- `external_features.critic2.qtaim.basin_companion_summary_v1`
  - population/laplacian/volume stats
  - status + implemented-candidate count

## Module B: Atom/Bond Proxy Alignment Deepening

### New atom-level proxies

- `atom_features.atomic_charge_laplacian_coupling_proxy_v1`
- `atom_features.atomic_local_reactivity_proxy_v1`
- `atom_features.lone_pair_environment_proxy_v1`

### New bond-level proxies

- `bond_features.bond_covalency_polarity_proxy_v1`
- `bond_features.bond_delocalization_localization_balance_proxy_v1`
- `bond_features.bond_elf_deloc_coupling_proxy_v1`
- `bond_features.bond_strength_pattern_proxy_v1`

All above fields have metadata availability semantics and explicit formulas.

## Module C: Molecule-Level Scarce Summary Deepening

Extended `global_features.proxy_family_summary_v1` with:

- `polarity_heterogeneity_proxy_v1`
- `basin_charge_asymmetry_proxy_v1`
- `localized_vs_delocalized_balance_proxy_v1`
- `conformer_sensitivity_proxy_v1`
- `electronic_compactness_proxy_v1`
- `lone_pair_driven_polarity_proxy_v1`

## Validation

Use:

```bash
python scripts/qcmol_alignment_deepening_validation.py \
  --unified-dir <unified_output_dir> \
  --output-json <report.json> \
  --output-md <report.md>
```

Report includes:

- available rates for new features
- implemented/partial/rejected judgements
- candidate status histograms
- top failure reasons
- category-aware sample payload (small / aromatic / flexible / contains O/N)

# qcMol Consumer Guide

This guide is for downstream users who want stable consumption without learning the full unified schema.

## Three Consumer Tiers

## 1) Minimal Set (default for retrieval/ranking/model baselines)

Use this when you need robustness first.

- identity/basic:
  - `molecule_info.molecule_id`
  - `molecule_info.smiles`
  - `molecule_info.formula`
  - `molecule_info.charge`
- global:
  - `global_features.dft.total_energy_hartree`
  - `global_features.dft.homo_lumo_gap_hartree`
  - `global_features.dft.dipole_moment_debye`
  - `global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree`
  - `global_features.rdkit.molecular_weight`
  - `global_features.geometry_size.bounding_box_diagonal_angstrom`
- atom/bond compact vectors:
  - `atom_features.atomic_charge_iao_proxy`
  - `atom_features.atomic_density_partition_charge_proxy.hirshfeld`
  - `atom_features.atomic_density_partition_charge_proxy.cm5`
  - `bond_features.bond_indices`
  - `bond_features.bond_orders_mayer`
  - `bond_features.bond_delocalization_index_proxy_v1`

## 2) Enhanced Set (higher information density)

Add these when you want richer electronic structure signals.

- Bader/basin:
  - `atom_features.atomic_density_partition_charge_proxy.bader`
  - `atom_features.atomic_density_partition_laplacian_proxy_v1.bader`
  - `global_features.basin_proxy_summary_v1`
- orbital/proxy:
  - `atom_features.atomic_lone_pair_heuristic_proxy`
  - `atom_features.atomic_orbital_descriptor_proxy_v1`
  - `bond_features.bond_orbital_localization_proxy`
  - `bond_features.bond_order_weighted_localization_proxy`
- molecule summaries:
  - `global_features.proxy_family_summary_v1`
  - `structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1`
- realspace:
  - `realspace_features.density_isosurface_area`
  - `realspace_features.density_isosurface_volume`
  - `realspace_features.density_sphericity_like`
  - `realspace_features.orbital_extent_homo`
  - `realspace_features.orbital_extent_lumo`

## 3) Caution Set (not default dependencies)

Use only with explicit null/availability guards.

- `atom_features.atomic_density_partition_volume_proxy.bader`
- `external_features.critic2.qtaim.bader_volumes`
- `external_bridge_roadmap.*` placeholders

## Lightweight Export Helper

Use this script to export three consumer views directly from unified outputs:

```bash
python scripts/qcmol_export_consumer_views.py \
  --unified-dir /home/sulixian/FastORCA/test_output_qcmol_default/A_main \
  --output-dir /home/sulixian/FastORCA/test_output_qcmol_default/consumer_exports
```

It writes:
- `qcmol_consumer_minimal.{jsonl,csv}`
- `qcmol_consumer_enhanced.{jsonl,csv}`
- `qcmol_consumer_caution.{jsonl,csv}`

## Consumption Rules

- Prefer minimal set for production defaults.
- Gate enhanced fields by availability metadata when needed.
- Treat caution fields as opportunistic only.
- Do not default-depend on `external_bridge_roadmap` placeholders.

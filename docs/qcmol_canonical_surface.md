# qcMol Canonical Surface (Productized Freeze)

This document freezes the **recommended consumer read surface** for the `qcmol_substitute_default` mainline.

Status legend:
- `implemented_exact`
- `implemented_proxy`
- `partial`
- `roadmap_only` (do not default-depend)

Stability legend:
- `stable`: safe default dependency
- `caution`: usable with guardrails/fallback
- `optional`: opportunistic or roadmap-only

## Consumer Whitelist (Default-Depend)

These are the default recommended dependencies.

### basic_information
- `molecule_info.molecule_id` (`implemented_exact`, stable)
- `molecule_info.smiles` (`implemented_exact`, stable)
- `molecule_info.inchi` (`implemented_exact`, stable)
- `molecule_info.inchikey` (`implemented_exact`, stable)
- `molecule_info.formula` (`implemented_exact`, stable)

### global_features
- `global_features.dft.total_energy_hartree` (`implemented_exact`, stable)
- `global_features.dft.homo_energy_hartree` (`implemented_exact`, stable)
- `global_features.dft.lumo_energy_hartree` (`implemented_exact`, stable)
- `global_features.dft.homo_lumo_gap_hartree` (`implemented_exact`, stable)
- `global_features.dft.dipole_moment_debye` (`implemented_exact`, stable)
- `global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree` (`implemented_proxy`, stable)
- `global_features.rdkit.molecular_weight` (`implemented_exact`, stable)
- `global_features.geometry_size.bounding_box_diagonal_angstrom` (`implemented_proxy`, stable)
- `global_features.proxy_family_summary_v1` (`implemented_proxy`, stable)
- `global_features.basin_proxy_summary_v1` (`implemented_proxy`, stable)

### atom_features
- `atom_features.atomic_charge_iao_proxy` (`implemented_proxy`, stable)
- `atom_features.atomic_density_partition_charge_proxy.hirshfeld` (`implemented_proxy`, stable)
- `atom_features.atomic_density_partition_charge_proxy.cm5` (`implemented_proxy`, stable)
- `atom_features.atomic_density_partition_charge_proxy.bader` (`implemented_proxy`, stable)
- `atom_features.atomic_density_partition_laplacian_proxy_v1.bader` (`implemented_proxy`, stable)
- `atom_features.atomic_lone_pair_heuristic_proxy` (`implemented_proxy`, stable)
- `atom_features.atomic_orbital_descriptor_proxy_v1` (`implemented_proxy`, stable)

### bond_features
- `bond_features.bond_indices` (`implemented_exact`, stable)
- `bond_features.bond_orders_mayer` (`implemented_proxy`, stable)
- `bond_features.bond_orders_wiberg` (`implemented_exact`, stable)
- `bond_features.elf_bond_midpoint` (`implemented_exact`, stable)
- `bond_features.bond_delocalization_index_proxy_v1` (`implemented_proxy`, stable)
- `bond_features.bond_orbital_localization_proxy` (`implemented_proxy`, stable)
- `bond_features.bond_order_weighted_localization_proxy` (`implemented_proxy`, stable)
- `bond_features.bond_stereo_info` (`implemented_proxy`, stable)

### structural_features
- `structural_features.optimized_3d_geometry` (`implemented_proxy`, stable)
- `structural_features.most_stable_conformation` (`implemented_proxy`, stable)
- `structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1` (`implemented_proxy`, stable)

### realspace_features
- `realspace_features.density_isosurface_area` (`implemented_exact`, stable)
- `realspace_features.density_isosurface_volume` (`implemented_exact`, stable)
- `realspace_features.density_sphericity_like` (`implemented_proxy`, stable)
- `realspace_features.esp_extrema_summary` (`implemented_proxy`, stable)
- `realspace_features.orbital_extent_homo` (`implemented_exact`, stable)
- `realspace_features.orbital_extent_lumo` (`implemented_exact`, stable)

### external_features / bridge
- `external_bridge.critic2` (`implemented_exact`, stable)
- `external_features.critic2.qtaim` (`implemented_proxy`, stable)

## Caution / Optional Fields

These are not default dependencies.

- `atom_features.atomic_density_partition_volume_proxy.bader` (`partial`, caution)
- `external_features.critic2.qtaim.bader_volumes` (`partial`, optional)
- `external_bridge_roadmap.*` placeholders (`roadmap_only`, optional)

For Bader volume, current dominant reason is upstream critic2 volume column absence/instability.

## Field Traffic Light (Red/Yellow/Green)

| path | status | stability | recommended default dependency | color |
|---|---|---|---|---|
| `molecule_info.molecule_id` | implemented_exact | stable | yes | green |
| `molecule_info.smiles` | implemented_exact | stable | yes | green |
| `global_features.dft.homo_lumo_gap_hartree` | implemented_exact | stable | yes | green |
| `global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree` | implemented_proxy | stable | yes | yellow |
| `global_features.geometry_size.bounding_box_diagonal_angstrom` | implemented_proxy | stable | yes | yellow |
| `atom_features.atomic_density_partition_charge_proxy.bader` | implemented_proxy | stable | yes | yellow |
| `bond_features.bond_orders_mayer` | implemented_proxy | stable | yes | yellow |
| `structural_features.optimized_3d_geometry` | implemented_proxy | stable | yes | yellow |
| `atom_features.atomic_density_partition_volume_proxy.bader` | partial | caution | no | red |
| `external_bridge_roadmap.atom_level.nao_descriptors` | rejected_as_exact | optional | no | red |

## Explicit Non-Default Dependencies

- exact-only archived family:
  - `external_bridge_roadmap.atom_level.nao_descriptors`
  - `external_bridge_roadmap.atom_level.li_values`
  - `external_bridge_roadmap.atom_level.adch_charges`
  - `external_bridge_roadmap.atom_level.npa_exact`
  - `external_bridge_roadmap.bond_level.nbo_bd`
  - `external_bridge_roadmap.bond_level.lbo`
- opportunistic/partial:
  - `atom_features.atomic_density_partition_volume_proxy.bader`
  - raw critic2-only volume companions

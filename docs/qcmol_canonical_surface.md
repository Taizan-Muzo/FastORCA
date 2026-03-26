# qcMol Substitute Canonical Surface (Recommended)

This is the recommended read-surface for default consumers.  
Schema stays unchanged; this list only defines the preferred consumption subset.

## Basic

- `molecule_info.molecule_id` (`implemented_exact`)
- `molecule_info.smiles` (`implemented_exact`)
- `molecule_info.inchi` (`implemented_exact`)
- `molecule_info.inchikey` (`implemented_exact`)
- `molecule_info.formula` (`implemented_exact`)

## Global

- `global_features.dft.total_energy_hartree` (`implemented_exact`)
- `global_features.dft.homo_energy_hartree` (`partial`)
- `global_features.dft.lumo_energy_hartree` (`implemented_exact`)
- `global_features.dft.homo_lumo_gap_hartree` (`implemented_exact`)
- `global_features.dft.dipole_moment_debye` (`implemented_exact`)
- `global_features.rdkit.molecular_weight` (`implemented_exact`)
- `global_features.geometry_size.bounding_box_diagonal_angstrom` (`implemented_proxy`)
- `global_features.basin_proxy_summary_v1` (`implemented_proxy`, critic2/basin companion summary block)
- `global_features.proxy_family_summary_v1` (`implemented_proxy`, high-value aggregate companion block)

## Atom-Level

- `atom_features.charge_mulliken` (`implemented_exact`)
- `atom_features.charge_hirshfeld` (`implemented_proxy`)
- `atom_features.charge_cm5` (`implemented_proxy`)
- `atom_features.charge_iao` (`implemented_proxy`)
- `atom_features.atomic_charge_iao_proxy` (`implemented_proxy`)
- `atom_features.atomic_density_partition_charge_proxy.hirshfeld` (`implemented_proxy`)
- `atom_features.atomic_density_partition_charge_proxy.cm5` (`implemented_proxy`)
- `atom_features.atomic_density_partition_charge_proxy.bader` (`implemented_proxy`, validation-gated)
- `atom_features.atomic_density_partition_laplacian_proxy_v1.bader` (`implemented_proxy`, critic2 integrated Lap companion)
- `atom_features.atomic_charge_laplacian_coupling_proxy_v1` (`implemented_proxy`)
- `atom_features.atomic_local_reactivity_proxy_v1` (`implemented_proxy`)
- `atom_features.lone_pair_environment_proxy_v1` (`implemented_proxy`)
- `atom_features.atomic_lone_pair_heuristic_proxy` (`implemented_proxy`)
- `atom_features.atomic_orbital_descriptor_proxy_v1` (`implemented_proxy`)

## Bond-Level

- `bond_features.bond_indices` (`implemented_exact`)
- `bond_features.bond_orders_mayer` (`partial`)
- `bond_features.bond_orders_wiberg` (`implemented_exact`)
- `bond_features.elf_bond_midpoint` (`implemented_exact`)
- `bond_features.bond_delocalization_index_proxy_v1` (`implemented_proxy`)
- `bond_features.bond_orbital_localization_proxy` (`implemented_proxy`)
- `bond_features.bond_order_weighted_localization_proxy` (`implemented_proxy`)
- `bond_features.bond_covalency_polarity_proxy_v1` (`implemented_proxy`)
- `bond_features.bond_delocalization_localization_balance_proxy_v1` (`implemented_proxy`)
- `bond_features.bond_elf_deloc_coupling_proxy_v1` (`implemented_proxy`)
- `bond_features.bond_strength_pattern_proxy_v1` (`implemented_proxy`)
- `bond_features.bond_stereo_info` (`implemented_proxy`)

## Structural

- `structural_features.optimized_3d_geometry` (`partial`)
- `structural_features.most_stable_conformation` (`implemented_proxy`)
- `structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1` (`implemented_proxy`, candidate-set statistics)

## Realspace

- `realspace_features.density_isosurface_area` (`implemented_exact`)
- `realspace_features.density_isosurface_volume` (`implemented_exact`)
- `realspace_features.density_sphericity_like` (`implemented_proxy`)
- `realspace_features.esp_extrema_summary` (`implemented_proxy`)
- `realspace_features.orbital_extent_homo` (`implemented_exact`)
- `realspace_features.orbital_extent_lumo` (`implemented_exact`)

## External / Bader

- `external_bridge.critic2` (`implemented_exact` execution contract)
- `external_features.critic2.qtaim` (`implemented_proxy`, raw parsed payload)
- `external_features.critic2.qtaim.stable_atomic_integrated_properties_v1` (`implemented_proxy`, curated critic2 integrated layer)
- `external_features.critic2.qtaim.atomic_integrated_property_candidate_assessment_v1` (`implemented_proxy`, implemented/partial/rejected decisions)
- `external_features.critic2.qtaim.basin_companion_summary_v1` (`implemented_proxy`, compact critic2 integrated summary)
- `atom_features.atomic_density_partition_charge_proxy.bader` (`implemented_proxy`, validated writeback)
- `atom_features.atomic_density_partition_volume_proxy.bader` (`partial`, optional)

## Not Recommended For Default Consumption

These remain roadmap placeholders and should not be default dependencies:

- `external_bridge_roadmap.atom_level.nao_descriptors`
- `external_bridge_roadmap.atom_level.adch_charges`
- `external_bridge_roadmap.atom_level.nbo_lp`
- `external_bridge_roadmap.atom_level.npa_exact`
- `external_bridge_roadmap.atom_level.li_values`
- `external_bridge_roadmap.bond_level.di_values_or_matrix`
- `external_bridge_roadmap.bond_level.nbo_bd`
- `external_bridge_roadmap.bond_level.lbo`

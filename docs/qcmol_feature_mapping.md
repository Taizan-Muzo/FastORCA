# qcMol Feature Mapping to FastORCA Schema (M5.5)

| qcMol feature name | layer | current FastORCA status | suggested implementation source |
|---|---|---|---|
| qcMol ID | basic_information | implemented_exact (`molecule_info.molecule_id`) | existing code |
| IUPAC name | basic_information | missing | external (RDKit name resolver / PubChem / OPSIN) |
| SMILES | basic_information | implemented_exact (`molecule_info.smiles`) | existing code |
| InChI | basic_information | implemented_exact (`molecule_info.inchi`) | existing code + RDKit |
| InChIKey | basic_information | implemented_exact (`molecule_info.inchikey`) | existing code + RDKit |
| chemical formula | basic_information | implemented_exact (`molecule_info.formula`) | existing code (RDKit) |
| SMART | basic_information | implemented_proxy (`molecule_info.smarts`, `needs_exact_qcmol_name`) | existing code (`RemoveHs + MolToSmarts`) + manual review |
| nickname / synonyms | basic_information | missing | external (PubChem/ChEBI) + manual review |
| HOMO-LUMO gap | global_features | implemented_exact (`global_features.dft.homo_lumo_gap_hartree`) | existing code (PySCF) |
| dipole moment | global_features | implemented_exact (`global_features.dft.dipole_moment_debye`) | existing code (PySCF) |
| isosurface area | global_features | implemented_exact (`realspace_features.density_isosurface_area`) | existing code |
| isosurface volume | global_features | implemented_exact (`realspace_features.density_isosurface_volume`) | existing code |
| sphericity parameters | global_features | implemented_proxy (`realspace_features.density_sphericity_like`) | existing code |
| molecule size | global_features | implemented_proxy (`global_features.geometry_size.bounding_box_diagonal_angstrom`) | existing code (geometry-based frozen definition) |
| molecular weight | global_features | implemented_exact (`global_features.rdkit.molecular_weight`) | existing code (RDKit) |
| ionization affinity / ionization-related quantity | global_features | partial (`needs_exact_qcmol_name`, currently HOMO proxy) | PySCF + manual review |
| charge | global_features | implemented_exact (`molecule_info.charge`) | existing code |
| element type | atom_features | implemented_exact (`geometry.atom_symbols`) | existing code |
| XYZ | atom_features | implemented_exact (`geometry.atom_coords_angstrom`) | existing code |
| NAO descriptors | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.nao_descriptors`, `needs_exact_qcmol_name`) | external/manual review (Multiwfn/NBO) |
| LI values | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.li_values`, `needs_exact_qcmol_name`) | external/manual review |
| ADCH charges | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.adch_charges`) | external (Multiwfn) |
| NBO-LP | atom_features | implemented_proxy (`atom_features.atomic_lone_pair_heuristic_proxy`) | existing code (IBO + IAO charge heuristic) |
| NPA | atom_features | implemented_proxy (`atom_features.atomic_charge_iao_proxy`) | existing code (IAO) |
| NPA (exact) | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.npa_exact`) | external (NBO/Multiwfn) |
| stereo info | bond_features | implemented_proxy (`bond_features.bond_stereo_info`) | existing code (RDKit) |
| DI values / DI matrix | bond_features | implemented_proxy (`bond_features.bond_delocalization_index_proxy_v1`, `needs_exact_qcmol_name`) | existing code (Mayer/Wiberg proxy v1) |
| ELF values | bond_features | implemented_exact (`bond_features.elf_bond_midpoint`) | existing code |
| NBO-BD | bond_features | missing (schema placeholder: `external_bridge_roadmap.bond_level.nbo_bd`) | external (NBO/Multiwfn) |
| LBO | bond_features | missing (schema placeholder: `external_bridge_roadmap.bond_level.lbo`) | external/manual review |
| Mayer BL | bond_features | partial (`bond_orders_mayer`, naming to confirm) | existing code (PySCF) |
| optimized 3D geometry | structural_features | partial (semantic reference to `geometry.atom_coords_angstrom`, no duplicated coordinates) | existing code + policy |
| most stable conformation | structural_features | implemented_proxy (RDKit ETKDG + MMFF/UFF ranking within current candidate set) | existing code (RDKit) |

## Notes

- `needs_exact_qcmol_name` has been added in schema mapping for terms where paper abbreviation/wording must be locked before implementation.
- Allowed status levels in this table: `implemented_exact / implemented_proxy / partial / missing`.
- This file is a working contract for M5.5 naming convergence; do not rename unknown terms without paper-exact confirmation.

## B1 Proxy Fields (Open-Source)

- `atom_features.atomic_charge_iao_proxy`
- `atom_features.atomic_density_partition_charge_proxy = {hirshfeld, cm5, bader}`
- `atom_features.atomic_density_partition_volume_proxy = {bader}`
- `bond_features.bond_delocalization_index_proxy_v1`
- `bond_features.bond_orbital_localization_proxy`
- `bond_features.bond_order_weighted_localization_proxy`
- `atom_features.atomic_orbital_descriptor_proxy_v1`

## B2 Proxy Fields (Open-Source, Narrow Scope)

- `atom_features.atomic_lone_pair_heuristic_proxy`
- Heuristic contract:
  - `is_heuristic = true`
  - `equivalent_to_nbo_lp = false`
  - only occupied IBOs are considered
  - if IBO inputs are unavailable, field remains `null` (not `0.0`)

## B3 Structural Semantics (Open-Source, Minimal)

- `structural_features.optimized_3d_geometry`:
  - semantic reference only (`coordinate_ref = geometry.atom_coords_angstrom`)
  - does not duplicate coordinates
- `structural_features.most_stable_conformation`:
  - `available=true` only when conformer generation + force-field ranking actually runs
  - `selection_scope` is restricted to candidate conformers generated/optimized in current run
  - `random_seed` is persisted for ETKDG reproducibility
  - candidate count strategy (B3.1): `n_requested = clamp(4 + 3*rotatable_bonds, 4, 24)`
  - energy dedup threshold uses forcefield native energy units (`energy_dedup_threshold = 1e-4`)

## Global Canonical Paths (Frozen v1)

| canonical feature | canonical path in unified schema | units | implementation status |
|---|---|---|---|
| HOMO-LUMO gap | `global_features.dft.homo_lumo_gap_hartree` | hartree | implemented_exact |
| dipole moment | `global_features.dft.dipole_moment_debye` | debye | implemented_exact |
| isosurface area | `realspace_features.density_isosurface_area` | angstrom^2 | implemented_exact |
| isosurface volume | `realspace_features.density_isosurface_volume` | angstrom^3 | implemented_exact |
| sphericity-like | `realspace_features.density_sphericity_like` | dimensionless | implemented_proxy |
| molecule size (primary) | `global_features.geometry_size.bounding_box_diagonal_angstrom` | angstrom | implemented_proxy |
| molecule size (auxiliary) | `global_features.geometry_size.heavy_atom_count_proxy` | count | implemented_proxy |
| molecular weight | `global_features.rdkit.molecular_weight` | g/mol | implemented_exact |
| total charge | `molecule_info.charge` | e | implemented_exact |

## Proxy/Structural Metadata Freeze (v1)

Objects with frozen metadata contract:
- `atom_features.metadata.atomic_charge_iao_proxy`
- `atom_features.metadata.atomic_density_partition_charge_proxy`
- `atom_features.metadata.atomic_density_partition_volume_proxy`
- `bond_features.metadata.bond_delocalization_index_proxy_v1`
- `bond_features.metadata.bond_orbital_localization_proxy`
- `bond_features.metadata.bond_order_weighted_localization_proxy`
- `atom_features.metadata.atomic_orbital_descriptor_proxy_v1`
- `atom_features.metadata.atomic_lone_pair_heuristic_proxy`
- `structural_features.optimized_3d_geometry` (semantic metadata in-object)
- `structural_features.most_stable_conformation` (selection metadata in-object)

Frozen formulas/constraints:
- `bond_delocalization_index_proxy_v1(i,j) = max(0, 0.5 * (max(0, Wiberg_ij) + max(0, Mayer_ij)))`
- `bond_orbital_localization_proxy` candidate IBO rules:
  - occupancy >= 1.5
  - c_i >= 0.20
  - c_j >= 0.20
  - c_i + c_j >= 0.65
- `bond_order_weighted_localization_proxy = bond_orbital_localization_proxy * bond_delocalization_index_proxy_v1`
- `atomic_orbital_descriptor_proxy_v1.contribution_entropy`:
  - For atom A, `p_k = c_{kA}/sum_k c_{kA}`
  - `H_A = -sum_k p_k ln p_k / ln(N_A)` where `N_A` is number of orbitals with `c_{kA}>0`

## External Bridge Contract

- Layer split:
  - `external_bridge`: execution contract only (status/failure/metadata/artifact refs/warnings).
  - `external_features`: parsed scientific features only.
  - `external_bridge_roadmap`: standardized placeholders for unimplemented or terminology-pending items.
- `external_bridge.<tool>.execution_status` enum is fixed to:
  - `not_attempted | success | failed | timeout | skipped | disabled`
- `external_bridge_roadmap` placeholder keys are fixed to:
  - `status | payload | needs_exact_qcmol_name | notes`
- Roadmap `status` enum is fixed to:
  - `missing | placeholder | implemented_proxy | implemented_exact`
- Semantics:
  - `needs_exact_qcmol_name` only indicates whether feature naming/term freeze is pending.
  - `notes` is only for implementation/definition remarks.
- Backward compatibility:
  - `external_bridge.<tool>.input_file/output_file/execution_time_seconds/critic2_version` are **deprecated**.
  - New code should read/write `metadata` and `artifact_refs` first.

### Bader writeback semantics (v1)

- Atom-level canonical paths:
  - `atom_features.atomic_density_partition_charge_proxy.bader`
  - `atom_features.atomic_density_partition_volume_proxy.bader`
- Availability status is stored in:
  - `atom_features.metadata.atomic_density_partition_charge_proxy.bader_status`
  - `atom_features.metadata.atomic_density_partition_charge_proxy.bader_volume_status`
- Status enum (field availability):
  - `not_attempted | unavailable | success`
- Semantics:
  - `not_attempted`: critic2 was not run (`external_bridge.critic2.execution_status in {not_attempted, disabled, skipped}`)
  - `unavailable`: critic2 attempted but failed/timeout, or returned missing/invalid-length arrays
  - `success`: parsed list exists, is aligned to `natm`, and passes consistency checks
- Validation and writeback rule:
  - `external_features.critic2.qtaim` may keep raw parsed arrays for diagnosis.
  - canonical atom fields only accept validated values:
    - `atom_features.atomic_density_partition_charge_proxy.bader = q_i = Z_i - N_i(Bader)`
    - charge writeback gate: `|sum(N_i)-N_expected| <= max(0.50 e, 0.02 * N_expected)`
    - if gate fails, canonical `bader` is set to `null` with `bader_status=unavailable`
- Volume semantics:
  - if critic2 output does not provide a usable volume column, keep
    `atom_features.atomic_density_partition_volume_proxy.bader = null`
  - distinguish reasons via `bader_volume_status_reason`, e.g.:
    - `bader_volume_column_not_reported_in_critic2_output`
    - `bader_volume_column_present_but_all_null`

### Orbital/Proxy availability semantics (v1)

- Unified availability enum for orbital-dependent outputs:
  - `success | skipped | unavailable | not_attempted`
- Upstream status:
  - `orbital_features.metadata.extraction_status` now follows the same enum.
  - `skip_reason` is used only when status is `skipped`.
  - `failure_reason` is used only when status is `unavailable`.
- Downstream proxy metadata nodes now carry explicit availability:
  - `atom_features.metadata.atomic_lone_pair_heuristic_proxy`
  - `atom_features.metadata.atomic_orbital_descriptor_proxy_v1`
  - `bond_features.metadata.bond_orbital_localization_proxy`
  - `bond_features.metadata.bond_order_weighted_localization_proxy`
- Required metadata keys (for each node above):
  - `availability_status`
  - `status_reason`
  - `skip_reason`
  - `failure_reason`
  - `upstream_orbital_extraction_status`
- Null semantics:
  - `not_attempted/skipped/unavailable` may all keep canonical value field as `null`, but are distinguished by metadata status/reason.
  - `success` allows true low/zero values without being confused with uncomputed/null.

### Critic2 exploitable outputs (current)

- Besides canonical Bader writeback fields, we now keep parsed atomic integrated columns in:
  - `external_features.critic2.qtaim.atomic_integrated_properties`
- Typical stable keys (when present in critic2 table):
  - `Pop` (electron population)
  - `Lap` (integrated Laplacian-like property from critic2 table)
  - `Volume` (if critic2 actually reports a volume column)
- Parser metadata exposed at:
  - `external_features.critic2.qtaim.metadata.atomic_property_*`
  - includes header tokens, parser note, parsed row count, and parsed column order.
- Reliability policy:
  - raw parsed integrated properties can be preserved in `external_features`.
  - canonical main fields (`atom_features.atomic_density_partition_*_proxy.bader`) still require dedicated validation gates.

### Bader coverage uplift notes (v1)

- Canonical Bader writeback remains validation-gated:
  - `atom_features.atomic_density_partition_charge_proxy.bader` is written only when population/length checks pass.
  - `atom_features.atomic_density_partition_volume_proxy.bader` may stay unavailable even when charge succeeds.
- Coverage uplift mechanism:
  - If critic2 succeeds but charge fails due population-sum mismatch, the pipeline can run one refined-density retry for critic2.
  - Refined retry metadata is recorded under `external_bridge.critic2.metadata.bader_refined_density_retry_*`.
- Unavailable reason categorization is exposed in atom metadata:
  - `bader_status_category`
  - `bader_volume_status_category`
- Batch-level Bader coverage report script:
  - `scripts/bader_coverage_uplift_report.py`
  - Supports baseline comparison via `--baseline-report <validation_round_report.json>`.

## Consolidation Default Profile (v1)

- Frozen default-delivery profile:
  - `configs/qcmol_substitute_default.json`
  - profile id: `qcmol_substitute_default`
- Canonical one-command entry:
  - `scripts/run_qcmol_substitute_default.py`
- Consolidated readiness report generator:
  - `scripts/qcmol_substitute_readiness_report.py`
- Recommended consumer read-surface:
  - see `docs/qcmol_canonical_surface.md`

## bond_indices Semantics

- `bond_indices` is a bond list aligned to all bond-level arrays in the same order.
- Current pipeline includes explicit-hydrogen bonds when explicit H atoms are present:
  - RDKit path: SMILES is parsed and then `Chem.AddHs(...)` is applied before bond extraction.
  - Fallback path: if RDKit topology is unavailable, bonds are inferred from Mayer threshold on the full atom set (including H atoms in wavefunction geometry).
- `bond_stereo_info` alignment rule:
  - `len(bond_stereo_info) == len(bond_indices)` must hold.
  - Each element corresponds to the same bond at the same index in `bond_indices`.
  - Allowed enum values: `none | any | cis | trans | e | z | unknown`.

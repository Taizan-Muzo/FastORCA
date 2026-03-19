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
| sphericity parameters | global_features | implemented_exact (`realspace_features.density_sphericity_like`) | existing code |
| molecule size | global_features | partial (currently heavy atom count proxy) | existing code + definition freeze |
| molecular weight | global_features | implemented_exact (`global_features.rdkit.molecular_weight`) | existing code (RDKit) |
| ionization affinity / ionization-related quantity | global_features | partial (`needs_exact_qcmol_name`, currently HOMO proxy) | PySCF + manual review |
| charge | global_features | implemented_exact (`molecule_info.charge`) | existing code |
| element type | atom_features | implemented_exact (`geometry.atom_symbols`) | existing code |
| XYZ | atom_features | implemented_exact (`geometry.atom_coords_angstrom`) | existing code |
| NAO descriptors | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.nao_descriptors`, `needs_exact_qcmol_name`) | external/manual review (Multiwfn/NBO) |
| LI values | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.li_values`, `needs_exact_qcmol_name`) | external/manual review |
| ADCH charges | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.adch_charges`) | external (Multiwfn) |
| NBO-LP | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.nbo_lp`) | external (NBO/Multiwfn) |
| NPA | atom_features | implemented_proxy (IAO charge proxy) | existing code + external (NBO) |
| NPA (exact) | atom_features | missing (schema placeholder: `external_bridge_roadmap.atom_level.npa_exact`) | external (NBO/Multiwfn) |
| stereo info | bond_features | implemented_proxy (`bond_features.bond_stereo_info`) | existing code (RDKit) |
| DI values / DI matrix | bond_features | missing (schema placeholder: `external_bridge_roadmap.bond_level.di_values_or_matrix`, `needs_exact_qcmol_name`) | external/manual review |
| ELF values | bond_features | implemented_exact (`bond_features.elf_bond_midpoint`) | existing code |
| NBO-BD | bond_features | missing (schema placeholder: `external_bridge_roadmap.bond_level.nbo_bd`) | external (NBO/Multiwfn) |
| LBO | bond_features | missing (schema placeholder: `external_bridge_roadmap.bond_level.lbo`) | external/manual review |
| Mayer BL | bond_features | partial (`bond_orders_mayer`, naming to confirm) | existing code (PySCF) |
| optimized 3D geometry | structural_features | partial (semantic slot available, no duplicated coordinates) | existing code + policy |
| most stable conformation | structural_features | missing | external conformer workflow (RDKit/xTB) |

## Notes

- `needs_exact_qcmol_name` has been added in schema mapping for terms where paper abbreviation/wording must be locked before implementation.
- Allowed status levels in this table: `implemented_exact / implemented_proxy / partial / missing`.
- This file is a working contract for M5.5 naming convergence; do not rename unknown terms without paper-exact confirmation.

## bond_indices Semantics

- `bond_indices` is a bond list aligned to all bond-level arrays in the same order.
- Current pipeline includes explicit-hydrogen bonds when explicit H atoms are present:
  - RDKit path: SMILES is parsed and then `Chem.AddHs(...)` is applied before bond extraction.
  - Fallback path: if RDKit topology is unavailable, bonds are inferred from Mayer threshold on the full atom set (including H atoms in wavefunction geometry).
- `bond_stereo_info` alignment rule:
  - `len(bond_stereo_info) == len(bond_indices)` must hold.
  - Each element corresponds to the same bond at the same index in `bond_indices`.
  - Allowed enum values: `none | any | cis | trans | e | z | unknown`.

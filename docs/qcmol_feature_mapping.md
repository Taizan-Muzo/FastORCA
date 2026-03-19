# qcMol Feature Mapping to FastORCA Schema (M5.5)

| qcMol feature name | layer | current FastORCA status | suggested implementation source |
|---|---|---|---|
| qcMol ID | basic_information | implemented (`molecule_info.molecule_id`) | existing code |
| IUPAC name | basic_information | missing | external (RDKit name resolver / PubChem / OPSIN) |
| SMILES | basic_information | implemented (`molecule_info.smiles`) | existing code |
| InChI | basic_information | implemented (schema present, extraction not fully wired in all paths) | existing code + RDKit |
| InChIKey | basic_information | implemented (schema present, extraction not fully wired in all paths) | existing code + RDKit |
| chemical formula | basic_information | implemented (`molecule_info.formula`) | existing code (RDKit) |
| SMART | basic_information | missing (`needs_exact_qcmol_name`) | manual review + RDKit SMARTS |
| nickname / synonyms | basic_information | missing | external (PubChem/ChEBI) + manual review |
| HOMO-LUMO gap | global_features | implemented (`global_features.dft.homo_lumo_gap_hartree`) | existing code (PySCF) |
| dipole moment | global_features | implemented (`global_features.dft.dipole_moment_debye`) | existing code (PySCF) |
| isosurface area | global_features | implemented (`realspace_features.density_isosurface_area`) | existing code |
| isosurface volume | global_features | implemented (`realspace_features.density_isosurface_volume`) | existing code |
| sphericity parameters | global_features | implemented (`realspace_features.density_sphericity_like`) | existing code |
| molecule size | global_features | partial (currently heavy atom count as proxy) | existing code + definition freeze |
| molecular weight | global_features | implemented (`global_features.rdkit.molecular_weight`) | existing code (RDKit) |
| ionization affinity / ionization-related quantity | global_features | partial (`needs_exact_qcmol_name`, currently HOMO as proxy) | PySCF + manual review |
| charge | global_features | implemented (`molecule_info.charge`) | existing code |
| element type | atom_features | implemented (`geometry.atom_symbols`) | existing code |
| XYZ | atom_features | implemented (`geometry.atom_coords_angstrom`) | existing code |
| NAO descriptors | atom_features | missing (`needs_exact_qcmol_name`) | external/manual review (Multiwfn/NBO) |
| LI values | atom_features | missing (`needs_exact_qcmol_name`) | external/manual review |
| ADCH charges | atom_features | missing | external (Multiwfn) |
| NBO-LP | atom_features | missing | external (NBO/Multiwfn) |
| NPA | atom_features | partial (IAO charge proxy) | existing code + external (NBO) |
| stereo info | bond_features | partial (RDKit stereo exists but not dedicated field yet) | existing code (RDKit) |
| DI values / DI matrix | bond_features | missing (`needs_exact_qcmol_name`) | external/manual review |
| ELF values | bond_features | implemented (`bond_features.elf_bond_midpoint`) | existing code |
| NBO-BD | bond_features | missing | external (NBO/Multiwfn) |
| LBO | bond_features | missing | external/manual review |
| Mayer BL | bond_features | partial (`bond_orders_mayer`, naming to confirm) | existing code (PySCF) |
| optimized 3D geometry | structural_features | partial (coordinates stored; explicit optimization provenance needs tightening) | existing code + policy |
| most stable conformation | structural_features | missing | external conformer workflow (RDKit/xTB) |

## Notes

- `needs_exact_qcmol_name` has been added in schema mapping for terms where paper abbreviation/wording must be locked before implementation.
- This file is a working contract for M5.5 naming convergence; do not rename unknown terms without paper-exact confirmation.

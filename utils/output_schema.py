"""
统一输出 Schema 构造器 (Milestone 1)

统一输出骨架，确保字段命名稳定、缺失字段显式标记为 null。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import time


class UnifiedOutputBuilder:
    """
    FastORCA 统一输出骨架构造器
    
    Schema 版本: 1.0.0
    
    设计原则:
    1. 顶层 section 始终保留，内部字段可为 null
    2. 字段命名使用 snake_case，稳定不变
    3. 所有缺失字段显式标记为 null
    4. bond_features 统一采用 list 模式，与 bond_indices 严格对齐
    
    Spin 定义:
    - spin = N_alpha - N_beta (未成对电子数)
    - multiplicity = spin + 1 = 2S + 1
    """
    
    SCHEMA_VERSION = "1.0.0"
    SOFTWARE_NAME = "FastORCA"
    # Contract constants (do not store as per-placeholder runtime fields)
    ROADMAP_STATUS_ENUM = ("missing", "placeholder", "implemented_proxy", "implemented_exact")
    EXTERNAL_BRIDGE_EXECUTION_STATUS_ENUM = ("not_attempted", "success", "failed", "timeout", "skipped", "disabled")
    BADER_FIELD_STATUS_ENUM = ("not_attempted", "unavailable", "success")
    
    def __init__(self, molecule_id: str, smiles: str):
        """
        初始化统一输出骨架
        
        Args:
            molecule_id: 分子唯一标识
            smiles: 原始 SMILES 字符串
        """
        self.data = self._init_skeleton(molecule_id, smiles)
        self.timers = {}
        self._start_time = time.time()
        
    def _init_skeleton(self, molecule_id: str, smiles: str) -> Dict[str, Any]:
        """初始化空骨架，所有字段显式设为 null"""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "schema_name": "FastORCA Unified Output Schema",
            
            "molecule_info": {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "smarts": None,
                "inchi": None,
                "inchikey": None,
                "formula": None,
                "natm": None,
                "charge": None,
                "spin": None,  # N_alpha - N_beta
                "multiplicity": None,  # spin + 1
                "representation_metadata": {
                    "smarts": {
                        "available": False,
                        "source": None,
                        "is_proxy": True,
                        "proxy_note": None,
                    }
                }
            },

            # qcMol paper name mapping (M5.5)
            # Unknown abbreviation / exact label is explicitly marked with needs_exact_qcmol_name.
            "qcmol_alignment": {
                "basic_information": {
                    "qcMol_ID": {"mapped_path": "molecule_info.molecule_id", "status": "implemented_exact"},
                    "IUPAC_name": {"mapped_path": None, "status": "missing"},
                    "SMILES": {"mapped_path": "molecule_info.smiles", "status": "implemented_exact"},
                    "InChI": {"mapped_path": "molecule_info.inchi", "status": "implemented_exact"},
                    "InChIKey": {"mapped_path": "molecule_info.inchikey", "status": "implemented_exact"},
                    "chemical_formula": {"mapped_path": "molecule_info.formula", "status": "implemented_exact"},
                    "SMART": {
                        "mapped_path": "molecule_info.smarts",
                        "status": "implemented_proxy",
                        "needs_exact_qcmol_name": False,
                        "notes": "Frozen substitute semantics: RDKit canonical SMARTS proxy (not qcMol exact SMART naming).",
                    },
                    "nickname_or_synonyms": {"mapped_path": None, "status": "missing"},
                },
                "global_features": {
                    "HOMO_LUMO_gap": {"mapped_path": "global_features.dft.homo_lumo_gap_hartree", "status": "implemented_exact"},
                    "dipole_moment": {"mapped_path": "global_features.dft.dipole_moment_debye", "status": "implemented_exact"},
                    "isosurface_area": {"mapped_path": "realspace_features.density_isosurface_area", "status": "implemented_exact"},
                    "isosurface_volume": {"mapped_path": "realspace_features.density_isosurface_volume", "status": "implemented_exact"},
                    "sphericity_parameters": {
                        "mapped_path": "realspace_features.density_shape_descriptor_family_v1.sphericity",
                        "status": "implemented_proxy",
                        "notes": "Frozen substitute semantics: density shape descriptor family v1 (canonical 0.95 scale) with multiscale companion family (0.50/0.90/0.95); legacy density_sphericity_like kept for backward compatibility.",
                    },
                    "molecule_size": {
                        "mapped_path": "global_features.geometry_size.bounding_box_diagonal_angstrom",
                        "status": "implemented_proxy",
                        "notes": "Frozen substitute semantics: bbox-diagonal primary with companion size proxies (radius_of_gyration/heavy_atom_count/total_atom_count/num_bonds/num_rings).",
                    },
                    "molecular_weight": {"mapped_path": "global_features.rdkit.molecular_weight", "status": "implemented_exact"},
                    "ionization_affinity_or_related": {
                        "mapped_path": "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree",
                        "status": "implemented_proxy",
                        "needs_exact_qcmol_name": False,
                        "notes": "Frozen substitute semantics: Koopmans-style ionization-related proxy derived from HOMO energy.",
                    },
                    "charge": {"mapped_path": "molecule_info.charge", "status": "implemented_exact"},
                },
                "atom_features": {
                    "element_type": {"mapped_path": "geometry.atom_symbols", "status": "implemented_exact"},
                    "XYZ": {"mapped_path": "geometry.atom_coords_angstrom", "status": "implemented_exact"},
                    "NAO_descriptors": {
                        "mapped_path": "external_bridge_roadmap.atom_level.nao_descriptors",
                        "status": "missing",
                        "needs_exact_qcmol_name": True,  # NAO descriptor definition must follow qcMol exact naming
                    },
                    "LI_values": {
                        "mapped_path": "external_bridge_roadmap.atom_level.li_values",
                        "status": "missing",
                        "needs_exact_qcmol_name": True,  # LI abbreviation needs exact qcMol expansion
                    },
                    "ADCH_charges": {"mapped_path": "external_bridge_roadmap.atom_level.adch_charges", "status": "missing"},
                    "NBO_LP": {"mapped_path": "atom_features.atomic_lone_pair_heuristic_proxy", "status": "implemented_proxy"},
                    "NPA": {"mapped_path": "atom_features.atomic_charge_iao_proxy", "status": "implemented_proxy"},
                    "NPA_exact": {"mapped_path": "external_bridge_roadmap.atom_level.npa_exact", "status": "missing"},
                },
                "bond_features": {
                    "stereo_info": {"mapped_path": "bond_features.bond_stereo_info", "status": "implemented_proxy"},
                    "DI_values_or_matrix": {
                        "mapped_path": "bond_features.bond_delocalization_index_proxy_v1",
                        "status": "implemented_proxy",
                        "needs_exact_qcmol_name": False,
                        "notes": "Frozen substitute-only DI semantics; this is not an exact DI matrix.",
                    },
                    "ELF_values": {"mapped_path": "bond_features.elf_bond_midpoint", "status": "implemented_exact"},
                    "NBO_BD": {"mapped_path": "external_bridge_roadmap.bond_level.nbo_bd", "status": "missing"},
                    "LBO": {"mapped_path": "external_bridge_roadmap.bond_level.lbo", "status": "missing"},
                    "Mayer_BL": {
                        "mapped_path": "bond_features.bond_orders_mayer",
                        "status": "implemented_proxy",
                        "notes": "Frozen substitute semantics: PySCF Mayer bond-order vector aligned to bond_indices.",
                    },
                },
                "structural_features": {
                    "optimized_3D_geometry": {
                        "mapped_path": "structural_features.optimized_3d_geometry",
                        "status": "implemented_proxy",
                        "notes": "Frozen semantic-reference representation with explicit link to geometry.atom_coords_angstrom.",
                    },
                    "most_stable_conformation": {"mapped_path": "structural_features.most_stable_conformation", "status": "implemented_proxy"},
                },
            },
            
            "calculation_status": {
                "overall_status": None,
                "invalid_input": False,
                "rdkit_parse_success": False,
                "geometry_optimization_success": False,
                "scf_convergence_success": False,
                "wavefunction_load_success": False,
                "core_features_success": False,
                "extended_features_success": False,
                "error_messages": []
            },
            
            "geometry": {
                "atom_symbols": None,
                "atom_coords_angstrom": None,
                "point_group": None
            },
            
            "global_features": {
                "dft": {
                    "total_energy_hartree": None,
                    "homo_energy_hartree": None,
                    "lumo_energy_hartree": None,
                    "homo_lumo_gap_hartree": None,
                    "ionization_related_proxy_v1": {
                        "available": False,
                        "definition_version": "v1",
                        "proxy_family": "koopmans_homo_related",
                        "homo_energy_hartree": None,
                        "koopmans_ip_proxy_hartree": None,
                        "koopmans_ip_proxy_ev": None,
                        "status": "unavailable",
                        "status_reason": "homo_energy_missing",
                        "limitations": [
                            "Koopmans-style related quantity only; not equivalent to adiabatic/vertical ionization affinity/exact qcMol target.",
                            "depends on single-point HOMO energy quality in current DFT setup",
                        ],
                    },
                    "dipole_moment_debye": None,
                    "dipole_vector_debye": None,
                    "scf_converged": None,
                    "dispersion_correction": None
                },
                "rdkit": {
                    "molecular_weight": None,
                    "logp": None,
                    "tpsa": None,
                    "h_bond_donors": None,
                    "h_bond_acceptors": None,
                    "rotatable_bonds": None,
                    "heavy_atom_count": None
                },
                "geometry_size": {
                    "bounding_box_diagonal_angstrom": None,
                    "radius_of_gyration_angstrom": None,
                    "heavy_atom_count_proxy": None,
                    "total_atom_count_proxy": None,
                    "num_bonds_proxy": None,
                    "num_rings_proxy": None,
                },
                "basin_proxy_summary_v1": {
                    "available": False,
                    "definition_version": "v1",
                    "is_proxy": True,
                    "bader_population_dispersion_proxy": None,
                    "bader_population_entropy_proxy_v1": None,
                    "hetero_basin_population_share_proxy_v1": None,
                    "hetero_bader_charge_extrema_proxy": None,
                    "bader_laplacian_extrema_proxy": None,
                    "bader_laplacian_dispersion_proxy": None,
                    "bader_laplacian_sign_balance_proxy_v1": None,
                    "bader_charge_laplacian_correlation_proxy_v1": None,
                    "atomwise_basin_companion_summary_proxy_v1": None,
                    "metadata": {
                        "candidate_set_scope": "single_optimized_geometry_current_run",
                        "status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "bader_population_dispersion_proxy_formula": "std(external_features.critic2.qtaim.stable_atomic_integrated_properties_v1.population_e)",
                        "bader_population_entropy_proxy_v1_formula": "normalized_shannon_entropy(population_e / sum(population_e))",
                        "hetero_basin_population_share_proxy_v1_formula": "sum(population_e on non C/H atoms) / sum(population_e)",
                        "hetero_bader_charge_extrema_proxy_formula": "extrema over non C/H atoms in atom_features.atomic_density_partition_charge_proxy.bader",
                        "bader_laplacian_extrema_proxy_formula": "min/max/mean/std over atom_features.atomic_density_partition_laplacian_proxy_v1.bader",
                        "bader_laplacian_dispersion_proxy_formula": "std(atom_features.atomic_density_partition_laplacian_proxy_v1.bader)",
                        "bader_laplacian_sign_balance_proxy_v1_formula": "(n_pos - n_neg) / n over atom_features.atomic_density_partition_laplacian_proxy_v1.bader",
                        "bader_charge_laplacian_correlation_proxy_v1_formula": "pearson_corr(bader_charge, bader_laplacian) when both vectors are valid",
                        "atomwise_basin_companion_summary_proxy_v1_formula": "compact atomwise summary over validated bader charge / population / laplacian vectors",
                        "candidate_assessment_v1": {},
                        "limitations": [
                            "requires critic2 qtaim integrated-property outputs",
                            "basin quantities are open-source proxy companions, not exact qcMol NBO-family descriptors"
                        ],
                    },
                },
                "proxy_family_summary_v1": {
                    "available": False,
                    "definition_version": "v1",
                    "is_proxy": True,
                    "atom_charge_dispersion_proxy": None,
                    "hetero_atom_charge_extrema_proxy": None,
                    "lone_pair_rich_atom_count_proxy": None,
                    "bond_delocalization_extrema_proxy": None,
                    "high_delocalization_bond_count_proxy": None,
                    "polarity_heterogeneity_proxy_v1": None,
                    "basin_charge_asymmetry_proxy_v1": None,
                    "localized_vs_delocalized_balance_proxy_v1": None,
                    "conformer_sensitivity_proxy_v1": None,
                    "electronic_compactness_proxy_v1": None,
                    "lone_pair_driven_polarity_proxy_v1": None,
                    "reactivity_concentration_proxy_v1": None,
                    "bond_pattern_heterogeneity_proxy_v1": None,
                    "lp_environment_polarization_proxy_v1": None,
                    "metadata": {
                        "candidate_set_scope": "single_optimized_geometry_current_run",
                        "status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "atom_charge_dispersion_proxy_formula": "std(atom_features.atomic_charge_iao_proxy)",
                        "hetero_atom_charge_extrema_proxy_formula": "extrema over non C/H atoms in atomic_charge_iao_proxy",
                        "lone_pair_rich_atom_count_proxy_formula": "count(score >= 0.6) over atomic_lone_pair_heuristic_proxy",
                        "bond_delocalization_extrema_proxy_formula": "min/max/mean/std over bond_delocalization_index_proxy_v1",
                        "high_delocalization_bond_count_proxy_formula": "count(di >= 1.0) over bond_delocalization_index_proxy_v1",
                        "polarity_heterogeneity_proxy_v1_formula": "std(charge_ref) * (1 + hetero_atom_fraction), charge_ref prefers validated bader then IAO",
                        "basin_charge_asymmetry_proxy_v1_formula": "mean(abs(bader_charge_i)) with signed charge-sum companions when bader charge is validated",
                        "localized_vs_delocalized_balance_proxy_v1_formula": "mean((loc_i - di_i)/(abs(loc_i)+abs(di_i)+1e-8)) over bonds",
                        "conformer_sensitivity_proxy_v1_formula": "weighted blend of normalized energy-span and geometry-span proxies from candidate_set_statistics_proxy_v1",
                        "electronic_compactness_proxy_v1_formula": "expected_electrons / realspace_features.density_isosurface_volume.value_angstrom3",
                        "lone_pair_driven_polarity_proxy_v1_formula": "mean(atomic_lone_pair_heuristic_proxy_i * abs(charge_ref_i))",
                        "reactivity_concentration_proxy_v1_formula": "sum(top3 atomic_local_reactivity_refined_proxy_v1) / sum(all atomic_local_reactivity_refined_proxy_v1)",
                        "bond_pattern_heterogeneity_proxy_v1_formula": "std(bond_features.bond_strength_pattern_proxy_v1)",
                        "lp_environment_polarization_proxy_v1_formula": "mean(lone_pair_environment_proxy_v1_i * abs(charge_ref_i))",
                        "limitations": [
                            "proxy summaries aggregate existing proxy vectors",
                            "not equivalent to exact qcMol external descriptors"
                        ],
                    },
                },
                "metadata": {
                    "homo_lumo_gap_hartree": {
                        "canonical_path": "global_features.dft.homo_lumo_gap_hartree",
                        "definition_version": "v1",
                        "units": "hartree",
                        "formula": "E_LUMO - E_HOMO",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "pyscf_mo_energies",
                        "limitations": []
                    },
                    "dipole_moment_debye": {
                        "canonical_path": "global_features.dft.dipole_moment_debye",
                        "definition_version": "v1",
                        "units": "debye",
                        "formula": "||dipole_vector_debye||_2",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "pyscf_dipole_moment",
                        "limitations": []
                    },
                    "isosurface_area_angstrom2": {
                        "canonical_path": "realspace_features.density_isosurface_area",
                        "definition_version": "v1",
                        "units": "angstrom^2",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "realspace_density_isosurface",
                        "limitations": [
                            "requires realspace_features extraction success"
                        ]
                    },
                    "isosurface_volume_angstrom3": {
                        "canonical_path": "realspace_features.density_isosurface_volume",
                        "definition_version": "v1",
                        "units": "angstrom^3",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "realspace_density_isosurface",
                        "limitations": [
                            "requires realspace_features extraction success"
                        ]
                    },
                    "sphericity_like_dimensionless": {
                        "canonical_path": "realspace_features.density_sphericity_like",
                        "definition_version": "v1",
                        "units": "dimensionless",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "realspace_density_shape_proxy",
                        "canonical_successor": "realspace_features.density_shape_descriptor_family_v1.sphericity",
                        "limitations": [
                            "proxy shape descriptor; not a paper-exact sphericity index"
                        ]
                    },
                    "molecule_size_bounding_box_diagonal_angstrom": {
                        "canonical_path": "global_features.geometry_size.bounding_box_diagonal_angstrom",
                        "definition_version": "v2",
                        "units": "angstrom",
                        "formula": "sqrt((x_max-x_min)^2 + (y_max-y_min)^2 + (z_max-z_min)^2)",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "geometry.atom_coords_angstrom",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "orientation-dependent geometric extent",
                            "not equivalent to exact qcMol molecule-size definition"
                        ]
                    },
                    "molecule_size_radius_of_gyration_angstrom": {
                        "canonical_path": "global_features.geometry_size.radius_of_gyration_angstrom",
                        "definition_version": "v2",
                        "units": "angstrom",
                        "formula": "sqrt(mean(||r_i - r_center||^2)), r_center=mean(atom_coords_angstrom)",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "geometry.atom_coords_angstrom_equal_weight",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "equal-weight atom-point compactness proxy",
                            "not mass-weighted and not equivalent to exact qcMol internal molecule-size definition",
                        ]
                    },
                    "molecule_size_heavy_atom_count_proxy": {
                        "canonical_path": "global_features.geometry_size.heavy_atom_count_proxy",
                        "definition_version": "v2",
                        "units": "count",
                        "formula": "prefer rdkit.heavy_atom_count else count(symbol != 'H') from geometry.atom_symbols",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "rdkit_heavy_atom_count_or_geometry_symbols",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "topology/count proxy, not a 3D size metric"
                        ]
                    },
                    "molecule_size_total_atom_count_proxy": {
                        "canonical_path": "global_features.geometry_size.total_atom_count_proxy",
                        "definition_version": "v2",
                        "units": "count",
                        "formula": "prefer molecule_info.natm else len(geometry.atom_symbols)",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "molecule_info_natm_or_geometry_symbols",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "count proxy, not a direct physical size metric"
                        ]
                    },
                    "molecule_size_num_bonds_proxy": {
                        "canonical_path": "global_features.geometry_size.num_bonds_proxy",
                        "definition_version": "v2",
                        "units": "count",
                        "formula": "prefer len(bond_features.bond_indices) else rdkit.GetNumBonds()",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "bond_indices_or_rdkit_bond_count",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "depends on bond topology availability/alignment in current run"
                        ]
                    },
                    "molecule_size_num_rings_proxy": {
                        "canonical_path": "global_features.geometry_size.num_rings_proxy",
                        "definition_version": "v2",
                        "units": "count",
                        "formula": "rdkit ring_info.NumRings()",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "rdkit_ring_info",
                        "availability_status": "not_attempted",
                        "status_reason": "not_computed_yet",
                        "limitations": [
                            "null when RDKit topology is unavailable"
                        ]
                    },
                    "molecular_weight_g_mol": {
                        "canonical_path": "global_features.rdkit.molecular_weight",
                        "definition_version": "v1",
                        "units": "g/mol",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "rdkit_mol_weight",
                        "limitations": []
                    },
                    "ionization_related_proxy_v1": {
                        "canonical_path": "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree",
                        "definition_version": "v1",
                        "units": "hartree",
                        "implementation_status": "implemented_proxy",
                        "is_proxy": True,
                        "source": "derived_from_global_features.dft.homo_energy_hartree",
                        "formula": "koopmans_ip_proxy_hartree = -E_HOMO",
                        "limitations": [
                            "related proxy only, not an exact ionization affinity / ionization potential target",
                            "sensitive to chosen functional/basis and Koopmans approximation assumptions",
                        ],
                    },
                    "total_charge_e": {
                        "canonical_path": "molecule_info.charge",
                        "definition_version": "v1",
                        "units": "e",
                        "implementation_status": "implemented_exact",
                        "is_proxy": False,
                        "source": "pyscf_molecule_charge",
                        "limitations": []
                    }
                }
            },
            
            "atom_features": {
                "atomic_number": None,
                "rdkit_degree": None,
                "rdkit_hybridization": None,
                "rdkit_aromatic": None,
                "charge_mulliken": None,
                "charge_hirshfeld": None,
                "charge_cm5": None,
                "charge_iao": None,
                "elf_value": None,
                "atomic_charge_iao_proxy": None,
                "atomic_density_partition_charge_proxy": {
                    "hirshfeld": None,
                    "cm5": None,
                    "bader": None,
                },
                "atomic_density_partition_volume_proxy": {
                    "bader": None,
                },
                "atomic_density_partition_laplacian_proxy_v1": {
                    "bader": None,
                },
                "atomic_charge_laplacian_coupling_proxy_v1": None,
                "atomic_local_reactivity_proxy_v1": None,
                "atomic_local_reactivity_refined_proxy_v1": None,
                "lone_pair_environment_proxy_v1": None,
                "lone_pair_polarization_proxy_v1": None,
                "atomic_lone_pair_heuristic_proxy": None,
                "atomic_orbital_descriptor_proxy_v1": {
                    "n_dominant_ibo": None,
                    "sum_ibo_occupancy": None,
                    "mean_localization_score": None,
                    "contribution_entropy": None,
                },
                "metadata": {
                    "atomic_charge_iao_proxy": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "units": "e",
                        "source": "pyscf_iao_population",
                        "formula": "atomic_number - IAO_population",
                        "limitations": [
                            "open-source IAO proxy; not equivalent to exact NPA"
                        ]
                    },
                    "atomic_density_partition_charge_proxy": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "units": "e",
                        "field_order": ["hirshfeld", "cm5", "bader"],
                        "sources": {
                            "hirshfeld": "pyscf_hirshfeld",
                            "cm5": "hirshfeld_plus_cm5_correction",
                            "bader": "derived_from_external_features.critic2.qtaim.bader_populations"
                        },
                        "bader_charge_formula": "q_i = Z_i - N_i(Bader)",
                        "bader_status": "not_attempted",
                        "bader_status_reason": "not_attempted_by_default",
                        "bader_volume_status": "not_attempted",
                        "bader_volume_status_reason": "not_attempted_by_default",
                        "limitations": [
                            "bader field may be null when external bridge is not executed"
                        ]
                    },
                    "atomic_density_partition_volume_proxy": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "units": "critic2_native_volume_units",
                        "field_order": ["bader"],
                        "sources": {
                            "bader": "external_features.critic2.qtaim.bader_volumes"
                        },
                        "bader_status": "not_attempted",
                        "bader_status_reason": "not_attempted_by_default",
                        "bader_numeric_count": None,
                        "bader_null_count": None,
                        "bader_non_numeric_count": None,
                        "limitations": [
                            "units follow critic2 parsed output and may depend on cube/input convention",
                            "bader volume field may be null when external bridge is not executed or parsing fails",
                        ],
                    },
                    "atomic_density_partition_laplacian_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "units": "critic2_integrated_laplacian_like_units",
                        "field_order": ["bader"],
                        "sources": {
                            "bader": "external_features.critic2.qtaim.stable_atomic_integrated_properties_v1.laplacian_integral"
                        },
                        "bader_status": "not_attempted",
                        "bader_status_reason": "not_attempted_by_default",
                        "bader_source_key": None,
                        "bader_validation_stage": None,
                        "bader_retry_attempted": False,
                        "bader_retry_success": False,
                        "limitations": [
                            "depends on critic2 integrated-property table containing Lap-like column",
                            "units/physical meaning follow critic2 integrated table conventions"
                        ],
                    },
                    "atomic_charge_laplacian_coupling_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "charge_source_priority": [
                            "atomic_density_partition_charge_proxy.bader",
                            "atomic_charge_iao_proxy",
                            "charge_hirshfeld"
                        ],
                        "charge_source": None,
                        "laplacian_source": "atomic_density_partition_laplacian_proxy_v1.bader",
                        "formula": "coupling_i = charge_ref_i * laplacian_bader_i",
                        "units": "e * critic2_integrated_laplacian_like_units",
                        "limitations": [
                            "requires Lap-like integrated quantity from critic2",
                            "charge_ref source may fallback from bader to IAO/Hirshfeld when bader is unavailable"
                        ],
                    },
                    "atomic_local_reactivity_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "minmax_over_atoms(sqrt(abs(charge_ref_i) * (abs(laplacian_i)+1e-8)) * (0.7 + 0.3*lp_i))",
                        "charge_source_priority": [
                            "atomic_density_partition_charge_proxy.bader",
                            "atomic_charge_iao_proxy",
                            "charge_hirshfeld"
                        ],
                        "charge_source": None,
                        "lp_source": "atomic_lone_pair_heuristic_proxy (missing -> 0)",
                        "lp_fallback_used": False,
                        "range": "[0, 1]",
                        "limitations": [
                            "heuristic local reactivity proxy, not a direct conceptual-DFT local descriptor",
                            "depends on critic2 Lap-like integral and chosen charge proxy"
                        ],
                    },
                    "atomic_local_reactivity_refined_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "minmax_over_atoms(0.60*atomic_local_reactivity_proxy_v1 + 0.25*abs(atomic_charge_laplacian_coupling_proxy_v1) + 0.15*lone_pair_environment_proxy_v1)",
                        "inputs_used": [
                            "atomic_local_reactivity_proxy_v1",
                            "atomic_charge_laplacian_coupling_proxy_v1",
                            "lone_pair_environment_proxy_v1"
                        ],
                        "range": "[0, 1]",
                        "limitations": [
                            "heuristic refinement over proxy components; not an exact local reactivity observable"
                        ],
                    },
                    "lone_pair_environment_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "lp_i * mean_j(bond_delocalization_index_proxy_v1(i,j)) over incident bonds",
                        "inputs_used": [
                            "atomic_lone_pair_heuristic_proxy",
                            "bond_features.bond_indices",
                            "bond_features.bond_delocalization_index_proxy_v1"
                        ],
                        "range": "[0, +inf)",
                        "limitations": [
                            "depends on orbital-derived lone pair proxy availability",
                            "zero for atoms without incident bonds in current topology"
                        ],
                    },
                    "lone_pair_polarization_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "atomic_lone_pair_heuristic_proxy_i * abs(charge_ref_i)",
                        "charge_source_priority": [
                            "atomic_density_partition_charge_proxy.bader",
                            "atomic_charge_iao_proxy",
                            "charge_hirshfeld"
                        ],
                        "charge_source": None,
                        "range": "[0, +inf)",
                        "limitations": [
                            "depends on lone pair heuristic proxy availability",
                            "charge_ref source may fallback when bader is unavailable"
                        ],
                    },
                    "atomic_lone_pair_heuristic_proxy": {
                        "is_proxy": True,
                        "is_heuristic": True,
                        "equivalent_to_nbo_lp": False,
                        "definition_version": "v1",
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "upstream_orbital_extraction_status": "not_attempted",
                        "inputs_used": [
                            "orbital_features.ibo_atom_contributions",
                            "orbital_features.ibo_occupancies",
                            "atom_features.atomic_charge_iao_proxy",
                            "geometry.atom_symbols",
                        ],
                        "normalization_rule": "score = clamp(lp_base * (0.7 + 0.3*charge_boost) * element_gate * non_negative_charge_penalty, 0, 1), where charge_boost increases as atom gets more negative; element_gate=1.0 for lone-pair-typical elements else 0.18; non_negative_charge_penalty=0.9 if q>=0 else 1.0",
                        "only_occupied_ibo_considered": True,
                        "limitations": [
                            "heuristic proxy only, not equivalent to NBO-LP",
                            "closed-shell IBO extraction dependency",
                            "transition-metal and strongly delocalized systems may be unreliable",
                        ],
                    },
                    "atomic_orbital_descriptor_proxy_v1": {
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "upstream_orbital_extraction_status": "not_attempted",
                        "field_order": [
                            "n_dominant_ibo",
                            "sum_ibo_occupancy",
                            "mean_localization_score",
                            "contribution_entropy",
                        ],
                        "definition_version": "v1",
                        "source_basis": "IAO",
                        "source_orbital_type": "occupied_IBO_only",
                        "dominant_ibo_rule": "Atom A is dominant for orbital k if c_{kA} is the largest atomic contribution and c_{kA} >= 0.50.",
                        "localization_score_definition": "mean_localization_score is the mean of orbital_locality_score over orbitals with c_{kA} >= 0.20 for atom A.",
                        "contribution_entropy_definition": "For atom A, p_k = c_{kA} / sum_k c_{kA} over orbitals with c_{kA} > 0; H_A = -sum_k p_k ln(p_k) / ln(N_A), with H_A=0 when N_A<=1.",
                        "normalization_notes": "contribution_entropy is normalized to [0,1] by ln(N_A); n_dominant_ibo and sum_ibo_occupancy are unbounded by molecular size.",
                        "limitations": [
                            "descriptor is proxy, not equivalent to NAO/NBO descriptors",
                            "depends on closed-shell IBO extraction quality",
                        ],
                    },
                },
            },
            
            "bond_features": {
                "bond_indices": None,
                "bond_types_rdkit": None,
                "bond_stereo_info": None,
                "bond_orders_mayer": None,
                "bond_orders_wiberg": None,
                "elf_bond_midpoint": None,
                "bond_delocalization_index_proxy_v1": None,
                "bond_orbital_localization_proxy": None,
                "bond_order_weighted_localization_proxy": None,
                "bond_covalency_polarity_proxy_v1": None,
                "bond_delocalization_localization_balance_proxy_v1": None,
                "bond_elf_deloc_coupling_proxy_v1": None,
                "bond_strength_pattern_proxy_v1": None,
                "bond_localization_tension_proxy_v1": None,
                "bond_polarized_delocalization_proxy_v1": None,
                "metadata": {
                    "bond_stereo_info": {
                        "available": False,
                        "source": None,
                        "is_proxy": True,
                        "proxy_note": None,
                        "enum_values": ["none", "any", "cis", "trans", "e", "z", "unknown"],
                    },
                    "bond_delocalization_index_proxy_v1": {
                        "formula": "max(0, 0.5 * (max(0, Wiberg_ij) + max(0, Mayer_ij)))",
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "substitute_scope": "qcMol_DI_values_or_matrix_substitute_only",
                        "exact_di_matrix_available": False,
                        "limitations": [
                            "proxy DI definition based on Mayer/Wiberg bond orders"
                        ],
                    },
                    "bond_orders_mayer": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "substitute_scope": "qcMol_Mayer_BL_substitute",
                        "alignment_note": "Per-bond Mayer bond-order values aligned with bond_features.bond_indices order.",
                        "limitations": [
                            "numeric convention depends on basis/set and implementation details",
                            "substitute alignment; not guaranteed identical to every external Mayer_BL convention"
                        ],
                    },
                    "bond_orbital_localization_proxy": {
                        "definition_version": "v1",
                        "formula": "max_k(c_{k,i}+c_{k,j}) over bonding-candidate IBOs",
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "upstream_orbital_extraction_status": "not_attempted",
                        "ibo_candidate_rules": {
                            "occupancy_min": 1.5,
                            "ci_min": 0.20,
                            "cj_min": 0.20,
                            "ci_plus_cj_min": 0.65,
                        },
                        "is_proxy": True,
                        "is_heuristic": True,
                        "limitations": [
                            "depends on occupied IBO extraction and heuristic candidate rules"
                        ],
                    },
                    "bond_order_weighted_localization_proxy": {
                        "definition_version": "v1",
                        "formula": "bond_orbital_localization_proxy * bond_delocalization_index_proxy_v1",
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "upstream_orbital_extraction_status": "not_attempted",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "limitations": [
                            "composite proxy from localization and DI proxies"
                        ],
                    },
                    "bond_covalency_polarity_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "abs(charge_ref_i - charge_ref_j) / (1 + max(0, 0.5*(max(0,mayer_ij)+max(0,wiberg_ij))))",
                        "charge_source_priority": [
                            "atomic_density_partition_charge_proxy.bader",
                            "atomic_charge_iao_proxy",
                            "charge_hirshfeld"
                        ],
                        "charge_source": None,
                        "limitations": [
                            "heuristic polarity-vs-covalency proxy",
                            "depends on chosen atomic charge proxy"
                        ],
                    },
                    "bond_delocalization_localization_balance_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "(loc_ij - di_ij) / (abs(loc_ij) + abs(di_ij) + 1e-8)",
                        "inputs_used": [
                            "bond_orbital_localization_proxy",
                            "bond_delocalization_index_proxy_v1"
                        ],
                        "localization_source": None,
                        "range": "[-1, 1]",
                        "limitations": [
                            "depends on orbital-derived localization proxy when available"
                        ],
                    },
                    "bond_elf_deloc_coupling_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "clamp(elf_bond_midpoint,0,1) * max(0,bond_delocalization_index_proxy_v1)",
                        "inputs_used": [
                            "elf_bond_midpoint",
                            "bond_delocalization_index_proxy_v1"
                        ],
                        "limitations": [
                            "captures coupled localization+delocalization tendency in a single scalar"
                        ],
                    },
                    "bond_strength_pattern_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "0.40*max(0,mayer) + 0.30*max(0,wiberg) + 0.20*clamp(elf,0,1) + 0.10*max(0,di)",
                        "inputs_used": [
                            "bond_orders_mayer",
                            "bond_orders_wiberg",
                            "elf_bond_midpoint",
                            "bond_delocalization_index_proxy_v1"
                        ],
                        "limitations": [
                            "heuristic composite proxy, not an exact quantum bond strength observable"
                        ],
                    },
                    "bond_localization_tension_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": False,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "abs(bond_delocalization_localization_balance_proxy_v1)",
                        "inputs_used": [
                            "bond_delocalization_localization_balance_proxy_v1"
                        ],
                        "range": "[0, 1]",
                        "limitations": [
                            "captures imbalance magnitude only; sign information is in balance proxy"
                        ],
                    },
                    "bond_polarized_delocalization_proxy_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "is_heuristic": True,
                        "availability_status": "not_attempted",
                        "status_reason": "not_attempted_by_default",
                        "skip_reason": None,
                        "failure_reason": None,
                        "formula": "bond_covalency_polarity_proxy_v1 * max(0,bond_delocalization_index_proxy_v1)",
                        "inputs_used": [
                            "bond_covalency_polarity_proxy_v1",
                            "bond_delocalization_index_proxy_v1"
                        ],
                        "limitations": [
                            "heuristic coupling between bond polarity and delocalization tendencies"
                        ],
                    },
                }
            },

            "structural_features": {
                "optimized_3d_geometry": {
                    "available": False,
                    "source": None,
                    "is_proxy": None,
                    "definition_version": "v1",
                    "proxy_family": "semantic_reference",
                    "coordinate_ref": "geometry.atom_coords_angstrom",
                    "coordinate_embedding": "reference_only",
                    "coordinate_source_of_truth": "geometry.atom_coords_angstrom",
                    "natm_reference": None,
                    "geometry_fingerprint_sha256": None,
                    "semantics": "reference_to_current_working_geometry",
                    "proxy_note": None,
                    "limitations": [],
                },
                "most_stable_conformation": {
                    "available": False,
                    "is_proxy": True,
                    "definition_version": "v1",
                    "proxy_family": "rdkit_forcefield_conformer_search",
                    "conformer_id": None,
                    "conformer_generation_method": None,
                    "selection_method": None,
                    "selection_scope": "lowest-energy conformer within generated+optimized candidate set of current run",
                    "n_conformers_requested": 0,
                    "n_conformers_generated": 0,
                    "n_conformers_optimized": 0,
                    "n_conformers_ranked": 0,
                    "forcefield_used": None,
                    "ranking_energy_type": None,
                    "ranking_energy_value": None,
                    "ranking_energy_unit": "forcefield_native_energy_units",
                    "duplicate_filter_applied": False,
                    "energy_dedup_threshold": None,
                    "energy_dedup_threshold_unit": "forcefield_native_energy_units",
                    "random_seed": None,
                    "source": None,
                    "proxy_note": "requires dedicated conformer generation + ranking workflow in current run",
                    "candidate_set_statistics_proxy_v1": {
                        "available": False,
                        "definition_version": "v1",
                        "candidate_set_scope": "generated_optimized_ranked_candidate_set_current_run",
                        "conformer_count_ranked": 0,
                        "conformer_energy_span_proxy": None,
                        "conformer_energy_std_proxy": None,
                        "geometry_size_variability_proxy": None,
                        "conformer_compactness_proxy_v1": None,
                        "limitations": [
                            "statistics are candidate-set summaries, not global conformer-space truth",
                            "forcefield native energies are not directly thermodynamic free energies"
                        ],
                    },
                    "limitations": [],
                },
            },
            
            "artifacts": {
                "wavefunction": {
                    "pkl_path": None,
                    "format": "pickle",
                    "loaded_successfully": None
                },
                "fock_matrix": {
                    "iao_matrix": None,
                    "atomic_slices": None,
                    "available": False
                },
                "cube_files": {
                    "density": None,
                    "elf": None
                },
                "external_bridge": {
                    "critic2_input": None,
                    "critic2_output": None,
                    "horton_output": None,
                    "psi4_output": None
                }
            },
            
            "plugin_status": {
                "rdkit": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                },
                "xtb": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                },
                "pyscf": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                }
            },
            
            "provenance": {
                "software": self.SOFTWARE_NAME,
                "version": self.SCHEMA_VERSION,
                "dft_functional": None,
                "basis_set": None,
                "geometry_optimization_method": None,
                "gpu_acceleration_used": None,
                "feature_definition_version": None,
                "realspace_definition_version": None,
                "calculation_timestamp": datetime.utcnow().isoformat() + "Z",
                "wall_time_seconds": None
            },
            "runtime_metadata": {
                "unified_extraction_time_seconds": None,
                "extraction_timestamp": None,
                "stage_timing_seconds": None,
                "elf_bond_midpoints_raw_count": None,
                "elf_bond_midpoints_aligned_count": None,
                "elf_bond_midpoints_dropped_count": None,
                "schema_build_finalized": False
            },
            
                "orbital_features": {
                    "local_orbital_method": None,
                    "ibo_count": None,
                    "ibo_occupancies": None,
                    "ibo_centers_angstrom": None,
                "ibo_atom_contributions": None,
                "ibo_class_heuristic": None,
                "orbital_locality_score": None,
                "iao_atom_mapping": None,
                    "metadata": {
                        "coefficient_basis": None,
                        "coefficient_metric": None,
                        "center_definition": None,
                        "contribution_definition": None,
                        "occupancy_definition": None,
                        "locality_score_formula": None,
                        "classification_is_heuristic": None,
                        "heuristic_classification_rules": None,
                        "extraction_status": "not_attempted",
                        "skip_reason": None,
                        "failure_reason": None,
                        "extraction_time_seconds": None,
                        "pyscf_version": None,
                    }
                },
            
            "realspace_features": {
                "density_isosurface_volume": None,
                "density_isosurface_area": None,
                "density_sphericity_like": None,
                "density_shape_descriptor_family_v1": None,
                "density_shape_multiscale_family_v1": None,
                "esp_extrema_summary": None,
                "orbital_extent_homo": None,
                "orbital_extent_lumo": None,
                "metadata": {
                    "realspace_definition_version": None,
                    "native_grid_unit": None,
                    "output_unit": None,
                    "conversion_factor_bohr_to_angstrom": None,
                    "density_isovalue": None,
                    "orbital_isovalue": None,
                    "cube_grid_shape": None,
                    "cube_spacing_angstrom": None,
                    "density_grid_resolution_angstrom": None,
                    "margin_angstrom": None,
                    "geometry_electronic_state_key": None,
                    "wavefunction_signature": None,
                    "feature_compatibility_key": None,
                    "artifact_compatibility_key": None,
                    "density_source_method": None,
                    "density_source_family": None,
                    "density_source_artifact_path": None,
                    "density_source_wavefunction_dependency": None,
                    "density_source_state_key": None,
                    "cache_reuse_mode": None,
                    "cache_hit": None,
                    "cache_entry_id": None,
                    "cache_root": None,
                    "required_artifacts": None,
                    "optional_artifacts": None,
                    "realspace_core_features_enabled": None,
                    "realspace_core_features_expected": None,
                    "realspace_core_features_status": None,
                    "realspace_extended_features_enabled": None,
                    "realspace_extended_features_expected": None,
                    "realspace_extended_features_status": None,
                    "realspace_extended_failure_reason": None,
                    "stage_timing_seconds": None,
                    "density_shape_descriptor_family_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "proxy_family": "density_shape_descriptor_family",
                        "mass_cutoff_default": 0.95,
                        "eps": 1e-12,
                        "coordinate_source": "realspace density cube grid",
                        "weight_definition": "density_value",
                        "covariance_definition": "weighted covariance over selected mass-cutoff points",
                        "eigenvalue_order": "lambda1>=lambda2>=lambda3>=0",
                        "normalization_rule": "lambda_i_norm = lambda_i / (lambda1 + lambda2 + lambda3 + eps)",
                        "formula_sphericity": "3*lambda3/(lambda1+lambda2+lambda3+eps)",
                        "formula_asphericity": "(lambda1 - 0.5*(lambda2+lambda3))/(lambda1+lambda2+lambda3+eps)",
                        "formula_anisotropy": "((lambda1-lambda2)^2 + (lambda2-lambda3)^2 + (lambda3-lambda1)^2)/(2*(lambda1+lambda2+lambda3)^2+eps)",
                        "formula_relative_anisotropy_kappa2": "1 - 3*(lambda1n*lambda2n + lambda2n*lambda3n + lambda3n*lambda1n), lambda_in=lambda_i/(lambda1+lambda2+lambda3+eps)",
                        "formula_elongation": "(lambda1 - lambda2)/(lambda1 + eps)",
                        "formula_planarity": "(lambda2 - lambda3)/(lambda1 + eps)",
                        "normalization_notes": "raw eigenvalues encode absolute cloud size; normalized eigenvalues and kappa2 encode relative shape",
                        "limitations": [
                            "open-source substitute descriptor family; not qcMol exact internal formula",
                            "depends on cube grid resolution/margin and mass_cutoff setting",
                        ],
                        "status": "not_attempted",
                        "status_reason": "not_computed_yet",
                    },
                    "density_shape_multiscale_family_v1": {
                        "definition_version": "v1",
                        "is_proxy": True,
                        "proxy_family": "density_shape_multiscale_family",
                        "default_scale": 0.95,
                        "available_scales": [0.5, 0.9, 0.95],
                        "eps": 1e-12,
                        "coordinate_source": "realspace density cube grid",
                        "weight_definition": "density_value",
                        "covariance_definition": "weighted covariance over per-scale mass-cutoff selected density points",
                        "eigenvalue_order": "lambda1>=lambda2>=lambda3>=0",
                        "normalization_rule": "lambda_i_norm = lambda_i / (lambda1 + lambda2 + lambda3 + eps)",
                        "formula_sphericity": "3*lambda3/(lambda1+lambda2+lambda3+eps)",
                        "formula_asphericity": "(lambda1 - 0.5*(lambda2+lambda3))/(lambda1+lambda2+lambda3+eps)",
                        "formula_anisotropy": "((lambda1-lambda2)^2 + (lambda2-lambda3)^2 + (lambda3-lambda1)^2)/(2*(lambda1+lambda2+lambda3)^2+eps)",
                        "formula_relative_anisotropy_kappa2": "1 - 3*(lambda1n*lambda2n + lambda2n*lambda3n + lambda3n*lambda1n)",
                        "formula_elongation": "(lambda1-lambda2)/(lambda1+eps)",
                        "formula_planarity": "(lambda2-lambda3)/(lambda1+eps)",
                        "scale_semantics": "0.50/0.90/0.95 capture dense core, intermediate shell, and near-complete electron cloud mass",
                        "normalization_notes": "raw eigenvalues are size-dependent; normalized eigenvalues and kappa2 are preferred for cross-size comparisons",
                        "canonical_single_scale_view": "realspace_features.density_shape_descriptor_family_v1",
                        "limitations": [
                            "open-source substitute descriptor family; not qcMol exact internal formula",
                            "cross-scale comparability depends on shared grid resolution/margin and density source",
                        ],
                        "status": "not_attempted",
                        "status_reason": "not_computed_yet",
                    },
                    "extraction_status": None,
                    "failure_reason": None,
                    "extraction_time_seconds": None,
                }
            },
            
            "external_bridge": {
                "critic2": {
                    "execution_status": "not_attempted",
                    "failure_reason": None,
                    # New contract fields (preferred readers/writers)
                    "metadata": {
                        "tool_version": None,
                        "execution_time_seconds": None,
                        "command": None,
                        "parser_version": None,
                        "environment": None,
                    },
                    "artifact_refs": {
                        "input_file": None,
                        "output_file": None,
                        "stdout_file": None,
                        "stderr_file": None,
                    },
                    "warnings": [],
                    # Deprecated compatibility fields (write/read discouraged; use metadata/artifact_refs)
                    "input_file": None,
                    "output_file": None,
                    "execution_time_seconds": None,
                    "critic2_version": None,
                }
            },
            
                "external_features": {
                    "critic2": {
                        "qtaim": {
                            "bader_charges": None,
                            "bader_populations": None,
                            "bader_volumes": None,
                            "n_bader_volumes": None,
                            "atomic_integrated_properties": None,
                            "stable_atomic_integrated_properties_v1": None,
                            "stable_atomic_integrated_property_summary_v1": None,
                            "atomic_integrated_property_candidate_assessment_v1": None,
                            "basin_companion_summary_v1": None,
                            "metadata": {
                                "analysis_type": None,
                                "convergence_status": None,
                                "atomic_property_parser": None,
                                "atomic_property_source": None,
                                "atomic_property_header_tokens": None,
                                "atomic_property_header_token_count": None,
                                "atomic_property_rows_parsed": None,
                                "atomic_property_pop_column": None,
                                "atomic_property_volume_column": None,
                                "atomic_property_volume_available": None,
                                "atomic_property_parse_note": None,
                                "atomic_property_table_excerpt": None,
                                "atomic_integrated_property_columns": None,
                            }
                        }
                    }
                },

            # External bridge roadmap placeholders (M5.5)
            # Keep exact qcMol naming decisions deferred where marked by needs_exact_qcmol_name.
            "external_bridge_roadmap": {
                "atom_level": {
                    "nao_descriptors": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": True,
                        "notes": "Placeholder for NAO descriptor payload from external bridge toolchain.",
                    },
                    "adch_charges": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": False,
                        "notes": "Placeholder for ADCH charges.",
                    },
                    "nbo_lp": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": False,
                        "notes": "Placeholder for NBO lone pair descriptors.",
                    },
                    "npa_exact": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": False,
                        "notes": "Exact NPA placeholder; do not confuse with IAO proxy.",
                    },
                    "li_values": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": True,
                        "notes": "Placeholder for LI values; abbreviation expansion pending paper-exact naming.",
                    },
                },
                "bond_level": {
                    "di_values_or_matrix": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": True,
                        "notes": "Placeholder for DI values/matrix with exact qcMol naming pending.",
                    },
                    "nbo_bd": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": False,
                        "notes": "Placeholder for NBO bond descriptors.",
                    },
                    "lbo": {
                        "status": "missing",
                        "payload": None,
                        "needs_exact_qcmol_name": False,
                        "notes": "Placeholder for localized bond order payload.",
                    },
                },
            }
        }
    
    # ============ Setter 方法 ============
    
    def set_molecule_info(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置分子基本信息"""
        for key, value in kwargs.items():
            if key in self.data["molecule_info"]:
                self.data["molecule_info"][key] = value
        return self

    def set_molecule_representation_metadata(self, representation: str, **kwargs) -> "UnifiedOutputBuilder":
        """设置分子表示层 metadata（如 SMARTS 的 proxy 信息）"""
        rep_meta = self.data["molecule_info"].get("representation_metadata", {})
        if representation in rep_meta and isinstance(rep_meta[representation], dict):
            for key, value in kwargs.items():
                if key in rep_meta[representation]:
                    rep_meta[representation][key] = value
        return self
    
    def set_status(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置计算状态"""
        for key, value in kwargs.items():
            if key in self.data["calculation_status"]:
                self.data["calculation_status"][key] = value
        return self
    
    def add_error(self, error_msg: str) -> "UnifiedOutputBuilder":
        """添加错误信息"""
        self.data["calculation_status"]["error_messages"].append(error_msg)
        return self
    
    def set_geometry(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置几何信息"""
        for key, value in kwargs.items():
            if key in self.data["geometry"]:
                self.data["geometry"][key] = value
        return self
    
    def set_global_dft(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 DFT 全局特征"""
        for key, value in kwargs.items():
            if key in self.data["global_features"]["dft"]:
                self.data["global_features"]["dft"][key] = value
        return self
    
    def set_global_rdkit(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 RDKit 全局特征"""
        for key, value in kwargs.items():
            if key in self.data["global_features"]["rdkit"]:
                self.data["global_features"]["rdkit"][key] = value
        return self

    def set_global_geometry_size(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 geometry-dependent 分子尺寸特征"""
        for key, value in kwargs.items():
            if key in self.data["global_features"]["geometry_size"]:
                self.data["global_features"]["geometry_size"][key] = value
        return self

    def set_global_metadata(self, feature: str, **kwargs) -> "UnifiedOutputBuilder":
        """设置 global_features.metadata 下某个 canonical 特征的 metadata"""
        meta = self.data["global_features"].get("metadata", {})
        if feature in meta and isinstance(meta[feature], dict):
            for key, value in kwargs.items():
                if key in meta[feature]:
                    meta[feature][key] = value
        return self
    
    def set_atom_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置原子特征"""
        for key, value in kwargs.items():
            if key in self.data["atom_features"]:
                self.data["atom_features"][key] = value
        return self

    def set_atom_metadata(self, feature: str, **kwargs) -> "UnifiedOutputBuilder":
        """设置原子特征 metadata（proxy 定义/约束等）"""
        meta = self.data["atom_features"].get("metadata", {})
        if feature in meta and isinstance(meta[feature], dict):
            for key, value in kwargs.items():
                if key in meta[feature]:
                    meta[feature][key] = value
        return self
    
    def set_bond_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置键特征（统一 list 模式）"""
        for key, value in kwargs.items():
            if key in self.data["bond_features"] and key != "metadata":
                self.data["bond_features"][key] = value
        return self

    def set_bond_metadata(self, feature: str, **kwargs) -> "UnifiedOutputBuilder":
        """设置键特征 metadata（如 stereo 的 proxy 信息）"""
        meta = self.data["bond_features"].get("metadata", {})
        if feature in meta and isinstance(meta[feature], dict):
            for key, value in kwargs.items():
                if key in meta[feature]:
                    meta[feature][key] = value
        return self

    def set_structural_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置结构层特征（optimized_3d_geometry / most_stable_conformation）"""
        for key, value in kwargs.items():
            if key in self.data["structural_features"]:
                self.data["structural_features"][key] = value
        return self
    
    def set_artifacts_wavefunction(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置波函数 artifacts"""
        for key, value in kwargs.items():
            if key in self.data["artifacts"]["wavefunction"]:
                self.data["artifacts"]["wavefunction"][key] = value
        return self
    
    def set_artifacts_fock(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 Fock 矩阵 artifacts"""
        for key, value in kwargs.items():
            if key in self.data["artifacts"]["fock_matrix"]:
                self.data["artifacts"]["fock_matrix"][key] = value
        return self
    
    def set_plugin_status(self, plugin: str, **kwargs) -> "UnifiedOutputBuilder":
        """
        设置插件状态
        
        Args:
            plugin: "rdkit" | "xtb" | "pyscf"
        """
        if plugin in self.data["plugin_status"]:
            for key, value in kwargs.items():
                if key in self.data["plugin_status"][plugin]:
                    self.data["plugin_status"][plugin][key] = value
        return self
    
    def set_provenance(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置溯源信息"""
        for key, value in kwargs.items():
            if key in self.data["provenance"]:
                self.data["provenance"][key] = value
        return self
    
    def set_orbital_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置局域轨道特征"""
        for key, value in kwargs.items():
            if key in self.data["orbital_features"] and key != "metadata":
                self.data["orbital_features"][key] = value
        return self
    
    def set_orbital_metadata(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置局域轨道 metadata"""
        for key, value in kwargs.items():
            if key in self.data["orbital_features"]["metadata"]:
                self.data["orbital_features"]["metadata"][key] = value
        return self
    
    def set_realspace_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置实空间特征"""
        for key, value in kwargs.items():
            if key in self.data["realspace_features"] and key != "metadata":
                self.data["realspace_features"][key] = value
        return self
    
    def set_realspace_metadata(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置实空间 metadata"""
        for key, value in kwargs.items():
            if key in self.data["realspace_features"]["metadata"]:
                self.data["realspace_features"]["metadata"][key] = value
        return self
    
    def set_cube_file(self, cube_type: str, info: dict) -> "UnifiedOutputBuilder":
        """
        设置 cube 文件信息
        
        Args:
            cube_type: "density" | "esp" | "homo" | "lumo"
            info: cube 文件信息字典
        """
        if cube_type in ["density", "esp", "homo", "lumo"]:
            # 注意：cube_files 现在可能在 artifacts 或 realspace_features 中
            if "cube_files" not in self.data["artifacts"]:
                self.data["artifacts"]["cube_files"] = {}
            self.data["artifacts"]["cube_files"][cube_type] = info
        return self
    
    def set_external_bridge(self, tool: str, **kwargs) -> "UnifiedOutputBuilder":
        """
        设置 external bridge 执行信息
        
        Args:
            tool: "critic2" | "horton" | "psi4"
            **kwargs: bridge 信息字段
        """
        if tool in self.data["external_bridge"]:
            node = self.data["external_bridge"][tool]
            for key, value in kwargs.items():
                if key == "execution_status":
                    if value not in self.EXTERNAL_BRIDGE_EXECUTION_STATUS_ENUM:
                        # invalid enum falls back to failed for contract safety
                        node["execution_status"] = "failed"
                        node["failure_reason"] = f"invalid_execution_status:{value}"
                    else:
                        node["execution_status"] = value
                elif key == "metadata" and isinstance(value, dict):
                    if "metadata" not in node or not isinstance(node["metadata"], dict):
                        node["metadata"] = {}
                    node["metadata"].update(value)
                elif key == "artifact_refs" and isinstance(value, dict):
                    if "artifact_refs" not in node or not isinstance(node["artifact_refs"], dict):
                        node["artifact_refs"] = {}
                    node["artifact_refs"].update(value)
                elif key == "warnings" and isinstance(value, list):
                    node["warnings"] = value
                elif key in node:
                    node[key] = value
        return self
    
    def set_external_features(self, tool: str, features: dict) -> "UnifiedOutputBuilder":
        """
        设置 external features
        
        Args:
            tool: "critic2" | "horton" | "psi4"
            features: 特征字典
        """
        if tool in self.data["external_features"]:
            self.data["external_features"][tool].update(features)
        return self

    def set_external_bridge_roadmap(self, level: str, feature: str, **kwargs) -> "UnifiedOutputBuilder":
        """
        设置 external bridge roadmap 占位字段

        Args:
            level: "atom_level" | "bond_level"
            feature: placeholder 特征名
        """
        roadmap = self.data.get("external_bridge_roadmap", {})
        if level in roadmap and feature in roadmap[level] and isinstance(roadmap[level][feature], dict):
            for key, value in kwargs.items():
                if key in roadmap[level][feature]:
                    if key == "status":
                        if value in self.ROADMAP_STATUS_ENUM:
                            roadmap[level][feature][key] = value
                    else:
                        roadmap[level][feature][key] = value
        return self
    
    # ============ 状态判定方法 ============
    
    def determine_overall_status(self) -> str:
        """
        确定 overall_status（严格顺序执行，短路返回）
        
        判定流程:
        1. invalid_input → SMILES 无法解析或参数非法
        2. failed_geometry → 几何优化失败
        3. failed_scf → SCF 未收敛
        4. failed_core_features → 核心特征计算失败
        5. core_success_partial_features → 核心成功但扩展失败
        6. fully_success → 全部成功
        
        注意：M5 后，推荐使用 utils.policy.StatusDeterminer 进行更精确的状态判定。
        此方法保持向后兼容。
        """
        # 尝试使用新的 StatusDeterminer
        try:
            from utils.policy.status_determiner import StatusDeterminer
            determiner = StatusDeterminer(self.data)
            return determiner.determine()
        except Exception:
            # 回退到旧逻辑
            pass
        
        status = self.data["calculation_status"]
        
        if status["invalid_input"]:
            return "invalid_input"
        
        if not status["geometry_optimization_success"]:
            return "failed_geometry"
        
        if not status["scf_convergence_success"]:
            return "failed_scf"
        
        if not status["core_features_success"]:
            return "failed_core_features"
        
        if not status["extended_features_success"]:
            return "core_success_partial_features"
        
        return "fully_success"
    
    def check_core_features_success(self) -> bool:
        """
        检查核心特征是否成功（放宽版判定规则）
        
        必须同时满足:
        1. wavefunction_load_success == true
        2. scf_convergence_success == true
        3. global_features.dft.total_energy 存在
        4. 至少一组原子电荷成功且长度匹配（Mulliken/Hirshfeld/IAO/CM5 任一）
        5. 多原子分子至少一组键级特征成功（单原子跳过此项）
        
        注意：M5 后，推荐使用 utils.policy.FieldTierChecker 进行更精确的字段检查。
        """
        # 尝试使用新的 FieldTierChecker
        try:
            from utils.policy.field_tiers import FieldTierChecker
            checker = FieldTierChecker()
            ok, _ = checker.check_core_required(self.data)
            return ok
        except Exception:
            # 回退到旧逻辑
            pass
        
        status = self.data["calculation_status"]
        dft = self.data["global_features"]["dft"]
        atom_feat = self.data["atom_features"]
        bond_feat = self.data["bond_features"]
        natm = self.data["molecule_info"]["natm"] or 0
        
        # 1 & 2. 波函数加载和 SCF 收敛
        if not (status["wavefunction_load_success"] and dft["scf_converged"]):
            return False
        
        # 3. 基本能量信息可得
        if dft["total_energy_hartree"] is None:
            return False
        
        # 4. 至少一组原子电荷成功且长度匹配
        charge_fields = [
            atom_feat["charge_mulliken"],
            atom_feat["charge_hirshfeld"],
            atom_feat["charge_iao"],
            atom_feat["charge_cm5"],
        ]
        valid_charge = any(
            c is not None and isinstance(c, (list, tuple)) and len(c) == natm
            for c in charge_fields
        )
        if not valid_charge:
            return False
        
        # 5. 多原子分子至少一组键级特征成功
        if natm > 1:
            bond_orders = [
                bond_feat["bond_orders_mayer"],
                bond_feat["bond_orders_wiberg"]
            ]
            valid_bond = any(
                b is not None and isinstance(b, (list, tuple)) and len(b) > 0
                for b in bond_orders
            )
            if not valid_bond:
                return False
        
        return True
    
    def check_extended_features_success(self) -> bool:
        """
        检查扩展特征是否成功（strongly_recommended 规则组）
        
        M5 后：检查 strongly_recommended 规则组是否全部满足。
        若不满足，则判定为 core_success_partial_features。
        """
        if not self.data["calculation_status"]["core_features_success"]:
            return False
        
        # 尝试使用新的 FieldTierChecker
        try:
            from utils.policy.field_tiers import FieldTierChecker
            checker = FieldTierChecker()
            violation_count, _ = checker.check_strongly_recommended(self.data)
            return violation_count == 0
        except Exception:
            # 回退到旧逻辑
            pass
        
        atom_feat = self.data["atom_features"]
        bond_feat = self.data["bond_features"]
        
        # CM5 电荷
        if atom_feat["charge_cm5"] is None:
            return False
        
        # ELF 特征
        if atom_feat["elf_value"] is None:
            return False
        if bond_feat["elf_bond_midpoint"] is None:
            return False
        
        return True
    
    def finalize(self) -> Dict[str, Any]:
        """
        最终化输出，计算状态并填充时间戳
        
        这是唯一的状态判定入口，确保 overall_status 由中心函数统一判定。
        
        Returns:
            完整的统一输出字典
        """
        # 计算 wall_time
        self.data["provenance"]["wall_time_seconds"] = time.time() - self._start_time
        
        # 判定核心和扩展特征状态
        self.data["calculation_status"]["core_features_success"] = self.check_core_features_success()
        self.data["calculation_status"]["extended_features_success"] = self.check_extended_features_success()
        
        # 判定 overall_status（中心唯一判定点）
        self.data["calculation_status"]["overall_status"] = self.determine_overall_status()
        
        # 标记已最终化
        self.data["runtime_metadata"]["schema_build_finalized"] = True
        
        return self.data
    
    def build(self) -> Dict[str, Any]:
        """
        构建并返回最终输出。
        
        注意：此方法显式调用 finalize() 确保状态一致性。
        所有提前返回点（如错误处理）也应显式调用 finalize()。
        
        Returns:
            完整的统一输出字典
        """
        return self.finalize()
    
    def build_early(self) -> Dict[str, Any]:
        """
        提前构建输出（用于错误退出时）。
        
        确保即使流程提前终止，也调用 finalize() 进行正确的状态判定。
        
        Returns:
            完整的统一输出字典
        """
        return self.finalize()

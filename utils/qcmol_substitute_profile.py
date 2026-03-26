"""
qcMol substitute default profile and canonical surface helpers.

This module freezes a recommended "default delivery" profile so callers do not
need to handcraft run_mode/plugin/runtime knobs for every run.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


PROFILE_ID = "qcmol_substitute_default"
PROFILE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / f"{PROFILE_ID}.json"


QCMOL_ALIGNMENT_ITEMS: List[Dict[str, Any]] = [
    # basic_information
    {"section": "basic_information", "name": "qcMol_ID", "mapped_path": "molecule_info.molecule_id", "status": "implemented_exact"},
    {"section": "basic_information", "name": "IUPAC_name", "mapped_path": None, "status": "missing"},
    {"section": "basic_information", "name": "SMILES", "mapped_path": "molecule_info.smiles", "status": "implemented_exact"},
    {"section": "basic_information", "name": "InChI", "mapped_path": "molecule_info.inchi", "status": "implemented_exact"},
    {"section": "basic_information", "name": "InChIKey", "mapped_path": "molecule_info.inchikey", "status": "implemented_exact"},
    {"section": "basic_information", "name": "chemical_formula", "mapped_path": "molecule_info.formula", "status": "implemented_exact"},
    {"section": "basic_information", "name": "SMART", "mapped_path": "molecule_info.smarts", "status": "implemented_proxy"},
    {"section": "basic_information", "name": "nickname_or_synonyms", "mapped_path": None, "status": "missing"},
    # global_features
    {"section": "global_features", "name": "HOMO_LUMO_gap", "mapped_path": "global_features.dft.homo_lumo_gap_hartree", "status": "implemented_exact"},
    {"section": "global_features", "name": "dipole_moment", "mapped_path": "global_features.dft.dipole_moment_debye", "status": "implemented_exact"},
    {"section": "global_features", "name": "isosurface_area", "mapped_path": "realspace_features.density_isosurface_area", "status": "implemented_exact"},
    {"section": "global_features", "name": "isosurface_volume", "mapped_path": "realspace_features.density_isosurface_volume", "status": "implemented_exact"},
    {"section": "global_features", "name": "sphericity_parameters", "mapped_path": "realspace_features.density_sphericity_like", "status": "implemented_proxy"},
    {"section": "global_features", "name": "molecule_size", "mapped_path": "global_features.geometry_size.bounding_box_diagonal_angstrom", "status": "implemented_proxy"},
    {"section": "global_features", "name": "molecular_weight", "mapped_path": "global_features.rdkit.molecular_weight", "status": "implemented_exact"},
    {"section": "global_features", "name": "ionization_affinity_or_related", "mapped_path": "global_features.dft.homo_energy_hartree", "status": "partial"},
    {"section": "global_features", "name": "charge", "mapped_path": "molecule_info.charge", "status": "implemented_exact"},
    # atom_features
    {"section": "atom_features", "name": "element_type", "mapped_path": "geometry.atom_symbols", "status": "implemented_exact"},
    {"section": "atom_features", "name": "XYZ", "mapped_path": "geometry.atom_coords_angstrom", "status": "implemented_exact"},
    {"section": "atom_features", "name": "NAO_descriptors", "mapped_path": "external_bridge_roadmap.atom_level.nao_descriptors", "status": "missing"},
    {"section": "atom_features", "name": "LI_values", "mapped_path": "external_bridge_roadmap.atom_level.li_values", "status": "missing"},
    {"section": "atom_features", "name": "ADCH_charges", "mapped_path": "external_bridge_roadmap.atom_level.adch_charges", "status": "missing"},
    {"section": "atom_features", "name": "NBO_LP", "mapped_path": "atom_features.atomic_lone_pair_heuristic_proxy", "status": "implemented_proxy"},
    {"section": "atom_features", "name": "NPA", "mapped_path": "atom_features.atomic_charge_iao_proxy", "status": "implemented_proxy"},
    {"section": "atom_features", "name": "NPA_exact", "mapped_path": "external_bridge_roadmap.atom_level.npa_exact", "status": "missing"},
    {"section": "atom_features", "name": "density_partition_charge_proxy.hirshfeld", "mapped_path": "atom_features.atomic_density_partition_charge_proxy.hirshfeld", "status": "implemented_proxy"},
    {"section": "atom_features", "name": "density_partition_charge_proxy.cm5", "mapped_path": "atom_features.atomic_density_partition_charge_proxy.cm5", "status": "implemented_proxy"},
    {"section": "atom_features", "name": "density_partition_charge_proxy.bader", "mapped_path": "atom_features.atomic_density_partition_charge_proxy.bader", "status": "implemented_proxy"},
    {"section": "atom_features", "name": "density_partition_volume_proxy.bader", "mapped_path": "atom_features.atomic_density_partition_volume_proxy.bader", "status": "partial"},
    {"section": "atom_features", "name": "atomic_orbital_descriptor_proxy_v1", "mapped_path": "atom_features.atomic_orbital_descriptor_proxy_v1", "status": "implemented_proxy"},
    # bond_features
    {"section": "bond_features", "name": "stereo_info", "mapped_path": "bond_features.bond_stereo_info", "status": "implemented_proxy"},
    {"section": "bond_features", "name": "DI_values_or_matrix", "mapped_path": "bond_features.bond_delocalization_index_proxy_v1", "status": "implemented_proxy"},
    {"section": "bond_features", "name": "ELF_values", "mapped_path": "bond_features.elf_bond_midpoint", "status": "implemented_exact"},
    {"section": "bond_features", "name": "NBO_BD", "mapped_path": "external_bridge_roadmap.bond_level.nbo_bd", "status": "missing"},
    {"section": "bond_features", "name": "LBO", "mapped_path": "external_bridge_roadmap.bond_level.lbo", "status": "missing"},
    {"section": "bond_features", "name": "Mayer_BL", "mapped_path": "bond_features.bond_orders_mayer", "status": "partial"},
    {"section": "bond_features", "name": "bond_orbital_localization_proxy", "mapped_path": "bond_features.bond_orbital_localization_proxy", "status": "implemented_proxy"},
    {"section": "bond_features", "name": "bond_order_weighted_localization_proxy", "mapped_path": "bond_features.bond_order_weighted_localization_proxy", "status": "implemented_proxy"},
    # structural_features
    {"section": "structural_features", "name": "optimized_3D_geometry", "mapped_path": "structural_features.optimized_3d_geometry", "status": "partial"},
    {"section": "structural_features", "name": "most_stable_conformation", "mapped_path": "structural_features.most_stable_conformation", "status": "implemented_proxy"},
]


CANONICAL_SURFACE_ITEMS: List[Dict[str, Any]] = [
    {"group": "basic", "path": "molecule_info.molecule_id", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.smiles", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.inchi", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.inchikey", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.formula", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.total_energy_hartree", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.homo_energy_hartree", "status": "partial"},
    {"group": "global", "path": "global_features.dft.lumo_energy_hartree", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.homo_lumo_gap_hartree", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.dipole_moment_debye", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.rdkit.molecular_weight", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.geometry_size.bounding_box_diagonal_angstrom", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.charge_mulliken", "status": "implemented_exact"},
    {"group": "atom", "path": "atom_features.charge_hirshfeld", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.charge_cm5", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.charge_iao", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_charge_iao_proxy", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_density_partition_charge_proxy.hirshfeld", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_density_partition_charge_proxy.cm5", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_density_partition_charge_proxy.bader", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_lone_pair_heuristic_proxy", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_orbital_descriptor_proxy_v1", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_indices", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.bond_orders_mayer", "status": "partial"},
    {"group": "bond", "path": "bond_features.bond_orders_wiberg", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.elf_bond_midpoint", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.bond_delocalization_index_proxy_v1", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_orbital_localization_proxy", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_order_weighted_localization_proxy", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_stereo_info", "status": "implemented_proxy"},
    {"group": "structural", "path": "structural_features.optimized_3d_geometry", "status": "partial"},
    {"group": "structural", "path": "structural_features.most_stable_conformation", "status": "implemented_proxy"},
    {"group": "realspace", "path": "realspace_features.density_isosurface_area", "status": "implemented_exact"},
    {"group": "realspace", "path": "realspace_features.density_isosurface_volume", "status": "implemented_exact"},
    {"group": "realspace", "path": "realspace_features.density_sphericity_like", "status": "implemented_proxy"},
    {"group": "realspace", "path": "realspace_features.esp_extrema_summary", "status": "implemented_proxy"},
    {"group": "realspace", "path": "realspace_features.orbital_extent_homo", "status": "implemented_exact"},
    {"group": "realspace", "path": "realspace_features.orbital_extent_lumo", "status": "implemented_exact"},
    {"group": "external", "path": "external_bridge.critic2", "status": "implemented_exact"},
    {"group": "external", "path": "external_features.critic2.qtaim", "status": "implemented_proxy"},
    {"group": "external", "path": "atom_features.atomic_density_partition_charge_proxy.bader", "status": "implemented_proxy"},
    {"group": "external", "path": "atom_features.atomic_density_partition_volume_proxy.bader", "status": "partial"},
]


DISCOURAGED_DEFAULT_FIELDS: List[str] = [
    "external_bridge_roadmap.atom_level.nao_descriptors",
    "external_bridge_roadmap.atom_level.adch_charges",
    "external_bridge_roadmap.atom_level.nbo_lp",
    "external_bridge_roadmap.atom_level.npa_exact",
    "external_bridge_roadmap.atom_level.li_values",
    "external_bridge_roadmap.bond_level.di_values_or_matrix",
    "external_bridge_roadmap.bond_level.nbo_bd",
    "external_bridge_roadmap.bond_level.lbo",
]


def load_qcmol_substitute_default_profile(config_path: Path | None = None) -> Dict[str, Any]:
    """Load frozen default profile from JSON."""
    cfg_path = Path(config_path) if config_path else PROFILE_CONFIG_PATH
    profile = json.loads(cfg_path.read_text(encoding="utf-8"))
    return profile


def build_batch_kwargs_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Translate profile object to run_batch kwargs."""
    execution = profile.get("execution", {}) or {}
    return {
        "run_mode": execution.get("run_mode", "full"),
        "artifact_policy": execution.get("artifact_policy", "keep_failed_only"),
        "plugin_config": deepcopy(profile.get("plugins", {}) or {}),
        "n_workers": int(execution.get("n_workers", 1)),
    }


def apply_bader_validation_profile(feature_extractor: Any, profile: Dict[str, Any]) -> None:
    """
    Apply frozen Bader validation/retry controls to FeatureExtractor instance.
    """
    bader_cfg = profile.get("bader_validation", {}) or {}
    tol_cfg = bader_cfg.get("population_sum_tolerance", {}) or {}
    refined_cfg = bader_cfg.get("refined_retry", {}) or {}
    rescue_cfg = bader_cfg.get("rescue_retry", {}) or {}

    if "abs_tol_e" in tol_cfg:
        feature_extractor.BADER_POPULATION_SUM_ABS_TOL_E = float(tol_cfg["abs_tol_e"])
    if "rel_tol" in tol_cfg:
        feature_extractor.BADER_POPULATION_SUM_REL_TOL = float(tol_cfg["rel_tol"])

    if "enabled" in refined_cfg:
        feature_extractor.BADER_REFINED_RETRY_ENABLED = bool(refined_cfg["enabled"])
    if "grid_resolution_angstrom" in refined_cfg:
        feature_extractor.BADER_REFINED_GRID_RES_ANGSTROM = float(refined_cfg["grid_resolution_angstrom"])
    if "margin_angstrom" in refined_cfg:
        feature_extractor.BADER_REFINED_MARGIN_ANGSTROM = float(refined_cfg["margin_angstrom"])
    if "max_points_per_dimension" in refined_cfg:
        feature_extractor.BADER_REFINED_MAX_POINTS_PER_DIM = int(refined_cfg["max_points_per_dimension"])
    if "max_total_grid_points" in refined_cfg:
        feature_extractor.BADER_REFINED_MAX_TOTAL_GRID_POINTS = int(refined_cfg["max_total_grid_points"])

    if "enabled" in rescue_cfg:
        feature_extractor.BADER_RESCUE_RETRY_ENABLED = bool(rescue_cfg["enabled"])
    if "grid_resolution_angstrom" in rescue_cfg:
        feature_extractor.BADER_RESCUE_GRID_RES_ANGSTROM = float(rescue_cfg["grid_resolution_angstrom"])
    if "margin_angstrom" in rescue_cfg:
        feature_extractor.BADER_RESCUE_MARGIN_ANGSTROM = float(rescue_cfg["margin_angstrom"])
    if "max_points_per_dimension" in rescue_cfg:
        feature_extractor.BADER_RESCUE_MAX_POINTS_PER_DIM = int(rescue_cfg["max_points_per_dimension"])
    if "max_total_grid_points" in rescue_cfg:
        feature_extractor.BADER_RESCUE_MAX_TOTAL_GRID_POINTS = int(rescue_cfg["max_total_grid_points"])


def summarize_alignment_status_counts(items: List[Dict[str, Any]] | None = None) -> Dict[str, int]:
    """Return counts of implemented_exact/proxy/partial/missing from alignment table."""
    rows = items if items is not None else QCMOL_ALIGNMENT_ITEMS
    out = {"implemented_exact": 0, "implemented_proxy": 0, "partial": 0, "missing": 0}
    for row in rows:
        status = str(row.get("status", "")).strip()
        if status in out:
            out[status] += 1
    return out

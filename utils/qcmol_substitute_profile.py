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


QCMOL_ALIGNMENT_MASTER_TABLE: List[Dict[str, Any]] = [
    # basic_information
    {
        "section": "basic_information",
        "qcMol_item_name": "qcMol_ID",
        "mapped_path": "molecule_info.molecule_id",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Regression remains green and mapping path stays stable.",
        "notes": "Canonical ID mapping is stable.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "IUPAC_name",
        "mapped_path": None,
        "current_status": "missing",
        "next_action": "roadmap_only",
        "completion_criterion": "Reliable open-source naming resolver integrated with deterministic fallback.",
        "notes": "Not blocking substitute default; requires external naming source.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "SMILES",
        "mapped_path": "molecule_info.smiles",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "SMILES roundtrip coverage remains stable.",
        "notes": "Stable canonical path.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "InChI",
        "mapped_path": "molecule_info.inchi",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "InChI generation remains deterministic for supported molecules.",
        "notes": "Stable canonical path.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "InChIKey",
        "mapped_path": "molecule_info.inchikey",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "InChIKey generation remains deterministic.",
        "notes": "Stable canonical path.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "chemical_formula",
        "mapped_path": "molecule_info.formula",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Formula consistency checks stay green.",
        "notes": "Stable canonical path.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "SMART",
        "mapped_path": "molecule_info.smarts",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Substitute semantics remain frozen as RDKit canonical SMARTS proxy with explicit non-exact note.",
        "notes": "Frozen substitute semantics; do not treat as qcMol exact SMART naming.",
    },
    {
        "section": "basic_information",
        "qcMol_item_name": "nickname_or_synonyms",
        "mapped_path": None,
        "current_status": "missing",
        "next_action": "roadmap_only",
        "completion_criterion": "Trusted synonym source integrated with deterministic cache policy.",
        "notes": "Optional metadata; not a substitute-blocking gap.",
    },
    # global_features
    {
        "section": "global_features",
        "qcMol_item_name": "HOMO_LUMO_gap",
        "mapped_path": "global_features.dft.homo_lumo_gap_hartree",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Path and unit remain frozen.",
        "notes": "Exact within current DFT stack.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "dipole_moment",
        "mapped_path": "global_features.dft.dipole_moment_debye",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Path and unit remain frozen.",
        "notes": "Exact within current DFT stack.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "isosurface_area",
        "mapped_path": "realspace_features.density_isosurface_area",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Realspace extraction success remains high and definition stable.",
        "notes": "Uses frozen realspace definition v2.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "isosurface_volume",
        "mapped_path": "realspace_features.density_isosurface_volume",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Realspace extraction success remains high and definition stable.",
        "notes": "Uses frozen realspace definition v2.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "sphericity_parameters",
        "mapped_path": "realspace_features.density_sphericity_like",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Proxy formula/version unchanged and documented.",
        "notes": "Explicitly proxy, not paper-exact sphericity.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "molecule_size",
        "mapped_path": "global_features.geometry_size.bounding_box_diagonal_angstrom",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Frozen substitute definition and metadata note remain stable for consumer interpretation.",
        "notes": "Frozen substitute semantics: geometry bounding-box diagonal; not qcMol exact size definition.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "molecular_weight",
        "mapped_path": "global_features.rdkit.molecular_weight",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Stable path/unit and parser consistency.",
        "notes": "RDKit molecular weight is stable.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "ionization_affinity_or_related",
        "mapped_path": "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Koopmans-style ionization-related proxy remains explicitly documented and reproducible.",
        "notes": "Frozen substitute semantics: ionization-related proxy from HOMO; not exact ionization affinity target.",
    },
    {
        "section": "global_features",
        "qcMol_item_name": "charge",
        "mapped_path": "molecule_info.charge",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Charge consistency checks remain green.",
        "notes": "Stable canonical path.",
    },
    # atom_features
    {
        "section": "atom_features",
        "qcMol_item_name": "element_type",
        "mapped_path": "geometry.atom_symbols",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Atom ordering/alignment remains stable.",
        "notes": "Exact path.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "XYZ",
        "mapped_path": "geometry.atom_coords_angstrom",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "Coordinate/unit convention remains frozen.",
        "notes": "Exact path.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "NAO_descriptors",
        "mapped_path": "external_bridge_roadmap.atom_level.nao_descriptors",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Open-source equivalent definition accepted or external exact route explicitly enabled.",
        "notes": "Exact NAO requires routes outside current open-source substitute constraints.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "LI_values",
        "mapped_path": "external_bridge_roadmap.atom_level.li_values",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Exact LI semantics + implementation path agreed and validated.",
        "notes": "Kept as roadmap placeholder under current constraints.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "ADCH_charges",
        "mapped_path": "external_bridge_roadmap.atom_level.adch_charges",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Exact ADCH route integrated with stable dependency policy.",
        "notes": "Out of scope for current no-Multiwfn mainline.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "NBO_LP",
        "mapped_path": "atom_features.atomic_lone_pair_heuristic_proxy",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Availability semantics remain stable and documented as heuristic.",
        "notes": "Explicitly not equivalent to exact NBO-LP.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "NPA",
        "mapped_path": "atom_features.atomic_charge_iao_proxy",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Proxy metadata and source remain stable.",
        "notes": "IAO proxy retained as open-source substitute.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "NPA_exact",
        "mapped_path": "external_bridge_roadmap.atom_level.npa_exact",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Exact NPA route and licensing/dependency policy approved.",
        "notes": "Exact NPA unavailable in current open-source-only constraints.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "density_partition_charge_proxy.hirshfeld",
        "mapped_path": "atom_features.atomic_density_partition_charge_proxy.hirshfeld",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Stable extraction and metadata semantics.",
        "notes": "Open-source density-partition companion.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "density_partition_charge_proxy.cm5",
        "mapped_path": "atom_features.atomic_density_partition_charge_proxy.cm5",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Stable extraction and metadata semantics.",
        "notes": "Open-source density-partition companion.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "density_partition_charge_proxy.bader",
        "mapped_path": "atom_features.atomic_density_partition_charge_proxy.bader",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Validated writeback rate remains >= 0.80 on default validation set.",
        "notes": "Current validated writeback ~0.90 with retry rescue path.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "density_partition_volume_proxy.bader",
        "mapped_path": "atom_features.atomic_density_partition_volume_proxy.bader",
        "current_status": "partial",
        "next_action": "upgrade",
        "completion_criterion": "Volume status reasons are explicit (missing/non-numeric/all-null/length-mismatch) and validated usable-rate reaches stable target when upstream supports volume output.",
        "notes": "Current primary blocker is critic2 volume column stability/availability, not bridge execution.",
    },
    {
        "section": "atom_features",
        "qcMol_item_name": "atomic_orbital_descriptor_proxy_v1",
        "mapped_path": "atom_features.atomic_orbital_descriptor_proxy_v1",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Orbital availability semantics and fixed-shape validation remain stable.",
        "notes": "Proxy descriptor family retained for open-source route.",
    },
    # bond_features
    {
        "section": "bond_features",
        "qcMol_item_name": "stereo_info",
        "mapped_path": "bond_features.bond_stereo_info",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Enum and mapping semantics remain stable.",
        "notes": "RDKit perception proxy.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "DI_values_or_matrix",
        "mapped_path": "bond_features.bond_delocalization_index_proxy_v1",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Substitute-only DI semantics remain frozen with explicit non-exact metadata contract.",
        "notes": "Frozen DI substitute-only path; not an exact DI matrix.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "ELF_values",
        "mapped_path": "bond_features.elf_bond_midpoint",
        "current_status": "implemented_exact",
        "next_action": "keep",
        "completion_criterion": "ELF extraction/alignment checks remain stable.",
        "notes": "Exact within current extraction definition.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "NBO_BD",
        "mapped_path": "external_bridge_roadmap.bond_level.nbo_bd",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Exact NBO BD route explicitly enabled and validated.",
        "notes": "Unavailable under no-NBO constraint.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "LBO",
        "mapped_path": "external_bridge_roadmap.bond_level.lbo",
        "current_status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "completion_criterion": "Exact LBO route explicitly enabled and validated.",
        "notes": "Unavailable under no-Multiwfn/NBO constraint.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "Mayer_BL",
        "mapped_path": "bond_features.bond_orders_mayer",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Mayer substitute metadata remains frozen with explicit alignment to bond_indices order.",
        "notes": "Frozen substitute semantics for qcMol Mayer_BL (PySCF Mayer bond-order vector).",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "bond_orbital_localization_proxy",
        "mapped_path": "bond_features.bond_orbital_localization_proxy",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Availability semantics and shape validation remain stable.",
        "notes": "IBO-derived localization proxy.",
    },
    {
        "section": "bond_features",
        "qcMol_item_name": "bond_order_weighted_localization_proxy",
        "mapped_path": "bond_features.bond_order_weighted_localization_proxy",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Availability semantics and shape validation remain stable.",
        "notes": "Composite proxy from localization and DI proxy.",
    },
    # structural_features
    {
        "section": "structural_features",
        "qcMol_item_name": "optimized_3D_geometry",
        "mapped_path": "structural_features.optimized_3d_geometry",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Semantic-reference contract stays explicit (source-of-truth path + geometry fingerprint + natm reference).",
        "notes": "Frozen semantic-reference representation; coordinates remain canonical in geometry.atom_coords_angstrom.",
    },
    {
        "section": "structural_features",
        "qcMol_item_name": "most_stable_conformation",
        "mapped_path": "structural_features.most_stable_conformation",
        "current_status": "implemented_proxy",
        "next_action": "keep",
        "completion_criterion": "Candidate-set metadata and reproducibility fields stay stable.",
        "notes": "Proxy scope is candidate-set minimum, not global minimum proof.",
    },
]


QCMOL_ALIGNMENT_ITEMS: List[Dict[str, Any]] = [
    {
        "section": row["section"],
        "name": row["qcMol_item_name"],
        "mapped_path": row["mapped_path"],
        "status": row["current_status"],
    }
    for row in QCMOL_ALIGNMENT_MASTER_TABLE
]


CANONICAL_SURFACE_ITEMS: List[Dict[str, Any]] = [
    {"group": "basic", "path": "molecule_info.molecule_id", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.smiles", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.inchi", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.inchikey", "status": "implemented_exact"},
    {"group": "basic", "path": "molecule_info.formula", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.total_energy_hartree", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.homo_energy_hartree", "status": "implemented_exact"},
    {"group": "global", "path": "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree", "status": "implemented_proxy"},
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
    {"group": "atom", "path": "atom_features.atomic_density_partition_laplacian_proxy_v1.bader", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_lone_pair_heuristic_proxy", "status": "implemented_proxy"},
    {"group": "atom", "path": "atom_features.atomic_orbital_descriptor_proxy_v1", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_indices", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.bond_orders_mayer", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_orders_wiberg", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.elf_bond_midpoint", "status": "implemented_exact"},
    {"group": "bond", "path": "bond_features.bond_delocalization_index_proxy_v1", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_orbital_localization_proxy", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_order_weighted_localization_proxy", "status": "implemented_proxy"},
    {"group": "bond", "path": "bond_features.bond_stereo_info", "status": "implemented_proxy"},
    {"group": "structural", "path": "structural_features.optimized_3d_geometry", "status": "implemented_proxy"},
    {"group": "structural", "path": "structural_features.most_stable_conformation", "status": "implemented_proxy"},
    {"group": "structural", "path": "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1", "status": "implemented_proxy"},
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
    {"group": "global", "path": "global_features.proxy_family_summary_v1", "status": "implemented_proxy"},
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


CONSUMER_MINIMAL_FIELDS: List[str] = [
    "molecule_info.molecule_id",
    "molecule_info.smiles",
    "molecule_info.formula",
    "molecule_info.charge",
    "global_features.dft.total_energy_hartree",
    "global_features.dft.homo_lumo_gap_hartree",
    "global_features.dft.dipole_moment_debye",
    "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree",
    "global_features.rdkit.molecular_weight",
    "global_features.geometry_size.bounding_box_diagonal_angstrom",
    "atom_features.atomic_charge_iao_proxy",
    "atom_features.atomic_density_partition_charge_proxy.hirshfeld",
    "atom_features.atomic_density_partition_charge_proxy.cm5",
    "bond_features.bond_indices",
    "bond_features.bond_orders_mayer",
    "bond_features.bond_delocalization_index_proxy_v1",
]


CONSUMER_ENHANCED_FIELDS: List[str] = CONSUMER_MINIMAL_FIELDS + [
    "atom_features.atomic_density_partition_charge_proxy.bader",
    "atom_features.atomic_density_partition_laplacian_proxy_v1.bader",
    "atom_features.atomic_lone_pair_heuristic_proxy",
    "atom_features.atomic_orbital_descriptor_proxy_v1",
    "bond_features.bond_orbital_localization_proxy",
    "bond_features.bond_order_weighted_localization_proxy",
    "global_features.proxy_family_summary_v1",
    "global_features.basin_proxy_summary_v1",
    "structural_features.most_stable_conformation.candidate_set_statistics_proxy_v1",
    "realspace_features.density_isosurface_area",
    "realspace_features.density_isosurface_volume",
    "realspace_features.density_sphericity_like",
    "realspace_features.orbital_extent_homo",
    "realspace_features.orbital_extent_lumo",
    "external_bridge.critic2",
]


CONSUMER_CAUTION_FIELDS: List[str] = [
    "atom_features.atomic_density_partition_volume_proxy.bader",
    "external_features.critic2.qtaim.bader_volumes",
    "external_bridge_roadmap.atom_level.nao_descriptors",
    "external_bridge_roadmap.atom_level.li_values",
    "external_bridge_roadmap.atom_level.adch_charges",
    "external_bridge_roadmap.atom_level.npa_exact",
    "external_bridge_roadmap.bond_level.nbo_bd",
    "external_bridge_roadmap.bond_level.lbo",
]


CANONICAL_SURFACE_TRAFFIC_LIGHT: List[Dict[str, Any]] = [
    {
        "path": "molecule_info.molecule_id",
        "status": "implemented_exact",
        "tier": "green",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Primary identity field.",
    },
    {
        "path": "molecule_info.smiles",
        "status": "implemented_exact",
        "tier": "green",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Canonical structure string for most downstream joins.",
    },
    {
        "path": "global_features.dft.homo_lumo_gap_hartree",
        "status": "implemented_exact",
        "tier": "green",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Core electronic scalar.",
    },
    {
        "path": "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree",
        "status": "implemented_proxy",
        "tier": "yellow",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Frozen substitute semantics (Koopmans-style proxy).",
    },
    {
        "path": "global_features.geometry_size.bounding_box_diagonal_angstrom",
        "status": "implemented_proxy",
        "tier": "yellow",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Frozen substitute size definition.",
    },
    {
        "path": "atom_features.atomic_density_partition_charge_proxy.bader",
        "status": "implemented_proxy",
        "tier": "yellow",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Validation-gated; good coverage under default profile.",
    },
    {
        "path": "atom_features.atomic_density_partition_volume_proxy.bader",
        "status": "partial",
        "tier": "red",
        "stability": "caution",
        "recommended_default_dependency": False,
        "notes": "Known upstream critic2 limitation (volume column often missing).",
    },
    {
        "path": "bond_features.bond_orders_mayer",
        "status": "implemented_proxy",
        "tier": "yellow",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Frozen substitute for qcMol Mayer_BL semantics.",
    },
    {
        "path": "structural_features.optimized_3d_geometry",
        "status": "implemented_proxy",
        "tier": "yellow",
        "stability": "stable",
        "recommended_default_dependency": True,
        "notes": "Reference-only geometry contract to canonical geometry block.",
    },
    {
        "path": "external_bridge_roadmap.atom_level.nao_descriptors",
        "status": "rejected_as_exact",
        "tier": "red",
        "stability": "optional",
        "recommended_default_dependency": False,
        "notes": "Exact-only archived family.",
    },
]


EXACT_ONLY_ARCHIVED_LIST: List[Dict[str, str]] = [
    {
        "qcMol_item_name": "NAO_descriptors",
        "mapped_path": "external_bridge_roadmap.atom_level.nao_descriptors",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "exact-only family outside current open-source substitute boundary (no NBO/Multiwfn route).",
    },
    {
        "qcMol_item_name": "LI_values",
        "mapped_path": "external_bridge_roadmap.atom_level.li_values",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "exact-only family outside current open-source substitute boundary.",
    },
    {
        "qcMol_item_name": "ADCH_charges",
        "mapped_path": "external_bridge_roadmap.atom_level.adch_charges",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "exact ADCH route requires excluded dependencies/toolchains.",
    },
    {
        "qcMol_item_name": "NPA_exact",
        "mapped_path": "external_bridge_roadmap.atom_level.npa_exact",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "exact NPA route is out-of-scope under open-source-only constraints.",
    },
    {
        "qcMol_item_name": "NBO_BD",
        "mapped_path": "external_bridge_roadmap.bond_level.nbo_bd",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "NBO exact family intentionally archived under no-NBO policy.",
    },
    {
        "qcMol_item_name": "LBO",
        "mapped_path": "external_bridge_roadmap.bond_level.lbo",
        "status": "rejected_as_exact",
        "next_action": "roadmap_only",
        "archive_reason": "exact LBO route intentionally archived under no-Multiwfn/NBO policy.",
    },
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
    """Return counts by status from alignment table."""
    rows = items if items is not None else QCMOL_ALIGNMENT_ITEMS
    out = {
        "implemented_exact": 0,
        "implemented_proxy": 0,
        "partial": 0,
        "missing": 0,
        "rejected_as_exact": 0,
    }
    for row in rows:
        status = str(row.get("status", row.get("current_status", ""))).strip()
        if status in out:
            out[status] += 1
    return out


def summarize_alignment_next_actions(master_table: List[Dict[str, Any]] | None = None) -> Dict[str, int]:
    """Count next_action values in master table."""
    rows = master_table if master_table is not None else QCMOL_ALIGNMENT_MASTER_TABLE
    out = {"keep": 0, "upgrade": 0, "redefine": 0, "roadmap_only": 0, "reject": 0}
    for row in rows:
        action = str(row.get("next_action", "")).strip()
        if action in out:
            out[action] += 1
    return out


def remaining_alignment_gaps(master_table: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """Return rows that are not yet implemented_exact/implemented_proxy keep-state."""
    rows = master_table if master_table is not None else QCMOL_ALIGNMENT_MASTER_TABLE
    out: List[Dict[str, Any]] = []
    for row in rows:
        status = str(row.get("current_status", "")).strip()
        action = str(row.get("next_action", "")).strip()
        if status in {"implemented_exact", "implemented_proxy"} and action == "keep":
            continue
        out.append(row)
    return out

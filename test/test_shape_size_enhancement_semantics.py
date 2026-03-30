import numpy as np

from analysis.realspace_features import BOHR_TO_ANGSTROM, RealspaceFeatureExtractor
from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder


class _FakeMol:
    def __init__(self, coords_ang):
        self._coords = np.array(coords_ang, dtype=float)

    def atom_coords(self, unit="A"):
        assert unit == "A"
        return self._coords


class _FakeRingInfo:
    def __init__(self, n_rings):
        self._n_rings = int(n_rings)

    def NumRings(self):
        return self._n_rings


class _FakeRdkitMol:
    def __init__(self, n_bonds, n_rings):
        self._n_bonds = int(n_bonds)
        self._ring_info = _FakeRingInfo(n_rings)

    def GetNumBonds(self):
        return self._n_bonds

    def GetRingInfo(self):
        return self._ring_info


def test_geometry_size_family_updates_with_source_priority():
    fx = FeatureExtractor()
    builder = UnifiedOutputBuilder(molecule_id="m_test", smiles="CCO")
    builder.set_molecule_info(natm=3)
    builder.set_geometry(atom_symbols=["C", "C", "O"])
    builder.set_global_rdkit(heavy_atom_count=3)

    mol = _FakeMol([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0]])
    rdkit = _FakeRdkitMol(n_bonds=2, n_rings=0)
    fx._update_geometry_size_family(builder=builder, mol=mol, mol_rdkit=rdkit, bond_indices=[[0, 1], [1, 2]])

    gsize = builder.data["global_features"]["geometry_size"]
    gmeta = builder.data["global_features"]["metadata"]

    assert gsize["bounding_box_diagonal_angstrom"] > 0.0
    assert gsize["radius_of_gyration_angstrom"] > 0.0
    assert gsize["heavy_atom_count_proxy"] == 3
    assert gsize["total_atom_count_proxy"] == 3
    assert gsize["num_bonds_proxy"] == 2
    assert gsize["num_rings_proxy"] == 0
    assert gmeta["molecule_size_num_bonds_proxy"]["source"] == "len(bond_features.bond_indices)"
    assert gmeta["molecule_size_radius_of_gyration_angstrom"]["definition_version"] == "v2"
    assert gmeta["molecule_size_total_atom_count_proxy"]["definition_version"] == "v2"


def test_density_shape_descriptor_family_v1_success_case():
    ex = RealspaceFeatureExtractor(config={"density_shape_mass_cutoff": 0.95})
    # 2x2x2 grid in bohr
    coords_bohr = np.array(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]],
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]],
        ],
        dtype=float,
    )
    rho = np.array(
        [
            [[0.8, 0.4], [0.2, 0.1]],
            [[0.5, 0.3], [0.15, 0.05]],
        ],
        dtype=float,
    )

    fam = ex._compute_density_shape_descriptor_family(rho_3d=rho, coords_3d_bohr=coords_bohr, mass_cutoff=0.95, eps=1e-12)
    assert fam["status"] == "success"
    assert fam["n_points_total"] == 8
    assert fam["n_points_selected"] >= 1
    assert fam["selected_density_fraction"] > 0.0
    assert len(fam["center_of_mass_angstrom"]) == 3
    assert len(fam["eigenvalues_raw"]) == 3
    assert fam["eigenvalues_raw"][0] >= fam["eigenvalues_raw"][1] >= fam["eigenvalues_raw"][2] >= 0.0
    assert len(fam["eigenvalues_normalized"]) == 3
    assert abs(sum(fam["eigenvalues_normalized"]) - 1.0) < 1e-6
    assert len(fam["shape_tensor"]) == 3
    assert len(fam["shape_tensor"][0]) == 3
    assert 0.0 <= fam["relative_anisotropy_kappa2"] <= 1.0
    assert 0.0 <= fam["sphericity"] <= 1.0
    assert fam["center_of_mass_angstrom"][0] >= 0.0 * BOHR_TO_ANGSTROM


def test_density_shape_descriptor_family_v1_density_all_zero():
    ex = RealspaceFeatureExtractor()
    coords_bohr = np.zeros((2, 2, 2, 3), dtype=float)
    rho = np.zeros((2, 2, 2), dtype=float)
    fam = ex._compute_density_shape_descriptor_family(rho_3d=rho, coords_3d_bohr=coords_bohr)
    assert fam["status"] == "unavailable"
    assert fam["status_reason"] == "density_all_zero"


def test_density_shape_multiscale_family_v1_monotonic_selected_points():
    ex = RealspaceFeatureExtractor(config={"density_shape_mass_cutoff": 0.95, "density_shape_multiscale_cutoffs": [0.5, 0.9, 0.95]})
    coords_bohr = np.array(
        [
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]],
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]],
        ],
        dtype=float,
    )
    rho = np.array(
        [
            [[0.8, 0.4], [0.2, 0.1]],
            [[0.5, 0.3], [0.15, 0.05]],
        ],
        dtype=float,
    )

    fam = ex._compute_density_shape_multiscale_family(
        rho_3d=rho,
        coords_3d_bohr=coords_bohr,
        scales=[0.5, 0.9, 0.95],
        default_scale=0.95,
        eps=1e-12,
    )
    assert fam["status"] in {"success", "partial"}
    assert fam["default_scale_key"] == "0.95"
    assert "0.50" in fam["scales"] and "0.90" in fam["scales"] and "0.95" in fam["scales"]

    s50 = fam["scales"]["0.50"]
    s90 = fam["scales"]["0.90"]
    s95 = fam["scales"]["0.95"]
    assert s50["status"] == "success"
    assert s90["status"] == "success"
    assert s95["status"] == "success"
    assert s50["n_points_selected"] <= s90["n_points_selected"] <= s95["n_points_selected"]

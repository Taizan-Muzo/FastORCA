from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder


def test_qcmol_alignment_redefine_items_are_frozen_to_proxy_semantics():
    builder = UnifiedOutputBuilder(molecule_id="m_test", smiles="CCO")
    align = builder.data["qcmol_alignment"]

    assert align["basic_information"]["SMART"]["status"] == "implemented_proxy"
    assert align["global_features"]["molecule_size"]["status"] == "implemented_proxy"
    assert align["global_features"]["ionization_affinity_or_related"]["status"] == "implemented_proxy"
    assert (
        align["global_features"]["ionization_affinity_or_related"]["mapped_path"]
        == "global_features.dft.ionization_related_proxy_v1.koopmans_ip_proxy_hartree"
    )
    assert align["bond_features"]["DI_values_or_matrix"]["status"] == "implemented_proxy"


def test_upgrade_items_mayer_and_optimized_geometry_have_frozen_proxy_status():
    builder = UnifiedOutputBuilder(molecule_id="m_test", smiles="CCO")
    align = builder.data["qcmol_alignment"]

    assert align["bond_features"]["Mayer_BL"]["status"] == "implemented_proxy"
    assert align["structural_features"]["optimized_3D_geometry"]["status"] == "implemented_proxy"
    assert "bond_orders_mayer" in builder.data["bond_features"]["metadata"]
    assert "atomic_density_partition_volume_proxy" in builder.data["atom_features"]["metadata"]


def test_ionization_related_proxy_v1_builder_from_homo_energy():
    fx = FeatureExtractor()
    proxy = fx._build_ionization_related_proxy_v1(-0.2)
    assert proxy["available"] is True
    assert proxy["status"] == "success"
    assert abs(proxy["koopmans_ip_proxy_hartree"] - 0.2) < 1e-12
    assert proxy["koopmans_ip_proxy_ev"] > 0.0


from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder


def _make_builder() -> UnifiedOutputBuilder:
    builder = UnifiedOutputBuilder(molecule_id="m_test", smiles="CCO")
    builder.set_molecule_info(natm=3, charge=0)
    return builder


def test_set_orbital_proxy_availability_skipped():
    fx = FeatureExtractor()
    builder = _make_builder()

    fx._set_orbital_proxy_availability(
        builder=builder,
        availability_status="skipped",
        status_reason="upstream_orbital_skipped:smoke_skip",
        upstream_orbital_extraction_status="skipped",
        skip_reason="smoke_skip",
        failure_reason=None,
    )

    atom_lp_meta = builder.data["atom_features"]["metadata"]["atomic_lone_pair_heuristic_proxy"]
    atom_desc_meta = builder.data["atom_features"]["metadata"]["atomic_orbital_descriptor_proxy_v1"]
    bond_loc_meta = builder.data["bond_features"]["metadata"]["bond_orbital_localization_proxy"]
    bond_wloc_meta = builder.data["bond_features"]["metadata"]["bond_order_weighted_localization_proxy"]

    assert atom_lp_meta["availability_status"] == "skipped"
    assert atom_desc_meta["availability_status"] == "skipped"
    assert bond_loc_meta["availability_status"] == "skipped"
    assert bond_wloc_meta["availability_status"] == "skipped"
    assert atom_lp_meta["skip_reason"] == "smoke_skip"
    assert bond_loc_meta["status_reason"].startswith("upstream_orbital_skipped:")


def test_orbital_status_normalization():
    fx = FeatureExtractor()
    assert fx._normalize_availability_status("success") == "success"
    assert fx._normalize_availability_status("failed") == "unavailable"
    assert fx._normalize_availability_status("timeout") == "unavailable"
    assert fx._normalize_availability_status("disabled") == "skipped"
    assert fx._normalize_availability_status(None) == "not_attempted"


def test_pick_external_integrated_property_values_prefers_alias():
    fx = FeatureExtractor()
    values, key = fx._pick_external_integrated_property_values(
        property_map={"Lap": [-0.1, -0.2], "Pop": [5.9, 8.1], "Volume": [11.0, 12.0]},
        aliases=["Population", "Pop"],
    )
    assert key == "Pop"
    assert values == [5.9, 8.1]

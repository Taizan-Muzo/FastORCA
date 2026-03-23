from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder


def _make_builder(natm: int = 3) -> UnifiedOutputBuilder:
    builder = UnifiedOutputBuilder(molecule_id="m_test", smiles="CCO")
    builder.set_molecule_info(natm=natm, charge=0)
    atomic_numbers = [6, 8, 1][:natm]
    builder.set_atom_features(
        atomic_number=atomic_numbers,
        atomic_density_partition_charge_proxy={
            "hirshfeld": [0.0] * natm,
            "cm5": [0.0] * natm,
            "bader": None,
        },
        atomic_density_partition_volume_proxy={"bader": None},
    )
    return builder


def test_bader_proxy_not_attempted_from_skipped_bridge():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="skipped", failure_reason="no_density_cube_available")

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "not_attempted"
    assert meta["bader_volume_status"] == "not_attempted"
    assert builder.data["atom_features"]["atomic_density_partition_charge_proxy"]["bader"] is None
    assert builder.data["atom_features"]["atomic_density_partition_volume_proxy"]["bader"] is None


def test_bader_proxy_success_for_charge_and_volume():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="success", failure_reason=None)
    builder.set_external_features(
        "critic2",
        {
            "qtaim": {
                "bader_populations": [5.9, 8.2, 0.9],
                "bader_volumes": [12.0, 8.0, 6.0],
                "n_bader_volumes": 3,
            }
        },
    )

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "success"
    assert meta["bader_volume_status"] == "success"
    assert builder.data["atom_features"]["atomic_density_partition_charge_proxy"]["bader"] == [0.1, -0.2, 0.1]
    assert builder.data["atom_features"]["atomic_density_partition_volume_proxy"]["bader"] == [12.0, 8.0, 6.0]


def test_bader_proxy_unavailable_on_length_mismatch():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="success", failure_reason=None)
    builder.set_external_features(
        "critic2",
        {
            "qtaim": {
                "bader_charges": [0.1, -0.1],  # len mismatch
                "bader_volumes": [12.0, 7.5],
                "n_bader_volumes": 2,
            }
        },
    )

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "unavailable"
    assert meta["bader_volume_status"] == "unavailable"
    assert "length_mismatch" in meta["bader_status_reason"]
    assert "length_mismatch" in meta["bader_volume_status_reason"]
    assert builder.data["atom_features"]["atomic_density_partition_charge_proxy"]["bader"] is None
    assert builder.data["atom_features"]["atomic_density_partition_volume_proxy"]["bader"] is None


def test_bader_volume_all_null_is_unavailable_even_when_bridge_success():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="success", failure_reason=None)
    builder.set_external_features(
        "critic2",
        {
            "qtaim": {
                "bader_populations": [5.9, 8.2, 0.9],
                "bader_volumes": [None, None, None],
                "n_bader_volumes": 3,
            }
        },
    )

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "success"
    assert meta["bader_volume_status"] == "unavailable"
    assert "all_null" in meta["bader_volume_status_reason"]
    assert builder.data["atom_features"]["atomic_density_partition_volume_proxy"]["bader"] is None


def test_bader_population_sum_mismatch_marks_charge_unavailable():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="success", failure_reason=None)
    builder.set_external_features(
        "critic2",
        {
            "qtaim": {
                # Sum(pop)=20.0 while expected electrons for C-O-H neutral is 15.0
                "bader_populations": [5.0, 14.0, 1.0],
                "bader_volumes": [12.0, 8.0, 6.0],
                "n_bader_volumes": 3,
            }
        },
    )

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "unavailable"
    assert "population_sum_mismatch" in meta["bader_status_reason"]
    assert meta["bader_volume_status"] == "success"
    assert builder.data["atom_features"]["atomic_density_partition_charge_proxy"]["bader"] is None

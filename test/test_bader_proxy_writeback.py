from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder
from analysis.external.adapters.critic2_adapter import Critic2Adapter


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
    charges = builder.data["atom_features"]["atomic_density_partition_charge_proxy"]["bader"]
    assert [round(x, 6) for x in charges] == [0.1, -0.2, 0.1]
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


def test_bader_volume_missing_after_success_has_specific_reason():
    fx = FeatureExtractor()
    builder = _make_builder(natm=3)
    builder.set_external_bridge("critic2", execution_status="success", failure_reason=None)
    builder.set_external_features(
        "critic2",
        {
            "qtaim": {
                "bader_populations": [5.9, 8.2, 0.9],
                "bader_volumes": None,
                "metadata": {
                    "atomic_property_parse_note": "integrated_atomic_properties_parsed_without_volume_column",
                },
            }
        },
    )

    fx._sync_bader_partition_proxy_from_external(builder, "m_test")
    meta = builder.data["atom_features"]["metadata"]["atomic_density_partition_charge_proxy"]
    assert meta["bader_status"] == "success"
    assert meta["bader_volume_status"] == "unavailable"
    assert "not_reported" in meta["bader_volume_status_reason"]
    assert builder.data["atom_features"]["atomic_density_partition_volume_proxy"]["bader"] is None


def test_integrated_atomic_property_parser_prefers_pop_column():
    adapter = Critic2Adapter()
    content = """
* Integrated atomic properties
# Integrable properties in this table: Pop Lap
# Id cp ncp Name Z mult Pop Lap
  1  1  1  C  6  1  5.900000  -0.111000
  2  2  2  O  8  1  8.200000  -0.222000
  3  3  3  H  1  1  0.900000  -0.333000

* Integrated molecular properties
""".strip("\n")

    charges, populations, volumes, parse_meta, integrated = adapter._parse_integrated_atomic_properties(content)

    assert populations == [5.9, 8.2, 0.9]
    assert [round(x, 6) for x in charges] == [0.1, -0.2, 0.1]
    assert volumes is None
    assert integrated is not None
    assert integrated["Pop"] == [5.9, 8.2, 0.9]
    assert integrated["Lap"] == [-0.111, -0.222, -0.333]
    assert parse_meta["atomic_property_pop_column"] == "Pop"
    assert parse_meta["atomic_property_volume_available"] is False

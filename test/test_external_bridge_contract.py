import copy

from scripts.check_external_bridge_contract import validate_unified_contract


def _valid_base():
    return {
        "external_bridge": {
            "critic2": {
                "execution_status": "success",
                "failure_reason": None,
                "metadata": {
                    "tool_version": "1.0",
                    "execution_time_seconds": 1.23,
                    "command": None,
                    "parser_version": "v1",
                    "environment": None,
                },
                "artifact_refs": {
                    "input_file": "/tmp/in",
                    "output_file": "/tmp/out",
                    "stdout_file": None,
                    "stderr_file": None,
                },
                "warnings": [],
            }
        },
        "external_features": {
            "critic2": {
                "qtaim": {
                    "bader_charges": [0.1, -0.1],
                    "n_bader_volumes": 2,
                }
            }
        },
        "external_bridge_roadmap": {
            "atom_level": {
                "nao_descriptors": {
                    "status": "missing",
                    "payload": None,
                    "needs_exact_qcmol_name": True,
                    "notes": None,
                }
            },
            "bond_level": {
                "di_values_or_matrix": {
                    "status": "placeholder",
                    "payload": None,
                    "needs_exact_qcmol_name": True,
                    "notes": "todo",
                }
            },
        },
    }


def _errors(violations):
    return [v for v in violations if v.get("severity") == "error"]


def _warnings(violations):
    return [v for v in violations if v.get("severity") == "warning"]


def test_valid_success_case():
    data = _valid_base()
    errs, warns = validate_unified_contract(data, "inmem.json")
    assert not errs
    assert not warns


def test_valid_skipped_case():
    data = _valid_base()
    data["external_bridge"]["critic2"]["execution_status"] = "skipped"
    data["external_bridge"]["critic2"]["failure_reason"] = "disabled_by_plan"
    data["external_features"]["critic2"] = {}
    errs, warns = validate_unified_contract(data, "inmem.json")
    assert not errs
    assert not warns


def test_invalid_execution_status():
    data = _valid_base()
    data["external_bridge"]["critic2"]["execution_status"] = "done"
    errs, _ = validate_unified_contract(data, "inmem.json")
    assert any(v["rule_id"] == "RULE_1" for v in errs)


def test_invalid_roadmap_status():
    data = _valid_base()
    data["external_bridge_roadmap"]["atom_level"]["nao_descriptors"]["status"] = "available"
    errs, _ = validate_unified_contract(data, "inmem.json")
    assert any(v["rule_id"] == "RULE_6" for v in errs)


def test_external_features_mixed_execution_field():
    data = _valid_base()
    data["external_features"]["critic2"]["execution_status"] = "success"
    errs, _ = validate_unified_contract(data, "inmem.json")
    assert any(v["rule_id"] == "RULE_3" for v in errs)


def test_roadmap_legacy_value_field_is_violation():
    data = _valid_base()
    ph = data["external_bridge_roadmap"]["atom_level"]["nao_descriptors"]
    ph["value"] = None
    errs, _ = validate_unified_contract(data, "inmem.json")
    assert any(v["rule_id"] == "RULE_5" and "forbidden fields" in v["message"] for v in errs)


def test_only_deprecated_bridge_fields_is_fail():
    data = _valid_base()
    data["external_bridge"]["critic2"] = {
        "execution_status": "success",
        "failure_reason": None,
        "input_file": "/tmp/in",
        "output_file": "/tmp/out",
        "execution_time_seconds": 1.0,
        "critic2_version": "1.0",
    }
    errs, _ = validate_unified_contract(data, "inmem.json")
    assert any(v["rule_id"] == "RULE_7" for v in errs)


def test_deprecated_plus_new_fields_is_warning_not_fail():
    data = _valid_base()
    tool = copy.deepcopy(data["external_bridge"]["critic2"])
    tool.update(
        {
            "input_file": "/tmp/in",
            "output_file": "/tmp/out",
            "execution_time_seconds": 1.0,
            "critic2_version": "1.0",
        }
    )
    data["external_bridge"]["critic2"] = tool
    errs, warns = validate_unified_contract(data, "inmem.json")
    assert not errs
    assert any(w["rule_id"] == "RULE_7" for w in warns)

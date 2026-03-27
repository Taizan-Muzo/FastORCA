from utils.policy.status_determiner import StatusDeterminer


def _build_success_data(smiles=None, invalid_input_flag=False):
    return {
        "molecule_info": {
            "molecule_id": "m_from_unified",
            "smiles": smiles,
            "natm": 2,
            "charge": 0,
        },
        "calculation_status": {
            "invalid_input": invalid_input_flag,
            "wavefunction_load_success": True,
            "scf_convergence_success": True,
            "core_features_success": True,
            "geometry_optimization_success": True,
        },
        "global_features": {
            "dft": {
                "scf_converged": True,
                "total_energy_hartree": -1.234,
                "homo_energy_hartree": -0.2,
                "lumo_energy_hartree": 0.1,
                "dipole_moment_debye": 1.0,
            }
        },
        "geometry": {
            "atom_symbols": ["H", "H"],
            "atom_coords_angstrom": [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]],
        },
        "atom_features": {
            "charge_mulliken": [0.0, 0.0],
            "charge_hirshfeld": [0.0, 0.0],
            "charge_iao": [0.0, 0.0],
            "charge_cm5": [0.0, 0.0],
            "elf_value": [1.0, 1.0],
        },
        "bond_features": {
            "bond_orders_wiberg": [0.9],
            "bond_orders_mayer": [1.0],
        },
        "orbital_features": {
            "metadata": {
                "extraction_status": "success",
                "failure_reason": None,
            }
        },
        "realspace_features": {
            "metadata": {
                "extraction_status": "success",
                "failure_reason": None,
            }
        },
        "external_bridge": {
            "critic2": {
                "execution_status": "success",
                "failure_reason": None,
            }
        },
        "plugin_status": {
            "pyscf": {"success": True},
        },
    }


def test_invalid_smiles_raw_input_is_invalid_input():
    data = {
        "molecule_info": {"molecule_id": "bad_smiles", "smiles": "C1(CC", "natm": 0},
        "calculation_status": {"invalid_input": True, "wavefunction_load_success": False},
        "global_features": {"dft": {"scf_converged": False}},
    }
    det = StatusDeterminer(data)
    assert det.determine() == "invalid_input"
    assert "invalid_smiles" in det.get_reason_codes()


def test_unified_dir_like_success_without_smiles_is_not_invalid_input():
    det = StatusDeterminer(_build_success_data(smiles=None, invalid_input_flag=False))
    assert det.determine() == "fully_success"
    assert "invalid_smiles" not in det.get_reason_codes()


def test_success_evidence_overrides_stale_invalid_input_flag():
    det = StatusDeterminer(_build_success_data(smiles=None, invalid_input_flag=True))
    assert det.determine() == "fully_success"
    assert "invalid_smiles" not in det.get_reason_codes()


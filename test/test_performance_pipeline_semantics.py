import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "loguru" not in sys.modules:
    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["loguru"] = types.SimpleNamespace(logger=_DummyLogger())

if "consumer.feature_extractor" not in sys.modules:
    sys.modules["consumer.feature_extractor"] = types.SimpleNamespace(FeatureExtractor=object)


from consumer.batch_runner import BatchRunner
from consumer.molecule_processor import MoleculeProcessor, MoleculeProcessorConfig
from utils.policy.status_determiner import MoleculeResult


def _minimal_success_unified(molecule_id: str) -> dict:
    return {
        "molecule_info": {
            "molecule_id": molecule_id,
            "natm": 3,
        },
        "calculation_status": {
            "invalid_input": False,
            "wavefunction_load_success": True,
            "geometry_optimization_success": True,
            "scf_convergence_success": True,
            "core_features_success": True,
        },
        "global_features": {
            "dft": {
                "total_energy_hartree": -1.234,
                "scf_converged": True,
                "homo_energy_hartree": -0.2,
                "lumo_energy_hartree": 0.1,
                "dipole_moment_debye": 1.2,
            }
        },
        "geometry": {
            "atom_symbols": ["C", "H", "H"],
            "atom_coords_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        },
        "atom_features": {
            "charge_mulliken": [0.1, -0.05, -0.05],
            "charge_hirshfeld": [0.1, -0.05, -0.05],
            "charge_iao": [0.1, -0.05, -0.05],
            "elf_value": [0.5, 0.4, 0.4],
        },
        "bond_features": {
            "bond_indices": [[0, 1], [0, 2]],
            "bond_orders_mayer": [1.0, 1.0],
            "bond_orders_wiberg": [1.0, 1.0],
            "elf_bond_midpoint": [0.7, 0.7],
        },
        "orbital_features": {
            "metadata": {
                "extraction_status": "success",
            }
        },
        "realspace_features": {
            "metadata": {
                "extraction_status": "success",
            }
        },
        "external_bridge": {
            "critic2": {
                "execution_status": "success",
            }
        },
        "runtime_metadata": {
            "unified_extraction_time_seconds": 1.23,
            "stage_timing_seconds": {
                "wavefunction_load_seconds": 0.1,
                "core_pre_orbital_feature_seconds": 0.3,
            },
        },
        "provenance": {
            "wall_time_seconds": 1.23,
        },
    }


class _FakePluginRegistry:
    def get_all_plugin_status(self, run_mode, natm):
        return {
            "orbital_features": {"should_execute": True},
            "realspace_features": {"should_execute": True},
            "critic2_bridge": {"should_execute": True},
        }


class _FakeArtifactManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def cleanup_molecule(self, molecule_id, overall_status):
        return {"molecule_id": molecule_id, "overall_status": overall_status}


class _FakeExtractor:
    use_multiwfn = False
    multiwfn_path = "Multiwfn"
    output_format = "json"
    BADER_POPULATION_SUM_ABS_TOL_E = 0.5
    BADER_POPULATION_SUM_REL_TOL = 0.02
    BADER_REFINED_RETRY_ENABLED = True
    BADER_REFINED_GRID_RES_ANGSTROM = 0.16
    BADER_REFINED_MARGIN_ANGSTROM = 5.5
    BADER_REFINED_MAX_POINTS_PER_DIM = 180
    BADER_REFINED_MAX_TOTAL_GRID_POINTS = 3000000
    BADER_RESCUE_RETRY_ENABLED = True
    BADER_RESCUE_GRID_RES_ANGSTROM = 0.14
    BADER_RESCUE_MARGIN_ANGSTROM = 6.0
    BADER_RESCUE_MAX_POINTS_PER_DIM = 220
    BADER_RESCUE_MAX_TOTAL_GRID_POINTS = 5000000

    def __init__(self):
        self.load_calls = 0
        self.extract_calls = 0
        self.last_loaded_wavefunction = None
        self.saved_paths = []

    def load_wavefunction(self, pkl_path):
        self.load_calls += 1
        mol = types.SimpleNamespace(natm=3)
        mf = object()
        return mol, mf

    def extract_unified(self, **kwargs):
        self.extract_calls += 1
        self.last_loaded_wavefunction = kwargs.get("loaded_wavefunction")
        return _minimal_success_unified(kwargs["molecule_id"])

    def save_unified_features(self, unified_data, output_path):
        self.saved_paths.append(output_path)
        return f"{output_path}.unified.json"


class MoleculeProcessorPerformanceTests(unittest.TestCase):
    def test_process_one_reuses_loaded_wavefunction_and_returns_summary_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = _FakeExtractor()
            processor = MoleculeProcessor(
                feature_extractor=extractor,
                plugin_registry=_FakePluginRegistry(),
                artifact_manager=_FakeArtifactManager(Path(tmpdir)),
                run_mode="full",
            )
            result = processor.process_one(
                pkl_path=Path(tmpdir) / "fake.pkl",
                molecule_id="mol_001",
                smiles="CCO",
                dft_config={"basis": "def2-TZVP"},
                return_data_mode="summary",
            )

            self.assertEqual(extractor.load_calls, 1)
            self.assertEqual(extractor.extract_calls, 1)
            self.assertIsNotNone(extractor.last_loaded_wavefunction)
            self.assertEqual(result.overall_status, "fully_success")
            self.assertIn("orbital_features", result.data)
            self.assertIn("realspace_features", result.data)
            self.assertIn("external_bridge", result.data)
            self.assertNotIn("atom_features", result.data)
            self.assertIn("processor_wall_time_seconds", result.metrics)


class BatchRunnerPerformanceTests(unittest.TestCase):
    def test_batch_runner_serial_uses_summary_mode_and_surfaces_molecule_results(self):
        class FakeProcessor:
            def __init__(self):
                self.feature_extractor = _FakeExtractor()
                self.return_modes = []

            def process_one(self, pkl_path, molecule_id, smiles=None, dft_config=None, return_data_mode="full"):
                self.return_modes.append(return_data_mode)
                return MoleculeResult(
                    molecule_id=molecule_id,
                    overall_status="fully_success",
                    reason_codes=[],
                    data={
                        "orbital_features": {"metadata": {"extraction_status": "success"}},
                        "realspace_features": {"metadata": {"extraction_status": "success"}},
                        "external_bridge": {"critic2": {"execution_status": "success"}},
                    },
                    wall_time_seconds=1.5,
                    metrics={"processor_wall_time_seconds": 1.5},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = FakeProcessor()
            config = MoleculeProcessorConfig(run_mode="full", artifact_policy="keep_failed_only", plugin_config={})
            runner = BatchRunner(
                processor=processor,
                config=config,
                output_dir=Path(tmpdir),
                n_workers=1,
            )
            summary = runner.run(
                molecule_list=[{"molecule_id": "mol_001", "pkl_path": str(Path(tmpdir) / "mol_001.pkl")}],
                include_molecule_results=True,
            )

            self.assertEqual(processor.return_modes, ["summary"])
            self.assertEqual(summary["molecule_counts"]["by_final_status"]["fully_success"], 1)
            self.assertEqual(len(summary["molecule_results"]), 1)
            self.assertEqual(summary["molecule_results"][0]["metrics"]["processor_wall_time_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main()

"""
统一输出 Schema 构造器 (Milestone 1)

统一输出骨架，确保字段命名稳定、缺失字段显式标记为 null。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import time


class UnifiedOutputBuilder:
    """
    FastORCA 统一输出骨架构造器
    
    Schema 版本: 1.0.0
    
    设计原则:
    1. 顶层 section 始终保留，内部字段可为 null
    2. 字段命名使用 snake_case，稳定不变
    3. 所有缺失字段显式标记为 null
    4. bond_features 统一采用 list 模式，与 bond_indices 严格对齐
    
    Spin 定义:
    - spin = N_alpha - N_beta (未成对电子数)
    - multiplicity = spin + 1 = 2S + 1
    """
    
    SCHEMA_VERSION = "1.0.0"
    SOFTWARE_NAME = "FastORCA"
    
    def __init__(self, molecule_id: str, smiles: str):
        """
        初始化统一输出骨架
        
        Args:
            molecule_id: 分子唯一标识
            smiles: 原始 SMILES 字符串
        """
        self.data = self._init_skeleton(molecule_id, smiles)
        self.timers = {}
        self._start_time = time.time()
        
    def _init_skeleton(self, molecule_id: str, smiles: str) -> Dict[str, Any]:
        """初始化空骨架，所有字段显式设为 null"""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "schema_name": "FastORCA Unified Output Schema",
            
            "molecule_info": {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "inchi": None,
                "inchikey": None,
                "formula": None,
                "natm": None,
                "charge": None,
                "spin": None,  # N_alpha - N_beta
                "multiplicity": None  # spin + 1
            },
            
            "calculation_status": {
                "overall_status": None,
                "invalid_input": False,
                "rdkit_parse_success": False,
                "geometry_optimization_success": False,
                "scf_convergence_success": False,
                "wavefunction_load_success": False,
                "core_features_success": False,
                "extended_features_success": False,
                "error_messages": []
            },
            
            "geometry": {
                "atom_symbols": None,
                "atom_coords_angstrom": None,
                "point_group": None
            },
            
            "global_features": {
                "dft": {
                    "total_energy_hartree": None,
                    "homo_energy_hartree": None,
                    "lumo_energy_hartree": None,
                    "homo_lumo_gap_hartree": None,
                    "dipole_moment_debye": None,
                    "dipole_vector_debye": None,
                    "scf_converged": None,
                    "dispersion_correction": None
                },
                "rdkit": {
                    "molecular_weight": None,
                    "logp": None,
                    "tpsa": None,
                    "h_bond_donors": None,
                    "h_bond_acceptors": None,
                    "rotatable_bonds": None,
                    "heavy_atom_count": None
                }
            },
            
            "atom_features": {
                "atomic_number": None,
                "rdkit_degree": None,
                "rdkit_hybridization": None,
                "rdkit_aromatic": None,
                "charge_mulliken": None,
                "charge_hirshfeld": None,
                "charge_cm5": None,
                "charge_iao": None,
                "elf_value": None
            },
            
            "bond_features": {
                "bond_indices": None,
                "bond_types_rdkit": None,
                "bond_orders_mayer": None,
                "bond_orders_wiberg": None,
                "elf_bond_midpoint": None
            },
            
            "artifacts": {
                "wavefunction": {
                    "pkl_path": None,
                    "format": "pickle",
                    "loaded_successfully": None
                },
                "fock_matrix": {
                    "iao_matrix": None,
                    "atomic_slices": None,
                    "available": False
                },
                "cube_files": {
                    "density": None,
                    "elf": None
                },
                "external_bridge": {
                    "critic2_input": None,
                    "critic2_output": None,
                    "horton_output": None,
                    "psi4_output": None
                }
            },
            
            "plugin_status": {
                "rdkit": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                },
                "xtb": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                },
                "pyscf": {
                    "available": False,
                    "used": False,
                    "success": False,
                    "errors": []
                }
            },
            
            "provenance": {
                "software": self.SOFTWARE_NAME,
                "version": self.SCHEMA_VERSION,
                "dft_functional": None,
                "basis_set": None,
                "geometry_optimization_method": None,
                "gpu_acceleration_used": None,
                "feature_definition_version": None,
                "realspace_definition_version": None,
                "calculation_timestamp": datetime.utcnow().isoformat() + "Z",
                "wall_time_seconds": None
            },
            "runtime_metadata": {
                "unified_extraction_time_seconds": None,
                "extraction_timestamp": None,
                "elf_bond_midpoints_raw_count": None,
                "elf_bond_midpoints_aligned_count": None,
                "elf_bond_midpoints_dropped_count": None,
                "schema_build_finalized": False
            },
            
            "orbital_features": {
                "local_orbital_method": None,
                "ibo_count": None,
                "ibo_occupancies": None,
                "ibo_centers_angstrom": None,
                "ibo_atom_contributions": None,
                "ibo_class_heuristic": None,
                "orbital_locality_score": None,
                "iao_atom_mapping": None,
                "metadata": {
                    "coefficient_basis": None,
                    "coefficient_metric": None,
                    "center_definition": None,
                    "contribution_definition": None,
                    "occupancy_definition": None,
                    "locality_score_formula": None,
                    "classification_is_heuristic": None,
                    "heuristic_classification_rules": None,
                    "extraction_status": None,
                    "failure_reason": None,
                    "extraction_time_seconds": None,
                    "pyscf_version": None,
                }
            },
            
            "realspace_features": {
                "density_isosurface_volume": None,
                "density_isosurface_area": None,
                "density_sphericity_like": None,
                "esp_extrema_summary": None,
                "orbital_extent_homo": None,
                "orbital_extent_lumo": None,
                "metadata": {
                    "realspace_definition_version": None,
                    "native_grid_unit": None,
                    "output_unit": None,
                    "conversion_factor_bohr_to_angstrom": None,
                    "density_isovalue": None,
                    "orbital_isovalue": None,
                    "cube_grid_shape": None,
                    "cube_spacing_angstrom": None,
                    "geometry_electronic_state_key": None,
                    "wavefunction_signature": None,
                    "feature_compatibility_key": None,
                    "artifact_compatibility_key": None,
                    "cache_reuse_mode": None,
                    "cache_hit": None,
                    "cache_entry_id": None,
                    "cache_root": None,
                    "required_artifacts": None,
                    "optional_artifacts": None,
                    "realspace_core_features_enabled": None,
                    "realspace_core_features_expected": None,
                    "realspace_core_features_status": None,
                    "realspace_extended_features_enabled": None,
                    "realspace_extended_features_expected": None,
                    "realspace_extended_features_status": None,
                    "realspace_extended_failure_reason": None,
                    "stage_timing_seconds": None,
                    "extraction_status": None,
                    "failure_reason": None,
                    "extraction_time_seconds": None,
                }
            },
            
            "external_bridge": {
                "critic2": {
                    "execution_status": None,
                    "failure_reason": None,
                    "input_file": None,
                    "output_file": None,
                    "execution_time_seconds": None,
                    "critic2_version": None,
                }
            },
            
            "external_features": {
                "critic2": {
                    "qtaim": {
                        "bader_charges": None,
                        "bader_volumes": None,
                        "n_bader_volumes": None,
                        "metadata": {
                            "analysis_type": None,
                            "convergence_status": None,
                        }
                    }
                }
            }
        }
    
    # ============ Setter 方法 ============
    
    def set_molecule_info(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置分子基本信息"""
        for key, value in kwargs.items():
            if key in self.data["molecule_info"]:
                self.data["molecule_info"][key] = value
        return self
    
    def set_status(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置计算状态"""
        for key, value in kwargs.items():
            if key in self.data["calculation_status"]:
                self.data["calculation_status"][key] = value
        return self
    
    def add_error(self, error_msg: str) -> "UnifiedOutputBuilder":
        """添加错误信息"""
        self.data["calculation_status"]["error_messages"].append(error_msg)
        return self
    
    def set_geometry(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置几何信息"""
        for key, value in kwargs.items():
            if key in self.data["geometry"]:
                self.data["geometry"][key] = value
        return self
    
    def set_global_dft(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 DFT 全局特征"""
        for key, value in kwargs.items():
            if key in self.data["global_features"]["dft"]:
                self.data["global_features"]["dft"][key] = value
        return self
    
    def set_global_rdkit(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 RDKit 全局特征"""
        for key, value in kwargs.items():
            if key in self.data["global_features"]["rdkit"]:
                self.data["global_features"]["rdkit"][key] = value
        return self
    
    def set_atom_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置原子特征"""
        for key, value in kwargs.items():
            if key in self.data["atom_features"]:
                self.data["atom_features"][key] = value
        return self
    
    def set_bond_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置键特征（统一 list 模式）"""
        for key, value in kwargs.items():
            if key in self.data["bond_features"]:
                self.data["bond_features"][key] = value
        return self
    
    def set_artifacts_wavefunction(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置波函数 artifacts"""
        for key, value in kwargs.items():
            if key in self.data["artifacts"]["wavefunction"]:
                self.data["artifacts"]["wavefunction"][key] = value
        return self
    
    def set_artifacts_fock(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置 Fock 矩阵 artifacts"""
        for key, value in kwargs.items():
            if key in self.data["artifacts"]["fock_matrix"]:
                self.data["artifacts"]["fock_matrix"][key] = value
        return self
    
    def set_plugin_status(self, plugin: str, **kwargs) -> "UnifiedOutputBuilder":
        """
        设置插件状态
        
        Args:
            plugin: "rdkit" | "xtb" | "pyscf"
        """
        if plugin in self.data["plugin_status"]:
            for key, value in kwargs.items():
                if key in self.data["plugin_status"][plugin]:
                    self.data["plugin_status"][plugin][key] = value
        return self
    
    def set_provenance(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置溯源信息"""
        for key, value in kwargs.items():
            if key in self.data["provenance"]:
                self.data["provenance"][key] = value
        return self
    
    def set_orbital_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置局域轨道特征"""
        for key, value in kwargs.items():
            if key in self.data["orbital_features"] and key != "metadata":
                self.data["orbital_features"][key] = value
        return self
    
    def set_orbital_metadata(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置局域轨道 metadata"""
        for key, value in kwargs.items():
            if key in self.data["orbital_features"]["metadata"]:
                self.data["orbital_features"]["metadata"][key] = value
        return self
    
    def set_realspace_features(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置实空间特征"""
        for key, value in kwargs.items():
            if key in self.data["realspace_features"] and key != "metadata":
                self.data["realspace_features"][key] = value
        return self
    
    def set_realspace_metadata(self, **kwargs) -> "UnifiedOutputBuilder":
        """设置实空间 metadata"""
        for key, value in kwargs.items():
            if key in self.data["realspace_features"]["metadata"]:
                self.data["realspace_features"]["metadata"][key] = value
        return self
    
    def set_cube_file(self, cube_type: str, info: dict) -> "UnifiedOutputBuilder":
        """
        设置 cube 文件信息
        
        Args:
            cube_type: "density" | "esp" | "homo" | "lumo"
            info: cube 文件信息字典
        """
        if cube_type in ["density", "esp", "homo", "lumo"]:
            # 注意：cube_files 现在可能在 artifacts 或 realspace_features 中
            if "cube_files" not in self.data["artifacts"]:
                self.data["artifacts"]["cube_files"] = {}
            self.data["artifacts"]["cube_files"][cube_type] = info
        return self
    
    def set_external_bridge(self, tool: str, **kwargs) -> "UnifiedOutputBuilder":
        """
        设置 external bridge 执行信息
        
        Args:
            tool: "critic2" | "horton" | "psi4"
            **kwargs: bridge 信息字段
        """
        if tool in self.data["external_bridge"]:
            for key, value in kwargs.items():
                self.data["external_bridge"][tool][key] = value
        return self
    
    def set_external_features(self, tool: str, features: dict) -> "UnifiedOutputBuilder":
        """
        设置 external features
        
        Args:
            tool: "critic2" | "horton" | "psi4"
            features: 特征字典
        """
        if tool in self.data["external_features"]:
            self.data["external_features"][tool].update(features)
        return self
    
    # ============ 状态判定方法 ============
    
    def determine_overall_status(self) -> str:
        """
        确定 overall_status（严格顺序执行，短路返回）
        
        判定流程:
        1. invalid_input → SMILES 无法解析或参数非法
        2. failed_geometry → 几何优化失败
        3. failed_scf → SCF 未收敛
        4. failed_core_features → 核心特征计算失败
        5. core_success_partial_features → 核心成功但扩展失败
        6. fully_success → 全部成功
        
        注意：M5 后，推荐使用 utils.policy.StatusDeterminer 进行更精确的状态判定。
        此方法保持向后兼容。
        """
        # 尝试使用新的 StatusDeterminer
        try:
            from utils.policy.status_determiner import StatusDeterminer
            determiner = StatusDeterminer(self.data)
            return determiner.determine()
        except Exception:
            # 回退到旧逻辑
            pass
        
        status = self.data["calculation_status"]
        
        if status["invalid_input"]:
            return "invalid_input"
        
        if not status["geometry_optimization_success"]:
            return "failed_geometry"
        
        if not status["scf_convergence_success"]:
            return "failed_scf"
        
        if not status["core_features_success"]:
            return "failed_core_features"
        
        if not status["extended_features_success"]:
            return "core_success_partial_features"
        
        return "fully_success"
    
    def check_core_features_success(self) -> bool:
        """
        检查核心特征是否成功（放宽版判定规则）
        
        必须同时满足:
        1. wavefunction_load_success == true
        2. scf_convergence_success == true
        3. global_features.dft.total_energy 存在
        4. 至少一组原子电荷成功且长度匹配（Mulliken/Hirshfeld/IAO/CM5 任一）
        5. 多原子分子至少一组键级特征成功（单原子跳过此项）
        
        注意：M5 后，推荐使用 utils.policy.FieldTierChecker 进行更精确的字段检查。
        """
        # 尝试使用新的 FieldTierChecker
        try:
            from utils.policy.field_tiers import FieldTierChecker
            checker = FieldTierChecker()
            ok, _ = checker.check_core_required(self.data)
            return ok
        except Exception:
            # 回退到旧逻辑
            pass
        
        status = self.data["calculation_status"]
        dft = self.data["global_features"]["dft"]
        atom_feat = self.data["atom_features"]
        bond_feat = self.data["bond_features"]
        natm = self.data["molecule_info"]["natm"] or 0
        
        # 1 & 2. 波函数加载和 SCF 收敛
        if not (status["wavefunction_load_success"] and dft["scf_converged"]):
            return False
        
        # 3. 基本能量信息可得
        if dft["total_energy_hartree"] is None:
            return False
        
        # 4. 至少一组原子电荷成功且长度匹配
        charge_fields = [
            atom_feat["charge_mulliken"],
            atom_feat["charge_hirshfeld"],
            atom_feat["charge_iao"],
            atom_feat["charge_cm5"],
        ]
        valid_charge = any(
            c is not None and isinstance(c, (list, tuple)) and len(c) == natm
            for c in charge_fields
        )
        if not valid_charge:
            return False
        
        # 5. 多原子分子至少一组键级特征成功
        if natm > 1:
            bond_orders = [
                bond_feat["bond_orders_mayer"],
                bond_feat["bond_orders_wiberg"]
            ]
            valid_bond = any(
                b is not None and isinstance(b, (list, tuple)) and len(b) > 0
                for b in bond_orders
            )
            if not valid_bond:
                return False
        
        return True
    
    def check_extended_features_success(self) -> bool:
        """
        检查扩展特征是否成功（strongly_recommended 规则组）
        
        M5 后：检查 strongly_recommended 规则组是否全部满足。
        若不满足，则判定为 core_success_partial_features。
        """
        if not self.data["calculation_status"]["core_features_success"]:
            return False
        
        # 尝试使用新的 FieldTierChecker
        try:
            from utils.policy.field_tiers import FieldTierChecker
            checker = FieldTierChecker()
            violation_count, _ = checker.check_strongly_recommended(self.data)
            return violation_count == 0
        except Exception:
            # 回退到旧逻辑
            pass
        
        atom_feat = self.data["atom_features"]
        bond_feat = self.data["bond_features"]
        
        # CM5 电荷
        if atom_feat["charge_cm5"] is None:
            return False
        
        # ELF 特征
        if atom_feat["elf_value"] is None:
            return False
        if bond_feat["elf_bond_midpoint"] is None:
            return False
        
        return True
    
    def finalize(self) -> Dict[str, Any]:
        """
        最终化输出，计算状态并填充时间戳
        
        这是唯一的状态判定入口，确保 overall_status 由中心函数统一判定。
        
        Returns:
            完整的统一输出字典
        """
        # 计算 wall_time
        self.data["provenance"]["wall_time_seconds"] = time.time() - self._start_time
        
        # 判定核心和扩展特征状态
        self.data["calculation_status"]["core_features_success"] = self.check_core_features_success()
        self.data["calculation_status"]["extended_features_success"] = self.check_extended_features_success()
        
        # 判定 overall_status（中心唯一判定点）
        self.data["calculation_status"]["overall_status"] = self.determine_overall_status()
        
        # 标记已最终化
        self.data["runtime_metadata"]["schema_build_finalized"] = True
        
        return self.data
    
    def build(self) -> Dict[str, Any]:
        """
        构建并返回最终输出。
        
        注意：此方法显式调用 finalize() 确保状态一致性。
        所有提前返回点（如错误处理）也应显式调用 finalize()。
        
        Returns:
            完整的统一输出字典
        """
        return self.finalize()
    
    def build_early(self) -> Dict[str, Any]:
        """
        提前构建输出（用于错误退出时）。
        
        确保即使流程提前终止，也调用 finalize() 进行正确的状态判定。
        
        Returns:
            完整的统一输出字典
        """
        return self.finalize()

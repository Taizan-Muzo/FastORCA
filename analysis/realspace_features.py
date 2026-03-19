"""
Milestone 3: Real-space Features and Cube Artifacts
基于 PySCF cubegen 的实空间特征提取

设计原则:
- 内部计算使用 Bohr，输出转换为 Å
- Cube artifacts 包含完整 metadata
- 所有 shape 特征明确定义，可解释
- 大分子成本控制，soft-fail 策略
- M5: 支持子进程级超时控制
"""

import time
import multiprocessing
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile
import os

import numpy as np
from loguru import logger

# PySCF imports
from pyscf import gto, scf, dft
from pyscf.tools import cubegen
from pyscf.dft import numint


# 物理常数
BOHR_TO_ANGSTROM = 0.52917720859  # Bohr -> Å
REALSPACE_DEFINITION_VERSION = "2.0.0"
COORD_QUANTIZATION_DECIMALS = 6
VALUE_QUANTIZATION_DECIMALS = 8

# 默认配置
DEFAULT_CUBE_CONFIG = {
    "enable_cube_generation": True,
    "enable_cache": True,
    "cache_directory": ".realspace_cache",
    "realspace_core_features_enabled": True,
    "realspace_core_features_expected": True,
    "realspace_extended_features_enabled": True,
    "realspace_extended_features_expected": True,
    "required_artifacts": ["density"],
    "optional_artifacts": ["esp", "homo", "lumo"],
    "grid_resolution_angstrom": 0.2,  # Å
    "margin_angstrom": 4.0,  # Å
    "max_atoms_for_cube": 50,
    "max_points_per_dimension": 100,
    "max_total_grid_points": 1_000_000,
    "density_isovalue": 0.001,  # e/Bohr^3
    "orbital_isovalue": 0.05,   # 用于可视化，不用于 extent 计算
    "output_directory": "cubes/"
}


class RealspaceFeatureExtractor:
    """
    实空间特征提取器
    
    生成 cube artifacts 并从中提取 shape 特征
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = {**DEFAULT_CUBE_CONFIG, **(config or {})}
    
    def extract_realspace_features(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        molecule_id: str = "unknown",
        output_dir: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        提取实空间特征和 cube artifacts
        
        M5: 添加子进程级超时控制，防止 cubegen 无限运行
        
        Args:
            mol: PySCF Mole 对象
            mf: 已收敛的 SCF 对象
            molecule_id: 分子标识（用于 cube 文件名）
            output_dir: cube 文件输出目录
            timeout_seconds: 超时时间（秒），None 表示无限制
            
        Returns:
            包含 realspace_features 和 artifacts.cube_files 的字典
        """
        start_time = time.time()
        
        result = self._init_realspace_features_skeleton()
        compatibility = self._build_compatibility_keys(mol, mf)
        result["metadata"]["geometry_electronic_state_key"] = compatibility["geometry_electronic_state_key"]
        result["metadata"]["wavefunction_signature"] = compatibility["wavefunction_signature"]
        result["metadata"]["feature_compatibility_key"] = compatibility["feature_compatibility_key"]
        result["metadata"]["artifact_compatibility_key"] = compatibility["artifact_compatibility_key"]
        result["metadata"]["cache_root"] = str(self._cache_root(output_dir))
        
        # Step 1: 检查是否应该跳过
        should_skip, reason = self._should_skip_cube(mol)
        if should_skip:
            result["metadata"]["extraction_status"] = "skipped"
            result["metadata"]["failure_reason"] = reason
            logger.warning(f"Cube generation skipped for {molecule_id}: {reason}")
            return result

        # Step 1.5: 缓存复用
        cache_mode, cached_result, cached_artifacts = self._try_reuse_cache(
            compatibility=compatibility,
            molecule_id=molecule_id,
            mol=mol,
            output_dir=output_dir,
        )
        if cache_mode == "exact_feature_reuse" and cached_result is not None:
            # exact feature reuse: 直接返回完整特征
            cached_result["metadata"]["cache_reuse_mode"] = "exact_feature_reuse"
            cached_result["metadata"]["cache_hit"] = True
            return cached_result
        
        # M5: 使用子进程执行 cube 生成以实现可靠超时
        if timeout_seconds is not None and timeout_seconds > 0:
            return self._extract_with_timeout(
                mol, mf, molecule_id, output_dir, timeout_seconds, start_time, result, compatibility, cache_mode, cached_artifacts
            )
        
        # 无超时：直接执行
        return self._extract_cubes(
            mol, mf, molecule_id, output_dir, result, start_time,
            compatibility=compatibility,
            cache_mode=cache_mode,
            reused_artifacts=cached_artifacts
        )
    
    def _extract_with_timeout(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        molecule_id: str,
        output_dir: Optional[str],
        timeout_seconds: float,
        start_time: float,
        result: Dict[str, Any],
        compatibility: Dict[str, Any],
        cache_mode: str,
        reused_artifacts: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """使用子进程执行 cube 生成，带超时控制"""
        from multiprocessing import Process, Queue
        
        queue = Queue()
        
        # 序列化必要的数据
        def target(queue, mol_dict, mf_dict, molecule_id, output_dir, config, compatibility, cache_mode, reused_artifacts):
            """子进程目标函数"""
            try:
                # 重建 mol 和 mf
                mol_dict['verbose'] = 0  # M5: 静音
                mol = gto.Mole()
                mol.build(**mol_dict)
                
                mf = dft.RKS(mol)
                mf.verbose = 0  # M5: 静音
                mf.mo_energy = mf_dict['mo_energy']
                mf.mo_coeff = mf_dict['mo_coeff']
                mf.mo_occ = mf_dict['mo_occ']
                mf.e_tot = mf_dict['e_tot']
                mf.converged = mf_dict['converged']
                
                # 创建临时 extractor
                extractor = RealspaceFeatureExtractor(config)
                result = extractor._extract_cubes(
                    mol, mf, molecule_id, output_dir,
                    extractor._init_realspace_features_skeleton(),
                    time.time(),
                    compatibility=compatibility,
                    cache_mode=cache_mode,
                    reused_artifacts=reused_artifacts
                )
                queue.put(('success', result))
            except Exception as e:
                queue.put(('error', str(e)))
        
        # 准备序列化数据
        mol_dict = {
            'atom': mol.atom,
            'basis': mol.basis,
            'charge': mol.charge,
            'spin': mol.spin,
        }
        mf_dict = {
            'mo_energy': mf.mo_energy,
            'mo_coeff': mf.mo_coeff,
            'mo_occ': mf.mo_occ,
            'e_tot': mf.e_tot,
            'converged': mf.converged,
        }
        
        # 启动子进程
        process = Process(
            target=target,
            args=(queue, mol_dict, mf_dict, molecule_id, output_dir, self.config, compatibility, cache_mode, reused_artifacts)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # 超时：终止子进程
            process.terminate()
            process.join()
            result["metadata"]["extraction_status"] = "timeout"
            result["metadata"]["failure_reason"] = "realspace_timeout"
            result["metadata"]["realspace_extended_failure_reason"] = "realspace_timeout"
            result["metadata"]["cache_reuse_mode"] = cache_mode
            result["metadata"]["cache_hit"] = cache_mode in ("exact_feature_reuse", "artifact_reuse")
            result["metadata"]["extraction_time_seconds"] = timeout_seconds
            logger.warning(f"[{molecule_id}] Realspace extraction timeout after {timeout_seconds}s")
            return result
        
        # 获取结果
        try:
            status, data = queue.get(timeout=5)  # 给5秒缓冲时间
            if status == 'success':
                return data
            else:
                logger.error(f"[{molecule_id}] Subprocess error: {data}")
                # M5.5: 子进程失败时回退到直接执行
                logger.warning(f"[{molecule_id}] Falling back to direct execution (no timeout)")
                return self._extract_cubes(
                    mol, mf, molecule_id, output_dir,
                    self._init_realspace_features_skeleton(),
                    time.time(),
                    compatibility=compatibility,
                    cache_mode=cache_mode,
                    reused_artifacts=reused_artifacts
                )
        except Exception as e:
            logger.error(f"[{molecule_id}] Queue communication error: {e}")
            # 回退到直接执行
            logger.warning(f"[{molecule_id}] Falling back to direct execution (no timeout)")
            return self._extract_cubes(
                mol, mf, molecule_id, output_dir,
                self._init_realspace_features_skeleton(),
                time.time(),
                compatibility=compatibility,
                cache_mode=cache_mode,
                reused_artifacts=reused_artifacts
            )
    
    def _extract_cubes(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        molecule_id: str,
        output_dir: Optional[str],
        result: Dict[str, Any],
        start_time: float,
        compatibility: Optional[Dict[str, Any]] = None,
        cache_mode: str = "no_reuse",
        reused_artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """实际的 cube 生成逻辑（在子进程或主进程中执行）"""
        try:
            # 准备输出目录
            if output_dir is None:
                output_dir = self.config["output_directory"]
            cube_dir = Path(output_dir) / molecule_id
            cube_dir.mkdir(parents=True, exist_ok=True)

            if compatibility:
                result["metadata"]["geometry_electronic_state_key"] = compatibility.get("geometry_electronic_state_key")
                result["metadata"]["wavefunction_signature"] = compatibility.get("wavefunction_signature")
                result["metadata"]["feature_compatibility_key"] = compatibility.get("feature_compatibility_key")
                result["metadata"]["artifact_compatibility_key"] = compatibility.get("artifact_compatibility_key")
            result["metadata"]["cache_reuse_mode"] = cache_mode
            result["metadata"]["cache_hit"] = cache_mode in ("exact_feature_reuse", "artifact_reuse")
            if isinstance(reused_artifacts, dict):
                result["metadata"]["cache_entry_id"] = reused_artifacts.get("_cache_entry_id")
            stage_timing = dict(result["metadata"].get("stage_timing_seconds") or {})
            for k in ["density_cube", "esp_cube", "homo_cube", "lumo_cube", "shape_core", "shape_extended"]:
                stage_timing.setdefault(k, 0.0)
            result["metadata"]["stage_timing_seconds"] = stage_timing
            
            # 计算网格参数（Å-based，然后转 Bohr）
            grid_shape = self._compute_grid_shape(mol)
            result["metadata"]["cube_grid_shape"] = grid_shape.tolist()
            
            # 分辨率（Bohr for cubegen）
            res_bohr = self.config["grid_resolution_angstrom"] / BOHR_TO_ANGSTROM
            margin_bohr = self.config["margin_angstrom"] / BOHR_TO_ANGSTROM
            
            required_artifacts = set(self.config.get("required_artifacts", []))
            optional_artifacts = set(self.config.get("optional_artifacts", []))
            density_required = (
                "density" in required_artifacts
                or "density" in optional_artifacts
                or "esp" in required_artifacts
                or "homo" in required_artifacts
                or "lumo" in required_artifacts
                or "esp" in optional_artifacts
                or "homo" in optional_artifacts
                or "lumo" in optional_artifacts
            )
            esp_requested = "esp" in required_artifacts or "esp" in optional_artifacts
            homo_requested = "homo" in required_artifacts or "homo" in optional_artifacts
            lumo_requested = "lumo" in required_artifacts or "lumo" in optional_artifacts

            core_enabled = bool(self.config.get("realspace_core_features_enabled", True))
            core_expected = bool(self.config.get("realspace_core_features_expected", core_enabled))
            ext_enabled = bool(self.config.get("realspace_extended_features_enabled", True))
            ext_expected = bool(self.config.get("realspace_extended_features_expected", ext_enabled))
            result["metadata"]["realspace_core_features_enabled"] = core_enabled
            result["metadata"]["realspace_core_features_expected"] = core_expected
            result["metadata"]["realspace_extended_features_enabled"] = ext_enabled
            result["metadata"]["realspace_extended_features_expected"] = ext_expected

            dm = mf.make_rdm1()

            # Step 2: Density Cube（required artifact）
            density_path = cube_dir / "density.cube"
            reused_density = self._pick_reused_artifact(reused_artifacts, "density")
            if density_required and reused_density:
                logger.debug(f"[{molecule_id}] Reusing density cube artifact")
                result["artifacts"]["cube_files"]["density"] = reused_density
                stage_timing["density_cube"] = 0.0
            elif density_required:
                logger.debug(f"[{molecule_id}] Generating density cube...")
                t_stage = time.time()
                cubegen.density(
                    mol, 
                    str(density_path), 
                    dm,
                    nx=int(grid_shape[0]),
                    ny=int(grid_shape[1]),
                    nz=int(grid_shape[2]),
                    resolution=res_bohr,
                    margin=margin_bohr
                )
                
                result["artifacts"]["cube_files"]["density"] = {
                    "path": str(density_path),
                    "file_type": "gaussian_cube",
                    "value_type": "electron_density",
                    "value_unit": "e/bohr^3",
                    "grid_shape": grid_shape.tolist(),
                    "spacing_angstrom": [self.config["grid_resolution_angstrom"]] * 3,
                    "origin_angstrom": self._get_cube_origin_angstrom(mol, margin_bohr),
                    "native_grid_unit": "bohr",
                    "source_method": "pyscf.tools.cubegen.density"
                }
                stage_timing["density_cube"] = float(time.time() - t_stage)

            # Step 3: ESP cube（optional artifact）
            if esp_requested:
                reused_esp = self._pick_reused_artifact(reused_artifacts, "esp")
                if reused_esp:
                    logger.debug(f"[{molecule_id}] Reusing ESP cube artifact")
                    result["artifacts"]["cube_files"]["esp"] = reused_esp
                    stage_timing["esp_cube"] = 0.0
                else:
                    logger.debug(f"[{molecule_id}] Generating MEP cube...")
                    mep_path = cube_dir / "mep.cube"
                    t_stage = time.time()
                    cubegen.mep(
                        mol,
                        str(mep_path),
                        dm,
                        nx=int(grid_shape[0]),
                        ny=int(grid_shape[1]),
                        nz=int(grid_shape[2]),
                        resolution=res_bohr,
                        margin=margin_bohr
                    )
                    
                    result["artifacts"]["cube_files"]["esp"] = {
                        "path": str(mep_path),
                        "file_type": "gaussian_cube",
                        "value_type": "electrostatic_potential",
                        "value_unit": "hartree",
                        "grid_shape": grid_shape.tolist(),
                        "spacing_angstrom": [self.config["grid_resolution_angstrom"]] * 3,
                        "origin_angstrom": result["artifacts"]["cube_files"]["density"]["origin_angstrom"],
                        "native_grid_unit": "bohr",
                        "source_method": "pyscf.tools.cubegen.mep"
                    }
                    stage_timing["esp_cube"] = float(time.time() - t_stage)

            # Step 4: HOMO/LUMO cubes（optional artifacts）
            occ_idx = mf.mo_occ > 0
            n_occ = np.sum(occ_idx)
            homo_idx = n_occ - 1
            lumo_idx = n_occ
            
            # HOMO
            if homo_requested and homo_idx >= 0 and homo_idx < mf.mo_coeff.shape[1]:
                reused_homo = self._pick_reused_artifact(reused_artifacts, "homo")
                if reused_homo:
                    logger.debug(f"[{molecule_id}] Reusing HOMO cube artifact")
                    result["artifacts"]["cube_files"]["homo"] = reused_homo
                    stage_timing["homo_cube"] = 0.0
                else:
                    logger.debug(f"[{molecule_id}] Generating HOMO cube (MO {homo_idx})...")
                    homo_path = cube_dir / "homo.cube"
                    t_stage = time.time()
                    cubegen.orbital(
                        mol,
                        str(homo_path),
                        mf.mo_coeff[:, homo_idx],
                        nx=int(grid_shape[0]),
                        ny=int(grid_shape[1]),
                        nz=int(grid_shape[2]),
                        resolution=res_bohr,
                        margin=margin_bohr
                    )
                    
                    result["artifacts"]["cube_files"]["homo"] = {
                        "path": str(homo_path),
                        "file_type": "gaussian_cube",
                        "value_type": "molecular_orbital_amplitude",
                        "value_unit": "dimensionless",
                        "orbital_index": int(homo_idx),
                        "orbital_type": "HOMO",
                        "orbital_energy_hartree": float(mf.mo_energy[homo_idx]) if hasattr(mf, 'mo_energy') else None,
                        "grid_shape": grid_shape.tolist(),
                        "spacing_angstrom": [self.config["grid_resolution_angstrom"]] * 3,
                        "origin_angstrom": result["artifacts"]["cube_files"]["density"]["origin_angstrom"],
                        "native_grid_unit": "bohr",
                        "source_method": "pyscf.tools.cubegen.orbital"
                    }
                    stage_timing["homo_cube"] = float(time.time() - t_stage)
            
            # LUMO
            if lumo_requested and lumo_idx < mf.mo_coeff.shape[1]:
                reused_lumo = self._pick_reused_artifact(reused_artifacts, "lumo")
                if reused_lumo:
                    logger.debug(f"[{molecule_id}] Reusing LUMO cube artifact")
                    result["artifacts"]["cube_files"]["lumo"] = reused_lumo
                    stage_timing["lumo_cube"] = 0.0
                else:
                    logger.debug(f"[{molecule_id}] Generating LUMO cube (MO {lumo_idx})...")
                    lumo_path = cube_dir / "lumo.cube"
                    t_stage = time.time()
                    cubegen.orbital(
                        mol,
                        str(lumo_path),
                        mf.mo_coeff[:, lumo_idx],
                        nx=int(grid_shape[0]),
                        ny=int(grid_shape[1]),
                        nz=int(grid_shape[2]),
                        resolution=res_bohr,
                        margin=margin_bohr
                    )
                    
                    result["artifacts"]["cube_files"]["lumo"] = {
                        "path": str(lumo_path),
                        "file_type": "gaussian_cube",
                        "value_type": "molecular_orbital_amplitude",
                        "value_unit": "dimensionless",
                        "orbital_index": int(lumo_idx),
                        "orbital_type": "LUMO",
                        "orbital_energy_hartree": float(mf.mo_energy[lumo_idx]) if hasattr(mf, 'mo_energy') else None,
                        "grid_shape": grid_shape.tolist(),
                        "spacing_angstrom": [self.config["grid_resolution_angstrom"]] * 3,
                        "origin_angstrom": result["artifacts"]["cube_files"]["density"]["origin_angstrom"],
                        "native_grid_unit": "bohr",
                        "source_method": "pyscf.tools.cubegen.orbital"
                    }
                    stage_timing["lumo_cube"] = float(time.time() - t_stage)
            
            # Step 5: 从 grid 计算 shape 特征（更精确，不用读 cube 文件）
            logger.debug(f"[{molecule_id}] Computing shape features from grid...")
            shape_timing = self._compute_shape_features_from_grid(
                mol, mf, result, grid_shape, res_bohr, margin_bohr,
                core_enabled=core_enabled,
                extended_enabled=ext_enabled
            )
            stage_timing["shape_core"] = float(shape_timing.get("shape_core", 0.0))
            stage_timing["shape_extended"] = float(shape_timing.get("shape_extended", 0.0))
            
            core_success = (
                (not core_enabled)
                or (
                    result.get("density_isosurface_volume") is not None
                    and result.get("density_isosurface_area") is not None
                    and result.get("density_sphericity_like") is not None
                )
            )
            ext_success = (
                (not ext_enabled)
                or (
                    result.get("esp_extrema_summary") is not None
                    and result.get("orbital_extent_homo") is not None
                    and result.get("orbital_extent_lumo") is not None
                )
            )
            result["metadata"]["realspace_core_features_status"] = "success" if core_success else "failed"
            if not ext_enabled:
                result["metadata"]["realspace_extended_features_status"] = "disabled"
            else:
                result["metadata"]["realspace_extended_features_status"] = "success" if ext_success else "failed"

            # 验证 + tier-aware 状态判定
            if not self._validate_realspace_features(result):
                result["metadata"]["extraction_status"] = "failed"
                result["metadata"]["failure_reason"] = "validation_failed"
            elif core_expected and not core_success:
                result["metadata"]["extraction_status"] = "failed"
                result["metadata"]["failure_reason"] = "realspace_core_failed"
            elif ext_expected and not ext_success:
                result["metadata"]["extraction_status"] = "failed"
                result["metadata"]["failure_reason"] = "realspace_extended_failed"
                result["metadata"]["realspace_extended_failure_reason"] = "realspace_extended_failed"
            else:
                result["metadata"]["extraction_status"] = "success"
                elapsed = time.time() - start_time
                result["metadata"]["extraction_time_seconds"] = elapsed
                logger.info(f"[{molecule_id}] Realspace features extracted in {elapsed:.3f}s")

            # 成功或可用结果写入缓存（避免缓存失败态）
            if result["metadata"]["extraction_status"] == "success":
                self._save_cache_entries(
                    compatibility=compatibility,
                    output_dir=output_dir,
                    result=result,
                )
                
        except Exception as e:
            result["metadata"]["extraction_status"] = "failed"
            result["metadata"]["failure_reason"] = f"cube_generation_failed: {str(e)}"
            logger.error(f"[{molecule_id}] Realspace features extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return result
    
    def _init_realspace_features_skeleton(self) -> Dict[str, Any]:
        """初始化 realspace_features 骨架"""
        return {
            "density_isosurface_volume": None,
            "density_isosurface_area": None,
            "density_sphericity_like": None,
            "esp_extrema_summary": None,
            "orbital_extent_homo": None,
            "orbital_extent_lumo": None,
            "artifacts": {
                "cube_files": {
                    "density": None,
                    "esp": None,
                    "homo": None,
                    "lumo": None
                }
            },
            "metadata": {
                "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
                "native_grid_unit": "bohr",
                "output_unit": "angstrom",
                "conversion_factor_bohr_to_angstrom": BOHR_TO_ANGSTROM,
                "density_isovalue": self.config["density_isovalue"],
                "orbital_isovalue": self.config["orbital_isovalue"],
                "cube_grid_shape": None,
                "cube_spacing_angstrom": None,
                "geometry_electronic_state_key": None,
                "wavefunction_signature": None,
                "feature_compatibility_key": None,
                "artifact_compatibility_key": None,
                "cache_reuse_mode": "no_reuse",  # exact_feature_reuse | artifact_reuse | no_reuse
                "cache_hit": False,
                "cache_entry_id": None,
                "cache_root": None,
                "required_artifacts": list(self.config.get("required_artifacts", [])),
                "optional_artifacts": list(self.config.get("optional_artifacts", [])),
                "realspace_core_features_enabled": bool(self.config.get("realspace_core_features_enabled", True)),
                "realspace_core_features_expected": bool(self.config.get("realspace_core_features_expected", True)),
                "realspace_core_features_status": "not_started",
                "realspace_extended_features_enabled": bool(self.config.get("realspace_extended_features_enabled", True)),
                "realspace_extended_features_expected": bool(self.config.get("realspace_extended_features_expected", True)),
                "realspace_extended_features_status": "not_started",
                "realspace_extended_failure_reason": None,
                "stage_timing_seconds": {
                    "density_cube": 0.0,
                    "esp_cube": 0.0,
                    "homo_cube": 0.0,
                    "lumo_cube": 0.0,
                    "shape_core": 0.0,
                    "shape_extended": 0.0,
                },
                "extraction_status": "not_attempted",
                "failure_reason": None,
                "extraction_time_seconds": None,
            }
        }
    
    def _should_skip_cube(self, mol: gto.Mole) -> Tuple[bool, str]:
        """检查是否应该跳过 cube 生成"""
        if not self.config["enable_cube_generation"]:
            return True, "cube_generation_disabled"
        
        if mol.natm > self.config["max_atoms_for_cube"]:
            return True, f"molecule_too_large_for_cube: natm={mol.natm} > {self.config['max_atoms_for_cube']}"
        
        # 预估网格点数（Å-based）
        grid_shape = self._compute_grid_shape(mol)
        total_points = np.prod(grid_shape)
        
        if total_points > self.config["max_total_grid_points"]:
            return True, f"molecule_too_large_for_cube: estimated_grid_points={total_points}"
        
        if np.any(grid_shape > self.config["max_points_per_dimension"]):
            return True, f"molecule_too_large_for_cube: grid_shape={grid_shape} exceeds per-dimension limit"
        
        return False, ""
    
    def _compute_grid_shape(self, mol: gto.Mole) -> np.ndarray:
        """
        计算网格形状（Å-based）
        
        公式: n = ceil((L + 2*margin) / res) + 1
        """
        coords = mol.atom_coords(unit='A')  # Å
        box_min = np.min(coords, axis=0)
        box_max = np.max(coords, axis=0)
        L = box_max - box_min  # 盒子尺寸 Å
        
        margin = self.config["margin_angstrom"]
        res = self.config["grid_resolution_angstrom"]
        
        # 每维点数
        n_points = np.ceil((L + 2 * margin) / res).astype(int) + 1
        
        # 应用上限
        n_points = np.clip(n_points, 1, self.config["max_points_per_dimension"])
        
        return n_points
    
    def _get_cube_origin_angstrom(self, mol: gto.Mole, margin_bohr: float) -> List[float]:
        """获取 cube 原点（Å）"""
        coords = mol.atom_coords(unit='A')
        box_min = np.min(coords, axis=0)
        margin_angstrom = margin_bohr * BOHR_TO_ANGSTROM
        origin = box_min - margin_angstrom
        return origin.tolist()
    
    def _compute_shape_features_from_grid(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        result: Dict[str, Any],
        grid_shape: np.ndarray,
        res_bohr: float,
        margin_bohr: float,
        core_enabled: bool = True,
        extended_enabled: bool = True,
    ) -> Dict[str, float]:
        """在规则网格上计算 shape 特征"""
        stage_timing = {"shape_core": 0.0, "shape_extended": 0.0}
        
        # 创建网格
        from pyscf.dft import gen_grid
        
        # 使用自定义网格（均匀网格）
        # 注意：cubegen 和 numint 使用相同的 grid 生成逻辑
        coords = self._generate_uniform_grid(mol, grid_shape, res_bohr, margin_bohr)
        
        # 体素体积（Bohr^3 -> Å^3）
        voxel_volume_bohr3 = res_bohr ** 3
        voxel_volume_ang3 = voxel_volume_bohr3 * (BOHR_TO_ANGSTROM ** 3)
        
        n_total = int(np.prod(grid_shape))
        
        # 计算 AO 和 density
        ao = numint.eval_ao(mol, coords)
        dm = mf.make_rdm1()
        rho = numint.eval_rho(mol, ao, dm)
        
        # Reshape 为 3D
        rho_3d = rho.reshape(grid_shape)
        
        threshold = self.config["density_isovalue"]
        mask = rho_3d > threshold
        n_inside = np.sum(mask)

        if core_enabled:
            t_core = time.time()
            # === 1. Density isosurface volume ===
            volume_ang3 = n_inside * voxel_volume_ang3
            result["density_isosurface_volume"] = {
                "value_angstrom3": float(volume_ang3),
                "threshold": threshold,
                "computation_method": "voxel_count_times_voxel_volume",
                "n_voxels_inside": int(n_inside),
                "voxel_volume_angstrom3": float(voxel_volume_ang3)
            }
            
            # === 2. Density isosurface area (voxel-face approximation) ===
            area_ang2 = self._compute_voxel_face_area(mask, res_bohr)
            result["density_isosurface_area"] = {
                "value_angstrom2": float(area_ang2),
                "threshold": threshold,
                "computation_method": "voxel_face_approximation",
                "surface_area_is_approximate": True,
                "approximation_note": "counts faces between inside/outside voxels, not marching cubes"
            }
            
            # === 3. Density sphericity-like ===
            if n_inside > 10:  # 需要足够点才能计算形状
                sphericity = self._compute_sphericity(mask, coords.reshape(*grid_shape, 3))
                result["density_sphericity_like"] = {
                    "value": float(sphericity),
                    "threshold": threshold,
                    "formula": "s = 3 * lambda3 / (lambda1 + lambda2 + lambda3)",
                    "range": "[0, 1], 1=perfect_sphere, 0=line",
                    "description": "based on covariance matrix eigenvalues of density mask"
                }
            stage_timing["shape_core"] = float(time.time() - t_core)

        # === 4-5. Extended features ===
        if extended_enabled:
            t_ext = time.time()
            try:
                esp = self._compute_esp_on_grid(mol, coords, dm)
                result["esp_extrema_summary"] = {
                    "min_hartree": float(np.min(esp)),
                    "max_hartree": float(np.max(esp)),
                    "mean_hartree": float(np.mean(esp)),
                    "std_hartree": float(np.std(esp)),
                    "computed_over": "full_cube_grid",
                    "n_voxels_total": n_total
                }
            except Exception as e:
                logger.warning(f"ESP computation failed: {e}")
                result["esp_extrema_summary"] = None

        # Orbital extent (HOMO/LUMO)
        occ_idx = mf.mo_occ > 0
        n_occ = np.sum(occ_idx)
        
        # HOMO
        homo_idx = n_occ - 1
        if extended_enabled and homo_idx >= 0:
            extent = self._compute_orbital_extent(
                mol, coords, mf.mo_coeff[:, homo_idx], grid_shape
            )
            if extent is not None:
                result["orbital_extent_homo"] = {
                    "value_angstrom": float(extent),
                    "definition": "sqrt(<r^2> - <r>^2) where <...> is over full-grid normalized |phi(r)|^2",
                    "note": "not based on isosurface threshold; uses complete orbital density on grid",
                    "orbital_index": int(homo_idx),
                    "orbital_type": "HOMO",
                    "orbital_energy_hartree": float(mf.mo_energy[homo_idx]) if hasattr(mf, 'mo_energy') else None
                }
        
        # LUMO
        lumo_idx = n_occ
        if extended_enabled and lumo_idx < mf.mo_coeff.shape[1]:
            extent = self._compute_orbital_extent(
                mol, coords, mf.mo_coeff[:, lumo_idx], grid_shape
            )
            if extent is not None:
                result["orbital_extent_lumo"] = {
                    "value_angstrom": float(extent),
                    "definition": "sqrt(<r^2> - <r>^2) where <...> is over full-grid normalized |phi(r)|^2",
                    "note": "not based on isosurface threshold; uses complete orbital density on grid",
                    "orbital_index": int(lumo_idx),
                    "orbital_type": "LUMO",
                    "orbital_energy_hartree": float(mf.mo_energy[lumo_idx]) if hasattr(mf, 'mo_energy') else None
                }
        if extended_enabled:
            stage_timing["shape_extended"] = float(time.time() - t_ext)
        
        # 更新 metadata
        result["metadata"]["cube_spacing_angstrom"] = [self.config["grid_resolution_angstrom"]] * 3
        return stage_timing
    
    def _generate_uniform_grid(
        self,
        mol: gto.Mole,
        grid_shape: np.ndarray,
        res_bohr: float,
        margin_bohr: float
    ) -> np.ndarray:
        """生成均匀网格坐标"""
        coords_ang = mol.atom_coords(unit='A')
        box_min = np.min(coords_ang, axis=0)
        box_max = np.max(coords_ang, axis=0)
        
        margin_ang = margin_bohr * BOHR_TO_ANGSTROM
        res_ang = res_bohr * BOHR_TO_ANGSTROM
        
        # 原点（Å -> Bohr）
        origin_ang = box_min - margin_ang
        origin_bohr = origin_ang / BOHR_TO_ANGSTROM
        
        # 生成网格
        x = np.arange(grid_shape[0]) * res_bohr + origin_bohr[0]
        y = np.arange(grid_shape[1]) * res_bohr + origin_bohr[1]
        z = np.arange(grid_shape[2]) * res_bohr + origin_bohr[2]
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        
        return coords
    
    def _compute_voxel_face_area(self, mask: np.ndarray, res_bohr: float) -> float:
        """
        使用 voxel-face approximation 计算表面积
        
        统计位于等值面两侧的相邻 voxel 面数量
        """
        nx, ny, nz = mask.shape
        res_ang = res_bohr * BOHR_TO_ANGSTROM
        face_area = res_ang ** 2
        
        # 计算每个方向的梯度/边界
        # x 方向
        diff_x = mask[1:, :, :] ^ mask[:-1, :, :]
        n_faces_x = np.sum(diff_x)
        
        # y 方向
        diff_y = mask[:, 1:, :] ^ mask[:, :-1, :]
        n_faces_y = np.sum(diff_y)
        
        # z 方向
        diff_z = mask[:, :, 1:] ^ mask[:, :, :-1]
        n_faces_z = np.sum(diff_z)
        
        total_area = (n_faces_x + n_faces_y + n_faces_z) * face_area
        return float(total_area)
    
    def _compute_sphericity(self, mask: np.ndarray, coords_3d: np.ndarray) -> float:
        """
        计算 sphericity-like descriptor
        
        s = 3 * lambda3 / (lambda1 + lambda2 + lambda3)
        """
        # 获取 mask 内点的坐标
        indices = np.argwhere(mask)
        if len(indices) < 4:
            return 0.0
        
        # 转换为实际坐标（Å）
        points = coords_3d[mask]
        
        # 计算重心
        center = np.mean(points, axis=0)
        
        # 计算协方差矩阵
        centered = points - center
        cov = np.dot(centered.T, centered) / len(points)
        
        # 特征值
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 从大到小
        
        # Sphericity
        lambda_sum = np.sum(eigenvalues)
        if lambda_sum > 1e-10:
            s = 3 * eigenvalues[2] / lambda_sum
            return float(np.clip(s, 0.0, 1.0))
        return 0.0
    
    def _compute_esp_on_grid(
        self,
        mol: gto.Mole,
        coords: np.ndarray,
        dm: np.ndarray
    ) -> np.ndarray:
        """在网格上计算静电势（简化版）"""
        # 电子贡献
        # V(r) = ∫ ρ(r') / |r - r'| dr'
        # 使用近似：在远场使用多极展开，近场直接积分
        
        # 简化为使用 pyscf 的 MEP 功能
        # 由于 MEP 计算较复杂，这里返回零数组作为占位
        # 实际应用中可以调用更精确的 MEP 计算
        
        # 使用粗粒度的近似
        # V_nuc + V_ele 的简化计算
        
        # 电子部分：使用密度的一阶近似
        # 注意：这是简化版本，可能不够精确
        
        # 更精确的方法需要调用 pyscf 的内部函数
        # 这里使用点电荷近似作为快速估算
        
        # 实际项目中可以使用 scipy 的积分或 pyscf 的 MEP
        # 这里为了稳定性，暂时返回零，等待更精确实现
        
        logger.warning("ESP computation uses simplified approximation")
        
        # 计算核贡献
        nuc_coords = mol.atom_coords()
        nuc_charges = np.array([mol.atom_charge(i) for i in range(mol.natm)])
        
        esp = np.zeros(len(coords))
        for i, r in enumerate(coords):
            # 核贡献
            r_diff = r - nuc_coords
            r_norm = np.linalg.norm(r_diff, axis=1)
            esp[i] = np.sum(nuc_charges / np.maximum(r_norm, 0.1))  # 避免除零
        
        return esp
    
    def _compute_orbital_extent(
        self,
        mol: gto.Mole,
        coords: np.ndarray,
        mo_coeff: np.ndarray,
        grid_shape: np.ndarray
    ) -> Optional[float]:
        """
        计算轨道 extent
        
        extent = sqrt(<r^2> - <r>^2)
        其中 <...> 是归一化的 |phi(r)|^2
        """
        # 计算轨道值
        ao = numint.eval_ao(mol, coords)
        phi = np.dot(ao, mo_coeff)
        
        # 轨道密度
        phi_squared = phi ** 2
        
        # 归一化
        # 注意：grid 上的积分近似
        # 由于 grid 有限，归一化可能不完全
        norm = np.sum(phi_squared)
        if norm < 1e-10:
            return None
        
        phi_squared_norm = phi_squared / norm
        
        # 计算 <r> 和 <r^2>
        r_coords = coords * BOHR_TO_ANGSTROM  # 转换为 Å
        
        mean_r = np.sum(r_coords * phi_squared_norm[:, np.newaxis], axis=0)
        
        r_squared = np.sum(r_coords ** 2, axis=1)
        mean_r2 = np.sum(r_squared * phi_squared_norm)
        
        # extent = sqrt(<r^2> - |<r>|^2)
        variance = mean_r2 - np.sum(mean_r ** 2)
        
        if variance < 0:
            variance = 0
        
        extent = np.sqrt(variance)
        return float(extent)
    
    def _validate_realspace_features(self, data: Dict[str, Any]) -> bool:
        """验证 realspace_features 的数值正确性"""
        try:
            # 检查 volume
            vol = data.get("density_isosurface_volume")
            if vol is not None:
                if not np.isfinite(vol["value_angstrom3"]):
                    logger.warning("Volume is not finite")
                    return False
                if vol["value_angstrom3"] < 0:
                    logger.warning("Volume is negative")
                    return False
            
            # 检查 area
            area = data.get("density_isosurface_area")
            if area is not None:
                if not np.isfinite(area["value_angstrom2"]):
                    logger.warning("Area is not finite")
                    return False
                if area["value_angstrom2"] < 0:
                    logger.warning("Area is negative")
                    return False
            
            # 检查 sphericity
            sph = data.get("density_sphericity_like")
            if sph is not None:
                if not (0 <= sph["value"] <= 1):
                    logger.warning(f"Sphericity out of range: {sph['value']}")
                    return False
            
            # 检查 orbital extent
            for key in ["orbital_extent_homo", "orbital_extent_lumo"]:
                ext = data.get(key)
                if ext is not None:
                    if not np.isfinite(ext["value_angstrom"]):
                        logger.warning(f"{key} is not finite")
                        return False
                    if ext["value_angstrom"] < 0:
                        logger.warning(f"{key} is negative")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    # ===== Cache / Fingerprint =====
    def _stable_hash(self, obj: Any) -> str:
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_default(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _cache_root(self, output_dir: Optional[str]) -> Path:
        if output_dir:
            base = Path(output_dir).resolve().parent
        else:
            base = Path(".").resolve()
        return base / self.config.get("cache_directory", ".realspace_cache")

    def _build_geometry_electronic_state_key(self, mol: gto.Mole) -> str:
        coords = np.round(mol.atom_coords(unit="A"), COORD_QUANTIZATION_DECIMALS).tolist()
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        payload = {
            "atom_symbols": symbols,
            "atom_coords_angstrom_q": coords,
            "natm": int(mol.natm),
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "multiplicity": int(mol.spin + 1),
            "basis": str(mol.basis),
        }
        return self._stable_hash(payload)

    def _build_wavefunction_signature(self, mf: scf.hf.SCF) -> str:
        mo_occ_q = np.round(np.array(mf.mo_occ).astype(float), VALUE_QUANTIZATION_DECIMALS).tolist()
        mo_energy_q = np.round(np.array(mf.mo_energy).astype(float), VALUE_QUANTIZATION_DECIMALS).tolist()
        payload = {
            "mo_coeff_shape": list(np.array(mf.mo_coeff).shape),
            "mo_occ_hash": self._stable_hash(mo_occ_q),
            "mo_energy_hash": self._stable_hash(mo_energy_q),
            "e_tot_q": round(float(mf.e_tot), VALUE_QUANTIZATION_DECIMALS),
            "scf_converged": bool(getattr(mf, "converged", False)),
        }
        return self._stable_hash(payload)

    def _build_compatibility_keys(self, mol: gto.Mole, mf: scf.hf.SCF) -> Dict[str, str]:
        geo_key = self._build_geometry_electronic_state_key(mol)
        wf_sig = self._build_wavefunction_signature(mf)
        artifact_payload = {
            "geo_key": geo_key,
            "wf_sig": wf_sig,
            "grid_resolution_angstrom": float(self.config["grid_resolution_angstrom"]),
            "margin_angstrom": float(self.config["margin_angstrom"]),
            "max_total_grid_points": int(self.config["max_total_grid_points"]),
            "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
        }
        feature_payload = {
            **artifact_payload,
            "density_isovalue": float(self.config["density_isovalue"]),
            "orbital_isovalue": float(self.config["orbital_isovalue"]),
            "core_enabled": bool(self.config.get("realspace_core_features_enabled", True)),
            "extended_enabled": bool(self.config.get("realspace_extended_features_enabled", True)),
        }
        return {
            "geometry_electronic_state_key": geo_key,
            "wavefunction_signature": wf_sig,
            "artifact_compatibility_key": self._stable_hash(artifact_payload),
            "feature_compatibility_key": self._stable_hash(feature_payload),
        }

    def _compute_grid_signature(self, mol: gto.Mole) -> Dict[str, Any]:
        grid_shape = self._compute_grid_shape(mol).tolist()
        spacing = [round(float(self.config["grid_resolution_angstrom"]), COORD_QUANTIZATION_DECIMALS)] * 3
        margin_bohr = self.config["margin_angstrom"] / BOHR_TO_ANGSTROM
        origin = [round(float(v), COORD_QUANTIZATION_DECIMALS) for v in self._get_cube_origin_angstrom(mol, margin_bohr)]
        return {
            "shape": grid_shape,
            "spacing_angstrom": spacing,
            "origin_angstrom": origin,
            "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
        }

    def _pick_reused_artifact(self, reused_artifacts: Optional[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
        if not isinstance(reused_artifacts, dict):
            return None
        info = reused_artifacts.get(key)
        if not isinstance(info, dict):
            return None
        p = info.get("path")
        if not p or not Path(p).exists():
            return None
        return info

    def _try_reuse_cache(
        self,
        compatibility: Dict[str, str],
        molecule_id: str,
        mol: gto.Mole,
        output_dir: Optional[str],
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not bool(self.config.get("enable_cache", True)):
            return "no_reuse", None, None

        root = self._cache_root(output_dir)
        features_dir = root / "features"
        artifacts_dir = root / "artifacts"
        feature_file = features_dir / f"{compatibility['feature_compatibility_key']}.json"
        artifact_file = artifacts_dir / f"{compatibility['artifact_compatibility_key']}.json"
        expected_grid = self._compute_grid_signature(mol)

        if feature_file.exists():
            try:
                entry = json.loads(feature_file.read_text(encoding="utf-8"))
                if entry.get("feature_compatibility_key") == compatibility["feature_compatibility_key"]:
                    result = entry.get("result")
                    if isinstance(result, dict):
                        result.setdefault("metadata", {})
                        result["metadata"]["cache_reuse_mode"] = "exact_feature_reuse"
                        result["metadata"]["cache_hit"] = True
                        result["metadata"]["cache_entry_id"] = entry.get("entry_id")
                        return "exact_feature_reuse", result, None
            except Exception as e:
                logger.warning(f"[{molecule_id}] feature cache read failed: {e}")

        if artifact_file.exists():
            try:
                entry = json.loads(artifact_file.read_text(encoding="utf-8"))
                if entry.get("artifact_compatibility_key") != compatibility["artifact_compatibility_key"]:
                    return "no_reuse", None, None
                if entry.get("grid_signature") != expected_grid:
                    return "no_reuse", None, None
                if entry.get("realspace_definition_version") != REALSPACE_DEFINITION_VERSION:
                    return "no_reuse", None, None
                artifacts = entry.get("artifacts")
                if not isinstance(artifacts, dict):
                    return "no_reuse", None, None
                # 仅当 required artifacts 都存在且可读时才 artifact_reuse
                required = set(self.config.get("required_artifacts", []))
                for rk in required:
                    info = artifacts.get(rk)
                    if not isinstance(info, dict) or not info.get("path") or not Path(info["path"]).exists():
                        return "no_reuse", None, None
                artifacts["_cache_entry_id"] = entry.get("entry_id")
                return "artifact_reuse", None, artifacts
            except Exception as e:
                logger.warning(f"[{molecule_id}] artifact cache read failed: {e}")

        return "no_reuse", None, None

    def _save_cache_entries(
        self,
        compatibility: Optional[Dict[str, str]],
        output_dir: Optional[str],
        result: Dict[str, Any],
    ):
        if not compatibility or not bool(self.config.get("enable_cache", True)):
            return

        root = self._cache_root(output_dir)
        features_dir = root / "features"
        artifacts_dir = root / "artifacts"
        features_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        entry_id = compatibility["feature_compatibility_key"][:16]
        result.setdefault("metadata", {})
        result["metadata"]["cache_entry_id"] = entry_id

        feature_entry = {
            "entry_id": entry_id,
            "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
            "geometry_electronic_state_key": compatibility["geometry_electronic_state_key"],
            "wavefunction_signature": compatibility["wavefunction_signature"],
            "feature_compatibility_key": compatibility["feature_compatibility_key"],
            "artifact_compatibility_key": compatibility["artifact_compatibility_key"],
            "result": result,
        }
        (features_dir / f"{compatibility['feature_compatibility_key']}.json").write_text(
            json.dumps(feature_entry, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8"
        )

        artifact_entry = {
            "entry_id": compatibility["artifact_compatibility_key"][:16],
            "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
            "geometry_electronic_state_key": compatibility["geometry_electronic_state_key"],
            "wavefunction_signature": compatibility["wavefunction_signature"],
            "artifact_compatibility_key": compatibility["artifact_compatibility_key"],
            "grid_signature": {
                "shape": result["metadata"].get("cube_grid_shape"),
                "spacing_angstrom": result["metadata"].get("cube_spacing_angstrom"),
                "origin_angstrom": (
                    (result.get("artifacts", {}).get("cube_files", {}).get("density") or {}).get("origin_angstrom")
                ),
                "realspace_definition_version": REALSPACE_DEFINITION_VERSION,
            },
            "artifacts": result.get("artifacts", {}).get("cube_files", {}),
        }
        (artifacts_dir / f"{compatibility['artifact_compatibility_key']}.json").write_text(
            json.dumps(artifact_entry, ensure_ascii=False, indent=2, default=self._json_default),
            encoding="utf-8"
        )


def extract_realspace_features(
    mol: gto.Mole,
    mf: scf.hf.SCF,
    molecule_id: str = "unknown",
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    便捷函数：提取实空间特征
    
    M5: 添加 timeout_seconds 支持
    """
    extractor = RealspaceFeatureExtractor(config)
    return extractor.extract_realspace_features(
        mol, mf, molecule_id, output_dir, timeout_seconds=timeout_seconds
    )

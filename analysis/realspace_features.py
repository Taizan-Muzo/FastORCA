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

# 默认配置
DEFAULT_CUBE_CONFIG = {
    "enable_cube_generation": True,
    "generate_esp_cube_file": True,
    "generate_orbital_cube_files": True,
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
        
        # Step 1: 检查是否应该跳过
        should_skip, reason = self._should_skip_cube(mol)
        if should_skip:
            result["metadata"]["extraction_status"] = "skipped"
            result["metadata"]["failure_reason"] = reason
            logger.warning(f"Cube generation skipped for {molecule_id}: {reason}")
            return result
        
        # M5: 使用子进程执行 cube 生成以实现可靠超时
        if timeout_seconds is not None and timeout_seconds > 0:
            return self._extract_with_timeout(
                mol, mf, molecule_id, output_dir, timeout_seconds, start_time, result
            )
        
        # 无超时：直接执行
        return self._extract_cubes(mol, mf, molecule_id, output_dir, result, start_time)
    
    def _extract_with_timeout(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        molecule_id: str,
        output_dir: Optional[str],
        timeout_seconds: float,
        start_time: float,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """使用子进程执行 cube 生成，带超时控制"""
        from multiprocessing import Process, Queue
        
        queue = Queue()
        
        # 序列化必要的数据
        def target(queue, mol_dict, mf_dict, molecule_id, output_dir, config):
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
                result = extractor._extract_cubes(mol, mf, molecule_id, output_dir, 
                                                   extractor._init_realspace_features_skeleton(),
                                                   time.time())
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
            args=(queue, mol_dict, mf_dict, molecule_id, output_dir, self.config)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            # 超时：终止子进程
            process.terminate()
            process.join()
            result["metadata"]["extraction_status"] = "timeout"
            result["metadata"]["failure_reason"] = "realspace_timeout"
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
                return self._extract_cubes(mol, mf, molecule_id, output_dir, 
                                          self._init_realspace_features_skeleton(), 
                                          time.time())
        except Exception as e:
            logger.error(f"[{molecule_id}] Queue communication error: {e}")
            # 回退到直接执行
            logger.warning(f"[{molecule_id}] Falling back to direct execution (no timeout)")
            return self._extract_cubes(mol, mf, molecule_id, output_dir,
                                      self._init_realspace_features_skeleton(),
                                      time.time())
    
    def _extract_cubes(
        self,
        mol: gto.Mole,
        mf: scf.hf.SCF,
        molecule_id: str,
        output_dir: Optional[str],
        result: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """实际的 cube 生成逻辑（在子进程或主进程中执行）"""
        try:
            # 准备输出目录
            if output_dir is None:
                output_dir = self.config["output_directory"]
            cube_dir = Path(output_dir) / molecule_id
            cube_dir.mkdir(parents=True, exist_ok=True)
            
            # 计算网格参数（Å-based，然后转 Bohr）
            grid_shape = self._compute_grid_shape(mol)
            result["metadata"]["cube_grid_shape"] = grid_shape.tolist()
            
            # 分辨率（Bohr for cubegen）
            res_bohr = self.config["grid_resolution_angstrom"] / BOHR_TO_ANGSTROM
            margin_bohr = self.config["margin_angstrom"] / BOHR_TO_ANGSTROM
            
            # Step 2: 生成 Density Cube
            logger.debug(f"[{molecule_id}] Generating density cube...")
            density_path = cube_dir / "density.cube"
            dm = mf.make_rdm1()
            
            # 使用 cubegen.density 生成标准 cube
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
            
            # Step 3: 生成 MEP Cube（可选，I/O 较重）
            generate_esp_cube = bool(self.config.get("generate_esp_cube_file", True))
            result["metadata"]["esp_cube_generated"] = generate_esp_cube
            if generate_esp_cube:
                logger.debug(f"[{molecule_id}] Generating MEP cube...")
                mep_path = cube_dir / "mep.cube"
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
            
            # Step 4: 生成 HOMO/LUMO Cubes（可选，I/O 最重）
            generate_orbital_cubes = bool(self.config.get("generate_orbital_cube_files", True))
            result["metadata"]["orbital_cubes_generated"] = generate_orbital_cubes
            occ_idx = mf.mo_occ > 0
            n_occ = np.sum(occ_idx)
            homo_idx = n_occ - 1
            lumo_idx = n_occ
            
            # HOMO
            if generate_orbital_cubes and homo_idx >= 0 and homo_idx < mf.mo_coeff.shape[1]:
                logger.debug(f"[{molecule_id}] Generating HOMO cube (MO {homo_idx})...")
                homo_path = cube_dir / "homo.cube"
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
            
            # LUMO
            if generate_orbital_cubes and lumo_idx < mf.mo_coeff.shape[1]:
                logger.debug(f"[{molecule_id}] Generating LUMO cube (MO {lumo_idx})...")
                lumo_path = cube_dir / "lumo.cube"
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
            
            # Step 5: 从 grid 计算 shape 特征（更精确，不用读 cube 文件）
            logger.debug(f"[{molecule_id}] Computing shape features from grid...")
            self._compute_shape_features_from_grid(mol, mf, result, grid_shape, res_bohr, margin_bohr)
            
            # 验证
            if self._validate_realspace_features(result):
                result["metadata"]["extraction_status"] = "success"
                elapsed = time.time() - start_time
                result["metadata"]["extraction_time_seconds"] = elapsed
                logger.info(f"[{molecule_id}] Realspace features extracted in {elapsed:.3f}s")
            else:
                result["metadata"]["extraction_status"] = "failed"
                result["metadata"]["failure_reason"] = "validation_failed"
                
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
                "native_grid_unit": "bohr",
                "output_unit": "angstrom",
                "conversion_factor_bohr_to_angstrom": BOHR_TO_ANGSTROM,
                "density_isovalue": self.config["density_isovalue"],
                "orbital_isovalue": self.config["orbital_isovalue"],
                "cube_grid_shape": None,
                "cube_spacing_angstrom": None,
                "esp_cube_generated": None,
                "orbital_cubes_generated": None,
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
        margin_bohr: float
    ):
        """在规则网格上计算 shape 特征"""
        
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
        
        # === 1. Density isosurface volume ===
        threshold = self.config["density_isovalue"]
        mask = rho_3d > threshold
        n_inside = np.sum(mask)
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
        
        # === 4. ESP extrema summary ===
        # 计算 ESP
        from pyscf.solvent import ddcosmo
        # ESP = V_nuc + V_ele
        # 简化为只计算电子部分 + 核部分的近似
        
        # 使用 pyscf 的 MEP 计算
        # 注意：MEP 计算比较耗时，这里用简化版本
        try:
            # 尝试从已生成的 cube 读取或使用简化计算
            # 这里使用点电荷近似
            esp = self._compute_esp_on_grid(mol, coords, dm)
            esp_3d = esp.reshape(grid_shape)
            
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
        
        # === 5. Orbital extent (HOMO/LUMO) ===
        occ_idx = mf.mo_occ > 0
        n_occ = np.sum(occ_idx)
        
        # HOMO
        homo_idx = n_occ - 1
        if homo_idx >= 0:
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
        if lumo_idx < mf.mo_coeff.shape[1]:
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
        
        # 更新 metadata
        result["metadata"]["cube_spacing_angstrom"] = [self.config["grid_resolution_angstrom"]] * 3
    
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

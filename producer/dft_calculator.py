"""
GPU DFT Calculator Module
使用 gpu4pyscf 进行高速量子化学计算 (Hybrid Architecture)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
import tempfile
import time
import cupy
from loguru import logger
import numpy as np

# GPU PySCF imports
from pyscf import gto, lib

try:
    from gpu4pyscf.dft import RKS as GPU_RKS
    GPU_AVAILABLE = True
    logger.info("gpu4pyscf loaded, GPU acceleration enabled")
except ImportError:
    logger.warning("gpu4pyscf not available, falling back to CPU PySCF")
    GPU_AVAILABLE = False

from pyscf import dft, grad
from pyscf.dft import rks

# xTB 导入
try:
    from xtb.interface import Calculator
    from xtb.libxtb import VERBOSITY_MINIMAL
    XTB_AVAILABLE = True
    logger.info("xtb-python loaded, fast geometry optimization available")
except ImportError:
    XTB_AVAILABLE = False
    logger.warning("=" * 70)
    logger.warning("xtb-python NOT AVAILABLE!")
    logger.warning("Geometry optimization will fall back to PySCF (GPU Hybrid) or be disabled")
    logger.warning("-" * 70)

# GPU 基组兼容性提示
def check_basis_gpu_compatibility(basis: str) -> tuple[bool, str]:
    basis_lower = basis.lower()
    gpu_friendly = ['sto-3g', '3-21g', '6-31g', '6-311g', 'mini', 'midi']
    gpu_limited = ['def2-svp', 'def2-tzvp', 'def2-svpp', 'def2-tzvpp',
                   'cc-pvdz', 'cc-pvtz', 'aug-cc-pvdz', 'aug-cc-pvtz']
    
    for b in gpu_friendly:
        if b in basis_lower:
            return True, f"基组 {basis} 应该支持 GPU 计算"
    
    for b in gpu_limited:
        if b in basis_lower:
            return False, f"基组 {basis} 包含 d/f 极化函数，需强制开启笛卡尔坐标 (cart=True)"
            
    if GPU_AVAILABLE:
        return True, f"基组 {basis} GPU 兼容性未知"
    else:
        return False, "GPU 不可用"

import pickle

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available, SMILES processing disabled")


class DFTCalculator:
    def __init__(
        self,
        functional: str = "B3LYP",
        basis: str = "def2-SVP",
        verbose: int = 3,
        max_memory: int = 8000, 
        scf_conv_tol: float = 1e-9,
        geometry_optimization: bool = True,
        geo_opt_method: str = "xtb", 
        geo_opt_maxsteps: int = 100,
    ):
        self.functional = functional
        self.basis = basis
        self.verbose = verbose
        self.max_memory = max_memory
        self.scf_conv_tol = scf_conv_tol
        self.geometry_optimization = geometry_optimization
        self.geo_opt_method = geo_opt_method
        self.geo_opt_maxsteps = geo_opt_maxsteps
        self._last_run_sp_backend = None
        
        logger.info(f"DFTCalculator initialized: {functional}/{basis}")
        logger.info(f"GPU available: {GPU_AVAILABLE}")
        logger.info(f"Geometry optimization: {geometry_optimization} ({geo_opt_method})")
        
        if geometry_optimization:
            # if geo_opt_method == "xtb" and not XTB_AVAILABLE:
            #     logger.warning("xtb-python not available, falling back to pyscf hybrid")
            self.geo_opt_method = "pyscf"
                
        if GPU_AVAILABLE:
            compatible, msg = check_basis_gpu_compatibility(basis)
            if not compatible:
                logger.warning(msg)
    
    def from_smiles(
        self,
        smiles: str,
        charge: int = 0,
        spin: int = 0,
        n_conformers: int = 1,
        random_seed: int = 42,
    ) -> gto.Mole:
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES processing")
        
        try:
            mol_rdkit = Chem.MolFromSmiles(smiles)
            if mol_rdkit is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            mol_rdkit = Chem.AddHs(mol_rdkit)
            AllChem.EmbedMolecule(mol_rdkit, randomSeed=random_seed)
            
            try:
                AllChem.MMFFOptimizeMolecule(mol_rdkit)
            except:
                AllChem.UFFOptimizeMolecule(mol_rdkit)
            
            conf = mol_rdkit.GetConformer()
            coords = conf.GetPositions()
            
            atoms = []
            for i, atom in enumerate(mol_rdkit.GetAtoms()):
                symbol = atom.GetSymbol()
                x, y, z = coords[i]
                atoms.append((symbol, (x, y, z)))
            
            mol = gto.Mole()
            mol.atom = atoms
            mol.basis = self.basis
            mol.charge = charge
            mol.spin = spin
            mol.verbose = self.verbose
            mol.max_memory = self.max_memory
            # mol.cart = True  # <-- 【核心修改 1】：强制笛卡尔坐标
            mol.build()
            
            logger.info(f"Created Mole from SMILES: {smiles[:30]}...")
            
            if self.geometry_optimization:
                mol = self.run_geometry_optimization(molecule_id="from_smiles", mol_obj=mol, charge=charge, spin=spin, uhf=(spin != 0))
            
            return mol
            
        except Exception as e:
            logger.error(f"Failed to process SMILES '{smiles}': {e}")
            raise
            
    def from_xyz(
        self,
        xyz_file: str,
        charge: int = 0,
        spin: int = 0,
    ) -> gto.Mole:
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
            
            n_atoms = int(lines[0].strip())
            
            atoms = []
            for i in range(2, 2 + n_atoms):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append((symbol, (x, y, z)))
            
            mol = gto.Mole()
            mol.atom = atoms
            mol.basis = self.basis
            mol.charge = charge
            mol.spin = spin
            mol.verbose = self.verbose
            mol.max_memory = self.max_memory
            # mol.cart = True  # <-- 【核心修改 2】：强制笛卡尔坐标
            mol.build()
            
            logger.info(f"Created Mole from XYZ: {xyz_file}")
            
            if self.geometry_optimization:
                uhf = (spin != 0)
                mol = self.run_geometry_optimization(molecule_id=xyz_file, mol_obj=mol, charge=charge, spin=spin, uhf=uhf)
            
            return mol
            
        except Exception as e:
            logger.error(f"Failed to read XYZ file '{xyz_file}': {e}")
            raise
    
    def run_geometry_optimization(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
        charge: int = 0,
        spin: int = 0,
        uhf: bool = False,
    ) -> gto.Mole:
        if not self.geometry_optimization:
            return mol_obj
        
        logger.info(f"[{molecule_id}] Starting geometry optimization ({self.geo_opt_method})...")
        start_time = time.time()
        
        try:
            if self.geo_opt_method == "xtb" and XTB_AVAILABLE:
                mol_opt = self._optimize_with_xtb(molecule_id, mol_obj, charge, spin, uhf)
            elif self.geo_opt_method == "pyscf":
                # 切换为 Hybrid 方法
                mol_opt = self._optimize_with_pyscf(molecule_id, mol_obj, charge, spin)
            else:
                logger.warning(f"[{molecule_id}] No geometry optimization method available, skipping")
                return mol_obj
            
            elapsed = time.time() - start_time
            logger.info(f"[{molecule_id}] Geometry optimization completed in {elapsed:.2f}s")
            return mol_opt
            
        except Exception as e:
            logger.error(f"[{molecule_id}] Geometry optimization failed: {e}, using original structure")
            return mol_obj
    
    def _optimize_with_xtb(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
        charge: int = 0,
        spin: int = 0,
        uhf: bool = False,
    ) -> gto.Mole:
        from xtb.interface import Calculator, Param
        from xtb.libxtb import VERBOSITY_MINIMAL
        from ase import Atoms
        from ase.optimize import BFGS
        
        logger.info(f"[{molecule_id}] Using xTB (GFN2-xTB) + ASE for geometry optimization")
        
        symbols = [mol_obj.atom_symbol(i) for i in range(mol_obj.natm)]
        positions = mol_obj.atom_coords()
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.charge = charge
        atoms.spin = 1 if uhf else 0
        
        class XTBCalculator:
            def __init__(self, atoms):
                self.atoms = atoms
                
            def get_potential_energy(self, atoms=None, force_consistent=False):
                return self._get_property('energy', atoms)
                
            def get_forces(self, atoms=None):
                return -self._get_property('forces', atoms)
                
            def _get_property(self, name, atoms=None):
                if atoms is None:
                    atoms = self.atoms
                numbers = atoms.get_atomic_numbers()
                positions = atoms.get_positions() * 0.529177 
                calc = Calculator(Param.GFN2xTB, numbers, positions, 
                                charge=getattr(atoms, 'charge', 0),
                                uhf=getattr(atoms, 'spin', 0))
                calc.set_verbosity(VERBOSITY_MINIMAL)
                res = calc.singlepoint()
                
                if name == 'energy':
                    return res.get_energy()
                elif name == 'forces':
                    return res.get_gradient() * 51.42208619083232
                    
            def calculation_required(self, atoms, quantities):
                return True
        
        atoms.calc = XTBCalculator(atoms)
        
        try:
            opt = BFGS(atoms, logfile=None) 
            opt.run(fmax=0.05, steps=self.geo_opt_maxsteps)
            
            opt_coords = atoms.get_positions()
            
            new_atoms = []
            for i in range(mol_obj.natm):
                symbol = mol_obj.atom_symbol(i)
                x, y, z = opt_coords[i]
                new_atoms.append((symbol, (x, y, z)))
            
            mol_opt = gto.Mole()
            mol_opt.atom = new_atoms
            mol_opt.basis = mol_obj.basis
            mol_opt.charge = mol_obj.charge
            mol_opt.spin = mol_obj.spin
            mol_opt.verbose = mol_obj.verbose
            mol_opt.max_memory = mol_obj.max_memory
            # mol_opt.cart = True # 保持笛卡尔
            mol_opt.build()
            
            final_energy = atoms.get_potential_energy()
            logger.info(f"[{molecule_id}] xTB optimization converged, final energy: {final_energy:.6f} Hartree")
            
            return mol_opt
            
        except Exception as e:
            logger.warning(f"[{molecule_id}] xTB optimization failed: {e}, using original structure")
            return mol_obj
    
    def _optimize_with_pyscf(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
        charge: int = 0,
        spin: int = 0,
    ) -> gto.Mole:
        """
        【核心修改 3】：完全重写为 GPU-CPU 混合计算架构。
        """
        from scipy.optimize import minimize
        from gpu4pyscf.dft import RKS as GPU_RKS
        import numpy as np
        
        logger.info(f"[{molecule_id}] 🚀 Switching to Hybrid Architecture: GPU SCF + CPU Gradient")
        
        # 二次确认笛卡尔坐标状态
        # if not getattr(mol_obj, 'cart', False):
        #     mol_obj.cart = True
        #     mol_obj.build()
        
        init_coords_bohr = mol_obj.atom_coords().flatten()
        iteration = [0]
        
        def cost_function(coords_1d):
            curr_coords = coords_1d.reshape(-1, 3)
            mol_step = mol_obj.copy()
            mol_step.set_geom_(curr_coords, unit='Bohr')
            mol_step.build()
            
            # GPU 计算能量
            mf_gpu = GPU_RKS(mol_step).density_fit(auxbasis='def2-svp-jkfit')
            mf_gpu.xc = self.functional

            # 【新增】：启用色散校正
            try:
                mf_gpu.disp = 'd3bj'
            except Exception:
                pass
            
            # 【新增】：启用 Level Shift 阻断高对称分子的轨道震荡
            mf_gpu.level_shift = 0.1

            mf_gpu.conv_tol = self.scf_conv_tol
            energy = mf_gpu.kernel()
            
            mf_cpu = mf_gpu.to_cpu()
            
            # 直接调用原生梯度方法，不需要传参数，它会自动读取内部的波函数
            grad_obj = mf_cpu.nuc_grad_method()
            grad_3d = grad_obj.kernel()
            
            grad_1d = np.asarray(grad_3d, dtype=float).flatten()
            
            iteration[0] += 1
            if iteration[0] % 1 == 0:
                grad_norm = np.linalg.norm(grad_1d)
                logger.debug(f"[{molecule_id}] Opt step {iteration[0]}: E = {energy:.8f}, |grad| = {grad_norm:.6f}")
            
            cupy.get_default_memory_pool().free_all_blocks()
            return energy, grad_1d

        logger.info(f"[{molecule_id}] Starting L-BFGS-B optimization (max_steps={self.geo_opt_maxsteps})...")
        
        res = minimize(
            cost_function,
            init_coords_bohr,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.geo_opt_maxsteps, 'ftol': 1e-6, 'gtol': 1e-4, 'disp': False}
        )
        
        mol_opt = mol_obj.copy()
        mol_opt.set_geom_(res.x.reshape(-1, 3), unit='Bohr')
        mol_opt.build()
        
        logger.info(f"[{molecule_id}] ✅ Optimization finished. Steps: {res.nit}, Converged: {res.success}")
        logger.info(f"[{molecule_id}]    Final energy: {res.fun:.8f} Hartree")
        
        return mol_opt
    
    def run_sp(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
    ):
        logger.info(f"[{molecule_id}] Starting DFT calculation...")
        start_time = time.time()
        
        # 二次确认确保 SP 阶段也是笛卡尔坐标
        # if not getattr(mol_obj, 'cart', False):
        #     mol_obj.cart = True
        #     mol_obj.build()
        
        try:
            if GPU_AVAILABLE:
                logger.info(f"[{molecule_id}] Trying GPU acceleration...")
                try:
                    self._last_run_sp_backend = "gpu"
                    mf = GPU_RKS(mol_obj).density_fit(auxbasis='def2-svp-jkfit')
                    mf.xc = self.functional

                    # 【新增】：启用色散校正
                    try:
                        mf.disp = 'd3bj'
                    except Exception:
                        pass
                        
                    # 【新增】：启用 Level Shift
                    mf.level_shift = 0.1

                    mf.conv_tol = self.scf_conv_tol
                    mf.max_cycle = 100
                    
                    energy = mf.kernel()
                    
                    if not mf.converged:
                        logger.warning(f"[{molecule_id}] SCF did not converge!")
                        mf = mf.newton()
                        energy = mf.kernel()
                    
                    if hasattr(mf, 'to_cpu'):
                        mf = mf.to_cpu()
                    
                    elapsed = time.time() - start_time
                    logger.info(f"[{molecule_id}] GPU DFT completed in {elapsed:.2f}s")
                    logger.info(f"[{molecule_id}] Energy: {energy:.6f} Hartree")
                    return mf
                    
                except Exception as gpu_error:
                    import traceback
                    self._last_run_sp_backend = "gpu_fallback_cpu"
                    logger.error(f"[{molecule_id}] ❌ GPU calculation failed!")
                    logger.error(f"[{molecule_id}] === TRACEBACK START ===")
                    logger.error(traceback.format_exc())
                    logger.error(f"[{molecule_id}] === TRACEBACK END ===")
                    logger.warning(f"[{molecule_id}] ⚠️ Falling back to CPU mode")
            
            mol_obj.verbose = 0  # M5: 静音
            if self._last_run_sp_backend is None:
                self._last_run_sp_backend = "cpu"
            if mol_obj.natm > 6: 
                logger.info(f"[{molecule_id}] Using density fitting for CPU calculation")
                mf = dft.RKS(mol_obj).density_fit()
            else:
                mf = dft.RKS(mol_obj)
            mf.verbose = 0  # M5: 静音
            mf.xc = self.functional
            try:
                mf.disp = 'd3bj'
            except Exception:
                pass
            mf.conv_tol = self.scf_conv_tol
            mf.max_cycle = 100
            
            energy = mf.kernel()
            
            if not mf.converged:
                mf = mf.newton()
                energy = mf.kernel()
            
            elapsed = time.time() - start_time
            logger.info(f"[{molecule_id}] CPU DFT completed in {elapsed:.2f}s")
            
            return mf
            
        except Exception as e:
            logger.error(f"[{molecule_id}] DFT calculation failed: {e}")
            raise
    
    def export_wavefunction(
        self,
        mf: dft.rks.RKS,
        molecule_id: str,
        output_dir: str = "temp/",
    ) -> str:
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            pkl_file = output_path / f"{molecule_id}.pkl"
            
            data = {
                'mol': mf.mol,
                'mo_energy': mf.mo_energy,
                'mo_coeff': mf.mo_coeff,
                'mo_occ': mf.mo_occ,
                'e_tot': mf.e_tot,
                'converged': mf.converged,
            }
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            
            return str(pkl_file)
            
        except Exception as e:
            logger.error(f"[{molecule_id}] Failed to export wavefunction: {e}")
            raise
    
    def calculate_and_export(
        self,
        molecule_id: str,
        mol_obj: gto.Mole,
        output_dir: str = "temp/",
    ) -> dict:
        result = {
            "molecule_id": molecule_id,
            "success": False,
            "energy": None,
            "pkl_file": None,
            "pkl_path": None,
            "error": None,
            "timing_seconds": {},
            "execution_backend": None,
        }
        
        try:
            calc_start = time.time()
            mf = self.run_sp(molecule_id, mol_obj)
            run_sp_elapsed = time.time() - calc_start
            result["energy"] = mf.e_tot
            result["converged"] = mf.converged
            result["execution_backend"] = self._last_run_sp_backend or ("gpu" if GPU_AVAILABLE else "cpu")
            
            export_start = time.time()
            pkl_path = self.export_wavefunction(mf, molecule_id, output_dir)
            export_elapsed = time.time() - export_start
            result["pkl_file"] = pkl_path
            result["pkl_path"] = pkl_path
            result["success"] = True
            result["timing_seconds"] = {
                "run_sp_seconds": run_sp_elapsed,
                "export_wavefunction_seconds": export_elapsed,
                "calculate_and_export_seconds": run_sp_elapsed + export_elapsed,
            }
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"[{molecule_id}] Calculation pipeline failed: {e}")
        
        return result

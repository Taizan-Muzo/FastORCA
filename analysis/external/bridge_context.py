"""
Bridge Context 定义
用于在 unified schema 和外部工具之间传递上下文信息
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class BridgeContext:
    """
    桥接上下文
    
    包含从 unified schema 提取的、外部工具所需的所有信息
    """
    # 分子标识
    molecule_id: str
    
    # 原子信息
    atom_symbols: List[str]
    atom_coords_angstrom: List[List[float]]  # [natm, 3]
    natm: int
    
    # 电子结构信息
    charge: int
    spin: int  # N_alpha - N_beta
    multiplicity: int  # 2S + 1
    
    # 必需输入文件
    density_cube_path: str
    
    # 可选输入文件
    homo_cube_path: Optional[str] = None
    lumo_cube_path: Optional[str] = None
    esp_cube_path: Optional[str] = None
    
    # 网格元数据（来自 cube）
    cube_grid_shape: Optional[List[int]] = None  # [nx, ny, nz]
    cube_spacing_angstrom: Optional[List[float]] = None  # [dx, dy, dz]
    cube_origin_angstrom: Optional[List[float]] = None  # [x0, y0, z0]
    
    # 单位信息
    geometry_coordinate_unit: str = "angstrom"
    cube_native_unit: str = "bohr"
    cube_output_unit: str = "angstrom"
    
    # 原子索引映射
    unified_to_external_atom_index_map: Optional[List[int]] = None
    external_to_unified_atom_index_map: Optional[List[int]] = None
    
    # 附加元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> tuple[bool, str]:
        """验证 context 完整性"""
        if len(self.atom_symbols) != self.natm:
            return False, f"atom_symbols length ({len(self.atom_symbols)}) != natm ({self.natm})"
        
        if len(self.atom_coords_angstrom) != self.natm:
            return False, f"atom_coords length ({len(self.atom_coords_angstrom)}) != natm ({self.natm})"
        
        for i, coord in enumerate(self.atom_coords_angstrom):
            if len(coord) != 3:
                return False, f"atom {i} coord length != 3"
        
        if not self.density_cube_path:
            return False, "density_cube_path is required"
        
        return True, ""
    
    def get_external_atom_index(self, unified_index: int) -> int:
        """将 unified schema 原子索引转换为外部工具索引"""
        if self.unified_to_external_atom_index_map:
            return self.unified_to_external_atom_index_map[unified_index]
        return unified_index
    
    def get_unified_atom_index(self, external_index: int) -> int:
        """将外部工具原子索引转换回 unified schema 索引"""
        if self.external_to_unified_atom_index_map:
            return self.external_to_unified_atom_index_map[external_index]
        return external_index

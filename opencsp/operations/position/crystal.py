# opencsp/operations/position/crystal.py
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class CrystalPositionUpdate(StructureOperation):
    """适用于晶体的位置更新操作"""
    
    def __init__(self, **kwargs):
        """
        初始化晶体位置更新操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=3, **kwargs)
        
    def apply(self, particle: Individual, velocity: np.ndarray) -> Individual:
        """
        更新晶体粒子的位置
        
        Args:
            particle: 当前粒子
            velocity: 速度
            
        Returns:
            更新后的粒子
        """
        # 复制粒子
        new_particle = particle.copy()
        structure = new_particle.structure
        
        # 确保是晶体
        dim = get_structure_dimensionality(structure)
        if dim != 3:
            raise ValueError(f"CrystalPositionUpdate只适用于晶体(3D)，但接收到的维度为{dim}")
        
        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            # 对于晶体，我们考虑笛卡尔坐标的更新
            coords = np.array([site.coords for site in structure.sites])
            
            # 根据速度更新位置
            new_coords = coords + velocity
            
            # 更新结构
            for i, site in enumerate(structure):
                structure.translate_sites(i, new_coords[i] - site.coords, frac_coords=False)
                
            # 对于超出晶胞的原子，将它们移回晶胞内（应用周期性边界条件）
            for i, site in enumerate(structure):
                # 获取分数坐标
                frac_coords = site.frac_coords
                
                # 将分数坐标调整到[0,1)范围内
                frac_coords = frac_coords - np.floor(frac_coords)
                
                # 更新原子位置
                structure.replace(i, site.species_string, frac_coords, coords_are_cartesian=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            
            # 根据速度更新位置
            new_positions = positions + velocity
            
            # 更新结构
            structure.set_positions(new_positions)
            
            # 应用周期性边界条件
            structure.wrap()
        
        return new_particle

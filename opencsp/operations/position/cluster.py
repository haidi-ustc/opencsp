# opencsp/operations/position/cluster.py
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class ClusterPositionUpdate(StructureOperation):
    """适用于团簇的位置更新操作"""
    
    def __init__(self, **kwargs):
        """
        初始化团簇位置更新操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=1, **kwargs)
        
    def apply(self, particle: Individual, velocity: np.ndarray) -> Individual:
        """
        更新团簇粒子的位置
        
        Args:
            particle: 当前粒子
            velocity: 速度
            
        Returns:
            更新后的粒子
        """
        # 复制粒子
        new_particle = particle.copy()
        structure = new_particle.structure
        
        # 确保是团簇
        dim = get_structure_dimensionality(structure)
        if dim != 1:
            raise ValueError(f"ClusterPositionUpdate只适用于团簇(1D)，但接收到的维度为{dim}")
        
        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            coords = np.array([site.coords for site in structure.sites])
            
            # 根据速度更新位置
            new_coords = coords + velocity
            
            # 更新结构
            for i, site in enumerate(structure):
                structure.translate_sites(i, new_coords[i] - site.coords, frac_coords=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            
            # 根据速度更新位置
            new_positions = positions + velocity
            
            # 更新结构
            structure.set_positions(new_positions)
        
        return new_particle

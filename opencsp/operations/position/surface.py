# opencsp/operations/position/surface.py
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class SurfacePositionUpdate(StructureOperation):
    """适用于表面的位置更新操作"""
    
    def __init__(self, **kwargs):
        """
        初始化表面位置更新操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=2, **kwargs)
        
    def apply(self, particle: Individual, velocity: np.ndarray) -> Individual:
        """
        更新表面粒子的位置
        
        Args:
            particle: 当前粒子
            velocity: 速度
            
        Returns:
            更新后的粒子
        """
        # 复制粒子
        new_particle = particle.copy()
        structure = new_particle.structure
        
        # 确保是表面
        dim = get_structure_dimensionality(structure)
        if dim != 2:
            raise ValueError(f"SurfacePositionUpdate只适用于表面(2D)，但接收到的维度为{dim}")
        
        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            # 对于表面，我们使用笛卡尔坐标
            coords = np.array([site.coords for site in structure.sites])
            
            # 根据速度更新位置
            new_coords = coords + velocity
            
            # 更新结构
            for i, site in enumerate(structure):
                structure.translate_sites(i, new_coords[i] - site.coords, frac_coords=False)
                
            # 对于表面结构，我们可能需要保持一些原子固定
            # 例如，底层原子可能应该保持固定
            # 这里我们假设底层原子是z坐标最小的原子
            z_coords = new_coords[:, 2]
            z_min = np.min(z_coords)
            bottom_indices = np.where(z_coords < z_min + 1.0)[0]  # 底层原子（z < z_min + 1埃）
            
            # 恢复底层原子的原始位置
            for i in bottom_indices:
                structure.translate_sites(i, coords[i] - new_coords[i], frac_coords=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            
            # 根据速度更新位置
            new_positions = positions + velocity
            
            # 更新结构
            structure.set_positions(new_positions)
            
            # 保持底层原子固定
            z_coords = new_positions[:, 2]
            z_min = np.min(z_coords)
            bottom_indices = np.where(z_coords < z_min + 1.0)[0]
            
            # 恢复底层原子的原始位置
            for i in bottom_indices:
                new_positions[i] = positions[i]
                
            structure.set_positions(new_positions)
        
        return new_particle

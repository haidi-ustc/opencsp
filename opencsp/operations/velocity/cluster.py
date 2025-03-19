# opencsp/operations/velocity/cluster.py
import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

def initialize_cluster_velocity(structure: Any) -> np.ndarray:
    """
    初始化团簇结构的速度
    
    Args:
        structure: 结构对象
        
    Returns:
        速度数组
    """
    # 获取原子数量
    if hasattr(structure, 'sites'):  # pymatgen Structure
        num_atoms = len(structure.sites)
    else:  # ASE Atoms
        num_atoms = len(structure)
    
    # 为每个原子生成随机速度
    return np.random.normal(0.0, 0.1, (num_atoms, 3))

class ClusterVelocityUpdate(StructureOperation):
    """适用于团簇的速度更新操作"""
    
    def __init__(self, **kwargs):
        """
        初始化团簇速度更新操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=1, **kwargs)
        
    def apply(self, current_velocity: np.ndarray, particle: Individual, 
              personal_best: Individual, global_best: Individual,
              inertia_weight: float, cognitive_factor: float, social_factor: float) -> np.ndarray:
        """
        更新团簇粒子的速度
        
        Args:
            current_velocity: 当前速度
            particle: 当前粒子
            personal_best: 个体最佳位置
            global_best: 全局最佳位置
            inertia_weight: 惯性权重
            cognitive_factor: 认知参数
            social_factor: 社会参数
            
        Returns:
            更新后的速度
        """
        # 获取结构
        structure = particle.structure
        personal_best_structure = personal_best.structure
        global_best_structure = global_best.structure
        
        # 确保是团簇
        dim = get_structure_dimensionality(structure)
        if dim != 1:
            raise ValueError(f"ClusterVelocityUpdate只适用于团簇(1D)，但接收到的维度为{dim}")
        
        # 获取当前位置、个体最佳位置和全局最佳位置
        if hasattr(structure, 'sites'):  # pymatgen Structure
            current_pos = np.array([site.coords for site in structure.sites])
            pbest_pos = np.array([site.coords for site in personal_best_structure.sites])
            gbest_pos = np.array([site.coords for site in global_best_structure.sites])
        else:  # ASE Atoms
            current_pos = structure.get_positions()
            pbest_pos = personal_best_structure.get_positions()
            gbest_pos = global_best_structure.get_positions()
        
        # 确保所有位置数组形状相同
        num_atoms = len(current_pos)
        if len(pbest_pos) != num_atoms or len(gbest_pos) != num_atoms:
            # 如果不匹配，可能需要截断或扩展
            pbest_pos = pbest_pos[:num_atoms] if len(pbest_pos) > num_atoms else np.pad(
                pbest_pos, ((0, num_atoms - len(pbest_pos)), (0, 0)), 'constant')
            gbest_pos = gbest_pos[:num_atoms] if len(gbest_pos) > num_atoms else np.pad(
                gbest_pos, ((0, num_atoms - len(gbest_pos)), (0, 0)), 'constant')
        
        # 更新速度
        r1 = random.random()
        r2 = random.random()
        
        new_velocity = (inertia_weight * current_velocity + 
                       cognitive_factor * r1 * (pbest_pos - current_pos) + 
                       social_factor * r2 * (gbest_pos - current_pos))
        
        # 可选：限制速度大小，防止过大
        max_speed = 1.0
        speeds = np.linalg.norm(new_velocity, axis=1)
        for i in range(num_atoms):
            if speeds[i] > max_speed:
                new_velocity[i] = (new_velocity[i] / speeds[i]) * max_speed
        
        return new_velocity

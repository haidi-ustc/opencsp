# opencsp/operations/velocity/crystal.py
import random
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

def initialize_crystal_velocity(structure: Any) -> np.ndarray:
    """
    初始化晶体结构的速度
    
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
    # 晶体结构：各向同性随机速度
    velocity = np.random.normal(0.0, 0.05, (num_atoms, 3))
    
    return velocity

class CrystalVelocityUpdate(StructureOperation):
    """适用于晶体的速度更新操作"""
    
    def __init__(self, **kwargs):
        """
        初始化晶体速度更新操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=3, **kwargs)
        
    def apply(self, current_velocity: np.ndarray, particle: Individual, 
              personal_best: Individual, global_best: Individual,
              inertia_weight: float, cognitive_factor: float, social_factor: float) -> np.ndarray:
        """
        更新晶体粒子的速度
        
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
        
        # 确保是晶体
        dim = get_structure_dimensionality(structure)
        if dim != 3:
            raise ValueError(f"CrystalVelocityUpdate只适用于晶体(3D)，但接收到的维度为{dim}")
        
        # 获取当前位置、个体最佳位置和全局最佳位置
        if hasattr(structure, 'sites'):  # pymatgen Structure
            # 对于晶体，我们考虑分数坐标
            current_pos = np.array([site.frac_coords for site in structure.sites])
            pbest_pos = np.array([site.frac_coords for site in personal_best_structure.sites])
            gbest_pos = np.array([site.frac_coords for site in global_best_structure.sites])
            
            # 计算分数坐标差异时考虑周期性边界条件
            pbest_diff = pbest_pos - current_pos
            pbest_diff = pbest_diff - np.round(pbest_diff)  # 应用最小图像约定
            
            gbest_diff = gbest_pos - current_pos
            gbest_diff = gbest_diff - np.round(gbest_diff)  # 应用最小图像约定
            
            # 转换回笛卡尔坐标
            pbest_diff_cart = np.zeros_like(pbest_diff)
            gbest_diff_cart = np.zeros_like(gbest_diff)
            
            for i in range(len(pbest_diff)):
                pbest_diff_cart[i] = np.dot(pbest_diff[i], structure.lattice.matrix)
                gbest_diff_cart[i] = np.dot(gbest_diff[i], structure.lattice.matrix)
                
            # 更新速度（在笛卡尔坐标系中）
            r1 = random.random()
            r2 = random.random()
            
            new_velocity = (inertia_weight * current_velocity + 
                           cognitive_factor * r1 * pbest_diff_cart + 
                           social_factor * r2 * gbest_diff_cart)
        else:  # ASE Atoms
            # 对于ASE，我们也考虑分数坐标
            current_pos = structure.get_scaled_positions()
            pbest_pos = personal_best_structure.get_scaled_positions()
            gbest_pos = global_best_structure.get_scaled_positions()
            
            # 计算分数坐标差异时考虑周期性边界条件
            pbest_diff = pbest_pos - current_pos
            pbest_diff = pbest_diff - np.round(pbest_diff)
            
            gbest_diff = gbest_pos - current_pos
            gbest_diff = gbest_diff - np.round(gbest_diff)
            
            # 转换回笛卡尔坐标
            cell = structure.get_cell()
            pbest_diff_cart = np.dot(pbest_diff, cell)
            gbest_diff_cart = np.dot(gbest_diff, cell)
            
            # 更新速度
            r1 = random.random()
            r2 = random.random()
            
            new_velocity = (inertia_weight * current_velocity + 
                           cognitive_factor * r1 * pbest_diff_cart + 
                           social_factor * r2 * gbest_diff_cart)
        
        # 限制速度大小
        max_speed = 0.5  # 晶体中速度应较小
        speeds = np.linalg.norm(new_velocity, axis=1)
        for i in range(len(speeds)):
            if speeds[i] > max_speed:
                new_velocity[i] = (new_velocity[i] / speeds[i]) * max_speed
        
        return new_velocity

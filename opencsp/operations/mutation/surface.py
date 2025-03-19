# opencsp/operations/mutation/surface.py
import random
import numpy as np
from typing import Any, Optional, Dict, List, Tuple

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class SurfaceMutation(StructureOperation):
    """适用于表面的变异操作"""
    
    def __init__(self, **kwargs):
        """
        初始化表面变异操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=2, **kwargs)
        self.mutation_strength = kwargs.get('mutation_strength', 0.3)
        self.method = kwargs.get('method', 'displacement')
        self.mutation_probability = kwargs.get('mutation_probability', 0.2)
        
    def apply(self, individual: Individual, **kwargs) -> Individual:
        """
        实现表面结构的变异
        
        Args:
            individual: 要变异的个体
            **kwargs: 其他参数，可以覆盖初始设置
            
        Returns:
            变异后的个体
        """
        # 使用传入的参数覆盖默认参数
        strength = kwargs.get('step_size', kwargs.get('mutation_strength', self.mutation_strength))
        method = kwargs.get('method', self.method)
        probability = kwargs.get('mutation_probability', self.mutation_probability)
        
        # 确保结构是表面
        dim = get_structure_dimensionality(individual.structure)
        if dim != 2:
            raise ValueError(f"SurfaceMutation只适用于表面(2D)，但接收到的维度为{dim}")
        
        # 根据方法选择不同的变异策略
        if method == 'displacement':
            return self._displacement_mutation(individual, strength, probability)
        elif method == 'adatom':
            return self._adatom_mutation(individual)
        elif method == 'rattle_surface':
            return self._rattle_surface_mutation(individual, strength, probability)
        else:
            raise ValueError(f"未知的变异方法：{method}")
    
    def _displacement_mutation(self, individual: Individual, strength: float, probability: float) -> Individual:
        """
        位移变异：随机移动表面原子
        
        Args:
            individual: 要变异的个体
            strength: 变异强度
            probability: 变异概率
            
        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure
        
        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            coords = np.array([site.coords for site in structure.sites])
            
            # 确定表面原子（假设z方向为垂直表面的方向）
            z_coords = coords[:, 2]
            z_threshold = np.max(z_coords) - 2.0  # 假设表面在最上层2埃处
            
            surface_indices = np.where(z_coords > z_threshold)[0]
            
            # 随机移动表面原子
            for i in surface_indices:
                if random.random() < probability:
                    # 生成随机位移（主要在xy平面内）
                    displacement = np.random.normal(0, strength, 3)
                    displacement[2] *= 0.3  # 减小z方向位移
                    
                    # 应用位移
                    structure.translate_sites(i, displacement, frac_coords=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            
            # 确定表面原子
            z_coords = positions[:, 2]
            z_threshold = np.max(z_coords) - 2.0
            
            surface_indices = np.where(z_coords > z_threshold)[0]
            
            # 随机移动表面原子
            for i in surface_indices:
                if random.random() < probability:
                    # 生成随机位移（主要在xy平面内）
                    displacement = np.random.normal(0, strength, 3)
                    displacement[2] *= 0.3  # 减小z方向位移
                    
                    # 应用位移
                    positions[i] += displacement
            
            # 更新结构
            structure.set_positions(positions)
        
        return new_individual
    
    def _adatom_mutation(self, individual: Individual) -> Individual:
        """
        吸附原子变异：在表面添加或移除吸附原子
        
        Args:
            individual: 要变异的个体
            
        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure
        
        # 获取原子坐标和元素
        if hasattr(structure, 'sites'):  # pymatgen Structure
            coords = np.array([site.coords for site in structure.sites])
            species = [site.species_string for site in structure.sites]
            lattice = structure.lattice
            
            # 确定表面原子
            z_coords = coords[:, 2]
            z_max = np.max(z_coords)
            z_threshold = z_max - 2.0
            
            surface_indices = np.where(z_coords > z_threshold)[0]
            
            # 随机决定添加或移除吸附原子
            if random.random() < 0.5 and len(surface_indices) > 0:
                # 移除表面原子
                remove_idx = random.choice(surface_indices)
                structure.remove_sites([remove_idx])
            else:
                # 添加吸附原子
                # 随机选择一个表面位置
                if len(surface_indices) > 0:
                    base_idx = random.choice(surface_indices)
                    base_pos = coords[base_idx]
                else:
                    # 如果没有明显的表面原子，使用最高的原子
                    max_z_idx = np.argmax(z_coords)
                    base_pos = coords[max_z_idx]
                
                # 随机选择一个元素（从现有元素中）
                new_element = random.choice(species)
                
                # 生成吸附位置（在基础位置上方1.5-2.5埃）
                adsorption_height = 1.5 + random.random() * 1.0
                new_pos = base_pos.copy()
                new_pos[2] += adsorption_height
                
                # 添加吸附原子
                structure.append(new_element, new_pos, coords_are_cartesian=True)
        else:  # ASE Atoms
            positions = structure.get_positions()
            symbols = structure.get_chemical_symbols()
            
            # 确定表面原子
            z_coords = positions[:, 2]
            z_max = np.max(z_coords)
            z_threshold = z_max - 2.0
            
            surface_indices = np.where(z_coords > z_threshold)[0]
            
            # 随机决定添加或移除吸附原子
            if random.random() < 0.5 and len(surface_indices) > 0:
                # 移除表面原子
                remove_idx = random.choice(surface_indices)
                
                # 创建新的原子列表，排除移除的原子
                new_symbols = [sym for i, sym in enumerate(symbols) if i != remove_idx]
                new_positions = np.delete(positions, remove_idx, axis=0)

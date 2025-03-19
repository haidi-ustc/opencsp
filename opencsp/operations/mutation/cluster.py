# opencsp/operations/mutation/cluster.py
import random
import numpy as np
from typing import Any, Optional, Dict, List, Tuple

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class ClusterMutation(StructureOperation):
    """适用于团簇的变异操作"""
    
    def __init__(self, **kwargs):
        """
        初始化团簇变异操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=1, **kwargs)
        self.mutation_strength = kwargs.get('mutation_strength', 0.5)
        self.method = kwargs.get('method', 'displacement')
        self.mutation_probability = kwargs.get('mutation_probability', 0.3)  # 每个原子变异的概率
        
    def apply(self, individual: Individual, **kwargs) -> Individual:
        """
        实现团簇结构的变异
        
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
        
        # 确保结构是团簇
        dim = get_structure_dimensionality(individual.structure)
        if dim != 1:
            raise ValueError(f"ClusterMutation只适用于团簇(1D)，但接收到的维度为{dim}")
        
        # 根据方法选择不同的变异策略
        if method == 'displacement':
            return self._displacement_mutation(individual, strength, probability)
        elif method == 'rattle':
            return self._rattle_mutation(individual, strength, probability)
        elif method == 'twist':
            return self._twist_mutation(individual, strength, probability)
        elif method == 'permutation':
            return self._permutation_mutation(individual)
        else:
            raise ValueError(f"未知的变异方法：{method}")
    
    def _displacement_mutation(self, individual: Individual, strength: float, probability: float) -> Individual:
        """
        位移变异：随机移动部分原子
        
        Args:
            individual: 要变异的个体
            strength: 变异强度，决定位移的大小
            probability: 每个原子变异的概率
            
        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure
        
        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            # 对于pymatgen结构，我们需要直接修改site_properties
            coords = np.array([site.coords for site in structure.sites])
            num_atoms = len(coords)
            
            # 随机选择原子并位移
            for i in range(num_atoms):
                if random.random() < probability:
                    # 生成随机位移
                    displacement = np.random.normal(0, strength, 3)
                    coords[i] += displacement
            
            # 更新结构
            for i, site in enumerate(structure):
                structure.translate_sites(i, coords[i] - site.coords, frac_coords=False)
        else:  # ASE Atoms
            # 对于ASE结构，直接修改positions
            positions = structure.get_positions()
            num_atoms = len(positions)
            
            # 随机选择原子并位移
            for i in range(num_atoms):
                if random.random() < probability:
                    # 生成随机位移
                    displacement = np.random.normal(0, strength, 3)
                    positions[i] += displacement
            
            # 更新结构
            structure.set_positions(positions)
        
        return new_individual
    
    def _rattle_mutation(self, individual: Individual, strength: float, probability: float) -> Individual:
        """
        抖动变异：在当前位置周围随机扰动所有原子
        # opencsp/operations/mutation/cluster.py (continued)
        抖动变异：在当前位置周围随机扰动所有原子

        Args:
            individual: 要变异的个体
            strength: 变异强度，决定抖动的大小
            probability: 每个原子变异的概率

        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure

        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            coords = np.array([site.coords for site in structure.sites])
            num_atoms = len(coords)

            # 对每个原子应用随机扰动
            for i in range(num_atoms):
                if random.random() < probability:
                    # 生成随机方向
                    direction = np.random.rand(3) - 0.5
                    direction = direction / np.linalg.norm(direction)

                    # 生成随机强度（在[0, strength]范围内）
                    magnitude = strength * random.random()

                    # 应用位移
                    coords[i] += direction * magnitude

            # 更新结构
            for i, site in enumerate(structure):
                structure.translate_sites(i, coords[i] - site.coords, frac_coords=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            num_atoms = len(positions)

            # 对每个原子应用随机扰动
            for i in range(num_atoms):
                if random.random() < probability:
                    # 生成随机方向
                    direction = np.random.rand(3) - 0.5
                    direction = direction / np.linalg.norm(direction)

                    # 生成随机强度
                    magnitude = strength * random.random()

                    # 应用位移
                    positions[i] += direction * magnitude

            # 更新结构
            structure.set_positions(positions)

        return new_individual

    def _twist_mutation(self, individual: Individual, strength: float, probability: float) -> Individual:
        """
        扭转变异：绕随机轴旋转一部分原子

        Args:
            individual: 要变异的个体
            strength: 变异强度，决定旋转角度的大小
            probability: 每个原子变异的概率

        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure

        # 获取原子坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            coords = np.array([site.coords for site in structure.sites])
        else:  # ASE Atoms
            coords = structure.get_positions()

        num_atoms = len(coords)

        # 计算团簇中心
        center = np.mean(coords, axis=0)

        # 生成随机旋转轴
        axis = np.random.rand(3) - 0.5
        axis = axis / np.linalg.norm(axis)

        # 随机旋转角度（取决于强度）
        angle = strength * 2 * np.pi * (random.random() - 0.5)

        # 创建旋转矩阵
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([
            [cos_angle + axis[0]**2 * (1 - cos_angle),
             axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
             axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
            [axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
             cos_angle + axis[1]**2 * (1 - cos_angle),
             axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
            [axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
             axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
             cos_angle + axis[2]**2 * (1 - cos_angle)]
        ])

        # 应用旋转
        new_coords = coords.copy()
        for i in range(num_atoms):
            if random.random() < probability:
                # 将原子移到中心
                rel_pos = coords[i] - center
                # 旋转
                rotated_pos = np.dot(rotation_matrix, rel_pos)
                # 移回原位置
                new_coords[i] = center + rotated_pos

        # 更新结构
        if hasattr(structure, 'sites'):  # pymatgen Structure
            for i, site in enumerate(structure):
                structure.translate_sites(i, new_coords[i] - site.coords, frac_coords=False)
        else:  # ASE Atoms
            structure.set_positions(new_coords)

        return new_individual

    def _permutation_mutation(self, individual: Individual) -> Individual:
        """
        置换变异：交换两个相同类型的原子位置

        Args:
            individual: 要变异的个体

        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure

        # 获取原子类型和坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            species = [site.species_string for site in structure.sites]
            coords = np.array([site.coords for site in structure.sites])
        else:  # ASE Atoms
            species = structure.get_chemical_symbols()
            coords = structure.get_positions()

        num_atoms = len(species)

        # 按元素类型分组
        element_groups = {}
        for i, element in enumerate(species):
            if element not in element_groups:
                element_groups[element] = []
            element_groups[element].append(i)

        # 尝试交换相同类型的原子
        for element, indices in element_groups.items():
            if len(indices) >= 2:  # 至少需要两个相同类型的原子
                # 随机选择两个不同的原子
                i, j = random.sample(indices, 2)

                # 交换位置
                if hasattr(structure, 'sites'):  # pymatgen Structure
                    temp = coords[i].copy()
                    structure.translate_sites(i, coords[j] - coords[i], frac_coords=False)
                    structure.translate_sites(j, temp - coords[j], frac_coords=False)
                else:  # ASE Atoms
                    positions = coords.copy()
                    positions[i], positions[j] = positions[j].copy(), positions[i].copy()
                    structure.set_positions(positions)

                break  # 每次只交换一对原子

        return new_individual

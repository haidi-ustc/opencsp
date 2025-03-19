# opencsp/operations/mutation/crystal.py
import random
import numpy as np
from typing import Any, Optional, Dict, List, Tuple

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class CrystalMutation(StructureOperation):
    """适用于晶体的变异操作"""
    
    def __init__(self, **kwargs):
        """
        初始化晶体变异操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=3, **kwargs)
        self.mutation_strength = kwargs.get('mutation_strength', 0.2)
        self.method = kwargs.get('method', 'lattice')
        self.mutation_probability = kwargs.get('mutation_probability', 0.2)
        
    def apply(self, individual: Individual, **kwargs) -> Individual:
        """
        实现晶体结构的变异
        
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
        
        # 确保结构是晶体
        dim = get_structure_dimensionality(individual.structure)
        if dim != 3:
            raise ValueError(f"CrystalMutation只适用于晶体(3D)，但接收到的维度为{dim}")
        
        # 根据方法选择不同的变异策略
        if method == 'lattice':
            return self._lattice_mutation(individual, strength)
        elif method == 'atomic_displacement':
            return self._atomic_displacement_mutation(individual, strength, probability)
        elif method == 'strain':
            return self._strain_mutation(individual, strength)
        else:
            raise ValueError(f"未知的变异方法：{method}")
    
    def _lattice_mutation(self, individual: Individual, strength: float) -> Individual:
        """
        晶格变异：扰动晶格参数
        
        Args:
            individual: 要变异的个体
            strength: 变异强度
            
        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure
        
        if hasattr(structure, 'lattice'):  # pymatgen Structure
            # 获取当前晶格参数
            a, b, c, alpha, beta, gamma = structure.lattice.parameters
            
            # 生成随机扰动（晶格长度）
            delta_a = a * (1.0 + (random.random() - 0.5) * 2 * strength)
            delta_b = b * (1.0 + (random.random() - 0.5) * 2 * strength)
            delta_c = c * (1.0 + (random.random() - 0.5) * 2 * strength)
            
            # 生成随机扰动（晶格角度，较小）
            angle_strength = strength * 0.3  # 角度变化应该较小
            delta_alpha = alpha + (random.random() - 0.5) * 2 * angle_strength * 10.0
            delta_beta = beta + (random.random() - 0.5) * 2 * angle_strength * 10.0
            delta_gamma = gamma + (random.random() - 0.5) * 2 * angle_strength * 10.0
            
            # 确保角度在合理范围内
            delta_alpha = min(max(delta_alpha, 60.0), 120.0)
            delta_beta = min(max(delta_beta, 60.0), 120.0)
            delta_gamma = min(max(delta_gamma, 60.0), 120.0)
            
            # 创建新晶格
            from pymatgen.core import Lattice
            new_lattice = Lattice.from_parameters(delta_a, delta_b, delta_c, delta_alpha, delta_beta, delta_gamma)
            
            # 更新结构
            structure.lattice = new_lattice
        else:  # ASE Atoms
            # 获取当前晶胞
            cell = structure.get_cell()
            
            # 生成随机扰动矩阵
            strain_matrix = np.eye(3) + (np.random.rand(3, 3) - 0.5) * 2 * strength
            
            # 应用扰动
            new_cell = np.dot(cell, strain_matrix)
            
            # 更新结构
            structure.set_cell(new_cell, scale_atoms=True)
        
        return new_individual
    
    def _atomic_displacement_mutation(self, individual: Individual, strength: float, probability: float) -> Individual:
        """
        原子位移变异：随机移动部分原子
        
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
            for i, site in enumerate(structure):
                if random.random() < probability:
                    # 生成随机位移
                    displacement = np.random.normal(0, strength, 3)
                    
                    # 应用位移
                    structure.translate_sites(i, displacement, frac_coords=False)
        else:  # ASE Atoms
            positions = structure.get_positions()
            
            # 随机移动原子
            for i in range(len(positions)):
                if random.random() < probability:
                    # 生成随机位移
                    displacement = np.random.normal(0, strength, 3)
                    
                    # 应用位移
                    positions[i] += displacement
            
            # 更新结构
            structure.set_positions(positions)
        
        return new_individual
    
    def _strain_mutation(self, individual: Individual, strength: float) -> Individual:
        """
        应变变异：应用随机应变张量
        
        Args:
            individual: 要变异的个体
            strength: 变异强度
            
        Returns:
            变异后的个体
        """
        # 复制个体
        new_individual = individual.copy()
        structure = new_individual.structure
        
        # 生成对称应变张量
        # 注意：应变张量应该是对称的，以确保体积变化合理
        strain = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    # 对角元素（拉伸/压缩）
                    strain[i, j] = (random.random() - 0.5) * 2 * strength
                else:
                    # 非对角元素（剪切）
                    strain[i, j] = (random.random() - 0.5) * 2 * strength * 0.5
                    strain[j, i] = strain[i, j]
        
        # 转换为形变矩阵
        deformation = np.eye(3) + strain
        
        if hasattr(structure, 'lattice'):  # pymatgen Structure
            # 应用应变到晶格
            matrix = structure.lattice.matrix
            new_matrix = np.dot(matrix, deformation)
            
            # 创建新晶格
            from pymatgen.core import Lattice
            new_lattice = Lattice(new_matrix)
            
            # 更新结构
            structure.lattice =new_lattice
        else:  # ASE Atoms
            # 应用应变到晶胞
            cell = structure.get_cell()
            new_cell = np.dot(cell, deformation)
            
            # 更新结构
            structure.set_cell(new_cell, scale_atoms=True)
        
        return new_individual

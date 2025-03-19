# opencsp/operations/crossover/cluster.py
import random
import numpy as np
from typing import List, Tuple, Any

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class ClusterCrossover(StructureOperation):
    """适用于团簇的交叉操作"""
    
    def __init__(self, **kwargs):
        """
        初始化团簇交叉操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=1, **kwargs)
        self.method = kwargs.get('method', 'cut_splice')
        
    def apply(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        实现团簇结构的交叉
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 确保结构是团簇
        dim1 = get_structure_dimensionality(parent1.structure)
        dim2 = get_structure_dimensionality(parent2.structure)
        
        if dim1 != 1 or dim2 != 1:
            raise ValueError(f"ClusterCrossover只适用于团簇(1D)，但接收到的维度为{dim1}和{dim2}")
        
        # 根据方法选择不同的交叉策略
        if self.method == 'cut_splice':
            return self._cut_splice_crossover(parent1, parent2)
        elif self.method == 'weighted_average':
            return self._weighted_average_crossover(parent1, parent2)
        else:
            raise ValueError(f"未知的交叉方法：{self.method}")
    
    def _cut_splice_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        切割拼接交叉操作
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 获取父代结构
        struct1 = parent1.structure
        struct2 = parent2.structure
        
        # 获取原子坐标和元素
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            coords1 = np.array([site.coords for site in struct1.sites])
            species1 = [site.species_string for site in struct1.sites]
            coords2 = np.array([site.coords for site in struct2.sites])
            species2 = [site.species_string for site in struct2.sites]
            lattice = struct1.lattice
        else:  # ASE Atoms
            coords1 = struct1.get_positions()
            species1 = struct1.get_chemical_symbols()
            coords2 = struct2.get_positions()
            species2 = struct2.get_chemical_symbols()
            cell = struct1.get_cell()
        
        # 确保两个结构有相同数量的原子
        if len(coords1) != len(coords2) or len(species1) != len(species2):
            # 如果原子数不同，尝试添加或移除原子使它们匹配
            min_atoms = min(len(coords1), len(coords2))
            coords1 = coords1[:min_atoms]
            species1 = species1[:min_atoms]
            coords2 = coords2[:min_atoms]
            species2 = species2[:min_atoms]
        
        # 计算团簇中心
        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        
        # 随机选择切割平面的法向量
        normal = np.random.rand(3)
        normal = normal / np.linalg.norm(normal)
        
        # 计算每个原子到切割平面的距离
        dist1 = np.dot(coords1 - center1, normal)
        dist2 = np.dot(coords2 - center2, normal)
        
        # 创建子代1：父代1的一半 + 父代2的一半
        child1_indices1 = np.where(dist1 >= 0)[0]
        child1_indices2 = np.where(dist2 < 0)[0]
        
        child1_coords = np.vstack([coords1[child1_indices1], coords2[child1_indices2]])
        child1_species = [species1[i] for i in child1_indices1] + [species2[i] for i in child1_indices2]
        
        # 创建子代2：父代1的另一半 + 父代2的另一半
        child2_indices1 = np.where(dist1 < 0)[0]
        child2_indices2 = np.where(dist2 >= 0)[0]
        
        child2_coords = np.vstack([coords1[child2_indices1], coords2[child2_indices2]])
        child2_species = [species1[i] for i in child2_indices1] + [species2[i] for i in child2_indices2]
        
        # 创建新结构
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            from pymatgen.core import Structure
            child1_struct = Structure(lattice, child1_species, child1_coords, coords_are_cartesian=True)
            child2_struct = Structure(lattice, child2_species, child2_coords, coords_are_cartesian=True)
        else:  # ASE Atoms
            from ase import Atoms
            child1_struct = Atoms(symbols=child1_species, positions=child1_coords, cell=cell, pbc=False)
            child2_struct = Atoms(symbols=child2_species, positions=child2_coords, cell=cell, pbc=False)
        
        # 创建子代个体
        child1 = Individual(child1_struct)
        child2 = Individual(child2_struct)
        
        return [child1, child2]
    
    def _weighted_average_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        加权平均交叉操作
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 获取父代结构
        struct1 = parent1.structure
        struct2 = parent2.structure
        
        # 获取原子坐标和元素
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            coords1 = np.array([site.coords for site in struct1.sites])
            species1 = [site.species_string for site in struct1.sites]
            coords2 = np.array([site.coords for site in struct2.sites])
            species2 = [site.species_string for site in struct2.sites]
            lattice = struct1.lattice
        else:  # ASE Atoms
            coords1 = struct1.get_positions()
            species1 = struct1.get_chemical_symbols()
            coords2 = struct2.get_positions()

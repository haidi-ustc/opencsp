# opencsp/operations/crossover/surface.py
import random
import numpy as np
from typing import List, Any

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class SurfaceCrossover(StructureOperation):
    """适用于表面的交叉操作"""
    
    def __init__(self, **kwargs):
        """
        初始化表面交叉操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=2, **kwargs)
        self.method = kwargs.get('method', 'layer_exchange')
        
    def apply(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        实现表面结构的交叉
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 确保结构是表面
        dim1 = get_structure_dimensionality(parent1.structure)
        dim2 = get_structure_dimensionality(parent2.structure)
        
        if dim1 != 2 or dim2 != 2:
            raise ValueError(f"SurfaceCrossover只适用于表面(2D)，但接收到的维度为{dim1}和{dim2}")
        
        # 根据方法选择不同的交叉策略
        if self.method == 'layer_exchange':
            return self._layer_exchange_crossover(parent1, parent2)
        elif self.method == 'region_exchange':
            return self._region_exchange_crossover(parent1, parent2)
        else:
            raise ValueError(f"未知的交叉方法：{self.method}")
    
    def _layer_exchange_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        层交换交叉：交换父代之间的层
        
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
            lattice1 = struct1.lattice
            lattice2 = struct2.lattice
        else:  # ASE Atoms
            coords1 = struct1.get_positions()
            species1 = struct1.get_chemical_symbols()
            coords2 = struct2.get_positions()
            species2 = struct2.get_chemical_symbols()
            cell1 = struct1.get_cell()
            cell2 = struct2.get_cell()
            pbc1 = struct1.get_pbc()
            pbc2 = struct2.get_pbc()
        
        # 为表面结构定义层
        # 假设z方向为垂直表面的方向
        z_coords1 = coords1[:, 2]
        z_coords2 = coords2[:, 2]
        
        # 确定层的z坐标范围
        z_min1, z_max1 = np.min(z_coords1), np.max(z_coords1)
        z_min2, z_max2 = np.min(z_coords2), np.max(z_coords2)
        
        # 确定层的分界点（随机选择）
        z_cut1 = z_min1 + random.random() * (z_max1 - z_min1)
        z_cut2 = z_min2 + random.random() * (z_max2 - z_min2)
        
        # 将原子分为上层和下层
        upper_indices1 = np.where(z_coords1 >= z_cut1)[0]
        lower_indices1 = np.where(z_coords1 < z_cut1)[0]
        upper_indices2 = np.where(z_coords2 >= z_cut2)[0]
        lower_indices2 = np.where(z_coords2 < z_cut2)[0]
        
        # 创建子代1：父代1的下层 + 父代2的上层
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            from pymatgen.core import Structure
            
            # 创建子代1
            lower_coords1 = coords1[lower_indices1]
            lower_species1 = [species1[i] for i in lower_indices1]
            
            upper_coords2 = coords2[upper_indices2]
            # 调整上层的z坐标，使其与父代1的上层位置匹配
            z_shift = (z_max1 - z_min1) - (z_max2 - z_min2)
            upper_coords2[:, 2] += z_shift
            upper_species2 = [species2[i] for i in upper_indices2]
            
            child1_coords = np.vstack([lower_coords1, upper_coords2])
            child1_species = lower_species1 + upper_species2
            
            child1_struct = Structure(lattice1, child1_species, child1_coords, coords_are_cartesian=True)
            
            # 创建子代2
            lower_coords2 = coords2[lower_indices2]
            lower_species2 = [species2[i] for i in lower_indices2]
            
            upper_coords1 = coords1[upper_indices1]
            # 调整上层的z坐标，使其与父代2的上层位置匹配
            upper_coords1[:, 2] -= z_shift
            upper_species1 = [species1[i] for i in upper_indices1]
            
            child2_coords = np.vstack([lower_coords2, upper_coords1])
            child2_species = lower_species2 + upper_species1
            
            child2_struct = Structure(lattice2, child2_species, child2_coords, coords_are_cartesian=True)
        else:  # ASE Atoms
            from ase import Atoms
            
            # 创建子代1
            lower_coords1 = coords1[lower_indices1]
            lower_species1 = [species1[i] for i in lower_indices1]
            
            upper_coords2 = coords2[upper_indices2]
            # 调整上层的z坐标，使其与父代1的上层位置匹配
            z_shift = (z_max1 - z_min1) - (z_max2 - z_min2)
            upper_coords2[:, 2] += z_shift
            upper_species2 = [species2[i] for i in upper_indices2]
            
            child1_coords = np.vstack([lower_coords1, upper_coords2])
            child1_species = lower_species1 + upper_species2
            
            child1_struct = Atoms(symbols=child1_species, positions=child1_coords, cell=cell1, pbc=pbc1)
            
            # 创建子代2
            lower_coords2 = coords2[lower_indices2]
            lower_species2 = [species2[i] for i in lower_indices2]
            
            upper_coords1 = coords1[upper_indices1]
            # 调整上层的z坐标，使其与父代2的上层位置匹配
            upper_coords1[:, 2] -= z_shift
            upper_species1 = [species1[i] for i in upper_indices1]
            
            child2_coords = np.vstack([lower_coords2, upper_coords1])
            child2_species = lower_species2 + upper_species1
            
            child2_struct = Atoms(symbols=child2_species, positions=child2_coords, cell=cell2, pbc=pbc2)
        
        # 创建子代个体
        child1 = Individual(child1_struct)
        child2 = Individual(child2_struct)
        
        return [child1, child2]
    
    def _region_exchange_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        区域交换交叉：交换表面上的不同区域
        
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
            lattice1 = struct1.lattice
            lattice2 = struct2.lattice
        else:  # ASE Atoms
            coords1 = struct1.get_positions()
            species1 = struct1.get_chemical_symbols()
            coords2 = struct2.get_positions()
            species2 = struct2.get_chemical_symbols()
            cell1 = struct1.get_cell()
            cell2 = struct2.get_cell()
            pbc1 = struct1.get_pbc()
            pbc2 = struct2.get_pbc()
        
        # 定义表面区域分割
        # 假设x和y方向是平行于表面的
        
        # 随机选择一个矩形区域
        x_min1, x_max1 = np.min(coords1[:, 0]), np.max(coords1[:, 0])
        y_min1, y_max1 = np.min(coords1[:, 1]), np.max(coords1[:, 1])
        
        x_cut_min = x_min1 + random.random() * 0.3 * (x_max1 - x_min1)
        x_cut_max = x_min1 + (0.7 + random.random() * 0.3) * (x_max1 - x_min1)
        y_cut_min = y_min1 + random.random() * 0.3 * (y_max1 - y_min1)
        y_cut_max = y_min1 + (0.7 + random.random() * 0.3) * (y_max1 - y_min1)
        
        # 定义区域内的原子索引
        region_indices1 = []
        region_indices2 = []
        
        for i, coord in enumerate(coords1):
            if (x_cut_min <= coord[0] <= x_cut_max and 
                y_cut_min <= coord[1] <= y_cut_max):
                region_indices1.append(i)
        
        # 对应父代2上的区域
        x_min2, x_max2 = np.min(coords2[:, 0]), np.max(coords2[:, 0])
        y_min2, y_max2 = np.min(coords2[:, 1]), np.max(coords2[:, 1])
        
        x_scale = (x_max2 - x_min2) / (x_max1 - x_min1)
        y_scale = (y_max2 - y_min2) / (y_max1 - y_min1)
        
        x_cut_min2 = x_min2 + (x_cut_min - x_min1) * x_scale
        x_cut_max2 = x_min2 + (x_cut_max - x_min1) * x_scale
        y_cut_min2 = y_min2 + (y_cut_min - y_min1) * y_scale
        y_cut_max2 = y_min2 + (y_cut_max - y_min1) * y_scale
        
        for i, coord in enumerate(coords2):
            if (x_cut_min2 <= coord[0] <= x_cut_max2 and 
                y_cut_min2 <= coord[1] <= y_cut_max2):
                region_indices2.append(i)
        
        # 区域外的原子索引
        outside_indices1 = [i for i in range(len(coords1)) if i not in region_indices1]
        outside_indices2 = [i for i in range(len(coords2)) if i not in region_indices2]
        
        # 创建子代
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            from pymatgen.core import Structure
            
            # 创建子代1：父代1的区域外 + 父代2的区域内（调整位置）
            outside_coords1 = coords1[outside_indices1]
            outside_species1 = [species1[i] for i in outside_indices1]
            
            region_coords2 = coords2[region_indices2]
            region_species2 = [species2[i] for i in region_indices2]
            
            # 调整区域内的原子位置，使其匹配父代1的位置比例
            region_coords2[:, 0] = ((region_coords2[:, 0] - x_min2) / (x_max2 - x_min2) * 
                                   (x_max1 - x_min1) + x_min1)
            region_coords2[:, 1] = ((region_coords2[:, 1] - y_min2) / (y_max2 - y_min2) * 
                                   (y_max1 - y_min1) + y_min1)
            
            child1_coords = np.vstack([outside_coords1, region_coords2])
            child1_species = outside_species1 + region_species2
            
            child1_struct = Structure(lattice1, child1_species, child1_coords, coords_are_cartesian=True)
            
            # 创建子代2：父代2的区域外 + 父代1的区域内（调整位置）
            outside_coords2 = coords2[outside_indices2]
            outside_species2 = [species2[i] for i in outside_indices2]
            
            region_coords1 = coords1[region_indices1]
            region_species1 = [species1[i] for i in region_indices1]
            
            # 调整区域内的原子位置，使其匹配父代2的位置比例
            region_coords1[:, 0] = ((region_coords1[:, 0] - x_min1) / (x_max1 - x_min1) * 
                                   (x_max2 - x_min2) + x_min2)
            region_coords1[:, 1] = ((region_coords1[:, 1] - y_min1) / (y_max1 - y_min1) * 
                                   (y_max2 - y_min2) + y_min2)
            
            child2_coords = np.vstack([outside_coords2, region_coords1])
            child2_species = outside_species2 + region_species1
            
            child2_struct = Structure(lattice2, child2_species, child2_coords, coords_are_cartesian=True)
        else:  # ASE Atoms
            from ase import Atoms
            
            # 创建子代1
            outside_coords1 = coords1[outside_indices1]
            outside_species1 = [species1[i] for i in outside_indices1]
            
            region_coords2 = coords2[region_indices2]
            region_species2 = [species2[i] for i in region_indices2]
            
            # 调整区域内的原子位置
            region_coords2[:, 0] = ((region_coords2[:, 0] - x_min2) / (x_max2 - x_min2) * 
                                   (x_max1 - x_min1) + x_min1)
            region_coords2[:, 1] = ((region_coords2[:, 1] - y_min2) / (y_max2 - y_min2) * 
                                   (y_max1 - y_min1) + y_min1)
            
            child1_coords = np.vstack([outside_coords1, region_coords2])
            child1_species = outside_species1 + region_species2
            
            child1_struct = Atoms(symbols=child1_species, positions=child1_coords, cell=cell1, pbc=pbc1)
            
            # 创建子代2
            outside_coords2 = coords2[outside_indices2]
            outside_species2 = [species2[i] for i in outside_indices2]
            
            region_coords1 = coords1[region_indices1]
            region_species1 = [species1[i] for i in region_indices1]
            
            # 调整区域内的原子位置
            region_coords1[:, 0] = ((region_coords1[:, 0] - x_min1) / (x_max1 - x_min1) * 
                                   (x_max2 - x_min2) + x_min2)
            region_coords1[:, 1] = ((region_coords1[:, 1] - y_min1) / (y_max1 - y_min1) * 
                                   (y_max2 - y_min2) + y_min2)
            
            child2_coords = np.vstack([outside_coords2, region_coords1])
            child2_species = outside_species2 + region_species1
            
            child2_struct = Atoms(symbols=child2_species, positions=child2_coords, cell=cell2, pbc=pbc2)
        
        # 创建子代个体
        child1 = Individual(child1_struct)
        child2 = Individual(child2_struct)
        
        return [child1, child2]

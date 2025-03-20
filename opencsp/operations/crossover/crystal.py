# opencsp/operations/crossover/crystal.py
import random
import numpy as np
from typing import List, Any

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

class CrystalCrossover(StructureOperation):
    """适用于晶体的交叉操作"""
    
    def __init__(self, **kwargs):
        """
        初始化晶体交叉操作
        
        Args:
            **kwargs: 其他参数
        """
        super().__init__(dimensionality=3, **kwargs)
        self.method = kwargs.get('method', 'plane_cut')
        
    def apply(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        实现晶体结构的交叉
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 确保结构是晶体
        dim1 = parent1.dimensionality 
        dim2 = parent2.dimensionality 
        
        if dim1 != 3 or dim2 != 3:
            raise ValueError(f"CrystalCrossover只适用于晶体(3D)，但接收到的维度为{dim1}和{dim2}")
        
        # 根据方法选择不同的交叉策略
        if self.method == 'plane_cut':
            return self._plane_cut_crossover(parent1, parent2)
        elif self.method == 'lattice_parameter':
            return self._lattice_parameter_crossover(parent1, parent2)
        else:
            raise ValueError(f"未知的交叉方法：{self.method}")
    
    def _plane_cut_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        平面切割交叉：沿随机平面切割并交换两个晶体的部分
        
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
            # 对于pymatgen结构，我们使用分数坐标
            coords1 = np.array([site.frac_coords for site in struct1.sites])
            species1 = [site.species_string for site in struct1.sites]
            coords2 = np.array([site.frac_coords for site in struct2.sites])
            species2 = [site.species_string for site in struct2.sites]
            lattice1 = struct1.lattice
            lattice2 = struct2.lattice
        else:  # ASE Atoms
            # 对于ASE结构，我们转换为分数坐标
            cell1 = struct1.get_cell()
            cell2 = struct2.get_cell()
            coords1 = struct1.get_scaled_positions()
            species1 = struct1.get_chemical_symbols()
            coords2 = struct2.get_scaled_positions()
            species2 = struct2.get_chemical_symbols()
            pbc1 = struct1.get_pbc()
            pbc2 = struct2.get_pbc()
        
        # 生成随机平面法向量（使用米勒指数）
        # 这里我们简化为基本平面：(100), (010), (001)
        planes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        plane = random.choice(planes)
        
        # 确定切割位置（0到1之间的随机值）
        cut_position = random.random()
        
        # 根据平面确定切割维度
        if plane == (1, 0, 0):
            dim = 0
        elif plane == (0, 1, 0):
            dim = 1
        else:  # (0, 0, 1)
            dim = 2
        
        # 将原子分为两部分
        indices1_part1 = np.where(coords1[:, dim] < cut_position)[0]
        indices1_part2 = np.where(coords1[:, dim] >= cut_position)[0]
        
        indices2_part1 = np.where(coords2[:, dim] < cut_position)[0]
        indices2_part2 = np.where(coords2[:, dim] >= cut_position)[0]
        
        # 创建子代1：父代1的第一部分 + 父代2的第二部分
        if hasattr(struct1, 'sites'):  # pymatgen Structure
            from pymatgen.core import Structure
            
            # 获取子代1的原子
            part1_coords1 = coords1[indices1_part1]
            part1_species1 = [species1[i] for i in indices1_part1]
            
            part2_coords2 = coords2[indices2_part2]
            part2_species2 = [species2[i] for i in indices2_part2]
            
            # 合并
            child1_coords = np.vstack([part1_coords1, part2_coords2])
            child1_species = part1_species1 + part2_species2
            
            # 创建子代1结构
            child1_struct = Structure(lattice1, child1_species, child1_coords, coords_are_cartesian=False)
            
            # 获取子代2的原子
            part1_coords2 = coords2[indices2_part1]
            part1_species2 = [species2[i] for i in indices2_part1]
            
            part2_coords1 = coords1[indices1_part2]
            part2_species1 = [species1[i] for i in indices1_part2]
            
            # 合并
            child2_coords = np.vstack([part1_coords2, part2_coords1])
            child2_species = part1_species2 + part2_species1
            
            # 创建子代2结构
            child2_struct = Structure(lattice2, child2_species, child2_coords, coords_are_cartesian=False)
        else:  # ASE Atoms
            from ase import Atoms
            
            # 获取子代1的原子
            part1_coords1 = coords1[indices1_part1]
            part1_species1 = [species1[i] for i in indices1_part1]
            
            part2_coords2 = coords2[indices2_part2]
            part2_species2 = [species2[i] for i in indices2_part2]
            
            # 合并
            child1_coords = np.vstack([part1_coords1, part2_coords2])
            child1_species = part1_species1 + part2_species2
            
            # 创建子代1结构
            child1_struct = Atoms(symbols=child1_species, scaled_positions=child1_coords, cell=cell1, pbc=pbc1)
            
            # 获取子代2的原子
            part1_coords2 = coords2[indices2_part1]
            part1_species2 = [species2[i] for i in indices2_part1]
            
            part2_coords1 = coords1[indices1_part2]
            part2_species1 = [species1[i] for i in indices1_part2]
            
            # 合并
            child2_coords = np.vstack([part1_coords2, part2_coords1])
            child2_species = part1_species2 + part2_species1
            
            # 创建子代2结构
            child2_struct = Atoms(symbols=child2_species, scaled_positions=child2_coords, cell=cell2, pbc=pbc2)
        
        # 创建子代个体
        child1 = Individual(child1_struct, dimensionality = parent1.dimensionality)
        child2 = Individual(child2_struct, dimensionality = parent2.dimensionality)
        
        return [child1, child2]
    
    def _lattice_parameter_crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        晶格参数交叉：混合两个晶体的晶格参数
        
        Args:
            parent1: 第一个父代个体
            parent2: 第二个父代个体
            
        Returns:
            两个子代个体的列表
        """
        # 获取父代结构
        struct1 = parent1.structure
        struct2 = parent2.structure
        
        if hasattr(struct1, 'lattice'):  # pymatgen Structure
            # 获取晶格参数
            a1, b1, c1, alpha1, beta1, gamma1 = struct1.lattice.parameters
            a2, b2, c2, alpha2, beta2, gamma2 = struct2.lattice.parameters
            
            # 生成随机权重
            w = random.random()
            
            # 创建混合晶格参数
            a_mix1 = a1 * w + a2 * (1 - w)
            b_mix1 = b1 * w + b2 * (1 - w)
            c_mix1 = c1 * w + c2 * (1 - w)
            alpha_mix1 = alpha1 * w + alpha2 * (1 - w)
            beta_mix1 = beta1 * w + beta2 * (1 - w)
            gamma_mix1 = gamma1 * w + gamma2 * (1 - w)
            
            a_mix2 = a1 * (1 - w) + a2 * w
            b_mix2 = b1 * (1 - w) + b2 * w
            c_mix2 = c1 * (1 - w) + c2 * w
            alpha_mix2 = alpha1 * (1 - w) + alpha2 * w
            beta_mix2 = beta1 * (1 - w) + beta2 * w
            gamma_mix2 = gamma1 * (1 - w) + gamma2 * w
            
            # 创建新晶格
            from pymatgen.core import Lattice
            lattice_mix1 = Lattice.from_parameters(a_mix1, b_mix1, c_mix1, alpha_mix1, beta_mix1, gamma_mix1)
            lattice_mix2 = Lattice.from_parameters(a_mix2, b_mix2, c_mix2, alpha_mix2, beta_mix2, gamma_mix2)
            
            # 创建子代结构
            from pymatgen.core import Structure
            
            # 子代1：父代1的原子位置 + 混合晶格1
            child1_struct = Structure(lattice_mix1, struct1.species, struct1.frac_coords)
            
            # 子代2：父代2的原子位置 + 混合晶格2
            child2_struct = Structure(lattice_mix2, struct2.species, struct2.frac_coords)
        else:  # ASE Atoms
            # 获取晶胞
            cell1 = struct1.get_cell()
            cell2 = struct2.get_cell()
            
            # 计算晶胞大小
            a1, b1, c1 = np.linalg.norm(cell1, axis=1)
            a2, b2, c2 = np.linalg.norm(cell2, axis=1)
            
            # 生成随机权重
            w = random.random()
            
            # 计算混合晶胞大小
            a_mix1 = a1 * w + a2 * (1 - w)
            b_mix1 = b1 * w + b2 * (1 - w)
            c_mix1 = c1 * w + c2 * (1 - w)
           
            # opencsp/operations/crossover/crystal.py (continued)
            a_mix2 = a1 * (1 - w) + a2 * w
            b_mix2 = b1 * (1 - w) + b2 * w
            c_mix2 = c1 * (1 - w) + c2 * w
            
            # 计算晶胞缩放因子
            scale1_a = a_mix1 / a1
            scale1_b = b_mix1 / b1
            scale1_c = c_mix1 / c1
            
            scale2_a = a_mix2 / a2
            scale2_b = b_mix2 / b2
            scale2_c = c_mix2 / c2
            
            # 创建新晶胞
            new_cell1 = cell1.copy()
            new_cell1[0] *= scale1_a
            new_cell1[1] *= scale1_b
            new_cell1[2] *= scale1_c
            
            new_cell2 = cell2.copy()
            new_cell2[0] *= scale2_a
            new_cell2[1] *= scale2_b
            new_cell2[2] *= scale2_c
            
            # 创建子代结构
            child1_struct = struct1.copy()
            child1_struct.set_cell(new_cell1, scale_atoms=True)
            
            child2_struct = struct2.copy()
            child2_struct.set_cell(new_cell2, scale_atoms=True)
        
        # 创建子代个体
        child1 = Individual(child1_struct, dimensionality=parent1.dimensionality)
        child2 = Individual(child2_struct, dimensionality=parent2.dimensionality)
        
        return [child1, child2]

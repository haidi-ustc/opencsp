# opencsp/core/constraints.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

from opencsp.core.individual import Individual

class Constraint(ABC):
    """约束条件基类"""
    
    def __init__(self, weight: float = 1.0):
        """
        初始化约束
        
        Args:
            weight: 违反约束的惩罚权重
        """
        self.weight = weight
        
    @abstractmethod
    def is_satisfied(self, structure: Any) -> bool:
        """
        检查结构是否满足约束
        
        Args:
            structure: 结构对象
            
        Returns:
            是否满足约束
        """
        pass
        
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        """
        评估约束违反程度，返回惩罚值
        
        Args:
            individual: 要评估的个体
            
        Returns:
            惩罚值（为0表示完全满足约束）
        """
        pass


class MinimumDistanceConstraint(Constraint):
    """最小原子间距约束"""
    
    def __init__(self, min_distance: Dict[str, float], weight: float = 1.0):
        """
        初始化最小距离约束
        
        Args:
            min_distance: 最小原子间距（可以是字典，指定不同原子对的最小距离）
            weight: 违反约束的惩罚权重
        """
        super().__init__(weight)
        self.min_distance = min_distance
        
    def is_satisfied(self, structure: Any) -> bool:
        """检查是否满足最小距离约束"""
        import numpy as np
        
        # 获取原子符号和坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            symbols = [site.species_string for site in structure.sites]
            coords = np.array([site.coords for site in structure.sites])
        else:  # ASE Atoms
            symbols = structure.get_chemical_symbols()
            coords = structure.get_positions()
            
        n_atoms = len(symbols)
        
        # 检查所有原子对的距离
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # 确定原子对
                pair = f"{symbols[i]}-{symbols[j]}"
                reverse_pair = f"{symbols[j]}-{symbols[i]}"
                
                # 获取最小距离
                if pair in self.min_distance:
                    min_dist = self.min_distance[pair]
                elif reverse_pair in self.min_distance:
                    min_dist = self.min_distance[reverse_pair]
                elif 'default' in self.min_distance:
                    min_dist = self.min_distance['default']
                else:
                    min_dist = 1.0  # 默认最小距离
                    
                # 计算实际距离
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # 检查是否违反约束
                if dist < min_dist:
                    return False
                    
        return True
        
    def evaluate(self, individual: Individual) -> float:
        """计算违反最小距离约束的惩罚值"""
        import numpy as np
        
        structure = individual.structure
        
        # 获取原子符号和坐标
        if hasattr(structure, 'sites'):  # pymatgen Structure
            symbols = [site.species_string for site in structure.sites]
            coords = np.array([site.coords for site in structure.sites])
        else:  # ASE Atoms
            symbols = structure.get_chemical_symbols()
            coords = structure.get_positions()
            
        n_atoms = len(symbols)
        penalty = 0.0
        
        # 检查所有原子对的距离
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                # 确定原子对
                pair = f"{symbols[i]}-{symbols[j]}"
                reverse_pair = f"{symbols[j]}-{symbols[i]}"
                
                # 获取最小距离
                if pair in self.min_distance:
                    min_dist = self.min_distance[pair]
                elif reverse_pair in self.min_distance:
                    min_dist = self.min_distance[reverse_pair]
                elif 'default' in self.min_distance:
                    min_dist = self.min_distance['default']
                else:
                    min_dist = 1.0  # 默认最小距离
                    
                # 计算实际距离
                dist = np.linalg.norm(coords[i] - coords[j])
                
                # 计算惩罚
                if dist < min_dist:
                    penalty += self.weight * (min_dist - dist)**2
                    
        return penalty


class SymmetryConstraint(Constraint):
    """结构对称性约束"""
    
    def __init__(self, target_spacegroup: Optional[int] = None, tolerance: float = 0.1, weight: float = 1.0):
        """
        初始化对称性约束
        
        Args:
            target_spacegroup: 目标空间群编号
            tolerance: 对称性判断容差
            weight: 违反约束的惩罚权重
        """
        super().__init__(weight)
        self.target_spacegroup = target_spacegroup
        self.tolerance = tolerance
        
    def is_satisfied(self, structure: Any) -> bool:
        """检查是否满足对称性约束"""
        # 需要pymatgen的spglib支持
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            
            # 确保结构是pymatgen格式
            if hasattr(structure, 'get_positions'):  # ASE Atoms
                from pymatgen.io.ase import AseAtomsAdaptor
                structure = AseAtomsAdaptor.get_structure(structure)
                
            # 分析对称性
            sga = SpacegroupAnalyzer(structure, symprec=self.tolerance)
            spacegroup = sga.get_space_group_number()
            
            # 如果没有指定目标空间群，则任何对称性都可以
            if self.target_spacegroup is None:
                return True
                
            return spacegroup == self.target_spacegroup
            
        except ImportError:
            # 如果没有pymatgen，默认满足约束
            return True
        
    def evaluate(self, individual: Individual) -> float:
        """计算违反对称性约束的惩罚值"""
        # 如果满足约束，返回0惩罚
        if self.is_satisfied(individual.structure):
            return 0.0
            
        # 否则，返回固定惩罚值
        return self.weight

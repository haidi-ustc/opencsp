# opencsp/core/individual.py
import uuid
import copy
from typing import Any, Dict, Optional, Union

class Individual:
    """表示一个结构单元，包含结构信息、能量、适应度等属性"""
    
    def __init__(self, structure: Any, energy: Optional[float] = None, 
                 fitness: Optional[float] = None, properties: Optional[Dict[str, Any]] = None):
        """
        初始化一个个体
        
        Args:
            structure: 结构对象（可以是ASE的Atoms或pymatgen的Structure）
            energy: 能量值
            fitness: 适应度值
            properties: 其他属性的字典
        """
        self.structure = structure
        self.energy = energy
        self.fitness = fitness
        self.properties = properties or {}
        self.id = str(uuid.uuid4())[:8]  # 唯一标识符
        
    def copy(self) -> 'Individual':
        """创建当前个体的深拷贝"""
        new_individual = Individual(
            structure=copy.deepcopy(self.structure),
            energy=self.energy,
            fitness=self.fitness,
            properties=copy.deepcopy(self.properties)
        )
        return new_individual
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        from opencsp.utils.serialization import structure_to_dict
        
        return {
            'id': self.id,
            'structure': structure_to_dict(self.structure),
            'energy': self.energy,
            'fitness': self.fitness,
            'properties': self.properties
        }
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Individual':
        """从字典创建个体"""
        from opencsp.utils.serialization import dict_to_structure
        
        individual = cls(
            structure=dict_to_structure(d['structure']),
            energy=d.get('energy'),
            fitness=d.get('fitness'),
            properties=d.get('properties', {})
        )
        individual.id = d['id']
        return individual

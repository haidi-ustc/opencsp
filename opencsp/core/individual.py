# 完全重写 opencsp/core/individual.py
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
        self._structure = structure
        self._energy = energy
        self._fitness = fitness
        self._properties = properties or {}
        self._id = str(uuid.uuid4())[:8]  # 唯一标识符
    
    @property
    def id(self) -> str:
        """获取个体ID"""
        return self._id
    
    @id.setter
    def id(self, value: str):
        """设置个体ID"""
        self._id = value
    
    @property
    def structure(self) -> Any:
        """获取结构"""
        return self._structure
    
    @structure.setter
    def structure(self, value: Any):
        """设置结构"""
        self._structure = value
    
    @property
    def energy(self) -> Optional[float]:
        """获取能量"""
        return self._energy
    
    @energy.setter
    def energy(self, value: Optional[float]):
        """设置能量"""
        self._energy = value
    
    @property
    def fitness(self) -> Optional[float]:
        """获取适应度"""
        return self._fitness
    
    @fitness.setter
    def fitness(self, value: Optional[float]):
        """设置适应度"""
        self._fitness = value
    
    @property
    def properties(self) -> Dict[str, Any]:
        """获取属性字典"""
        return self._properties
    
    def copy(self) -> 'Individual':
        """创建当前个体的深拷贝"""
        print(f"EXPLICIT COPY - id: {self.id}, energy: {self.energy}, fitness: {self.fitness}")
        
        # 创建新个体并明确设置所有属性
        new_individual = Individual(
            structure=copy.deepcopy(self.structure),
            energy=self.energy,  # 明确复制能量
            fitness=self.fitness,  # 明确复制适应度
            properties=copy.deepcopy(self.properties)  # 深复制其他属性
        )
        
        print(f"EXPLICIT COPY RESULT - id: {new_individual.id}, energy: {new_individual.energy}, fitness: {new_individual.fitness}")
        
        return new_individual
    
    def __str__(self) -> str:
        """返回个体的字符串表示"""
        return f"Individual(id={self.id}, energy={self.energy}, fitness={self.fitness})"
    
    def __repr__(self) -> str:
        """返回个体的字符串表示"""
        return self.__str__()
    
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

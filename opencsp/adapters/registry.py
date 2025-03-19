# opencsp/adapters/registry.py
from typing import Dict, Optional, Any, Type

from opencsp.operations.base import StructureOperation

class OperationRegistry:
    """
    操作注册中心，管理不同类型、不同维度的操作
    """
    
    def __init__(self):
        """初始化操作注册表"""
        # 初始化操作注册表
        self.crossover_operations: Dict[int, Dict[str, StructureOperation]] = {1: {}, 2: {}, 3: {}}
        self.mutation_operations: Dict[int, Dict[str, StructureOperation]] = {1: {}, 2: {}, 3: {}}
        self.position_operations: Dict[int, Dict[str, StructureOperation]] = {1: {}, 2: {}, 3: {}}
        self.velocity_operations: Dict[int, Dict[str, StructureOperation]] = {1: {}, 2: {}, 3: {}}
        
    def __str__(self):
        return f"{self.crossover_operations}"+f"{self.mutation_operations}"

    def __repr__(self):
        return f"{self.crossover_operations}"+f"{self.mutation_operations}"

    def register_crossover(self, operation: StructureOperation, dim: Optional[int] = None, name: Optional[str] = None) -> None:
        """注册交叉操作"""
        if dim is None:
            dim = operation.dimensionality
        if dim is None:
            raise ValueError("Dimensionality must be specified")
            
        if name is None:
            name = operation.__class__.__name__
            
        if dim not in self.crossover_operations:
            self.crossover_operations[dim] = {}
            
        self.crossover_operations[dim][name] = operation
        
    def register_mutation(self, operation: StructureOperation, dim: Optional[int] = None, name: Optional[str] = None) -> None:
        """注册变异操作"""
        if dim is None:
            dim = operation.dimensionality
        if dim is None:
            raise ValueError("Dimensionality must be specified")
            
        if name is None:
            name = operation.__class__.__name__
            
        if dim not in self.mutation_operations:
            self.mutation_operations[dim] = {}
            
        self.mutation_operations[dim][name] = operation
        
    def register_position(self, operation: StructureOperation, dim: Optional[int] = None, name: Optional[str] = None) -> None:
        """注册位置更新操作"""
        if dim is None:
            dim = operation.dimensionality
        if dim is None:
            raise ValueError("Dimensionality must be specified")
            
        if name is None:
            name = operation.__class__.__name__
            
        if dim not in self.position_operations:
            self.position_operations[dim] = {}
            
        self.position_operations[dim][name] = operation
        
    def register_velocity(self, operation: StructureOperation, dim: Optional[int] = None, name: Optional[str] = None) -> None:
        """注册速度更新操作"""
        if dim is None:
            dim = operation.dimensionality
        if dim is None:
            raise ValueError("Dimensionality must be specified")
            
        if name is None:
            name = operation.__class__.__name__
            
        if dim not in self.velocity_operations:
            self.velocity_operations[dim] = {}
            
        self.velocity_operations[dim][name] = operation
        
    def get_crossover_operation(self, dim: int, name: Optional[str] = None) -> Optional[StructureOperation]:
        """获取交叉操作"""
        if dim not in self.crossover_operations:
            return None
            
        if name is None and self.crossover_operations[dim]:
            # 返回第一个注册的操作
            return next(iter(self.crossover_operations[dim].values()))
            
        return self.crossover_operations[dim].get(name)
        
    def get_mutation_operation(self, dim: int, name: Optional[str] = None) -> Optional[StructureOperation]:
        """获取变异操作"""
        if dim not in self.mutation_operations:
            return None
            
        if name is None and self.mutation_operations[dim]:
            return next(iter(self.mutation_operations[dim].values()))
            
        return self.mutation_operations[dim].get(name)
        
    def get_position_operation(self, dim: int, name: Optional[str] = None) -> Optional[StructureOperation]:
        """获取位置更新操作"""
        if dim not in self.position_operations:
            return None
            
        if name is None and self.position_operations[dim]:
            return next(iter(self.position_operations[dim].values()))
            
        return self.position_operations[dim].get(name)
        
    def get_velocity_operation(self, dim: int, name: Optional[str] = None) -> Optional[StructureOperation]:
        """获取速度更新操作"""
        if dim not in self.velocity_operations:
            return None
            
        if name is None and self.velocity_operations[dim]:
            return next(iter(self.velocity_operations[dim].values()))
            
        return self.velocity_operations[dim].get(name)

# opencsp/adapters/dimension_aware.py
from typing import Dict, Any, Optional, TypeVar, Callable

from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality

T = TypeVar('T')

class DimensionAwareAdapter:
    """
    维度感知的操作适配器，根据结构维度选择合适的操作
    """
    
    def __init__(self):
        """初始化适配器"""
        self.dimension_operators: Dict[int, StructureOperation] = {}
        
    def add_operator(self, dimensionality: int, operator: StructureOperation) -> None:
        """添加适用于特定维度的操作"""
        self.dimension_operators[dimensionality] = operator
        
    def get_operator(self, dimensionality: int) -> Optional[StructureOperation]:
        """获取适用于特定维度的操作"""
        return self.dimension_operators.get(dimensionality)
        
    def apply(self, *args, **kwargs) -> Any:
        """
        根据结构维度应用合适的操作
        
        Args:
            *args: 传递给底层操作的位置参数
            dimensionality: 结构维度，如果不提供则尝试从参数推断
            **kwargs: 传递给底层操作的关键字参数
        """
        # 尝试获取维度
        if 'dimensionality' in kwargs:
            dim = kwargs.pop('dimensionality')
        else:
            # 尝试从第一个参数推断维度
            if args and hasattr(args[0], 'structure'):
                dim = get_structure_dimensionality(args[0].structure)
            else:
                raise ValueError("Cannot determine dimensionality")
                
        # 获取合适的操作器
        operator = self.get_operator(dim)
        #print(operator)

        if operator is None:
            raise ValueError(f"No operator available for dimensionality {dim}")
            
        # 应用操作
        return operator.apply(*args, **kwargs)

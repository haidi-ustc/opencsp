# opencsp/operations/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple, TypeVar, Dict

from opencsp.utils.structure import get_structure_dimensionality

T = TypeVar('T')

class StructureOperation(ABC):
    """
    结构操作的抽象基类
    所有结构变换操作（如交叉、变异）都应继承此类
    """
    
    def __init__(self, dimensionality: Optional[Union[int, List[int], Tuple[int, ...]]] = None, **kwargs):
        """
        初始化结构操作
        
        Args:
            dimensionality: 操作适用的维度（None表示适用于所有维度）
            kwargs: 其他参数
        """
        self.dimensionality = dimensionality  # 可以是None, 1, 2, 3或[1,2,3]等
        self.params = kwargs
        
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """应用操作"""
        pass
        
    def is_applicable(self, structure: Any) -> bool:
        """检查操作是否适用于给定结构"""
        if self.dimensionality is None:
            return True
        
        dim = get_structure_dimensionality(structure)
        
        if isinstance(self.dimensionality, (list, tuple)):
            return dim in self.dimensionality
        
        return dim == self.dimensionality

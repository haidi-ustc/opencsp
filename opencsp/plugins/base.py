# opencsp/plugins/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

from opencsp.adapters.registry import OperationRegistry
from opencsp.algorithms.optimizer import OptimizerFactory

class Plugin(ABC):
    """
    插件基类，所有插件都应继承此类
    """
    
    def __init__(self, **kwargs):
        """初始化插件"""
        self.params = kwargs
        
    @abstractmethod
    def initialize(self, operation_registry: OperationRegistry, optimizer_factory: OptimizerFactory) -> None:
        """
        初始化插件并注册其组件
        
        Args:
            operation_registry: 操作注册中心
            optimizer_factory: 优化器工厂
        """
        pass

# opencsp/plugins/manager.py
from typing import Dict, Any, Type

from opencsp.plugins.base import Plugin
from opencsp.adapters.registry import OperationRegistry
from opencsp.algorithms.optimizer import OptimizerFactory

class PluginManager:
    """
    插件管理器，用于动态加载和注册操作和优化器
    """
    
    def __init__(self, operation_registry: OperationRegistry, optimizer_factory: OptimizerFactory):
        """
        初始化插件管理器
        
        Args:
            operation_registry: 操作注册中心
            optimizer_factory: 优化器工厂
        """
        self.operation_registry = operation_registry
        self.optimizer_factory = optimizer_factory
        self.plugins: Dict[str, Type[Plugin]] = {}
        
    def register_plugin(self, plugin_name: str, plugin_class: Type[Plugin]) -> None:
        """注册插件"""
        self.plugins[plugin_name] = plugin_class
        
    def load_plugin(self, plugin_name: str, **kwargs) -> Plugin:
        """
        加载插件
        
        Args:
            plugin_name: 插件名称
            **kwargs: 插件参数
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Unknown plugin: {plugin_name}")
            
        plugin_class = self.plugins[plugin_name]
        plugin = plugin_class(**kwargs)
        
        # 初始化插件并注册其组件
        plugin.initialize(self.operation_registry, self.optimizer_factory)
        
        return plugin

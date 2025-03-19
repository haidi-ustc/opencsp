# opencsp/api.py
from opencsp.core.calculator import ASECalculatorWrapper, MLCalculator
from opencsp.core.evaluator import Evaluator
from opencsp.core.structure_generator import RandomStructureGenerator, SymmetryBasedStructureGenerator
from opencsp.adapters.registry import OperationRegistry
from opencsp.algorithms.optimizer import OptimizerFactory
from opencsp.runners.csp_runner import CSPRunner, OptimizationConfig
from opencsp.plugins.manager import PluginManager

class OpenCSP:
    """
    openCSP主类，提供简洁的API接口
    """
    
    def __init__(self):
        """初始化openCSP"""
        self.operation_registry = OperationRegistry()
        self.optimizer_factory = OptimizerFactory()
        self.optimizer_factory.operation_registry = self.operation_registry
        self.plugin_manager = PluginManager(self.operation_registry, self.optimizer_factory)
        
        # 注册默认优化器和操作
        self._register_defaults()
        
    def _register_defaults(self):
        """注册默认优化器和操作"""
        from opencsp.algorithms.genetic import GeneticAlgorithm
        from opencsp.algorithms.pso import ParticleSwarmOptimization
        
        # 注册默认优化器
        self.optimizer_factory.register_optimizer('ga', GeneticAlgorithm)
        self.optimizer_factory.register_optimizer('pso', ParticleSwarmOptimization)
        
        # 注册默认操作
        self._register_default_operations()
        
    def _register_default_operations(self):
        """注册默认操作"""
        from opencsp.operations.crossover.cluster import ClusterCrossover
        from opencsp.operations.crossover.surface import SurfaceCrossover
        from opencsp.operations.crossover.crystal import CrystalCrossover
        
        from opencsp.operations.mutation.cluster import ClusterMutation
        from opencsp.operations.mutation.surface import SurfaceMutation
        from opencsp.operations.mutation.crystal import CrystalMutation
        
        from opencsp.operations.position.cluster import ClusterPositionUpdate
        from opencsp.operations.position.surface import SurfacePositionUpdate
        from opencsp.operations.position.crystal import CrystalPositionUpdate
        
        from opencsp.operations.velocity.cluster import ClusterVelocityUpdate
        from opencsp.operations.velocity.surface import SurfaceVelocityUpdate
        from opencsp.operations.velocity.crystal import CrystalVelocityUpdate
        
        # 注册交叉操作
        self.operation_registry.register_crossover(ClusterCrossover(), 1)
        self.operation_registry.register_crossover(SurfaceCrossover(), 2)
        self.operation_registry.register_crossover(CrystalCrossover(), 3)
        
        # 注册变异操作
        self.operation_registry.register_mutation(ClusterMutation(), 1)
        self.operation_registry.register_mutation(SurfaceMutation(), 2)
        self.operation_registry.register_mutation(CrystalMutation(), 3)
        
        # 注册位置更新操作
        self.operation_registry.register_position(ClusterPositionUpdate(), 1)
        self.operation_registry.register_position(SurfacePositionUpdate(), 2)
        self.operation_registry.register_position(CrystalPositionUpdate(), 3)
        
        # 注册速度更新操作
        self.operation_registry.register_velocity(ClusterVelocityUpdate(), 1)
        self.operation_registry.register_velocity(SurfaceVelocityUpdate(), 2)
        self.operation_registry.register_velocity(CrystalVelocityUpdate(), 3)
        
    def create_calculator(self, calculator_type, **kwargs):
        """创建计算器"""
        if calculator_type == 'ase':
            return ASECalculatorWrapper(kwargs.pop('ase_calculator', None), **kwargs)
        elif calculator_type == 'ml':
            return MLCalculator(**kwargs)
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
            
    def create_evaluator(self, calculator, **kwargs):
        """创建评估器"""
        return Evaluator(calculator, **kwargs)
        
    def create_structure_generator(self, generator_type, composition, **kwargs):
        """创建结构生成器"""
        if generator_type == 'random':
            return RandomStructureGenerator(composition, **kwargs)
        elif generator_type == 'symmetry':
            return SymmetryBasedStructureGenerator(composition, **kwargs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
    def create_optimization_config(self, optimizer_type, dimensionality=None):
        """创建优化配置"""
        return OptimizationConfig(optimizer_type, dimensionality)
        
    def create_runner(self, structure_generator, evaluator, optimization_config=None, **kwargs):
        """创建CSP运行器"""
        return CSPRunner(structure_generator, evaluator, optimization_config, **kwargs)
        
    def load_plugin(self, plugin_name, **kwargs):
        """加载插件"""
        return self.plugin_manager.load_plugin(plugin_name, **kwargs)
        
    def register_plugin(self, plugin_name, plugin_class):
        """注册插件"""
        self.plugin_manager.register_plugin(plugin_name, plugin_class)

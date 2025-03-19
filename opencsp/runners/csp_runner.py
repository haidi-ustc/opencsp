# opencsp/runners/csp_runner.py
import os
import json
from typing import Dict, Any, Optional, List, Callable

from opencsp.core.evaluator import Evaluator
from opencsp.core.structure_generator import StructureGenerator
from opencsp.adapters.registry import OperationRegistry
from opencsp.algorithms.optimizer import OptimizerFactory, Optimizer

class OptimizationConfig:
    """
    优化配置类，用于管理算法参数和操作策略
    """
    
    def __init__(self, optimizer_type: str, dimensionality: Optional[int] = None):
        """
        初始化优化配置
        
        Args:
            optimizer_type: 优化器类型 ('ga', 'pso', 等)
            dimensionality: 结构维度 (1, 2, 3 或 None)
        """
        self.optimizer_type = optimizer_type
        self.dimensionality = dimensionality
        self.params: Dict[str, Any] = {}
        self.operations: Dict[str, Dict[int, Any]] = {}
        
    def set_param(self, name: str, value: Any) -> 'OptimizationConfig':
        """设置算法参数"""
        self.params[name] = value
        return self
        
    def set_operation(self, operation_type: str, operation: Any, dim: Optional[int] = None) -> 'OptimizationConfig':
        """
        设置操作策略
        
        Args:
            operation_type: 操作类型 ('crossover', 'mutation', 'position', 'velocity')
            operation: 操作实例
            dim: 操作适用的维度，如果为None则使用operation的dimensionality
        """
        if dim is None:
            dim = operation.dimensionality
            
        if operation_type not in self.operations:
            self.operations[operation_type] = {}
            
        self.operations[operation_type][dim] = operation
        return self
        
    def build(self, registry: OperationRegistry, factory: OptimizerFactory) -> Optimizer:
        """
        构建优化器和相关配置
        
        Args:
            registry: 操作注册中心
            factory: 优化器工厂
            
        Returns:
            configured_optimizer: 配置好的优化器实例
        """
        # 注册操作
        for op_type, dim_ops in self.operations.items():
            for dim, op in dim_ops.items():
                if op_type == 'crossover':
                    registry.register_crossover(op, dim)
                elif op_type == 'mutation':
                    registry.register_mutation(op, dim)
                elif op_type == 'position':
                    registry.register_position(op, dim)
                elif op_type == 'velocity':
                    registry.register_velocity(op, dim)
                    
        # 创建优化器
        return factory.create_optimizer(self.optimizer_type, **self.params)


class CSPRunner:
    """
    晶体结构预测运行器，协调整个搜索过程
    """
    
    def __init__(self, structure_generator: StructureGenerator,
                 evaluator: Evaluator, 
                 optimization_config: Optional[OptimizationConfig] = None,
                 operation_registry: OperationRegistry = None,
                 optimizer_factory: OptimizerFactory = None,
                 **kwargs):
        """
        初始化CSP运行器
        
        Args:
            structure_generator: 结构生成器
            evaluator: 评估器
            optimization_config: 优化配置
            **kwargs: 其他参数
        """
        self.structure_generator = structure_generator
        self.evaluator = evaluator
        self.optimization_config = optimization_config
        
        # 创建操作注册中心和优化器工厂
        self.operation_registry = operation_registry or  OperationRegistry()
        self.optimizer_factory = optimizer_factory or  OptimizerFactory()
        self.optimizer_factory.operation_registry = self.operation_registry
        
        # 注册优化器类型
        from opencsp.algorithms.genetic import GeneticAlgorithm
        from opencsp.algorithms.pso import ParticleSwarmOptimization
        self.optimizer_factory.register_optimizer('ga', GeneticAlgorithm)
        self.optimizer_factory.register_optimizer('pso', ParticleSwarmOptimization)
        
        # 运行参数
        self.population_size = kwargs.get('population_size', 50)
        self.max_steps = kwargs.get('max_steps', 100)
        self.callbacks: List[Callable[[Optimizer, int], None]] = kwargs.get('callbacks', [])
        self.output_dir = kwargs.get('output_dir', './csp_results')
        
        # 检查输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
    def configure(self, optimization_config: OptimizationConfig) -> 'CSPRunner':
        """设置优化配置"""
        self.optimization_config = optimization_config
        return self
        
    def set_population_size(self, size: int) -> 'CSPRunner':
        """设置种群/粒子数量"""
        self.population_size = size
        return self
        
    def set_max_steps(self, steps: int) -> 'CSPRunner':
        """设置最大迭代次数"""
        self.max_steps = steps
        return self
        
    def add_callback(self, callback: Callable[[Optimizer, int], None]) -> 'CSPRunner':
        """添加回调函数"""
        self.callbacks.append(callback)
        return self
        
    def run(self) -> Any:
        """
        运行晶体结构预测
        
        Returns:
            best_individual: 最佳结构个体
        """
        if self.optimization_config is None:
            raise ValueError("Optimization configuration is not set")
            
        # 构建优化器
        optimizer = self.optimization_config.build(
            self.operation_registry, 
            self.optimizer_factory
        )
        
        # 运行优化
        best_individual = optimizer.run(
            self.structure_generator,
            population_size=self.population_size,
            max_steps=self.max_steps,
            callbacks=self.callbacks
        )

        # 保存结果
        self._save_results(optimizer, best_individual)

        return best_individual

    def _save_results(self, optimizer: Optimizer, best_individual: Any) -> None:
        """
        保存搜索结果

        Args:
            optimizer: 优化器实例
            best_individual: 最佳个体
        """
        # 创建结果目录
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Handle the case when best_individual is None
        if best_individual is None:
            print("Warning: No best individual found. Saving limited results.")
            
            # Save empty best info
            best_info_path = os.path.join(results_dir, 'best_info.json')
            with open(best_info_path, 'w') as f:
                json.dump({"error": "No valid solution found"}, f, indent=2)
                
            # Still try to save history
            if hasattr(optimizer, 'history') and optimizer.history:
                try:
                    # Try to save a simplified version of history
                    simplified_history = []
                    for entry in optimizer.history:
                        simplified_entry = {}
                        for key, value in entry.items():
                            if isinstance(value, (int, float, str, bool, type(None))):
                                simplified_entry[key] = value
                        simplified_history.append(simplified_entry)
                        
                    history_path = os.path.join(results_dir, 'history.json')
                    with open(history_path, 'w') as f:
                        json.dump(simplified_history, f, indent=2)
                except Exception as e:
                    print(f"Error saving history: {e}")
                    
            # Save configuration - only save simple types
            try:
                config_path = os.path.join(results_dir, 'config.json')
                config_dict = {
                    "optimizer_type": self.optimization_config.optimizer_type,
                    "max_steps": self.max_steps,
                    "population_size": self.population_size
                }
                
                # Add serializable params
                params_dict = {}
                for key, value in self.optimization_config.params.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        params_dict[key] = value
                
                config_dict["parameters"] = params_dict
                
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            except Exception as e:
                print(f"Error saving config: {e}")
                
            return

        # 保存最佳结构
        best_structure_path = os.path.join(results_dir, 'best_structure.cif')
        try:
            from pymatgen.io.cif import CifWriter
            if hasattr(best_individual.structure, 'sites'):
                # pymatgen Structure
                CifWriter(best_individual.structure).write_file(best_structure_path)
            else:
                # ASE Atoms
                from ase.io import write
                write(best_structure_path, best_individual.structure)
        except Exception as e:
            print(f"Error saving best structure: {e}")

        # 保存最佳结构信息
        best_info_path = os.path.join(results_dir, 'best_info.json')
        with open(best_info_path, 'w') as f:
            info = {
                'energy': best_individual.energy,
                'fitness': best_individual.fitness,
                'properties': {k: v for k, v in best_individual.properties.items() 
                               if isinstance(v, (int, float, str, bool, type(None)))}
            }
            json.dump(info, f, indent=2)

        # 保存历史记录
        try:
            # Try to save a simplified version of history
            simplified_history = []
            for entry in optimizer.history:
                simplified_entry = {}
                for key, value in entry.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        simplified_entry[key] = value
                simplified_history.append(simplified_entry)
                
            history_path = os.path.join(results_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(simplified_history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

        # 保存运行配置
        try:
            config_path = os.path.join(results_dir, 'config.json')
            config_dict = {
                "optimizer_type": self.optimization_config.optimizer_type,
                "max_steps": self.max_steps,
                "population_size": self.population_size
            }
            
            # Add serializable params
            params_dict = {}
            for key, value in self.optimization_config.params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    params_dict[key] = value
            
            config_dict["parameters"] = params_dict
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

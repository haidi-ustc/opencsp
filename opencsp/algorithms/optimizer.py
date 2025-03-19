# opencsp/algorithms/optimizer.py
import traceback
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, TypeVar

from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from opencsp.core.structure_generator import StructureGenerator

T = TypeVar('T', bound=Individual)

class Optimizer(ABC):
    """
    优化算法抽象基类
    """
    
    def __init__(self, evaluator: Evaluator, **kwargs):
        """
        初始化优化器
        
        Args:
            evaluator: 评估器对象
            kwargs: 算法特定参数
        """
        self.evaluator = evaluator
        self.best_individual = None
        self.history = []
        self.params = kwargs
        
    @abstractmethod
    def initialize(self, structure_generator: StructureGenerator, population_size: int) -> None:
        """初始化搜索状态"""
        pass
        
    @abstractmethod
    def step(self) -> None:
        """执行一步优化"""
        pass
        
    def run(self, structure_generator: StructureGenerator, population_size: int = 50, 
            max_steps: int = 100, callbacks: Optional[List[Callable[['Optimizer', int], None]]] = None) -> T:
        """
        运行优化算法
        
        Args:
            structure_generator: 结构生成器
            population_size: 种群/粒子数量
            max_steps: 最大步数/代数
            callbacks: 回调函数列表
        
        Returns:
            最佳个体
        """
        try:
            self.initialize(structure_generator, population_size)
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            return None
            
            # 初始化后立即更新最佳个体
        if hasattr(self, 'population') and self.population:
           current_best = self.population.get_best()
           if current_best:
               print(f"Initial best: id={current_best.id}, energy={current_best.energy}, fitness={current_best.fitness}")

               # 手动创建和复制最佳个体
               new_best = Individual(structure=copy.deepcopy(current_best.structure))
               new_best.energy = current_best.energy
               new_best.fitness = current_best.fitness
               for key, value in current_best.properties.items():
                   new_best.properties[key] = copy.deepcopy(value)

               self.best_individual = new_best
               print(f"After setting best_individual: id={self.best_individual.id}, energy={self.best_individual.energy}, fitness={self.best_individual.fitness}")


        
        try:
            for step in range(max_steps):
                self.step()
                
                # 确保最佳个体不为None
                if self.best_individual is None:
                    print(f"Warning: Lost best individual at step {step}. Terminating optimization.")
                    break
                
                # 更新历史记录
                self.history.append(self.get_state())
                
                # 执行回调
                if callbacks:
                    for callback in callbacks:
                        callback(self, step)
                        
                # 检查终止条件
                if self.check_termination():
                    break
        except Exception as e:
            print(f"Optimization failed at step {step}: {str(e)}")
            traceback.print_exc()
                
        return self.best_individual
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取当前优化状态"""
        pass
        
    def check_termination(self) -> bool:
        """检查是否满足终止条件"""
        return False


class OptimizerFactory:
    """
    优化器工厂，负责创建和配置优化器
    """
    
    def __init__(self):
        self.registered_optimizers = {}
        self.operation_registry = None
        
    def register_optimizer(self, name: str, optimizer_class: Type[Optimizer]) -> None:
        """注册优化器类"""
        self.registered_optimizers[name] = optimizer_class
        
    def create_optimizer(self, name: str, evaluator: Evaluator, **kwargs) -> Optimizer:
        """
        创建优化器实例
        
        Args:
            name: 优化器名称（如 'ga', 'pso'）
            evaluator: 评估器实例
            **kwargs: 优化器参数
        """
        if name not in self.registered_optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
            
        optimizer_class = self.registered_optimizers[name]
        
        # 根据优化器类型创建适当的适配器
        if name == 'ga':
            from opencsp.adapters.dimension_aware import DimensionAwareAdapter
            
            # 为遗传算法创建交叉和变异适配器
            crossover_adapter = DimensionAwareAdapter()
            #print(crossover_adapter)
            mutation_adapter = DimensionAwareAdapter()
            #print(mutation_adapter)
            
            # 添加各维度的交叉操作
            print(f"check: {self.operation_registry}")
            for dim in [1, 2, 3]:
                if self.operation_registry:
                    op = self.operation_registry.get_crossover_operation(dim)
                    if op:
                        crossover_adapter.add_operator(dim, op)
                        
                    op = self.operation_registry.get_mutation_operation(dim)
                    if op:
                        mutation_adapter.add_operator(dim, op)
            
            return optimizer_class(
                evaluator=evaluator,
                crossover_adapter=crossover_adapter,
                mutation_adapter=mutation_adapter,
                **kwargs
            )
            
        elif name == 'pso':
            from opencsp.adapters.dimension_aware import DimensionAwareAdapter
            
            # 为粒子群算法创建位置和速度适配器
            position_adapter = DimensionAwareAdapter()
            velocity_adapter = DimensionAwareAdapter()
            
            # 添加各维度的位置和速度更新操作
            for dim in [1, 2, 3]:
                if self.operation_registry:
                    op = self.operation_registry.get_position_operation(dim)
                    if op:
                        position_adapter.add_operator(dim, op)
                        
                    op = self.operation_registry.get_velocity_operation(dim)
                    if op:
                        velocity_adapter.add_operator(dim, op)
            
            return optimizer_class(
                evaluator=evaluator,
                position_adapter=position_adapter,
                velocity_adapter=velocity_adapter,
                **kwargs
            )
            
        else:
            # 对于其他优化器，直接创建
            return optimizer_class(evaluator=evaluator, **kwargs)

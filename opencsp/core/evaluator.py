# opencsp/core/evaluator.py
from typing import List, Callable, Optional, Any, Dict, Union

from opencsp.core.individual import Individual
from opencsp.core.calculator import Calculator

class Evaluator:
    """评估个体的适应度"""
    
    def __init__(self, calculator: Calculator, fitness_function: Optional[Callable[[Individual], float]] = None, 
                 constraints: Optional[List[Any]] = None):
        """
        初始化评估器
        
        Args:
            calculator: 计算引擎
            fitness_function: 适应度函数，默认为能量的负值
            constraints: 约束条件列表
        """
        self.calculator = calculator
        self.fitness_function = fitness_function or (lambda indiv: -indiv.energy if indiv.energy is not None else None)
        self.constraints = constraints or []
        self.evaluation_count = 0
        
    def evaluate(self, individual: Individual) -> float:
        """
        评估单个个体
        
        Args:
            individual: 要评估的个体
            
        Returns:
            适应度值
        """
        if individual.energy is None:
            try:
                energy = self.calculator.calculate(individual.structure)
                individual.energy = energy
                
                # 计算其他属性
                properties = self.calculator.get_properties(individual.structure)
                individual.properties.update(properties)
            except Exception as e:
                individual.energy = float('inf')
                individual.properties['error'] = str(e)
                
        # 应用约束条件
        penalty = 0.0
        for constraint in self.constraints:
            penalty += constraint.evaluate(individual)
            
        # 计算适应度
        if individual.energy is not None:
            individual.fitness = self.fitness_function(individual) - penalty
        
        self.evaluation_count += 1
        return individual.fitness if individual.fitness is not None else float('-inf')
        
    def evaluate_population(self, individuals: List[Individual], parallel: bool = True, n_jobs: int = -1) -> None:
        """
        评估整个种群
        
        Args:
            individuals: 要评估的个体列表
            parallel: 是否使用并行计算
            n_jobs: 并行作业数，-1表示使用所有可用核心
        """
        if parallel and len(individuals) > 1:
            try:
                from joblib import Parallel, delayed
                
                # 使用joblib并行执行评估
                Parallel(n_jobs=n_jobs)(
                    delayed(self.evaluate)(individual) for individual in individuals
                )
            except ImportError:
                # 如果没有joblib，回退到串行评估
                for individual in individuals:
                    self.evaluate(individual)
        else:
            # 串行评估
            for individual in individuals:
                self.evaluate(individual)

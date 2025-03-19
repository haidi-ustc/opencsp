# opencsp/algorithms/genetic.py
import random
import copy
from typing import List, Dict, Any, Optional

from opencsp.algorithms.optimizer import Optimizer
from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from opencsp.core.population import Population
from opencsp.core.structure_generator import StructureGenerator
from opencsp.adapters.dimension_aware import DimensionAwareAdapter

class GeneticAlgorithm(Optimizer):
    """
    遗传算法实现，支持不同维度的结构操作
    """
    
    def __init__(self, evaluator: Evaluator, crossover_adapter: Optional[DimensionAwareAdapter] = None, 
                 mutation_adapter: Optional[DimensionAwareAdapter] = None, **kwargs):
        """
        初始化遗传算法
        
        Args:
            evaluator: 评估器对象
            crossover_adapter: 交叉操作适配器
            mutation_adapter: 变异操作适配器
            kwargs: 其他参数
        """
        super().__init__(evaluator, **kwargs)
        self.crossover_adapter = crossover_adapter or DimensionAwareAdapter()
        self.mutation_adapter = mutation_adapter or DimensionAwareAdapter()
        self.population = None
        self.selection_method = kwargs.get('selection_method', 'tournament')
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.mutation_rate = kwargs.get('mutation_rate', 0.2)
        self.elitism = kwargs.get('elitism', 1)
        
    def initialize(self, structure_generator: StructureGenerator, population_size: int) -> None:
        """初始化种群"""
        # 生成初始种群
        structures = structure_generator.generate(n=population_size)
        
        # Filter out None structures
        valid_structures = [s for s in structures if s is not None]
        
        if not valid_structures:
            raise ValueError("Failed to generate any valid structures. Please check the structure generator parameters.")
            
        individuals = [Individual(structure) for structure in valid_structures]
        
        # 评估初始种群
        self.evaluator.evaluate_population(individuals)
        
        # 创建Population对象
        self.population = Population(individuals, max_size=population_size)
        
        # 更新最佳个体
        self.best_individual = self.population.get_best()
        
        # Handle the case where there might not be any valid individuals
        if self.best_individual is None:
            print("Warning: No valid individuals in the initial population. This may cause optimization issues.")
        
    def step(self) -> None:
        """执行一代遗传算法操作"""
        # 1. 选择父代
        if self.selection_method == 'tournament':
            parents = self.population.select_tournament(n=self.population.size)
        elif self.selection_method == 'roulette':
            parents = self.population.select_roulette(n=self.population.size)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        # 2. 创建下一代
        offspring = []
        
        # 交叉操作
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if random.random() < self.crossover_rate:
                    # 使用维度感知适配器执行交叉操作
                    children = self.crossover_adapter.apply(parents[i], parents[i+1])
                    offspring.extend(children)
                else:
                    offspring.extend([parents[i].copy(), parents[i+1].copy()])
        
        # 变异操作
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                # 使用维度感知适配器执行变异操作
                offspring[i] = self.mutation_adapter.apply(offspring[i])
        
        # 3. 评估新个体
        self.evaluator.evaluate_population(offspring)
        
        # 4. 更新种群（精英保留策略）
        self.population.update(offspring, elitism=self.elitism)
        
        # 5. 更新最佳个体
        current_best = self.population.get_best()
        print(f"STEP BEST - id: {current_best.id}, energy: {current_best.energy}, fitness: {current_best.fitness}")

        # 判断是否需要更新全局最佳个体
        update_best = False
        if self.best_individual is None:
            update_best = True
        elif current_best.fitness is not None and (self.best_individual.fitness is None or current_best.fitness > self.best_individual.fitness):
            update_best = True

        if update_best:
            # 完全创建新对象
            new_best = Individual(
                structure=copy.deepcopy(current_best.structure),
                energy=current_best.energy,
                fitness=current_best.fitness
            )

            # 复制所有属性
            for key, value in current_best.properties.items():
                new_best.properties[key] = copy.deepcopy(value)

            self.best_individual = new_best
            print(f"NEW GLOBAL BEST - id: {self.best_individual.id}, energy: {self.best_individual.energy}, fitness: {self.best_individual.fitness}")

    def get_state1(self) -> Dict[str, Any]:
        """获取当前优化状态"""
        print(f"In get_state: best_individual = {self.best_individual}")
        if self.best_individual:
           print(f"  best_individual details: id={self.best_individual.id}, energy={self.best_individual.energy}, fitness={self.best_individual.fitness}")

        return {
            'generation': self.population.generation,
            'best_fitness': self.best_individual.fitness if self.best_individual else None,
            'best_energy': self.best_individual.energy if self.best_individual else None,
            'avg_fitness': self.population.get_average_fitness(),
            'diversity': self.population.get_diversity(),
            'evaluations': self.evaluator.evaluation_count
        }
    # 修复 opencsp/algorithms/optimizer.py 中的 get_state 方法
    def get_state(self) -> Dict[str, Any]:
        """获取当前优化状态"""
        print(f"GET STATE - best individual: {self.best_individual}")
    
        if self.best_individual:
            return {
                'generation': getattr(self, 'generation', 0),
                'best_fitness': self.best_individual.fitness,
                'best_energy': self.best_individual.energy,
                'avg_fitness': getattr(self, 'population', None) and self.population.get_average_fitness(),
                'diversity': getattr(self, 'population', None) and self.population.get_diversity(),
                'evaluations': self.evaluator.evaluation_count
            }
        else:
            return {
                'generation': getattr(self, 'generation', 0),
                'best_fitness': None,
                'best_energy': None,
                'avg_fitness': float('nan'),
                'diversity': 0.0,
                'evaluations': self.evaluator.evaluation_count
            }

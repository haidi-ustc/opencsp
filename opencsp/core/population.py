# opencsp/core/population.py
import random
from typing import List, Optional, Any, Callable, TypeVar, Union
from opencsp.core.individual import Individual

T = TypeVar('T', bound=Individual)

class Population:
    """管理个体集合，提供种群操作功能"""
    
    def __init__(self, individuals: Optional[List[T]] = None, max_size: int = 100):
        """
        初始化种群
        
        Args:
            individuals: 初始个体列表
            max_size: 种群最大大小
        """
        self.individuals = individuals or []
        self.max_size = max_size
        self.generation = 0
        
    @property
    def size(self) -> int:
        """获取当前种群大小"""
        return len(self.individuals)
        
    def add_individual(self, individual: T) -> None:
        """添加个体到种群"""
        self.individuals.append(individual)
        
    def get_best(self, n: int = 1) -> Union[T, List[T]]:
        """获取适应度最好的n个个体"""
        # Filter out None values before sorting
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            return None if n == 1 else []
        
        sorted_individuals = sorted(
            valid_individuals, 
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=True
        )
        
        if n == 1:
            return sorted_individuals[0] if sorted_individuals else None
        else:
            return sorted_individuals[:n]
            
    def select_tournament(self, n: int, tournament_size: int = 3) -> List[T]:
        """锦标赛选择n个个体"""
        # Filter out None values before selection
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            return []
            
        selected = []
        for _ in range(n):
            competitors = random.sample(valid_individuals, min(tournament_size, len(valid_individuals)))
            winner = max(competitors, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
            selected.append(winner)
        return selected
        
    def select_roulette(self, n: int) -> List[T]:
        """轮盘赌选择n个个体"""
        # Filter out None values before selection
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            return []
            
        # 计算适应度总和（需要将适应度值调整为非负值）
        min_fitness = min((ind.fitness for ind in valid_individuals if ind.fitness is not None), default=0)
        adjusted_fitness = [(ind.fitness - min_fitness + 1e-6) if ind.fitness is not None else 1e-6 
                            for ind in valid_individuals]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness <= 0:
            # 如果总适应度为零，使用均匀选择
            return random.choices(valid_individuals, k=n) if valid_individuals else []
        
        # 进行选择
        selected = []
        for _ in range(n):
            r = random.uniform(0, total_fitness)
            cum_sum = 0
            for i, fitness in enumerate(adjusted_fitness):
                cum_sum += fitness
                if cum_sum >= r:
                    selected.append(valid_individuals[i])
                    break
            else:
                # 如果由于浮点误差没有选择到个体，选择最后一个
                selected.append(valid_individuals[-1])
                
        return selected
        
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """按适应度排序个体"""
        # Filter out None values
        self.individuals = [ind for ind in self.individuals if ind is not None]
        
        self.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=reverse
        )
        
    def update(self, new_individuals: List[T], elitism: int = 1) -> None:
        """
        使用新个体更新种群，保留精英个体
        
        Args:
            new_individuals: 新个体列表
            elitism: 保留的精英个体数量
        """
        # Filter out None values from new individuals
        new_individuals = [ind for ind in new_individuals if ind is not None]
        
        # 保留精英个体
        if elitism > 0:
            elites = self.get_best(n=elitism)
            if not isinstance(elites, list):
                elites = [elites] if elites is not None else []
        else:
            elites = []
            
        # 更新种群
        combined = elites + new_individuals
        self.individuals = combined[:self.max_size]
        self.generation += 1
        
    def get_average_fitness(self) -> float:
        """计算种群平均适应度"""
        # Filter out None individuals
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            return float('nan')
            
        valid_fitness = [ind.fitness for ind in valid_individuals if ind.fitness is not None]
        if not valid_fitness:
            return float('nan')
        return sum(valid_fitness) / len(valid_fitness)
        
    def get_diversity(self) -> float:
        """计算种群多样性（结构差异的平均值）"""
        # 这需要一个能计算结构差异的函数
        from opencsp.utils.structure import calculate_structure_distance
        
        # Filter out None individuals
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if len(valid_individuals) <= 1:
            return 0.0
            
        total_distance = 0.0
        count = 0
        
        for i in range(len(valid_individuals)):
            for j in range(i + 1, len(valid_individuals)):
                total_distance += calculate_structure_distance(
                    valid_individuals[i].structure,
                    valid_individuals[j].structure
                )
                count += 1
                
        if count == 0:
            return 0.0
        return total_distance / count

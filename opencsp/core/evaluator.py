# opencsp/core/evaluator.py
from typing import List, Callable, Optional, Any, Dict, Union
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from pymatgen.core import Structure            
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
        """评估单个个体"""
        print(f"Evaluating individual {individual.id}")
        structure = individual.structure
        if isinstance(structure,Structure):
           structure = AseAtomsAdaptor().get_atoms(structure)
        if individual.energy is None:
            try:
                energy = self.calculator.calculate(structure)
                print(f"  Calculated energy: {energy}")
                individual.energy = energy
                
                # 计算其他属性
                properties = self.calculator.get_properties(structure)
                individual.properties.update(properties)
                
                # 增加评估计数 - 确保这里是真正的累加
                self.evaluation_count += 1
                print(f"  Increased evaluation count to {self.evaluation_count}")
            except Exception as e:
                print(f"  Error calculating energy: {e}")
                individual.energy = float('inf')
                individual.properties['error'] = str(e)
                
        # 应用约束条件
        penalty = 0.0
        for constraint in self.constraints:
            penalty += constraint.evaluate(individual)
        
        # 计算适应度
        if individual.energy is not None:
            individual.fitness = self.fitness_function(individual) - penalty
            print(f"  Set fitness to {individual.fitness}")
        
        return individual.fitness if individual.fitness is not None else float('-inf')

    def evaluate1(self, individual: Individual) -> float:
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
                print(f"  Calculated energy: {energy}")
                
                # 计算其他属性
                properties = self.calculator.get_properties(individual.structure)
                individual.properties.update(properties)
                self.evaluation_count += 1
                print(f"  Increased evaluation count to {self.evaluation_count}")


            except Exception as e:
                print(f"  Error calculating energy: {e}")
                individual.energy = float('inf')
                individual.properties['error'] = str(e)
                
        # 应用约束条件
        penalty = 0.0
        for constraint in self.constraints:
            penalty += constraint.evaluate(individual)
            
        # 计算适应度
        if individual.energy is not None:
            individual.fitness = self.fitness_function(individual) - penalty
            print(f"  Set fitness to {individual.fitness}")
        
        print(individual)
        self.evaluation_count += 1
        return individual.fitness if individual.fitness is not None else float('-inf')
        
    def evaluate_population1(self, individuals: List[Individual], parallel: bool = False, n_jobs: int = -1) -> None:
        """
        评估整个种群
        
        Args:
            individuals: 要评估的个体列表
            parallel: 是否使用并行计算
            n_jobs: 并行作业数，-1表示使用所有可用核心
        """
        start_count = self.evaluation_count
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
        evaluated_count = sum(1 for ind in individuals if ind.energy is not None)
        print(f"Population evaluation: evaluated {evaluated_count} individuals")
        print(f"Total evaluations so far: {self.evaluation_count}")


    def evaluate_population(self, individuals: List[Individual], parallel: bool = False, n_jobs: int = -1) -> None:
        """评估整个种群"""
        # 记录开始前的计数
        start_count = self.evaluation_count
    
        if parallel and len(individuals) > 1:
            try:
                from joblib import Parallel, delayed
    
                # 创建一个辅助函数，只返回必要的评估结果，而不是修改对象
                def eval_wrapper(ind):
                    if ind.energy is None:
                        try:
                            energy = self.calculator.calculate(ind.structure)
                            properties = self.calculator.get_properties(ind.structure)
                            # 返回计算结果，而不是修改对象
                            return ind.id, energy, properties
                        except Exception as e:
                            return ind.id, float('inf'), {'error': str(e)}
                    return None
    
                # 并行执行评估，获取结果列表
                results = Parallel(n_jobs=n_jobs)(
                    delayed(eval_wrapper)(individual) for individual in individuals
                )
    
                # 手动更新每个个体的能量和属性
                for result in results:
                    if result is not None:
                        ind_id, energy, props = result
                        # 找到对应的个体
                        for ind in individuals:
                            if ind.id == ind_id:
                                # 更新能量和属性
                                ind.energy = energy
                                ind.properties.update(props)
                                # 计算适应度
                                penalty = sum(constraint.evaluate(ind) for constraint in self.constraints)
                                ind.fitness = self.fitness_function(ind) - penalty
                                # 累加评估计数
                                self.evaluation_count += 1
                                break
            except ImportError:
                # 如果没有joblib，回退到串行评估
                for individual in individuals:
                    self.evaluate(individual)
        else:
            # 串行评估
            for individual in individuals:
                self.evaluate(individual)
    
        # 输出实际完成的评估数量
        end_count = self.evaluation_count
        print(f"Population evaluation complete: {end_count - start_count} new evaluations performed")
        print(f"Total evaluations so far: {end_count}")

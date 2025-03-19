# opencsp/algorithms/pso.py
import random
from typing import List, Dict, Any, Optional

from opencsp.algorithms.optimizer import Optimizer
from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from opencsp.core.structure_generator import StructureGenerator
from opencsp.adapters.dimension_aware import DimensionAwareAdapter
from opencsp.utils.structure import get_structure_dimensionality

class ParticleSwarmOptimization(Optimizer):
    """
    粒子群算法实现，支持不同维度的结构操作
    """
    
    def __init__(self, evaluator: Evaluator, position_adapter: Optional[DimensionAwareAdapter] = None, 
                 velocity_adapter: Optional[DimensionAwareAdapter] = None, **kwargs):
        """
        初始化粒子群算法
        
        Args:
            evaluator: 评估器对象
            position_adapter: 位置更新适配器
            velocity_adapter: 速度更新适配器
            kwargs: 其他参数
        """
        super().__init__(evaluator, **kwargs)
        self.position_adapter = position_adapter or DimensionAwareAdapter()
        self.velocity_adapter = velocity_adapter or DimensionAwareAdapter()
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.inertia_weight = kwargs.get('inertia_weight', 0.7)
        self.cognitive_factor = kwargs.get('cognitive_factor', 1.5)
        self.social_factor = kwargs.get('social_factor', 1.5)
        self.iteration = 0
        
    def initialize(self, structure_generator: StructureGenerator, population_size: int) -> None:
        """初始化粒子群"""
        # 生成初始粒子位置（结构）
        structures = structure_generator.generate(n=population_size)
        self.particles = [Individual(structure) for structure in structures]
        
        # 评估初始粒子
        self.evaluator.evaluate_population(self.particles)
        
        # 初始化个体最佳位置和全局最佳位置
        self.personal_best = [p.copy() for p in self.particles]
        self.global_best = max(self.particles, key=lambda p: p.fitness if p.fitness is not None else float('-inf')).copy()
        self.best_individual = self.global_best.copy()
        
        # 初始化速度
        self.velocities = []
        for p in self.particles:
            dim = get_structure_dimensionality(p.structure)
            # 使用正确的维度初始化速度
            if dim == 1:
                from opencsp.operations.velocity.cluster import initialize_cluster_velocity
                self.velocities.append(initialize_cluster_velocity(p.structure))
            elif dim == 2:
                from opencsp.operations.velocity.surface import initialize_surface_velocity
                # opencsp/algorithms/pso.py (continued)
                self.velocities.append(initialize_surface_velocity(p.structure))
            elif dim == 3:
                from opencsp.operations.velocity.crystal import initialize_crystal_velocity
                self.velocities.append(initialize_crystal_velocity(p.structure))
            else:
                # 默认情况
                self.velocities.append(None)

        self.iteration = 0

    def step(self) -> None:
        """执行一步粒子群算法"""
        for i, particle in enumerate(self.particles):
            # 1. 更新速度
            new_velocity = self.velocity_adapter.apply(
                current_velocity=self.velocities[i],
                particle=particle,
                personal_best=self.personal_best[i],
                global_best=self.global_best,
                inertia_weight=self.inertia_weight,
                cognitive_factor=self.cognitive_factor,
                social_factor=self.social_factor
            )
            self.velocities[i] = new_velocity

            # 2. 更新位置
            new_particle = self.position_adapter.apply(
                particle=particle,
                velocity=self.velocities[i]
            )

            # 3. 评估新位置
            self.evaluator.evaluate(new_particle)
            self.particles[i] = new_particle

            # 4. 更新个体最佳位置
            if (new_particle.fitness is not None and
                (self.personal_best[i].fitness is None or
                 new_particle.fitness > self.personal_best[i].fitness)):
                self.personal_best[i] = new_particle.copy()

                # 5. 更新全局最佳位置
                if (self.global_best.fitness is None or
                    new_particle.fitness > self.global_best.fitness):
                    self.global_best = new_particle.copy()
                    self.best_individual = self.global_best.copy()

        self.iteration += 1

    def get_state(self) -> Dict[str, Any]:
        """获取当前优化状态"""
        return {
            'iteration': self.iteration,
            'best_fitness': self.global_best.fitness if self.global_best else None,
            'best_energy': self.global_best.energy if self.global_best else None,
            'avg_fitness': sum(p.fitness for p in self.particles if p.fitness is not None) /
                          sum(1 for p in self.particles if p.fitness is not None),
            'diversity': self._calculate_diversity(),
            'evaluations': self.evaluator.evaluation_count
        }

    def _calculate_diversity(self) -> float:
        """计算粒子群的多样性"""
        from opencsp.utils.structure import calculate_structure_distance

        if len(self.particles) <= 1:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                total_distance += calculate_structure_distance(
                    self.particles[i].structure,
                    self.particles[j].structure
                )
                count += 1

        if count == 0:
            return 0.0
        return total_distance / count

# opencsp/core/__init__.py
"""
核心模块包含openCSP的基本数据结构和功能组件。
"""

from opencsp.core.individual import Individual
from opencsp.core.population import Population
from opencsp.core.evaluator import Evaluator
from opencsp.core.calculator import Calculator, ASECalculatorWrapper
from opencsp.core.structure_generator import StructureGenerator, RandomStructureGenerator, SymmetryBasedStructureGenerator
from opencsp.core.constraints import Constraint, MinimumDistanceConstraint, SymmetryConstraint

__all__ = [
    'Individual',
    'Population',
    'Evaluator',
    'Calculator',
    'ASECalculatorWrapper',
    'StructureGenerator',
    'RandomStructureGenerator',
    'SymmetryBasedStructureGenerator',
    'Constraint',
    'MinimumDistanceConstraint',
    'SymmetryConstraint'
]

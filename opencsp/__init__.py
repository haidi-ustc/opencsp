# opencsp/__init__.py
"""
OpenCSP: Open-Source Crystal Structure Prediction Software
==========================================================

OpenCSP is a comprehensive Python library for crystal structure prediction 
and optimization. It supports various dimensional systems (clusters, surfaces, 
crystals) and multiple global optimization algorithms.

Key Features:
- Multi-dimensional structure prediction (1D, 2D, 3D)
- Supports multiple global optimization algorithms
- Flexible computational engine integration
- Extensible plugin system
- Advanced API for easy use

Key Optimization Algorithms:
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Basin Hopping
- And more...

Dependencies:
- numpy
- scipy
- ase
- pymatgen (optional)

Quick Example:
    >>> from opencsp.api import OpenCSP
    >>> from ase.calculators.emt import EMT
    
    >>> # Create OpenCSP instance
    >>> csp = OpenCSP()
    
    >>> # Create ASE calculator
    >>> calculator = csp.create_calculator('ase', ase_calculator=EMT())
    >>> evaluator = csp.create_evaluator(calculator)
    
    >>> # Create structure generator
    >>> structure_gen = csp.create_structure_generator(
    ...     'random', 
    ...     composition={'Si': 10}, 
    ...     dimensionality=1
    ... )
    
    >>> # Configure genetic algorithm
    >>> ga_config = csp.create_optimization_config('ga')
    >>> ga_config.set_param('evaluator', evaluator)
    
    >>> # Create and run optimization
    >>> runner = csp.create_runner(
    ...     structure_generator=structure_gen, 
    ...     evaluator=evaluator, 
    ...     optimization_config=ga_config
    ... )
    >>> best_structure = runner.run()
"""

# Package version follows semantic versioning
__version__ = '0.1.0'

# Package authorship
__author__ = 'OpenCSP Development Team'
__email__ = 'dev@opencsp.org'

# Package metadata
__description__ = 'Open-Source Crystal Structure Prediction Software'
__url__ = 'https://github.com/opencsp/opencsp'
__license__ = 'MIT'

# Core imports for convenient access
from opencsp.api import OpenCSP
from opencsp.core.individual import Individual
from opencsp.core.evaluator import Evaluator
from opencsp.core.structure_generator import StructureGenerator
from opencsp.core.calculator import ASECalculatorWrapper
from opencsp.core.constraints import MinimumDistanceConstraint, SymmetryConstraint
from opencsp.runners.csp_runner import CSPRunner, OptimizationConfig

# List of symbols to export when using `from opencsp import *`
__all__ = [
    # Main API entry point
    'OpenCSP',
    
    # Core classes
    'Individual', 
    'Evaluator', 
    'StructureGenerator',
    
    # Calculators
    'ASECalculatorWrapper',
    
    # Constraints
    'MinimumDistanceConstraint',
    'SymmetryConstraint',
    
    # Optimization components
    'OptimizationConfig',
    'CSPRunner',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    '__url__',
    '__license__'
]


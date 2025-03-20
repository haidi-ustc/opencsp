# openCSP: Open-Source Crystal Structure Prediction Software

openCSP is an open-source software for crystal structure prediction and optimization, supporting different dimensional systems (clusters, surfaces, crystals) and multiple global optimization algorithms (genetic algorithm, particle swarm optimization, basin hopping, etc.).

## Main Features

- **Multi-dimensional support**: Handles one-dimensional clusters, two-dimensional surfaces, and three-dimensional crystal structures
- **Multiple optimization algorithms**: Supports genetic algorithms, particle swarm optimization, basin hopping algorithm, and more
- **Flexible computational engine integration**: Supports ASE calculators and machine learning models
- **Extensible plugin system**: Easily add new optimization algorithms and operation strategies
- **Advanced API interface**: Provides a clean and easy-to-use programming interface
- **Based on pymatgen and ASE**: Seamlessly integrates with two popular materials science computation libraries

## Installation

```bash
pip install opencsp
```

## Quick Start

```python
from opencsp.api import OpenCSP
from ase.calculators.emt import EMT

# Initialize openCSP
csp = OpenCSP()

# Create calculator and evaluator
calculator = csp.create_calculator('ase', ase_calculator=EMT())
evaluator = csp.create_evaluator(calculator)

# Create structure generator
structure_gen = csp.create_structure_generator(
    'random', 
    composition={'Si': 10}, 
    dimensionality=1,
    volume_range=(100, 300)
)

# Configure optimization algorithm
ga_config = csp.create_optimization_config('ga')
ga_config.set_param('evaluator', evaluator)
ga_config.set_param('crossover_rate', 0.8)
ga_config.set_param('mutation_rate', 0.2)

# Create and run a CSP job
runner = csp.create_runner(
    structure_generator=structure_gen, 
    evaluator=evaluator, 
    optimization_config=ga_config,
    population_size=20,
    max_steps=50
)

# Get the best structure
best_structure = runner.run()
```

## Project Structure

```
openCSP/
├── README.md
├── setup.py
├── requirements.txt
├── docs/
│   ├── index.md
│   ├── tutorials/
│   └── api/
├── examples/
│   ├── ga_cluster.py
│   ├── pso_surface.py
│   └── ml_crystal.py
├── tests/
│   ├── __init__.py
│   ├── test_core/
│   ├── test_algorithms/
│   └── test_operations/
└── opencsp/
    ├── __init__.py
    ├── api.py
    ├── core/
    │   ├── __init__.py
    │   ├── individual.py
    │   ├── population.py
    │   ├── evaluator.py
    │   ├── calculator.py
    │   ├── structure_generator.py
    │   └── constraints.py
    ├── algorithms/
    │   ├── __init__.py
    │   ├── optimizer.py
    │   ├── genetic.py
    │   ├── pso.py
    │   └── basin_hopping.py
    ├── operations/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── crossover/
    │   ├── mutation/
    │   ├── position/
    │   └── velocity/
    ├── adapters/
    │   ├── __init__.py
    │   ├── dimension_aware.py
    │   └── registry.py
    ├── runners/
    │   ├── __init__.py
    │   └── csp_runner.py
    ├── plugins/
    │   ├── __init__.py
    │   ├── manager.py
    │   └── base.py
    └── utils/
        ├── __init__.py
        ├── structure.py
        ├── logger.py
        └── serialization.py
```

## Advanced Usage

### Custom Calculator

```python
# Using a custom calculator with openCSP
from opencsp.api import OpenCSP
from ase.calculators.lammps import LAMMPS

# Create LAMMPS calculator
lammps_params = {"pair_style": "eam/alloy", "pair_coeff": ["* * Cu_mishin.eam.alloy Cu"]}
calc = LAMMPS(parameters=lammps_params)

# Create openCSP calculator wrapper
csp = OpenCSP()
calculator = csp.create_calculator('ase', ase_calculator=calc)
```

### Machine Learning Models

```python
# Using a machine learning model
from opencsp.api import OpenCSP

csp = OpenCSP()
ml_calculator = csp.create_calculator('ml', model_path='path/to/model.pt')
evaluator = csp.create_evaluator(ml_calculator)
```

### Custom Constraints

```python
# Adding constraints to structure generation
from opencsp.api import OpenCSP
from opencsp.core.constraints import MinimumDistanceConstraint

csp = OpenCSP()

# Define minimum atomic distances
min_distances = {
    'Si-Si': 2.2,  # Minimum Si-Si distance in Å
    'Si-O': 1.6,   # Minimum Si-O distance in Å
    'O-O': 2.0,    # Minimum O-O distance in Å
    'default': 1.5  # Default minimum distance
}

# Create constraint
min_dist_constraint = MinimumDistanceConstraint(min_distances)

# Create structure generator with constraint
structure_gen = csp.create_structure_generator(
    'random', 
    composition={'Si': 8, 'O': 16}, 
    dimensionality=3,
    volume_range=(100, 200),
    constraints=[min_dist_constraint]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

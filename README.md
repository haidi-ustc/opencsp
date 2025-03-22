# OpenCSP: Open-Source Crystal Structure Prediction

## Overview

OpenCSP is a comprehensive, open-source Python library for crystal structure prediction and optimization. It provides a flexible framework for exploring and discovering new crystal structures using advanced computational methods.

## Features

### Key Capabilities
- Multi-dimensional structure prediction (0D clusters, 2D surfaces, 3D crystals)
- Multiple global optimization algorithms
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
- Flexible computational engine integration
- Advanced structure generation and mutation strategies
- Composition and symmetry constraints

### Supported Dimensionalities
- 0D: Atomic clusters
- 2D: Surface structures
- 3D: Bulk crystal structures

## Installation

### Prerequisites
- Python 3.9+
- Dependencies:
  - NumPy
  - SciPy
  - ASE (Atomic Simulation Environment)
  - Pymatgen
  - Optional: PyXtal for advanced symmetry generation

### Install via pip
```bash
pip install opencsp
```

### Optional Dependencies
```bash
pip install pyxtal torch pymatgen
```

## Quick Start Example

```python
from opencsp.api import OpenCSP
from ase.calculators.emt import EMT

# Create OpenCSP instance
csp = OpenCSP(optimizer_type='ga', dimensionality=3)

# Create energy calculator
calculator = csp.create_calculator('ase', ase_calculator=EMT())
evaluator = csp.create_evaluator(calculator)

# Define structure generator
structure_gen = csp.create_structure_generator(
    'random', 
    composition={'Si': 8, 'O': 16}
)

# Configure optimization
ga_config = csp.create_optimization_config()
ga_config.set_param('crossover_rate', 0.8)
ga_config.set_param('mutation_rate', 0.2)

# Create and run optimization
runner = csp.create_runner(
    structure_generator=structure_gen,
    evaluator=evaluator,
    optimization_config=ga_config,
    population_size=50,
    max_steps=100
)

# Execute structure prediction
best_structure = runner.run()
print(f"Best structure energy: {best_structure.energy} eV")
```

## Optimization Algorithms

### Genetic Algorithm
- Crossover strategies
  - Plane-cut crossover
  - Lattice parameter crossover
- Mutation operations
  - Atomic displacement
  - Lattice deformation
  - Strain-based mutations

### Particle Swarm Optimization
- Position update strategies
- Velocity-based exploration

## Advanced Configuration

### Constraints
```python
# Minimum distance constraint
min_dist_constraint = csp.create_constraint(
    'minimum_distance', 
    min_distance={'Si-Si': 2.2, 'Si-O': 1.6}
)

# Symmetry constraint
symmetry_constraint = csp.create_constraint(
    'symmetry', 
    target_spacegroup=225,  # Cubic space group
    tolerance=0.1
)
```

## Computational Engines

### Supported Calculators
- ASE Calculators
- Universal force fields

## Visualization and Analysis

### Output Formats
- CIF files
- JSON structure representation
- Energy and fitness tracking
- Optimization history logging

## Contributing

### How to Contribute
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Setup
```bash
git clone https://gitee.com/haidi-hfut/opencsp.git
cd opencsp
pip install -e .[dev]
pytest
```

## License

MIT License

## Citation

If you use OpenCSP in your research, please cite:
```
OpenCSP Development Team. (2025). 
OpenCSP: Open-Source Crystal Structure Prediction Software. 
GitHub Repository, https://gitee.com/haidi-hfut/opencsp
```

## Contact

- Email:  haidi@hfut.edu.cn
- GitHub: [OpenCSP GitHub](https://gitee.com/haidi-hfut/opencsp)

## Acknowledgments

- Developed with support from the scientific computing community
- Inspired by challenges in materials discovery and computational chemistry
```


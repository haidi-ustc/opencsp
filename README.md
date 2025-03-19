# openCSP: 开源晶体结构预测软件

openCSP是一个用于晶体结构预测和优化的开源软件，支持不同维度的系统（团簇、表面、晶体）和多种全局优化算法（遗传算法、粒子群算法、盆地跳跃算法等）。

## 主要特点

- **多维度支持**：可以处理一维团簇、二维表面和三维晶体结构
- **多种优化算法**：支持遗传算法、粒子群算法、盆地跳跃算法等
- **灵活的计算引擎集成**：支持ASE计算器和机器学习模型
- **可扩展的插件系统**：可以方便地添加新的优化算法和操作策略
- **高级API接口**：提供简洁易用的编程接口
- **基于pymatgen和ASE**：无缝集成两大流行的材料科学计算库

## 安装

```bash
pip install opencsp
```

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
    │   │   ├── __init__.py
    │   │   ├── cluster.py
    │   │   ├── surface.py
    │   │   └── crystal.py
    │   ├── mutation/
    │   │   ├── __init__.py
    │   │   ├── cluster.py
    │   │   ├── surface.py
    │   │   └── crystal.py
    │   ├── position/
    │   │   ├── __init__.py
    │   │   ├── cluster.py
    │   │   ├── surface.py
    │   │   └── crystal.py
    │   └── velocity/
    │       ├── __init__.py
    │       ├── cluster.py
    │       ├── surface.py
    │       └── crystal.py
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

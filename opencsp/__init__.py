# opencsp/__init__.py
"""
openCSP: 开源晶体结构预测软件
==================================

openCSP是一个用于晶体结构预测和优化的开源软件包，支持不同维度的系统（团簇、表面、晶体）
和多种全局优化算法（遗传算法、粒子群算法等）。

主要特点:
- 支持多种全局优化算法
- 自动处理不同维度的结构（1D、2D、3D）
- 支持多种计算引擎（通过ASE和自定义适配器）
- 可扩展的插件系统
- 灵活的配置选项

简单使用示例:
    >>> from opencsp.api import OpenCSP
    >>> csp = OpenCSP()
    >>> calculator = csp.create_calculator('ase', ase_calculator=EMT())
    >>> evaluator = csp.create_evaluator(calculator)
    >>> structure_gen = csp.create_structure_generator('random', composition={'Si': 10}, dimensionality=1)
    >>> ga_config = csp.create_optimization_config('ga')
    >>> ga_config.set_param('evaluator', evaluator)
    >>> runner = csp.create_runner(structure_generator=structure_gen, evaluator=evaluator, optimization_config=ga_config)
    >>> best_structure = runner.run()
"""

__version__ = '0.1.0'
__author__ = 'openCSP Development Team'

# 导出关键API
#from opencsp.api import OpenCSP

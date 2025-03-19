# examples/pso_surface.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencsp.api import OpenCSP

def main():
    """使用粒子群算法优化MgO表面的示例"""
    
    # 创建openCSP实例
    csp = OpenCSP()
    
    # 1. 创建简单的测试计算器
    try:
        from ase.calculators.lj import LennardJones
        calculator = csp.create_calculator('ase', ase_calculator=LennardJones())
    except ImportError:
        print("ASE未安装或LennardJones计算器不可用。请安装ASE: pip install ase")
        return
    
    # 2. 创建评估器
    evaluator = csp.create_evaluator(calculator)
    
    # 3. 创建结构生成器
    try:
        from ase.build import fcc111
        import numpy as np
        
        # 先创建一个表面结构作为模板
        slab = fcc111('Mg', size=(3, 3, 2), vacuum=10.0)
        # 替换顶层原子为O
        symbols = slab.get_chemical_symbols()
        positions = slab.get_positions()
        for i in range(len(symbols)):
            if positions[i, 2] > 10:  # 顶层原子
                symbols[i] = 'O'
        slab.set_chemical_symbols(symbols)
        
        # 计算成分
        unique, counts = np.unique(symbols, return_counts=True)
        composition = dict(zip(unique, counts))
        
        # 使用模板结构创建生成器
        structure_gen = csp.create_structure_generator(
            'random',
            composition=composition,
            volume_range=(slab.get_volume(), slab.get_volume()),  # 固定体积
            dimensionality=2  # 表面
        )
    except ImportError:
        print("创建表面结构失败。请确保安装了ASE及其依赖库。")
        return
    
    # 4. 配置粒子群算法
    pso_config = csp.create_optimization_config('pso', dimensionality=2)
    pso_config.set_param('evaluator', evaluator)
    pso_config.set_param('inertia_weight', 0.7)
    pso_config.set_param('cognitive_factor', 1.5)
    pso_config.set_param('social_factor', 1.5)
    
    # 5. 创建和运行优化
    runner = csp.create_runner(
        structure_generator=structure_gen,
        evaluator=evaluator,
        optimization_config=pso_config,
        output_dir='./MgO_surface_pso',
        population_size=30,
        max_steps=20
    )
    
    # 添加进度回调
    def progress_callback(optimizer, step):
        state = optimizer.get_state()
        print(f"Step {step}: Best energy = {state.get('best_energy', 'N/A')}, "
              f"Evaluations = {state.get('evaluations', 'N/A')}")
    
    runner.add_callback(progress_callback)
    
    print("开始MgO表面结构优化...")
    best_structure = runner.run()
    
    print(f"\n优化完成!")
    print(f"最佳结构能量: {best_structure.energy} eV")
    print(f"总计算次数: {evaluator.evaluation_count}")
    print(f"结果保存在: {os.path.abspath('./MgO_surface_pso/results')}")

if __name__ == "__main__":
    main()

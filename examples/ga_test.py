# examples/ga_cluster.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencsp.api import OpenCSP

def main():
    
    # 创建openCSP实例
    csp = OpenCSP()
    
    # 1. 创建ASE计算器
    try:
        from ase.calculators.emt import EMT
        calculator = csp.create_calculator('ase', ase_calculator=EMT())
    except ImportError:
        print("ASE未安装或EMT计算器不可用。请安装ASE: pip install ase")
        return
    
    # 2. 创建评估器
    evaluator = csp.create_evaluator(calculator)
    
    # 3. 创建结构生成器
    structure_gen = csp.create_structure_generator(
        'random',
        composition={'Al': 10},  # Al10团簇
        volume_range=(200, 400),  # 更大的体积范围
        dimensionality=3,  # 团簇
        min_distance=1.8   # 更小的原子间距
    )
    
    # 4. 配置遗传算法
    ga_config = csp.create_optimization_config('ga', dimensionality=3)
    ga_config.set_param('evaluator', evaluator)
    ga_config.set_param('population_size', 10)
    ga_config.set_param('max_generations', 10)
    ga_config.set_param('crossover_rate', 0.8)
    ga_config.set_param('mutation_rate', 0.2)
    
    # 5. 创建和运行优化
    runner = csp.create_runner(
        structure_generator=structure_gen,
        evaluator=evaluator,
        optimization_config=ga_config,
        output_dir='.',
        population_size=10,
        max_steps=50
    )
    
    # 添加一个简单的回调函数用于输出进度
    def progress_callback(optimizer, step):
        state = optimizer.get_state()
        print(f"Step {step}: Best energy = {state.get('best_energy', 'N/A')}, "
              f"Evaluations = {state.get('evaluations', 'N/A')}")
    
    runner.add_callback(progress_callback)
    
    print("开始Al10体相的结构优化...")
    try:
        best_structure = runner.run()
        
        if best_structure is None:
            print("\n优化失败: 无法找到有效结构。")
            print("请检查结构生成器参数和评估器是否正确配置。")
        else:
            print(f"\n优化完成!")
            print(f"最佳结构能量: {best_structure.energy} eV")
            print(f"总计算次数: {evaluator.evaluation_count}")
            print(f"结果保存在: {os.path.abspath('./result/results')}")
    except Exception as e:
        print(f"\n优化过程中出错: {str(e)}")
        print("请检查配置参数和日志以获取更多信息。")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Example script demonstrating how to use openCSP for crystal structure prediction
with a genetic algorithm on a silicon-oxygen system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opencsp.api import OpenCSP
from opencsp.utils.logging import setup_logging, get_logger

def main():
    # Set up logging
    setup_logging(log_file='ga_bulk.log', log_level='info')
    logger = get_logger(__name__)
    
    # Initialize OpenCSP
    csp = OpenCSP('ga',dimensionality=3)
    
    try:
        # Setup calculator
        from ase.calculators.emt import EMT
        from opencsp.utils.utils import get_universal_calculator
        
        logger.info("Setting up MatSimulator calculator")
        model_name = "mattersim-v1.0.0-5M.pth"
        mattersim = get_universal_calculator("mattersim", model_file=model_name)
        calculator = csp.create_calculator('ase', ase_calculator=mattersim, relax=True)
    except ImportError:
        logger.error("ASE not installed or EMT calculator not available. Please install ASE: pip install ase")
        return
    
    # Create evaluator
    evaluator = csp.create_evaluator(calculator)
    
    # Create structure generator
    structure_gen = csp.create_structure_generator(
        'symmetry',
        composition={'Si': 4, "O": 8}
    )
    
    # Configure GA optimization
    ga_config = csp.create_optimization_config()
    ga_config.set_param('evaluator', evaluator)
    ga_config.set_param('crossover_rate', 0.8)
    ga_config.set_param('mutation_rate', 0.2)
    
    # Setup CSP runner with the optimization parameters
    runner = csp.create_runner(
        structure_generator=structure_gen,
        evaluator=evaluator,
        optimization_config=ga_config,
        population_size=50, 
        max_steps=50,
        output_dir='./csprun'
    )
    
    # Add a callback to monitor progress
    def progress_callback(optimizer, step):
        state = optimizer.get_state()
        best_energy = state.get('best_energy', 'N/A')
        evaluations = state.get('evaluations', 'N/A')
        logger.info(f"Step {step}: Best energy = {best_energy}, Evaluations = {evaluations}")
    
    runner.add_callback(progress_callback)
    
    # Run the optimization
    logger.info("Starting SiO2 crystal structure optimization...")
    try:
        best_structure = runner.run()
        
        if best_structure is None:
            logger.error("Optimization failed: Could not find valid structures.")
            logger.error("Please check structure generator parameters and evaluator configuration.")
        else:
            logger.info(f"Optimization completed!")
            logger.info(f"Best structure energy: {best_structure.energy} eV")
            logger.info(f"Total evaluations: {evaluator.evaluation_count}")
            logger.info(f"Results saved in: {os.path.abspath('./result/results')}")
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.error("Please check configuration parameters and logs for more information.")

if __name__ == "__main__":
    main()

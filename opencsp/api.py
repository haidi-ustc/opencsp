"""
OpenCSP API module: Provides a simplified interface to the OpenCSP library components.

This module contains the main OpenCSP class that serves as the entry point for users,
offering factory methods for creating various components needed for crystal structure prediction.
"""

from typing import Dict, Any, Optional, List, Union
import os

from opencsp.core.calculator import ASECalculatorWrapper, MLCalculator
from opencsp.core.evaluator import Evaluator
from opencsp.core.structure_generator import (
    RandomStructureGenerator, 
    SymmetryBasedStructureGenerator
)
from opencsp.core.constraints import Constraint, MinimumDistanceConstraint, SymmetryConstraint
from opencsp.adapters.registry import OperationRegistry
from opencsp.algorithms.optimizer import OptimizerFactory
from opencsp.runners.csp_runner import CSPRunner, OptimizationConfig
from opencsp.plugins.manager import PluginManager
from opencsp.utils.logging import get_logger

class OpenCSP:
    """
    Main OpenCSP class providing a simplified API interface for crystal structure prediction.
    
    This class serves as the entry point for most users and offers factory methods
    for creating various components needed for a CSP workflow.
    """
    
    def __init__(self):
        """
        Initialize the OpenCSP API.
        """
        # Set up logging
        self.logger = get_logger(__name__)
        self.logger.info("Initializing OpenCSP")
        
        # Create core components
        self.operation_registry = OperationRegistry()
        self.optimizer_factory = OptimizerFactory()
        self.optimizer_factory.operation_registry = self.operation_registry
        self.plugin_manager = PluginManager(self.operation_registry, self.optimizer_factory)
        
        # Register default optimizers and operations
        self._register_defaults()
        
    def _register_defaults(self):
        """Register default optimizers and operations"""
        self.logger.info("Registering default components")
        
        from opencsp.algorithms.genetic import GeneticAlgorithm
        from opencsp.algorithms.pso import ParticleSwarmOptimization
        
        # Register default optimizers
        self.optimizer_factory.register_optimizer('ga', GeneticAlgorithm)
        self.optimizer_factory.register_optimizer('pso', ParticleSwarmOptimization)
        
        # Register default operations
        self._register_default_operations()
        
    def _register_default_operations(self):
        """Register default operations for different structure dimensions"""
        # Import operations
        from opencsp.operations.crossover.cluster import ClusterCrossover
        from opencsp.operations.crossover.surface import SurfaceCrossover
        from opencsp.operations.crossover.crystal import CrystalCrossover
        
        from opencsp.operations.mutation.cluster import ClusterMutation
        from opencsp.operations.mutation.surface import SurfaceMutation
        from opencsp.operations.mutation.crystal import CrystalMutation
        
        from opencsp.operations.position.cluster import ClusterPositionUpdate
        from opencsp.operations.position.surface import SurfacePositionUpdate
        from opencsp.operations.position.crystal import CrystalPositionUpdate
        
        from opencsp.operations.velocity.cluster import ClusterVelocityUpdate
        from opencsp.operations.velocity.surface import SurfaceVelocityUpdate
        from opencsp.operations.velocity.crystal import CrystalVelocityUpdate
        
        # Register crossover operations
        self.operation_registry.register_crossover(ClusterCrossover(), 1)
        self.operation_registry.register_crossover(SurfaceCrossover(), 2)
        self.operation_registry.register_crossover(CrystalCrossover(), 3)
        
        # Register mutation operations
        self.operation_registry.register_mutation(ClusterMutation(), 1)
        self.operation_registry.register_mutation(SurfaceMutation(), 2)
        self.operation_registry.register_mutation(CrystalMutation(), 3)
        
        # Register position update operations
        self.operation_registry.register_position(ClusterPositionUpdate(), 1)
        self.operation_registry.register_position(SurfacePositionUpdate(), 2)
        self.operation_registry.register_position(CrystalPositionUpdate(), 3)
        
        # Register velocity update operations
        self.operation_registry.register_velocity(ClusterVelocityUpdate(), 1)
        self.operation_registry.register_velocity(SurfaceVelocityUpdate(), 2)
        self.operation_registry.register_velocity(CrystalVelocityUpdate(), 3)
        
    def create_calculator(self, calculator_type: str, **kwargs) -> Union[ASECalculatorWrapper, MLCalculator]:
        """
        Create a calculator for energy evaluation.
        
        Args:
            calculator_type: Type of calculator ('ase' or 'ml')
            **kwargs: Additional parameters for the calculator
                For 'ase': ase_calculator (required)
                For 'ml': model_path (required)
                
        Returns:
            Calculator instance
            
        Raises:
            ValueError: If an unknown calculator type is provided
        """
        self.logger.info(f"Creating {calculator_type} calculator")
        
        if calculator_type == 'ase':
            return ASECalculatorWrapper(kwargs.pop('ase_calculator', None), **kwargs)
        elif calculator_type == 'ml':
            return MLCalculator(**kwargs)
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
            
    def create_evaluator(self, calculator: Union[ASECalculatorWrapper, MLCalculator], 
                         constraints: Optional[List[Constraint]] = None,
                         **kwargs) -> Evaluator:
        """
        Create an evaluator for fitness assessment.
        
        Args:
            calculator: Calculator instance for energy evaluation
            constraints: List of constraints to apply during evaluation
            **kwargs: Additional parameters for the evaluator
                
        Returns:
            Evaluator instance
        """
        self.logger.info("Creating evaluator")
        return Evaluator(calculator, constraints=constraints, **kwargs)
        
    def create_structure_generator(self, generator_type: str, 
                                  composition: Dict[str, int], 
                                  **kwargs) -> Union[RandomStructureGenerator, SymmetryBasedStructureGenerator]:
        """
        Create a structure generator.
        
        Args:
            generator_type: Type of generator ('random' or 'symmetry')
            composition: Chemical composition as {element: count}
            **kwargs: Additional parameters for the generator
                Common parameters:
                - dimensionality: Structure dimension (1, 2, or 3)
                - volume_range: Tuple of (min_volume, max_volume)
                - constraints: List of constraints
                For 'symmetry':
                - spacegroup: Space group number
                
        Returns:
            StructureGenerator instance
            
        Raises:
            ValueError: If an unknown generator type is provided
        """
        self.logger.info(f"Creating {generator_type} structure generator")
        
        if generator_type == 'random':
            return RandomStructureGenerator(composition, **kwargs)
        elif generator_type == 'symmetry':
            return SymmetryBasedStructureGenerator(composition, **kwargs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
    def create_constraint(self, constraint_type: str, **kwargs) -> Constraint:
        """
        Create a structure constraint.
        
        Args:
            constraint_type: Type of constraint ('minimum_distance' or 'symmetry')
            **kwargs: Additional parameters for the constraint
                For 'minimum_distance': min_distance (dict or float)
                For 'symmetry': target_spacegroup, tolerance
                
        Returns:
            Constraint instance
            
        Raises:
            ValueError: If an unknown constraint type is provided
        """
        self.logger.info(f"Creating {constraint_type} constraint")
        
        if constraint_type == 'minimum_distance':
            return MinimumDistanceConstraint(**kwargs)
        elif constraint_type == 'symmetry':
            return SymmetryConstraint(**kwargs)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
            
    def create_optimization_config(self, optimizer_type: str, dimensionality: Optional[int] = None) -> OptimizationConfig:
        """
        Create an optimization configuration.
        
        Args:
            optimizer_type: Type of optimizer ('ga', 'pso', etc.)
            dimensionality: Structure dimension (1, 2, or 3)
                
        Returns:
            OptimizationConfig instance
        """
        self.logger.info(f"Creating optimization config for {optimizer_type}")
        return OptimizationConfig(optimizer_type, dimensionality)
        
    def create_runner(self, structure_generator: Union[RandomStructureGenerator, SymmetryBasedStructureGenerator],
                     evaluator: Evaluator, 
                     optimization_config: Optional[OptimizationConfig] = None,
                     **kwargs) -> CSPRunner:
        """
        Create a CSP runner to execute the structure prediction.
        
        This method integrates all the components needed for structure prediction and
        creates a runner instance that can execute the prediction process.
        
        Args:
            structure_generator: Structure generator instance
            evaluator: Evaluator instance
            optimization_config: Optimization configuration
            **kwargs: Additional parameters:
                - population_size: Number of structures per generation
                - max_steps: Maximum number of optimization steps
                - output_dir: Directory to save results (default: './csprun')
                - callbacks: List of callback functions
                
        Returns:
            CSPRunner instance
            
        Note:
            The optimization_config should have the evaluator already set as a parameter.
            Population size and max_steps can be specified here or when calling run().
        """
        self.logger.info("Creating CSP runner")
        
        # Create output directory
        output_dir = kwargs.get('output_dir', './csprun')
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure optimization_config has evaluator set
        if optimization_config and 'evaluator' not in optimization_config.params:
            optimization_config.set_param('evaluator', evaluator)
            
        # Create the runner
        runner = CSPRunner(
            structure_generator,
            evaluator, 
            optimization_config,
            operation_registry=self.operation_registry,  
            optimizer_factory=self.optimizer_factory, 
            **kwargs
        )
        
        # If population_size was specified in both optimization_config and kwargs, 
        # log a warning about potential redundancy
        if (optimization_config and optimization_config.params.get('population_size') is not None and
            kwargs.get('population_size') is not None):
            self.logger.warning(
                f"Population size specified in both optimization_config ({optimization_config.params['population_size']}) "
                f"and runner parameters ({kwargs['population_size']}). The runner parameter will take precedence."
            )
            
        # Same for max_steps/max_generations
        if (optimization_config and optimization_config.params.get('max_generations') is not None and
            kwargs.get('max_steps') is not None):
            self.logger.warning(
                f"Maximum steps/generations specified in both optimization_config ({optimization_config.params['max_generations']}) "
                f"and runner parameters ({kwargs['max_steps']}). The runner parameter will take precedence."
            )
        
        return runner
        
    def load_plugin(self, plugin_name: str, **kwargs) -> Any:
        """
        Load a plugin by name.
        
        Args:
            plugin_name: Name of the plugin
            **kwargs: Parameters to pass to the plugin
                
        Returns:
            Plugin instance
        """
        self.logger.info(f"Loading plugin: {plugin_name}")
        return self.plugin_manager.load_plugin(plugin_name, **kwargs)
        
    def register_plugin(self, plugin_name: str, plugin_class: Any) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_name: Name to register the plugin as
            plugin_class: Plugin class
        """
        self.logger.info(f"Registering plugin: {plugin_name}")
        self.plugin_manager.register_plugin(plugin_name, plugin_class)

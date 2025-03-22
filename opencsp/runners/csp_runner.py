# opencsp/runners/csp_runner.py
"""
CSP Runner module for executing crystal structure prediction workflows.

This module provides the CSPRunner class that coordinates the entire CSP process,
including structure generation, evaluation, and optimization. It also includes the
OptimizationConfig class for configuring optimization searchers.
"""

import os
import json
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar, Union

from monty.json import MSONable, MontyEncoder

from opencsp.core.evaluator import Evaluator
from opencsp.core.structure_generator import StructureGenerator
from opencsp.adapters.registry import OperationRegistry
from opencsp.searchers.base import SearcherFactory, Searcher

# Configure logger
from opencsp.utils.logging import get_logger
logger = get_logger(__name__)

T = TypeVar('T', bound='OptimizationConfig')

class OptimizationConfig(MSONable):
    """
    Configuration class for optimization searchers and operations.
    
    This class manages algorithm parameters and operation strategies for different
    dimensionalities, providing a convenient builder interface for setting up
    optimization searchers.
    
    Attributes:
        optimizer_type (str): Type of optimizer ('ga', 'pso', etc.)
        dimensionality (Optional[int]): Structure dimensionality (1, 2, 3, or None)
        params (Dict[str, Any]): Algorithm parameters
        operations (Dict[str, Dict[int, Any]]): Operations for different dimensions
    """
    
    def __init__(self, optimizer_type: str, dimensionality: Optional[int] = None):
        """
        Initialize an optimization configuration.
        
        Args:
            optimizer_type: Type of optimizer ('ga', 'pso', etc.)
            dimensionality: Structure dimensionality (1, 2, 3, or None)
            
        Example:
            >>> from opencsp.runners.csp_runner import OptimizationConfig
            >>> config = OptimizationConfig('ga', dimensionality=3)
            >>> config.set_param('crossover_rate', 0.8)
            >>> config.set_param('mutation_rate', 0.2)
        """
        self.optimizer_type = optimizer_type
        self.dimensionality = dimensionality
        self.params: Dict[str, Any] = {}
        self.operations: Dict[str, Dict[int, Any]] = {}
        logger.debug(f"Initialized {optimizer_type} configuration with dimensionality={dimensionality}")
        
    def set_param(self, name: str, value: Any) -> 'OptimizationConfig':
        """
        Set an algorithm parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
            
        Example:
            >>> config = OptimizationConfig('ga')
            >>> config.set_param('crossover_rate', 0.8).set_param('mutation_rate', 0.2)
        """
        self.params[name] = value
        logger.debug(f"Set parameter {name} = {value}")
        return self
        
    def set_operation(self, operation_type: str, operation: Any, dim: Optional[int] = None) -> 'OptimizationConfig':
        """
        Set an operation strategy for a specific dimensionality.
        
        Args:
            operation_type: Operation type ('crossover', 'mutation', 'position', 'velocity')
            operation: Operation instance
            dim: Target dimensionality (default: use operation's dimensionality)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> from opencsp.operations.crossover.crystal import CrystalCrossover
            >>> config = OptimizationConfig('ga')
            >>> config.set_operation('crossover', CrystalCrossover(), dim=3)
        """
        if dim is None:
            dim = operation.dimensionality
            
        if operation_type not in self.operations:
            self.operations[operation_type] = {}
            
        self.operations[operation_type][dim] = operation
        logger.debug(f"Set {operation_type} operation for dimensionality {dim}")
        return self
        
    def build(self, registry: OperationRegistry, factory: SearcherFactory) -> Searcher:
        """
        Build an optimizer with the specified configuration.
        
        This method registers operations with the operation registry and creates
        an optimizer instance through the factory. It only registers the relevant
        operations for each optimizer type (crossover and mutation for GA,
        position and velocity for PSO).
        
        Args:
            registry: Operation registry
            factory: Searcher factory
            
        Returns:
            Configured optimizer instance
            
        Example:
            >>> optimizer = config.build(operation_registry, optimizer_factory)
        """
        logger.info(f"Building {self.optimizer_type} optimizer")
        
        # Register only the relevant operations based on optimizer type
        if self.optimizer_type.lower() == 'ga':
            # For Genetic Algorithm, register crossover and mutation operations
            for op_type, dim_ops in self.operations.items():
                if op_type in ['crossover', 'mutation']:
                    for dim, op in dim_ops.items():
                        if op_type == 'crossover':
                            registry.register_crossover(op, dim)
                            logger.debug(f"Registered crossover operation for dimension {dim}")
                        elif op_type == 'mutation':
                            registry.register_mutation(op, dim)
                            logger.debug(f"Registered mutation operation for dimension {dim}")
        
        elif self.optimizer_type.lower() == 'pso':
            # For Particle Swarm Optimization, register position and velocity operations
            for op_type, dim_ops in self.operations.items():
                if op_type in ['position', 'velocity']:
                    for dim, op in dim_ops.items():
                        if op_type == 'position':
                            registry.register_position(op, dim)
                            logger.debug(f"Registered position operation for dimension {dim}")
                        elif op_type == 'velocity':
                            registry.register_velocity(op, dim)
                            logger.debug(f"Registered velocity operation for dimension {dim}")
        
        else:
            # For other optimizers, register all operations
            # This is included for extensibility with future optimizers
            logger.warning(f"Unknown optimizer type '{self.optimizer_type}'. Registering all operations.")
            for op_type, dim_ops in self.operations.items():
                for dim, op in dim_ops.items():
                    if op_type == 'crossover':
                        registry.register_crossover(op, dim)
                    elif op_type == 'mutation':
                        registry.register_mutation(op, dim)
                    elif op_type == 'position':
                        registry.register_position(op, dim)
                    elif op_type == 'velocity':
                        registry.register_velocity(op, dim)
                    logger.debug(f"Registered {op_type} operation for dimension {dim}")
                        
        # Create optimizer
        optimizer = factory.create_optimizer(self.optimizer_type, **self.params)
        logger.info(f"Created {self.optimizer_type} optimizer with {len(self.params)} parameters")
        return optimizer

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the optimization configuration to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the configuration
            
        Example:
            >>> config_dict = config.as_dict()
            >>> import json
            >>> json_str = json.dumps(config_dict)
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "optimizer_type": self.optimizer_type,
            "dimensionality": self.dimensionality,
            "params": self.params
        }
        
        # Handle operations (these may not be directly serializable)
        operations_dict = {}
        for op_type, dim_ops in self.operations.items():
            operations_dict[op_type] = {}
            for dim, op in dim_ops.items():
                if hasattr(op, 'as_dict'):
                    operations_dict[op_type][str(dim)] = op.as_dict()
                else:
                    operations_dict[op_type][str(dim)] = str(op.__class__.__name__)
                    
        d["operations"] = operations_dict
        
        return d
    
    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Create an optimization configuration from a dictionary.
        
        Args:
            d: Dictionary containing configuration data
            
        Returns:
            Reconstructed optimization configuration
            
        Example:
            >>> config = OptimizationConfig.from_dict(config_dict)
        """
        config = cls(
            optimizer_type=d["optimizer_type"],
            dimensionality=d.get("dimensionality")
        )
        
        # Restore parameters
        config.params = d.get("params", {})
        
        # Note: Operations cannot be fully reconstructed from serialized data
        # because they require actual operation instances. The operations field
        # is mainly informational in the serialized form.
        
        return config


class CSPRunner(MSONable):
    """
    Runner for crystal structure prediction workflows.
    
    This class coordinates the entire CSP process, including structure generation,
    evaluation, and optimization. It manages the various components needed for
    structure prediction and handles result output.
    
    Attributes:
        structure_generator (StructureGenerator): Generator for initial structures
        evaluator (Evaluator): Energy and fitness evaluator
        optimization_config (OptimizationConfig): Configuration for the optimizer
        operation_registry (OperationRegistry): Registry of operations
        optimizer_factory (SearcherFactory): Factory for creating optimizers
        population_size (int): Size of the population
        max_steps (int): Maximum number of optimization steps
        callbacks (List[Callable]): Callback functions for monitoring
        output_dir (str): Directory for output files
        results_dir (str): Directory for result files
    """
    
    def __init__(self, structure_generator: StructureGenerator,
                 evaluator: Evaluator, 
                 optimization_config: Optional[OptimizationConfig] = None,
                 operation_registry: Optional[OperationRegistry] = None,
                 optimizer_factory: Optional[SearcherFactory] = None,
                 **kwargs):
        """
        Initialize a CSP runner.
        
        Args:
            structure_generator: Generator for creating initial structures
            evaluator: Energy and fitness evaluator
            optimization_config: Configuration for the optimizer (optional)
            operation_registry: Registry of operations (optional)
            optimizer_factory: Factory for creating optimizers (optional)
            **kwargs: Additional parameters:
                - population_size: Size of the population (default: 50)
                - max_steps: Maximum number of optimization steps (default: 100)
                - callbacks: Callback functions for monitoring (default: [])
                - output_dir: Directory for output files (default: './csp_results')
                
        Example:
            >>> from opencsp.core.evaluator import Evaluator
            >>> from opencsp.core.structure_generator import RandomStructureGenerator
            >>> from opencsp.runners.csp_runner import CSPRunner, OptimizationConfig
            >>> 
            >>> # Create components
            >>> structure_gen = RandomStructureGenerator({'Si': 8}, volume_range=(100, 200))
            >>> evaluator = Evaluator(calculator)
            >>> config = OptimizationConfig('ga')
            >>> config.set_param('evaluator', evaluator)
            >>> 
            >>> # Create runner
            >>> runner = CSPRunner(
            ...     structure_generator=structure_gen,
            ...     evaluator=evaluator,
            ...     optimization_config=config,
            ...     output_dir='results'
            ... )
        """
        self.structure_generator = structure_generator
        self.evaluator = evaluator
        self.optimization_config = optimization_config
        
        # Create operation registry and optimizer factory if not provided
        self.operation_registry = operation_registry or OperationRegistry()
        self.optimizer_factory = optimizer_factory or SearcherFactory()
        self.optimizer_factory.operation_registry = self.operation_registry
        
        # Register optimizers
        from opencsp.searchers.genetic import GA
        from opencsp.searchers.pso import PSO
        if self.optimization_config.optimizer_type =='ga':
           self.optimizer_factory.register_optimizer('ga', GA)

        elif self.optimization_config.optimizer_type =='pso':
             self.optimizer_factory.register_optimizer('pso', ParticleSwarmOptimization)
        else:
            raise RuntimeError("Unsupported optimizer")
        
        # Set run parameters
        self.population_size = kwargs.get('population_size', 50)
        self.max_steps = kwargs.get('max_steps', 100)
        self.callbacks: List[Callable[[Searcher, int], None]] = kwargs.get('callbacks', [])
        self.output_dir = kwargs.get('output_dir', './csprun')
        self.results_dir = os.path.join(self.output_dir, 'results')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Initialized CSPRunner with population_size={self.population_size}, "
                   f"max_steps={self.max_steps}, output_dir='{self.output_dir}'")
        
    def configure(self, optimization_config: OptimizationConfig) -> 'CSPRunner':
        """
        Set the optimization configuration.
        
        Args:
            optimization_config: Configuration for the optimizer
            
        Returns:
            Self for method chaining
            
        Example:
            >>> config = OptimizationConfig('ga')
            >>> runner.configure(config)
        """
        self.optimization_config = optimization_config
        logger.info(f"Set {optimization_config.optimizer_type} optimization configuration")
        return self
        
    def set_population_size(self, size: int) -> 'CSPRunner':
        """
        Set the population size.
        
        Args:
            size: Population size
            
        Returns:
            Self for method chaining
            
        Example:
            >>> runner.set_population_size(100)
        """
        self.population_size = size
        logger.info(f"Set population size to {size}")
        return self
        
    def set_max_steps(self, steps: int) -> 'CSPRunner':
        """
        Set the maximum number of optimization steps.
        
        Args:
            steps: Maximum number of steps
            
        Returns:
            Self for method chaining
            
        Example:
            >>> runner.set_max_steps(200)
        """
        self.max_steps = steps
        logger.info(f"Set maximum steps to {steps}")
        return self
        
    def add_callback(self, callback: Callable[[Searcher, int], None]) -> 'CSPRunner':
        """
        Add a callback function for monitoring optimization progress.
        
        Args:
            callback: Callback function that takes (optimizer, step) as arguments
            
        Returns:
            Self for method chaining
            
        Example:
            >>> def monitor_callback(optimizer, step):
            ...     best = optimizer.best_individual
            ...     print(f"Step {step}: Best energy = {best.energy}")
            >>> 
            >>> runner.add_callback(monitor_callback)
        """
        self.callbacks.append(callback)
        logger.debug(f"Added callback {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")
        return self
        
    def run(self) -> Any:
        """
        Run the crystal structure prediction process.
        
        This method executes the complete CSP workflow, including structure generation,
        evaluation, and optimization.
        
        Returns:
            Best individual found during optimization
            
        Raises:
            ValueError: If optimization configuration is not set
            
        Example:
            >>> best_individual = runner.run()
            >>> print(f"Best energy: {best_individual.energy}")
            >>> print(f"Best fitness: {best_individual.fitness}")
        """
        if self.optimization_config is None:
            logger.error("Optimization configuration is not set")
            raise ValueError("Optimization configuration is not set")
            
        logger.info(f"Starting CSP run with {self.optimization_config.optimizer_type}")
        
        # Build optimizer
        optimizer = self.optimization_config.build(
            self.operation_registry, 
            self.optimizer_factory
        )
        
        # Run optimization
        logger.info(f"Running optimization with population_size={self.population_size}, max_steps={self.max_steps}")
        best_individual = optimizer.run(
            self.structure_generator,
            population_size=self.population_size,
            max_steps=self.max_steps,
            callbacks=self.callbacks,
            output_dir=self.output_dir
        )

        # Save results
        logger.info("Optimization complete, saving results")
        self._save_results(optimizer, best_individual)

        if best_individual:
            logger.info(f"Best individual: energy={best_individual.energy:.6f}, fitness={best_individual.fitness:.6f}")
        else:
            logger.warning("No best individual found")
            
        return best_individual

    def _save_results(self, optimizer: Searcher, best_individual: Any) -> None:
        """
        Save optimization results to files.
        
        This method saves the best structure, its properties, optimization history,
        and configuration to files in the results directory.
        
        Args:
            optimizer: Searcher instance used for the run
            best_individual: Best individual found during optimization
            
        Example:
            >>> runner._save_results(optimizer, best_individual)
        """
        logger.info(f"Saving results to {self.results_dir}")

        # Handle the case when best_individual is None
        if best_individual is None:
            logger.warning("No best individual found. Saving limited results.")
            
            # Save empty best info
            best_info_path = os.path.join(self.results_dir, 'best_info.json')
            with open(best_info_path, 'w') as f:
                json.dump({"error": "No valid solution found"}, f, indent=2)
                
            # Try to save history
            if hasattr(optimizer, 'history') and optimizer.history:
                try:
                    # Save a simplified version of history
                    simplified_history = []
                    for entry in optimizer.history:
                        simplified_entry = {}
                        for key, value in entry.items():
                            if isinstance(value, (int, float, str, bool, type(None))):
                                simplified_entry[key] = value
                        simplified_history.append(simplified_entry)
                        
                    history_path = os.path.join(self.results_dir, 'history.json')
                    with open(history_path, 'w') as f:
                        json.dump(simplified_history, f, indent=2)
                    
                    logger.info(f"Saved optimization history to {history_path}")
                except Exception as e:
                    logger.error(f"Error saving history: {e}")
                    
            # Save configuration
            try:
                config_path = os.path.join(self.results_dir, 'config.json')
                config_dict = {
                    "optimizer_type": self.optimization_config.optimizer_type,
                    "max_steps": self.max_steps,
                    "population_size": self.population_size
                }
                
                # Add serializable params
                params_dict = {}
                for key, value in self.optimization_config.params.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        params_dict[key] = value
                
                config_dict["parameters"] = params_dict
                
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                logger.info(f"Saved configuration to {config_path}")
            except Exception as e:
                logger.error(f"Error saving config: {e}")
                
            return

        # Save best structure as CIF
        best_structure_path = os.path.join(self.results_dir, 'best_structure.cif')
        try:
            from pymatgen.io.cif import CifWriter
            if hasattr(best_individual.structure, 'sites'):
                # pymatgen Structure
                CifWriter(best_individual.structure).write_file(best_structure_path)
                logger.info(f"Saved best structure to {best_structure_path}")
            else:
                # ASE Atoms
                from ase.io import write
                write(best_structure_path, best_individual.structure)
                logger.info(f"Saved best structure to {best_structure_path}")
        except Exception as e:
            logger.error(f"Error saving best structure: {e}")

        # Save best structure info
        best_info_path = os.path.join(self.results_dir, 'best_info.json')
        try:
            best_individual.save( best_info_path , strict=False)
            logger.info(f"Saved best structure info to {best_info_path}")
        except Exception as e:
            logger.error(f"Error saving best info: {e}")

        # Save history
        try:
            # Save a simplified version of history
            simplified_history = []
            for entry in optimizer.history:
                simplified_entry = {}
                for key, value in entry.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        simplified_entry[key] = value
                simplified_history.append(simplified_entry)
                
            history_path = os.path.join(self.results_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(simplified_history, f, indent=2)
            
            logger.info(f"Saved optimization history to {history_path}")
        except Exception as e:
            logger.error(f"Error saving history: {e}")

        # Save configuration
        try:
            config_path = os.path.join(self.results_dir, 'config.json')
            self.optimization_config.save(config_path, strict=False)
            
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the CSP runner to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the runner
            
        Example:
            >>> runner_dict = runner.as_dict()
            >>> import json
            >>> json_str = json.dumps(runner_dict, cls=MontyEncoder)
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "population_size": self.population_size,
            "max_steps": self.max_steps,
            "output_dir": self.output_dir
        }
        
        # Add serializable components
        if hasattr(self.structure_generator, 'as_dict'):
            d["structure_generator"] = self.structure_generator.as_dict()
            
        if hasattr(self.evaluator, 'as_dict'):
            d["evaluator"] = self.evaluator.as_dict()
            
        if self.optimization_config and hasattr(self.optimization_config, 'as_dict'):
            d["optimization_config"] = self.optimization_config.as_dict()
        
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CSPRunner':
        """
        Create a CSP runner from a dictionary.
        
        Args:
            d: Dictionary containing runner data
            
        Returns:
            Reconstructed CSP runner
            
        Note:
            This method cannot fully reconstruct a CSPRunner from serialized data
            because it requires actual structure_generator and evaluator instances.
            
        Example:
            >>> # This can only partially reconstruct a runner
            >>> partial_runner = CSPRunner.from_dict(runner_dict)
        """
        logger.warning("CSPRunner.from_dict() only creates a partial reconstruction")
        
        # This can only create a partial reconstruction without actual instances
        return cls.__new__(cls)

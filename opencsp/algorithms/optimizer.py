# opencsp/algorithms/optimizer.py
"""
This module contains the abstract base Optimizer class and OptimizerFactory.

The Optimizer class provides a common interface for all optimization algorithms
in the openCSP framework, including genetic algorithms, particle swarm optimization,
and other global optimization methods. The OptimizerFactory facilitates creating
and configuring optimizer instances with appropriate adapters.
"""

import traceback
import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, TypeVar, Union

from monty.json import MSONable

from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from opencsp.core.structure_generator import StructureGenerator

# Configure logger
logger = logging.getLogger(__name__)
T = TypeVar('T', bound=Individual)

class Optimizer(ABC, MSONable):
    """
    Abstract base class for optimization algorithms.
    
    This class defines the interface that all optimization algorithms must implement
    in the openCSP framework. It provides common functionality like running the
    optimization process, tracking the best solution, and maintaining history.
    
    Attributes:
        evaluator (Evaluator): Evaluator object for calculating fitness
        best_individual (Individual): Best solution found during optimization
        history (List[Dict[str, Any]]): History of optimization states
        params (Dict[str, Any]): Algorithm-specific parameters
    """
    
    def __init__(self, evaluator: Evaluator, **kwargs):
        """
        Initialize the optimizer.
        
        Args:
            evaluator: Evaluator object for calculating fitness values
            **kwargs: Algorithm-specific parameters
            
        Example:
            >>> from opencsp.core.evaluator import Evaluator
            >>> from opencsp.algorithms.genetic import GeneticAlgorithm
            >>> evaluator = Evaluator(calculator)
            >>> optimizer = GeneticAlgorithm(evaluator, crossover_rate=0.8, mutation_rate=0.2)
        """
        self.evaluator = evaluator
        self.best_individual = None
        self.history = []
        self.params = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with parameters: {kwargs}")
        
    @abstractmethod
    def initialize(self, structure_generator: StructureGenerator, population_size: int) -> None:
        """
        Initialize the search state.
        
        This method should initialize the optimization algorithm's state by generating
        initial structures and evaluating them. It must be implemented by subclasses.
        
        Args:
            structure_generator: Generator for creating initial structures
            population_size: Size of the population or number of particles
            
        Example:
            >>> optimizer.initialize(structure_generator, population_size=50)
        """
        pass
        
    @abstractmethod
    def step(self) -> None:
        """
        Execute one optimization step.
        
        This method should perform one generation/iteration of the optimization
        algorithm, which may include selection, recombination, mutation, etc.
        It must be implemented by subclasses.
        
        Example:
            >>> optimizer.step()  # Perform one optimization step
        """
        pass
        
    def run(self, structure_generator: StructureGenerator,
            population_size: int = 50, 
            max_steps: int = 100,
            callbacks: Optional[List[Callable[['Optimizer', int], None]]] = None,
            output_dir: Optional[str] = None) -> T:
        """
        Run the optimization algorithm.
        
        This method executes the complete optimization process by initializing the
        population and iteratively applying the optimization steps until a termination
        condition is met or the maximum number of steps is reached.
        
        Args:
            structure_generator: Generator for creating initial structures
            population_size: Size of population or number of particles (default: 50)
            max_steps: Maximum number of optimization steps (default: 100)
            callbacks: List of callback functions to execute after each step
        
        Returns:
            The best individual found during optimization
            
        Example:
            >>> best = optimizer.run(
            ...     structure_generator, 
            ...     population_size=50,
            ...     max_steps=100
            ... )
            >>> print(f"Best energy: {best.energy}")
        """
        # Initialize the optimization
        try:
            logger.info(f"Starting optimization with population size={population_size}, max_steps={max_steps}")
            self.initialize(structure_generator, population_size)
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return None
            
        # Update best individual after initialization
        if hasattr(self, 'population') and self.population:
            current_best = self.population.get_best()
            if current_best:
                logger.info(f"Initial best: id={current_best.id}, energy={current_best.energy:.6f}, "
                           f"fitness={current_best.fitness:.6f}")

                # Create a deep copy of the best individual
                new_best = Individual(structure=copy.deepcopy(current_best.structure))
                new_best.energy = current_best.energy
                new_best.fitness = current_best.fitness
                for key, value in current_best.properties.items():
                    new_best.properties[key] = copy.deepcopy(value)

                self.best_individual = new_best
                logger.debug(f"Set initial best_individual: id={self.best_individual.id}")

        # Main optimization loop
        try:
            for step in range(max_steps):
                self.population.save(f"{output_dir}/popu-{step}.json", strict=False)
                logger.info(f"Starting optimization step {step+1}/{max_steps}")
                self.step()
                
                # Ensure best individual exists
                if self.best_individual is None:
                    logger.warning(f"Lost best individual at step {step+1}. Terminating optimization.")
                    break
                
                # Update history
                state = self.get_state()
                self.history.append(state)
                
                # Log current state
                if hasattr(state, 'get'):
                    best_energy = state.get('best_energy')
                    best_fitness = state.get('best_fitness')
                    logger.info(f"Step {step+1} completed: energy={best_energy:.6f}, fitness={best_fitness:.6f}")
                
                # Execute callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, step)
                        
                # Check termination condition
                if self.check_termination():
                    logger.info(f"Termination condition met at step {step+1}")
                    break
                    
        except Exception as e:
            logger.error(f"Optimization failed at step {step+1}: {str(e)}")
            logger.debug(traceback.format_exc())
                
        if self.best_individual:
            logger.info(f"Optimization completed. Best energy: {self.best_individual.energy:.6f}, "
                       f"fitness: {self.best_individual.fitness:.6f}")
        else:
            logger.warning("Optimization completed without finding a valid solution")
            
        return self.best_individual
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current optimization state.
        
        This method should return a dictionary containing the current state of the
        optimization, including metrics like best fitness, average fitness, etc.
        It must be implemented by subclasses.
        
        Returns:
            Dictionary containing the current optimization state
            
        Example:
            >>> state = optimizer.get_state()
            >>> print(f"Best fitness: {state['best_fitness']}")
        """
        pass
        
    def check_termination(self) -> bool:
        """
        Check if the optimization should terminate.
        
        This method checks if termination conditions are met. The default implementation
        always returns False (no early termination). Subclasses can override to implement
        custom termination conditions.
        
        Returns:
            True if termination conditions are met, False otherwise
            
        Example:
            >>> if optimizer.check_termination():
            ...     print("Optimization converged")
        """
        return False
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the optimizer to a JSON-serializable dictionary.
        
        This method is required by MSONable for serialization.
        
        Returns:
            Dictionary representation of the optimizer
            
        Example:
            >>> optimizer_dict = optimizer.as_dict()
            >>> import json
            >>> json_str = json.dumps(optimizer_dict)
        """
        # Basic optimizer info
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "params": self.params,
        }
        
        # Add best individual if available
        if self.best_individual:
            d["best_individual"] = self.best_individual.as_dict()
        
        # Add history (filter out non-serializable items)
        d["history"] = []
        for state in self.history:
            serializable_state = {}
            for k, v in state.items():
                if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                    serializable_state[k] = v
            d["history"].append(serializable_state)
            
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Optimizer':
        """
        Create an optimizer from a dictionary.
        
        This method is required by MSONable for deserialization. Note that since
        Optimizer is an abstract class, this method should be implemented by subclasses
        to properly reconstruct optimizer-specific attributes.
        
        Args:
            d: Dictionary containing optimizer data
            
        Returns:
            Reconstructed optimizer
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("from_dict must be implemented by Optimizer subclasses")


class OptimizerFactory:
    """
    Factory class for creating and configuring optimizers.
    
    This class manages the registration and creation of optimizer instances,
    configuring them with appropriate dimension-aware adapters for different
    structure operations.
    
    Attributes:
        registered_optimizers (Dict[str, Type[Optimizer]]): Registered optimizer classes
        operation_registry: Registry for dimension-specific operations
    """
    
    def __init__(self):
        """
        Initialize the optimizer factory.
        
        Example:
            >>> factory = OptimizerFactory()
            >>> factory.register_optimizer('ga', GeneticAlgorithm)
            >>> factory.register_optimizer('pso', ParticleSwarmOptimization)
        """
        self.registered_optimizers = {}
        self.operation_registry = None
        logger.debug("Initialized OptimizerFactory")
        
    def register_optimizer(self, name: str, optimizer_class: Type[Optimizer]) -> None:
        """
        Register an optimizer class.
        
        Args:
            name: Name to register the optimizer under (e.g., 'ga', 'pso')
            optimizer_class: Optimizer class to register
            
        Example:
            >>> from opencsp.algorithms.genetic import GeneticAlgorithm
            >>> factory.register_optimizer('ga', GeneticAlgorithm)
        """
        self.registered_optimizers[name] = optimizer_class
        logger.debug(f"Registered optimizer: {name} -> {optimizer_class.__name__}")
        
    def create_optimizer(self, name: str, evaluator: Evaluator, **kwargs) -> Optimizer:
        """
        Create an optimizer instance.
        
        This method creates and configures an optimizer instance of the specified type,
        setting up appropriate adapters for dimension-specific operations.
        
        Args:
            name: Optimizer name (e.g., 'ga', 'pso')
            evaluator: Evaluator instance
            **kwargs: Additional parameters for the optimizer
            
        Returns:
            Configured optimizer instance
            
        Raises:
            ValueError: If the specified optimizer name is not registered
            
        Example:
            >>> evaluator = Evaluator(calculator)
            >>> ga_optimizer = factory.create_optimizer('ga', evaluator, 
            ...                                       crossover_rate=0.8, mutation_rate=0.2)
        """
        if name not in self.registered_optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
            
        optimizer_class = self.registered_optimizers[name]
        logger.info(f"Creating optimizer: {name}")
        
        # Configure optimizer with appropriate adapters
        if name == 'ga':
            from opencsp.adapters.dimension_aware import DimensionAwareAdapter
            
            # Create adapters for genetic algorithm
            crossover_adapter = DimensionAwareAdapter()
            mutation_adapter = DimensionAwareAdapter()
            
            # Add dimension-specific operations
            for dim in [1, 2, 3]:
                if self.operation_registry:
                    # Add crossover operations
                    op = self.operation_registry.get_crossover_operation(dim)
                    if op:
                        crossover_adapter.add_operator(dim, op)
                        logger.debug(f"Added crossover operator for dimension {dim}")
                        
                    # Add mutation operations
                    op = self.operation_registry.get_mutation_operation(dim)
                    if op:
                        mutation_adapter.add_operator(dim, op)
                        logger.debug(f"Added mutation operator for dimension {dim}")
            
            return optimizer_class(
                evaluator=evaluator,
                crossover_adapter=crossover_adapter,
                mutation_adapter=mutation_adapter,
                **kwargs
            )
            
        elif name == 'pso':
            from opencsp.adapters.dimension_aware import DimensionAwareAdapter
            
            # Create adapters for particle swarm optimization
            position_adapter = DimensionAwareAdapter()
            velocity_adapter = DimensionAwareAdapter()
            
            # Add dimension-specific operations
            for dim in [1, 2, 3]:
                if self.operation_registry:
                    # Add position update operations
                    op = self.operation_registry.get_position_operation(dim)
                    if op:
                        position_adapter.add_operator(dim, op)
                        logger.debug(f"Added position operator for dimension {dim}")
                        
                    # Add velocity update operations
                    op = self.operation_registry.get_velocity_operation(dim)
                    if op:
                        velocity_adapter.add_operator(dim, op)
                        logger.debug(f"Added velocity operator for dimension {dim}")
            
            return optimizer_class(
                evaluator=evaluator,
                position_adapter=position_adapter,
                velocity_adapter=velocity_adapter,
                **kwargs
            )
            
        else:
            # For other optimizers, create directly
            logger.debug(f"Creating {name} optimizer without adapters")
            return optimizer_class(evaluator=evaluator, **kwargs)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the optimizer factory to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the factory
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "registered_optimizers": {
                name: {
                    "@module": cls.__module__,
                    "@class": cls.__name__
                }
                for name, cls in self.registered_optimizers.items()
            }
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'OptimizerFactory':
        """
        Create an optimizer factory from a dictionary.
        
        Args:
            d: Dictionary containing factory data
            
        Returns:
            Reconstructed optimizer factory
        """
        factory = cls()
        
        # Register optimizers
        for name, optimizer_data in d.get("registered_optimizers", {}).items():
            try:
                # Import the optimizer class
                module_path = optimizer_data["@module"]
                class_name = optimizer_data["@class"]
                
                # Import the module
                module = __import__(module_path, fromlist=[class_name])
                
                # Get the optimizer class
                optimizer_class = getattr(module, class_name)
                
                # Register the optimizer
                factory.register_optimizer(name, optimizer_class)
                
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load optimizer '{name}': {str(e)}")
        
        return factory

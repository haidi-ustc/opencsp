"""
Genetic Algorithm implementation for crystal structure prediction.

This module implements a genetic algorithm (GA) for global optimization of crystal
structures. The GA works with different dimensionalities (clusters, surfaces, crystals)
through dimension-aware adapters for crossover and mutation operations.
"""

import random
import copy
import logging
from typing import List, Dict, Any, Optional, Type, ClassVar

from monty.json import MSONable
from monty.serialization import dumpfn,loadfn

from opencsp.searchers.base import Searcher
from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from opencsp.core.population import Population
from opencsp.core.structure_generator import StructureGenerator
from opencsp.adapters.dimension_aware import DimensionAwareAdapter

# Configure logger
logger = logging.getLogger(__name__)

class GA(Searcher):
    """
    Genetic Algorithm implementation for structure optimization.
    
    This class implements a genetic algorithm for global optimization of atomic structures
    with support for different dimensionalities. It uses dimension-aware adapters to apply
    appropriate crossover and mutation operations based on the structure type.
    
    Attributes:
        evaluator (Evaluator): Evaluator for fitness calculation
        crossover_adapter (DimensionAwareAdapter): Adapter for crossover operations
        mutation_adapter (DimensionAwareAdapter): Adapter for mutation operations
        population (Population): Population of individuals (structures)
        selection_method (str): Method for parent selection ('tournament' or 'roulette')
        crossover_rate (float): Probability of crossover
        mutation_rate (float): Probability of mutation
        elitism (int): Number of best individuals to preserve between generations
    """
    
    def __init__(self, evaluator: Evaluator, crossover_adapter: Optional[DimensionAwareAdapter] = None, 
                 mutation_adapter: Optional[DimensionAwareAdapter] = None, **kwargs):
        """
        Initialize the genetic algorithm.
        
        Args:
            evaluator: Evaluator object for fitness calculation
            crossover_adapter: Adapter for dimension-specific crossover operations (optional)
            mutation_adapter: Adapter for dimension-specific mutation operations (optional)
            **kwargs: Additional parameters:
                - selection_method: Method for parent selection ('tournament' or 'roulette', default: 'tournament')
                - crossover_rate: Probability of crossover (default: 0.8)
                - mutation_rate: Probability of mutation (default: 0.2)
                - elitism: Number of best individuals to preserve (default: 1)
                
        Example:
            >>> from opencsp.core.evaluator import Evaluator
            >>> from opencsp.adapters.dimension_aware import DimensionAwareAdapter
            >>> from opencsp.searchers.genetic import GA
            >>> evaluator = Evaluator(calculator)
            >>> crossover_adapter = DimensionAwareAdapter()
            >>> mutation_adapter = DimensionAwareAdapter()
            >>> ga = GA(
            ...     evaluator, 
            ...     crossover_adapter=crossover_adapter,
            ...     mutation_adapter=mutation_adapter,
            ...     crossover_rate=0.8,
            ...     mutation_rate=0.2
            ... )
        """
        super().__init__(evaluator, **kwargs)
        self.crossover_adapter = crossover_adapter or DimensionAwareAdapter()
        self.mutation_adapter = mutation_adapter or DimensionAwareAdapter()
        self.population = None
        self.selection_method = kwargs.get('selection_method', 'tournament')
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.mutation_rate = kwargs.get('mutation_rate', 0.2)
        self.elitism = kwargs.get('elitism', 1)
        
        logger.info(f"Initialized GA with: selection={self.selection_method}, "
                   f"crossover_rate={self.crossover_rate}, mutation_rate={self.mutation_rate}, "
                   f"elitism={self.elitism}")
        
    def initialize(self, structure_generator: StructureGenerator, population_size: int) -> None:
        """
        Initialize the population.
        
        This method generates initial structures, evaluates them, and creates
        the initial population for the genetic algorithm.
        
        Args:
            structure_generator: Generator for creating initial structures
            population_size: Size of the population
            
        Raises:
            ValueError: If no valid structures could be generated
            
        Example:
            >>> ga.initialize(structure_generator, population_size=50)
        """
        logger.info(f"Initializing population with size {population_size}")
        
        # Generate initial population
        structures = structure_generator.generate(n=population_size)
        
        # Filter out None structures
        valid_structures = [s for s in structures if s is not None]
        
        if not valid_structures:
            logger.error("Failed to generate any valid structures")
            raise ValueError("Failed to generate any valid structures. Please check the structure generator parameters.")
            
        logger.info(f"Generated {len(valid_structures)}/{population_size} valid structures")
        individuals = [Individual(structure, dimensionality = structure_generator.dimensionality) for structure in valid_structures]
        
        # Evaluate initial population
        logger.info("Evaluating initial population")
        self.evaluator.evaluate_population(individuals)
        
        # Create Population object
        self.population = Population(individuals, max_size=population_size)
        
        # Update best individual
        self.best_individual = self.population.get_best()
        
        # Handle the case where there might not be any valid individuals
        if self.best_individual is None:
            logger.warning("No valid individuals in the initial population. This may cause optimization issues.")
        else:
            logger.info(f"Initial best individual: id={self.best_individual.id}, "
                       f"energy={self.best_individual.energy:.6f}, fitness={self.best_individual.fitness:.6f}")
        
    def step(self) -> None:
        """
        Execute one generation of the genetic algorithm.
        
        This method performs one complete cycle of selection, crossover, mutation,
        evaluation, and population update, advancing the genetic algorithm by
        one generation.
        
        Example:
            >>> ga.step()  # Perform one generation
        """
        logger.info(f"-"*20)
        logger.info(f"Starting generation {self.population.generation + 1}")
        
        # 1. Select parents
        if self.selection_method == 'tournament':
            parents = self.population.select_tournament(n=self.population.size)
            logger.debug(f"Selected {len(parents)} parents using tournament selection")
        elif self.selection_method == 'roulette':
            parents = self.population.select_roulette(n=self.population.size)
            logger.debug(f"Selected {len(parents)} parents using roulette selection")
        else:
            logger.error(f"Unknown selection method: {self.selection_method}")
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        # 2. Create offspring
        offspring = []
        logger.debug("Performing crossover operations")
        
        # Perform crossover
        crossover_count = 0
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                if random.random() < self.crossover_rate:
                    # Use dimension-aware adapter for crossover
                    try:
                        children = self.crossover_adapter.apply(parents[i], parents[i+1])
                        offspring.extend(children)
                        crossover_count += 1
                    except Exception as e:
                        logger.warning(f"Crossover failed: {str(e)}")
                        # If crossover fails, just copy the parents
                        offspring.extend([parents[i].copy(), parents[i+1].copy()])
                else:
                    offspring.extend([parents[i].copy(), parents[i+1].copy()])
        
        logger.debug(f"Performed {crossover_count} crossover operations")
        
        # Perform mutation
        logger.debug("Performing mutation operations")
        mutation_count = 0
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                try:
                    # Use dimension-aware adapter for mutation
                    offspring[i] = self.mutation_adapter.apply(offspring[i])
                    mutation_count += 1
                except Exception as e:
                    logger.warning(f"Mutation failed: {str(e)}")
        
        logger.debug(f"Performed {mutation_count} mutation operations")
        
        # 3. Evaluate new individuals
        logger.debug(f"Evaluating {len(offspring)} offspring")
        self.evaluator.evaluate_population(offspring)
        
        # 4. Update population (with elitism)
        old_gen = self.population.generation
        self.population.update(offspring, elitism=self.elitism)
        logger.debug(f"Updated population: generation {old_gen} -> {self.population.generation}")
         
        # 5. Update best individual
        current_best = self.population.get_best()
        if current_best:
            logger.info(f"Current best: id={current_best.id}, "
                      f"energy={current_best.energy:.6f}, fitness={current_best.fitness:.6f}")

            # Check if we need to update the global best
            update_best = False
            if self.best_individual is None:
                update_best = True
                logger.debug("No previous best individual, updating")
            elif (current_best.fitness is not None and 
                 (self.best_individual.fitness is None or 
                  current_best.fitness > self.best_individual.fitness)):
                update_best = True
                logger.debug(f"New best fitness: {current_best.fitness:.6f} > {self.best_individual.fitness:.6f}")

            if update_best:
                # Create a complete copy of the new best individual
                new_best = Individual(
                    structure=copy.deepcopy(current_best.structure),
                    energy=current_best.energy,
                    fitness=current_best.fitness
                )

                # Copy all properties
                for key, value in current_best.properties.items():
                    new_best.properties[key] = copy.deepcopy(value)

                self.best_individual = new_best
                logger.info(f"New global best: id={self.best_individual.id}, "
                           f"energy={self.best_individual.energy:.6f}, fitness={self.best_individual.fitness:.6f}")
        else:
            logger.warning("No best individual found in current population")


    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the genetic algorithm.
        
        This method returns a dictionary containing the current state of the
        optimization, including the generation number, best fitness, best energy,
        average fitness, population diversity, and evaluation count.
        
        Returns:
            Dictionary with the current optimization state
            
        Example:
            >>> state = ga.get_state()
            >>> print(f"Generation {state['generation']}, Best energy: {state['best_energy']}")
        """
        logger.debug(f"Getting state, best individual: {self.best_individual}")
    
        if self.best_individual:
            return {
                'generation': self.population.generation if hasattr(self, 'population') else 0,
                'best_fitness': self.best_individual.fitness,
                'best_energy': self.best_individual.energy,
                'avg_fitness': (self.population.get_average_fitness() 
                              if hasattr(self, 'population') and self.population else float('nan')),
                'diversity': (self.population.get_diversity() 
                            if hasattr(self, 'population') and self.population else 0.0),
                'evaluations': self.evaluator.evaluation_count
            }
        else:
            return {
                'generation': self.population.generation if hasattr(self, 'population') else 0,
                'best_fitness': None,
                'best_energy': None,
                'avg_fitness': float('nan'),
                'diversity': 0.0,
                'evaluations': self.evaluator.evaluation_count
            }
            
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the GA to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the genetic algorithm
            
        Example:
            >>> ga_dict = ga.as_dict()
            >>> import json
            >>> json_str = json.dumps(ga_dict)
        """
        d = super().as_dict()
        
        # Add GA-specific attributes
        d.update({
            "selection_method": self.selection_method,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "elitism": self.elitism,
            "population_generation": self.population.generation if self.population else 0,
        })
        
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GA':
        """
        Create a GA from a dictionary.
        
        Args:
            d: Dictionary containing GA data
            
        Returns:
            Reconstructed GA instance
            
        Note:
            This method reconstructs the GA's parameters but not its state.
            To fully restore a running optimization, additional steps are needed.
            
        Example:
            >>> ga_dict = ga.as_dict()
            >>> reconstructed_ga = GA.from_dict(ga_dict)
        """
        # This can only partially restore the GA without a proper evaluator
        # In practice, you would need to provide an evaluator separately
        params = {
            "selection_method": d.get("selection_method", "tournament"),
            "crossover_rate": d.get("crossover_rate", 0.8),
            "mutation_rate": d.get("mutation_rate", 0.2),
            "elitism": d.get("elitism", 1)
        }
        
        # Add other parameters from the dict
        params.update(d.get("params", {}))
        
        # Create the GA instance
        # Note: This requires an evaluator, which must be provided externally
        # This is a limitation of serializing optimizers that depend on complex objects
        logger.warning("Reconstructing GA requires an Evaluator to be provided")
        
        if "evaluator" in params:
            evaluator = params.pop("evaluator")
            return cls(evaluator=evaluator, **params)
        else:
            # Return a partially initialized object that will need an evaluator later
            # This is not ideal but allows for partial reconstruction
            return cls.__new__(cls)

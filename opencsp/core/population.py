import random
from typing import List, Optional, Any, Callable, TypeVar, Union, Dict, Type, ClassVar

import numpy as np
from monty.json import MSONable

from opencsp.core.individual import Individual
from opencsp.utils.logging import get_logger

"""
Population module for managing collections of individuals in optimization algorithms.

This module provides the Population class that manages collections of Individual objects,
offering methods for selection, sorting, diversity calculation, and other population-level
operations essential for evolutionary algorithms.
"""
# Configure logger
logger = get_logger(__name__)

T = TypeVar('T', bound=Individual)

class Population(MSONable):
    """
    Manages a collection of individuals and provides population-level operations.
    
    The Population class manages a group of Individual objects and provides methods
    for selection, sorting, diversity calculation, and other operations needed by
    evolutionary algorithms. It supports features like elitism, tournament selection,
    and roulette wheel selection.
    
    Attributes:
        individuals (List[T]): List of individuals in the population
        max_size (int): Maximum population size
        generation (int): Current generation number
    """
    
    def __init__(self, individuals: Optional[List[T]] = None, max_size: int = 100):
        """
        Initialize a population.
        
        Args:
            individuals: Initial list of individuals (default: empty list)
            max_size: Maximum population size (default: 100)
            
        Example:
            >>> from opencsp.core.individual import Individual
            >>> from opencsp.core.population import Population
            >>> individuals = [Individual(structure) for structure in structures]
            >>> population = Population(individuals, max_size=50)
        """
        self.individuals = individuals or []
        self.max_size = max_size
        self.generation = 0
        logger.debug(f"Initialized population with {len(self.individuals)} individuals, max_size={max_size}")
        
    @property
    def size(self) -> int:
        """
        Get the current population size.
        
        Returns:
            Current number of individuals in the population
            
        Example:
            >>> population.size
            50
        """
        return len(self.individuals)
        
    def add_individual(self, individual: T) -> None:
        """
        Add an individual to the population.
        
        Args:
            individual: Individual to add to the population
            
        Example:
            >>> new_individual = Individual(structure)
            >>> population.add_individual(new_individual)
        """
        self.individuals.append(individual)
        logger.debug(f"Added individual {individual.id} to population. New size: {self.size}")
        
    def get_best(self, n: int = 1) -> Union[T, List[T]]:
        """
        Get the best n individuals based on fitness.
        
        Args:
            n: Number of best individuals to return (default: 1)
            
        Returns:
            If n=1, returns the best individual or None if the population is empty.
            If n>1, returns a list of the n best individuals (may be shorter if
            population size < n).
            
        Example:
            >>> best_individual = population.get_best()
            >>> top_five = population.get_best(n=5)
        """
        # Filter out None values before sorting
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            logger.warning("No valid individuals found in population")
            return None if n == 1 else []
        
        sorted_individuals = sorted(
            valid_individuals, 
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=True
        )
        
        if n == 1:
            best = sorted_individuals[0] if sorted_individuals else None
            if best:
                logger.debug(f"Best individual: id={best.id}, fitness={best.fitness}")
            return best
        else:
            best_n = sorted_individuals[:n]
            logger.debug(f"Selected {len(best_n)} best individuals")
            return best_n
            
    def select_tournament(self, n: int, tournament_size: int = 3) -> List[T]:
        """
        Select n individuals using tournament selection.
        
        In tournament selection, for each selection, tournament_size individuals are
        randomly chosen, and the best among them is selected.
        
        Args:
            n: Number of individuals to select
            tournament_size: Number of competitors per tournament (default: 3)
            
        Returns:
            List of selected individuals
            
        Example:
            >>> selected = population.select_tournament(n=population.size)
            >>> selected = population.select_tournament(n=20, tournament_size=5)
        """
        # Filter out None values before selection
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            logger.warning("No valid individuals for tournament selection")
            return []
            
        selected = []
        actual_tournament_size = min(tournament_size, len(valid_individuals))
        
        for i in range(n):
            competitors = random.sample(valid_individuals, actual_tournament_size)
            winner = max(competitors, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))
            selected.append(winner)
            
        logger.debug(f"Tournament selection: selected {len(selected)} individuals with tournament size {actual_tournament_size}")
        return selected
        
    def select_roulette(self, n: int) -> List[T]:
        """
        Select n individuals using roulette wheel (fitness proportionate) selection.
        
        In roulette wheel selection, individuals are selected with probability
        proportional to their fitness values.
        
        Args:
            n: Number of individuals to select
            
        Returns:
            List of selected individuals
            
        Example:
            >>> selected = population.select_roulette(n=population.size)
        """
        # Filter out None values before selection
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            logger.warning("No valid individuals for roulette selection")
            return []
            
        # Calculate fitness sum (adjust to non-negative values)
        min_fitness = min((ind.fitness for ind in valid_individuals if ind.fitness is not None), default=0)
        adjusted_fitness = [(ind.fitness - min_fitness + 1e-6) if ind.fitness is not None else 1e-6 
                            for ind in valid_individuals]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness <= 0:
            # If total fitness is zero or negative, use uniform selection
            logger.warning("Total adjusted fitness <= 0, using uniform selection instead")
            return random.choices(valid_individuals, k=n) if valid_individuals else []
        
        # Perform selection
        selected = []
        for i in range(n):
            r = random.uniform(0, total_fitness)
            cum_sum = 0
            for j, fitness in enumerate(adjusted_fitness):
                cum_sum += fitness
                if cum_sum >= r:
                    selected.append(valid_individuals[j])
                    break
            else:
                # If no individual selected due to floating point precision issues, select the last one
                selected.append(valid_individuals[-1])
                
        logger.debug(f"Roulette selection: selected {len(selected)} individuals")
        return selected
        
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """
        Sort individuals by fitness.
        
        Args:
            reverse: If True, sort in descending order (best first), otherwise ascending (default: True)
            
        Example:
            >>> population.sort_by_fitness()  # Sort with best first
            >>> population.sort_by_fitness(reverse=False)  # Sort with worst first
        """
        # Filter out None values
        self.individuals = [ind for ind in self.individuals if ind is not None]
        
        self.individuals.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'),
            reverse=reverse
        )
        
        logger.debug(f"Sorted {len(self.individuals)} individuals by fitness (reverse={reverse})")
        
    def update(self, new_individuals: List[T], elitism: int = 1) -> None:
        """
        Update the population with new individuals, preserving elites.
        
        This method replaces the current population with a combination of elite
        individuals from the current population and new individuals, up to max_size.
        
        Args:
            new_individuals: List of new individuals
            elitism: Number of best individuals to preserve (default: 1)
            
        Example:
            >>> offspring = create_offspring(population)
            >>> population.update(offspring, elitism=2)
        """
        # Filter out None values from new individuals
        new_individuals = [ind for ind in new_individuals if ind is not None]
        logger.debug(f"Updating population with {len(new_individuals)} new individuals, elitism={elitism}")
        
        # Get elite individuals
        if elitism > 0:
            elites = self.get_best(n=elitism)
            if not isinstance(elites, list):
                elites = [elites] if elites is not None else []
            logger.debug(f"Preserving {len(elites)} elite individuals")
        else:
            elites = []
            
        # Update population
        combined = elites + new_individuals
        old_size = len(self.individuals)
        self.individuals = combined[:self.max_size]
        self.generation += 1
        
        logger.info(f"Population updated: size {old_size} -> {len(self.individuals)}, generation {self.generation}")
        
    def get_average_fitness(self) -> float:
        """
        Calculate the average fitness of the population.
        
        Returns:
            Average fitness value, or NaN if no valid fitness values exist
            
        Example:
            >>> avg_fitness = population.get_average_fitness()
        """
        # Filter out None individuals
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if not valid_individuals:
            logger.warning("No valid individuals for average fitness calculation")
            return float('nan')
            
        valid_fitness = [ind.fitness for ind in valid_individuals if ind.fitness is not None]
        if not valid_fitness:
            logger.warning("No valid fitness values for average fitness calculation")
            return float('nan')
        
        avg = sum(valid_fitness) / len(valid_fitness)
        logger.debug(f"Average fitness: {avg:.6f}")
        return avg
        
    def get_diversity(self) -> float:
        """
        Calculate population diversity based on structural differences.
        
        This method calculates the average pairwise distance between all individuals
        in the population, which serves as a measure of population diversity.
        
        Returns:
            Average distance between individuals (diversity measure)
            
        Example:
            >>> diversity = population.get_diversity()
        """
        try:
            from opencsp.utils.structure import calculate_structure_distance
        except ImportError:
            logger.error("Cannot import calculate_structure_distance function")
            return 0.0
        
        # Filter out None individuals
        valid_individuals = [ind for ind in self.individuals if ind is not None]
        
        if len(valid_individuals) <= 1:
            logger.debug("Too few individuals for diversity calculation")
            return 0.0
            
        total_distance = 0.0
        count = 0
        
        for i in range(len(valid_individuals)):
            for j in range(i + 1, len(valid_individuals)):
                try:
                    distance = calculate_structure_distance(
                        valid_individuals[i].structure,
                        valid_individuals[j].structure
                    )
                    total_distance += distance
                    count += 1
                except Exception as e:
                    logger.warning(f"Error calculating structure distance: {e}")
                
        if count == 0:
            logger.warning("No valid distances calculated for diversity")
            return 0.0
            
        diversity = total_distance / count
        logger.debug(f"Population diversity: {diversity:.6f}")
        return diversity
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the Population to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the population
            
        Example:
            >>> population_dict = population.as_dict()
            >>> import json
            >>> json_str = json.dumps(population_dict)
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "individuals": [ind.as_dict() for ind in self.individuals if ind is not None],
            "max_size": self.max_size,
            "generation": self.generation
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Population':
        """
        Create a Population from a dictionary.
        
        Args:
            d: Dictionary containing population data
            
        Returns:
            Reconstructed Population instance
            
        Example:
            >>> population_dict = population.as_dict()
            >>> reconstructed = Population.from_dict(population_dict)
        """
        from opencsp.core.individual import Individual
        
        # Recreate individuals
        individuals = []
        for ind_dict in d.get("individuals", []):
            try:
                individual = Individual.from_dict(ind_dict)
                individuals.append(individual)
            except Exception as e:
                logger.warning(f"Error reconstructing individual: {e}")
        
        # Create population with reconstructed individuals
        population = cls(
            individuals=individuals,
            max_size=d.get("max_size", 100)
        )
        population.generation = d.get("generation", 0)
        
        return population

from typing import List, Callable, Optional, Any, Dict, Union, Type, TypeVar

import numpy as np
from monty.json import MSONable
from opencsp.core.composition_manager import CompositionManager
from opencsp.core.individual import Individual
from opencsp.core.calculator import Calculator
from opencsp.utils.logging import get_logger
"""
Evaluator module for assessing fitness of individuals in optimization algorithms.

This module provides the Evaluator class that calculates energies and fitness values
for individual structures during optimization. It supports both serial and parallel
evaluation, constraint application, and customizable fitness functions.
"""

# Configure logger
logger = get_logger(__name__)

# Type variable for constraints
C = TypeVar('C')

class Evaluator(MSONable):
    """
    Evaluates individuals to determine their fitness in optimization algorithms.
    
    The Evaluator class is responsible for calculating the energy and fitness
    of individuals using a calculator and fitness function. It can apply constraints
    to penalize invalid structures, and can evaluate both single individuals and
    entire populations, with optional parallel processing.
    
    Attributes:
        calculator (Calculator): Engine for energy calculations
        fitness_function (Callable): Function to convert energy to fitness
        constraints (List): List of constraints to apply during evaluation
        evaluation_count (int): Total number of evaluations performed
    """
    
    def __init__(self, calculator: Calculator, 
                 fitness_function: Optional[Callable[[Individual], float]] = None, 
                 constraints: Optional[List[Any]] = None):
        """
        Initialize an evaluator.
        
        Args:
            calculator: Calculator for energy computation
            fitness_function: Function that converts energy to fitness (default: negative energy)
            constraints: List of constraints to apply during evaluation
            
        Example:
            >>> from opencsp.core.calculator import ASECalculatorWrapper
            >>> from opencsp.core.evaluator import Evaluator
            >>> from ase.calculators.emt import EMT
            >>> calculator = ASECalculatorWrapper(EMT())
            >>> evaluator = Evaluator(calculator)
        """
        self.calculator = calculator
        self.fitness_function = fitness_function or (lambda indiv: -indiv.energy if indiv.energy is not None else None)
        self.constraints = constraints or []
        self.evaluation_count = 0
        
        logger.info(f"Initialized Evaluator with calculator: {calculator.__class__.__name__}")
        logger.debug(f"Using {len(self.constraints)} constraints")

    def evaluate(self, individual: Individual) -> float:
        """
        Evaluate a single individual.
        
        This method calculates the energy of an individual using the calculator,
        applies any constraints, and computes the fitness value.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness value, or -inf if evaluation failed
            
        Example:
            >>> individual = Individual(structure)
            >>> fitness = evaluator.evaluate(individual)
        """
        logger.debug(f"Evaluating individual {individual.id}")
        
        # Ensure structure is in correct format for calculator
        structure = individual.structure
        try:
            from pymatgen.core import Structure
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase import Atoms
            
            # Convert pymatgen Structure to ASE Atoms if needed
            if isinstance(structure, Structure):
                structure = AseAtomsAdaptor().get_atoms(structure)
                logger.debug("Converted pymatgen Structure to ASE Atoms")
        except ImportError as e:
            logger.warning(f"Import error during structure conversion: {e}")
        
        # Calculate energy if not already done
        if individual.energy is None:
            try:
                energy = self.calculator.calculate(structure)
                logger.info(f"Individual {individual.id}: energy: {energy:.6f} formula: {individual.structure.formula} ")
                individual.energy = energy
                
                # Calculate other properties
                properties = self.calculator.get_properties(structure)
                individual.properties.update(properties)
                
                # Increment evaluation counter
                self.evaluation_count += 1
                logger.debug(f"Increased evaluation count to {self.evaluation_count}")
            except Exception as e:
                logger.error(f"Error calculating energy for individual {individual.id}: {e}")
                individual.energy = float('inf')
                individual.properties['error'] = str(e)
                
        # Apply constraints
        penalty = 0.0
        for i, constraint in enumerate(self.constraints):
            try:
                constraint_penalty = constraint.evaluate(individual)
                penalty += constraint_penalty
                if constraint_penalty > 0:
                    logger.debug(f"Constraint {i} applied penalty: {constraint_penalty:.6f}")
            except Exception as e:
                logger.error(f"Error applying constraint {i}: {e}")
        
        # Calculate fitness
        if individual.energy is not None:
            try:
                raw_fitness = self.fitness_function(individual)
                individual.fitness = raw_fitness - penalty
                logger.debug(f"Individual {individual.id}: fitness = {individual.fitness:.6f} " 
                            f"(raw: {raw_fitness:.6f}, penalty: {penalty:.6f}")
            except Exception as e:
                logger.error(f"Error calculating fitness: {e}")
                individual.fitness = float('-inf')
        
        return individual.fitness if individual.fitness is not None else float('-inf')

    def evaluate_population(self, individuals: List[Individual], 
                           parallel: bool = False, n_jobs: int = -1) -> None:
        """
        Evaluate an entire population of individuals.
        
        This method evaluates multiple individuals, with optional parallel processing
        using joblib for improved performance.
        
        Args:
            individuals: List of individuals to evaluate
            parallel: Whether to use parallel processing (default: False)
            n_jobs: Number of parallel jobs (-1 for all cores, default: -1)
            
        Example:
            >>> population = [Individual(structure) for structure in structures]
            >>> evaluator.evaluate_population(population, parallel=True)
        """
        # Record starting count
        start_count = self.evaluation_count
        valid_individuals = [ind for ind in individuals if ind is not None]
        
        if not valid_individuals:
            logger.warning("No valid individuals to evaluate")
            return
            
        logger.info(f"Evaluating population of {len(valid_individuals)} individuals "
                   f"(parallel={parallel}, n_jobs={n_jobs})")
    
        if parallel and len(valid_individuals) > 1:
            try:
                from joblib import Parallel, delayed
                logger.debug("Using parallel evaluation with joblib")
    
                # Create a helper function that only returns evaluation results
                def eval_wrapper(ind):
                    if ind.energy is None:
                        try:
                            # Ensure structure is in correct format
                            structure = ind.structure
                            try:
                                from pymatgen.core import Structure
                                from pymatgen.io.ase import AseAtomsAdaptor
                                
                                if isinstance(structure, Structure):
                                    structure = AseAtomsAdaptor().get_atoms(structure)
                            except ImportError:
                                pass  # Use structure as-is if conversion fails
                                
                            energy = self.calculator.calculate(structure)
                            properties = self.calculator.get_properties(structure)
                            # Return results without modifying the object
                            return ind.id, energy, properties
                        except Exception as e:
                            logger.error(f"Error in parallel evaluation of {ind.id}: {e}")
                            return ind.id, float('inf'), {'error': str(e)}
                    return None
    
                # Execute evaluations in parallel
                results = Parallel(n_jobs=n_jobs)(
                    delayed(eval_wrapper)(individual) for individual in valid_individuals
                )
    
                # Update each individual with its results
                valid_results = [r for r in results if r is not None]
                logger.debug(f"Received {len(valid_results)} valid results from parallel evaluation")
                
                for result in valid_results:
                    if result is not None:
                        ind_id, energy, props = result
                        # Find the corresponding individual
                        for ind in valid_individuals:
                            if ind.id == ind_id:
                                # Update energy and properties
                                ind.energy = energy
                                ind.properties.update(props)
                                # Calculate fitness
                                try:
                                    penalty = sum(constraint.evaluate(ind) for constraint in self.constraints)
                                    ind.fitness = self.fitness_function(ind) - penalty
                                except Exception as e:
                                    logger.error(f"Error calculating fitness for {ind.id}: {e}")
                                    ind.fitness = float('-inf')
                                # Increment evaluation count
                                self.evaluation_count += 1
                                break
            except ImportError as e:
                logger.warning(f"Joblib not available for parallel processing: {e}")
                logger.info("Falling back to serial evaluation")
                # Fall back to serial evaluation
                for individual in valid_individuals:
                    self.evaluate(individual)
        else:
            # Serial evaluation
            logger.debug("Using serial evaluation")
            for individual in valid_individuals:
                self.evaluate(individual)
    
        # Report evaluation statistics
        end_count = self.evaluation_count
        new_evaluations = end_count - start_count
        logger.info(f"Population evaluation complete: {new_evaluations} new evaluations performed")
        logger.info(f"Total evaluations so far: {end_count}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the Evaluator to a JSON-serializable dictionary.
        
        This method is required by MSONable for serialization.
        
        Returns:
            Dictionary representation of the evaluator
            
        Example:
            >>> evaluator_dict = evaluator.as_dict()
            >>> import json
            >>> json_str = json.dumps(evaluator_dict)
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "evaluation_count": self.evaluation_count
        }
        
        # Handle calculator (may not be directly serializable)
        if hasattr(self.calculator, 'as_dict'):
            d['calculator'] = self.calculator.as_dict()
        else:
            d['calculator_type'] = self.calculator.__class__.__name__
            
        # Handle constraints
        if self.constraints:
            d['constraints'] = []
            for constraint in self.constraints:
                if hasattr(constraint, 'as_dict'):
                    d['constraints'].append(constraint.as_dict())
                else:
                    d['constraints'].append(str(constraint))
                    
        # Fitness function cannot be easily serialized,
        # just note if it's the default or custom
        d['using_default_fitness'] = (self.fitness_function.__code__.co_code == 
                                     (lambda indiv: -indiv.energy if indiv.energy is not None else None).__code__.co_code)
                                     
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Evaluator':
        """
        Create an Evaluator from a dictionary.
        
        Args:
            d: Dictionary containing evaluator data
            
        Returns:
            Reconstructed Evaluator object
            
        Note:
            This method requires a calculator to be provided separately, as
            calculator instances typically cannot be fully serialized.
            
        Example:
            >>> # Partial reconstruction, requires calculator
            >>> partial_evaluator = Evaluator.from_dict(evaluator_dict)
            >>> # Complete reconstruction with calculator
            >>> full_evaluator = Evaluator(calculator, fitness_function=custom_fitness)
            >>> full_evaluator.evaluation_count = evaluator_dict['evaluation_count']
        """
        # This cannot fully reconstruct an evaluator without a calculator
        logger.warning("Reconstructing Evaluator requires a Calculator to be provided separately")
        
        # Create a placeholder evaluator
        evaluator = cls.__new__(cls)
        evaluator.calculator = None
        evaluator.fitness_function = lambda indiv: -indiv.energy if indiv.energy is not None else None
        evaluator.constraints = []
        evaluator.evaluation_count = d.get('evaluation_count', 0)
        
        return evaluator

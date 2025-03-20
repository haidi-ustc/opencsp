"""
Crystal Mutation Operations for Structural Optimization

This module provides advanced mutation strategies for crystal structures,
supporting both fixed and variable composition scenarios.
"""

import random
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union

from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality
from opencsp.utils.logging import get_logger

logger = get_logger(__name__)


class CrystalMutation(StructureOperation):
    """
    Advanced mutation operations for crystal structures.
    
    Supports different mutation strategies with configurable composition constraints.
    
    Attributes:
        dimensionality (int): Target structural dimensionality (3 for crystals)
        mutation_strength (float): Intensity of structural mutations
        method (str): Mutation method ('lattice', 'atomic_displacement', 'strain')
        mutation_probability (float): Probability of mutation for each atom
        variable_composition (bool): Allow composition changes during mutation
    """
    
    def __init__(self, 
                 dimensionality: int = 3, 
                 mutation_strength: float = 0.2, 
                 method: str = 'lattice', 
                 mutation_probability: float = 0.2,
                 variable_composition: bool = False,
                 **kwargs):
        """
        Initialize crystal mutation operation.
        
        Args:
            dimensionality: Target structural dimensionality (default: 3)
            mutation_strength: Intensity of mutations (default: 0.2)
            method: Mutation strategy (default: 'lattice')
            mutation_probability: Probability of atom mutation (default: 0.2)
            variable_composition: Allow composition changes (default: False)
            **kwargs: Additional configuration parameters
        """
        super().__init__(dimensionality=dimensionality, **kwargs)
        self.mutation_strength = mutation_strength
        self.method = method
        self.mutation_probability = mutation_probability
        self.variable_composition = variable_composition
    
    def apply(self, individual: Individual, **kwargs) -> Individual:
        """
        Apply mutation to a crystal structure.
        
        Args:
            individual: Structure to mutate
            **kwargs: Optional override parameters
        
        Returns:
            Mutated individual
        """
        # Update parameters from kwargs
        strength = kwargs.get('mutation_strength', self.mutation_strength)
        method = kwargs.get('method', self.method)
        probability = kwargs.get('mutation_probability', self.mutation_probability)
        variable_composition = kwargs.get('variable_composition', self.variable_composition)
        
        # Select mutation strategy
        mutation_methods = {
            'lattice': self._lattice_mutation,
            'atomic_displacement': self._atomic_displacement_mutation,
            'strain': self._strain_mutation
        }
        
        if method not in mutation_methods:
            raise ValueError(f"Unknown mutation method: {method}")
        
        # Apply selected mutation method
        mutated_individual = mutation_methods[method](
            individual, 
            strength, 
            probability, 
            variable_composition
        )
        logger.info(f"mutated parent Individual : {individual.id}")
        logger.info(f"mutated child Individual  : {mutated_individual.id}")

        return mutated_individual
    
    def _lattice_mutation(self, 
                           individual: Individual, 
                           strength: float, 
                           probability: float, 
                           variable_composition: bool) -> Individual:
        """
        Mutate crystal lattice parameters.
        
        Args:
            individual: Structure to mutate
            strength: Mutation intensity
            probability: Mutation probability (unused in lattice mutation)
            variable_composition: Allow composition changes
        
        Returns:
            Mutated individual
        """
        # Create a copy of the individual to avoid modifying the original
        mutated_individual = individual.copy()
        structure = mutated_individual.structure
        
        # Pymatgen Structure handling
        if hasattr(structure, 'lattice'):
            # Current lattice parameters
            a, b, c, alpha, beta, gamma = structure.lattice.parameters
            
            # Random perturbations with controlled strength
            delta_a = a * (1.0 + (random.random() - 0.5) * 2 * strength)
            delta_b = b * (1.0 + (random.random() - 0.5) * 2 * strength)
            delta_c = c * (1.0 + (random.random() - 0.5) * 2 * strength)
            
            # Angle mutations (with smaller range)
            angle_strength = strength * 0.3
            delta_alpha = alpha + (random.random() - 0.5) * 2 * angle_strength * 10.0
            delta_beta = beta + (random.random() - 0.5) * 2 * angle_strength * 10.0
            delta_gamma = gamma + (random.random() - 0.5) * 2 * angle_strength * 10.0
            
            # Constrain angles within reasonable ranges
            delta_alpha = min(max(delta_alpha, 60.0), 120.0)
            delta_beta = min(max(delta_beta, 60.0), 120.0)
            delta_gamma = min(max(delta_gamma, 60.0), 120.0)
            
            # Create new lattice
            from pymatgen.core import Lattice
            new_lattice = Lattice.from_parameters(delta_a, delta_b, delta_c, delta_alpha, delta_beta, delta_gamma)
            
            # Update structure lattice
            structure.lattice = new_lattice
        
        # ASE Atoms handling
        else:
            cell = structure.get_cell()
            
            # Generate strain matrix
            strain_matrix = np.eye(3) + (np.random.rand(3, 3) - 0.5) * 2 * strength
            
            # Apply strain
            new_cell = np.dot(cell, strain_matrix)
            
            # Update structure with scaled atoms
            structure.set_cell(new_cell, scale_atoms=True)
        
        return mutated_individual
    
    def _atomic_displacement_mutation(self, 
                                      individual: Individual, 
                                      strength: float, 
                                      probability: float, 
                                      variable_composition: bool) -> Individual:
        """
        Mutate atomic positions with optional composition flexibility.
        
        Args:
            individual: Structure to mutate
            strength: Mutation displacement intensity
            probability: Probability of mutating each atom
            variable_composition: Allow composition changes
        
        Returns:
            Mutated individual
        """
        mutated_individual = individual.copy()
        structure = mutated_individual.structure
        
        # Track original composition
        original_composition = {}
        if hasattr(structure, 'sites'):  # Pymatgen Structure
            original_composition = dict(structure.composition.element_composition)
            coords = [site.frac_coords for site in structure.sites]
            species = [site.species_string for site in structure.sites]
        else:  # ASE Atoms
            original_composition = {
                symbol: list(structure.get_chemical_symbols()).count(symbol)
                for symbol in set(structure.get_chemical_symbols())
            }
            coords = structure.get_scaled_positions()
            species = structure.get_chemical_symbols()
        
        # Perform atomic mutations
        mutated_coords = []
        mutated_species = []
        
        for i, (coord, atom_species) in enumerate(zip(coords, species)):
            # Apply random displacement
            if random.random() < probability:
                displacement = np.random.normal(0, strength, 3)
                new_coord = coord + displacement
                
                # Optional composition change
                if variable_composition:
                    # Small chance to swap or change atom type
                    if random.random() < 0.1 and len(set(species)) > 1:
                        other_species = [s for s in set(species) if s != atom_species]
                        atom_species = random.choice(other_species)
            else:
                new_coord = coord
                
            mutated_coords.append(new_coord)
            mutated_species.append(atom_species)
        
        # Reconstruct structure with new coordinates and species
        if hasattr(structure, 'lattice'):  # Pymatgen Structure
            from pymatgen.core import Structure
            mutated_structure = Structure(
                structure.lattice, 
                mutated_species, 
                mutated_coords, 
                coords_are_cartesian=False
            )
        else:  # ASE Atoms
            import copy
            mutated_structure = copy.deepcopy(structure)
            mutated_structure.set_scaled_positions(mutated_coords)
            mutated_structure.set_chemical_symbols(mutated_species)
        
        # Update individual with new structure
        mutated_individual.structure = mutated_structure
        
        return mutated_individual
    
    def _strain_mutation(self, 
                         individual: Individual, 
                         strength: float, 
                         probability: float, 
                         variable_composition: bool) -> Individual:
        """
        Apply strain mutation to the crystal structure.
        
        Args:
            individual: Structure to mutate
            strength: Mutation strain intensity
            probability: Unused in strain mutation
            variable_composition: Unused in strain mutation
        
        Returns:
            Mutated individual
        """
        mutated_individual = individual.copy()
        structure = mutated_individual.structure
        
        # Generate symmetric strain tensor
        strain = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    # Diagonal elements (stretching/compression)
                    strain[i, j] = (random.random() - 0.5) * 2 * strength
                else:
                    # Off-diagonal elements (shear)
                    strain[i, j] = (random.random() - 0.5) * 2 * strength * 0.5
                    strain[j, i] = strain[i, j]
        
        # Create deformation matrix
        deformation = np.eye(3) + strain
        
        # Pymatgen Structure handling
        if hasattr(structure, 'lattice'):
            matrix = structure.lattice.matrix
            new_matrix = np.dot(matrix, deformation)
            
            from pymatgen.core import Lattice
            new_lattice = Lattice(new_matrix)
            structure.lattice = new_lattice
        
        # ASE Atoms handling
        else:
            cell = structure.get_cell()
            new_cell = np.dot(cell, deformation)
            structure.set_cell(new_cell, scale_atoms=True)
        
        return mutated_individual


"""
Crystal Structure Crossover Operations for Structural Optimization

This module provides advanced crossover strategies for crystal structures,
supporting both fixed and variable composition scenarios.
"""

import random
import numpy as np
from typing import List, Dict, Any, Union

from pymatgen.core import Structure, Lattice
from opencsp.core.individual import Individual
from opencsp.operations.base import StructureOperation
from opencsp.utils.structure import get_structure_dimensionality
from opencsp.utils.logging import get_logger

logger = get_logger(__name__)

class CrystalCrossover(StructureOperation):
    """
    Advanced crossover operations for crystal structures using pymatgen.

    Supports different crossover strategies with configurable composition constraints.
    """

    def __init__(self,
                 dimensionality: int = 3,
                 method: str = 'plane_cut',
                 variable_composition: bool = False,
                 **kwargs):
        """
        Initialize crystal crossover operation.

        Args:
            dimensionality: Target structural dimensionality (default: 3)
            method: Crossover strategy (default: 'plane_cut')
            variable_composition: Allow composition changes (default: False)
            **kwargs: Additional configuration parameters
        """
        super().__init__(dimensionality=dimensionality, **kwargs)
        self.method = method
        self.variable_composition = variable_composition

    def apply(self, parent1: Individual, parent2: Individual, **kwargs) -> List[Individual]:
        """
        Perform crossover between two crystal structures.

        Args:
            parent1: First parent individual
            parent2: Second parent individual
            **kwargs: Optional override parameters

        Returns:
            List of two child individuals
        """
        # Validate input is pymatgen Structure
        if not isinstance(parent1.structure, Structure) or not isinstance(parent2.structure, Structure):
            raise ValueError("CrystalCrossover requires pymatgen Structure objects")

        # Update parameters from kwargs
        method = kwargs.get('method', self.method)
        variable_composition = kwargs.get('variable_composition', self.variable_composition)

        # Validate dimensionality
        dim1 = get_structure_dimensionality(parent1.structure)
        dim2 = get_structure_dimensionality(parent2.structure)

        if dim1 != 3 or dim2 != 3:
            raise ValueError(f"CrystalCrossover only supports 3D structures, received dimensions {dim1} and {dim2}")

        # Select crossover method
        crossover_methods = {
            'plane_cut': self._plane_cut_crossover,
            'lattice_parameter': self._lattice_parameter_crossover
        }

        if method not in crossover_methods:
            raise ValueError(f"Unknown crossover method: {method}")

        # Perform crossover with robust error handling
        try:
            children = crossover_methods[method](
                parent1.structure,
                parent2.structure,
                variable_composition
            )
        except Exception as e:
            logger.warning(f"Crossover failed: {str(e)}")
            # Return parent structures if crossover fails
            return [parent1, parent2]

        # Create child individuals
        child_individuals = [
            Individual(child_struct, dimensionality=parent1.dimensionality)
            for child_struct in children
        ]
        logger.info(f"crossover parent Individuals : {parent1.id} and {parent2.id}")
        logger.info(f"crossover child Individuals  : {child_individuals[0].id} and {child_individuals[1].id}")

        return child_individuals

    def _plane_cut_crossover(self,
                              struct1: Structure,
                              struct2: Structure,
                              variable_composition: bool) -> List[Structure]:
        """
        Perform plane-cut crossover between two crystal structures.

        Args:
            struct1: First parent structure
            struct2: Second parent structure
            variable_composition: Allow composition changes

        Returns:
            List of two child structures
        """
        # Ensure we have numerical indices
        coords1 = np.array([site.frac_coords for site in struct1.sites])
        species1 = [str(site.species_string) for site in struct1.sites]
        coords2 = np.array([site.frac_coords for site in struct2.sites])
        species2 = [str(site.species_string) for site in struct2.sites]

        # Ensure we have at least two sites in each structure
        if len(coords1) < 2 or len(coords2) < 2:
            return [struct1, struct2]

        # Select random cutting plane (basic planes: x, y, z)
        planes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        plane = random.choice(planes)

        # Determine cutting dimension and position
        cut_position = random.random()
        dim = planes.index(plane)

        # Ensure these are integer indices
        indices1_part1 = np.where(coords1[:, dim] < cut_position)[0].tolist()
        indices1_part2 = np.where(coords1[:, dim] >= cut_position)[0].tolist()

        indices2_part1 = np.where(coords2[:, dim] < cut_position)[0].tolist()
        indices2_part2 = np.where(coords2[:, dim] >= cut_position)[0].tolist()

        # Ensure we have sites to swap
        if not indices1_part1 or not indices2_part2 or not indices2_part1 or not indices1_part2:
            return [struct1, struct2]

        # Composition tracking
        parent1_composition = dict(struct1.composition.element_composition)
        parent2_composition = dict(struct2.composition.element_composition)

        # Create child structures
        def create_child_structure(coords_part1, species_part1,
                                   coords_part2, species_part2,
                                   lattice, parent_composition):
            # Combine coordinates and species
            try:
                child_coords = np.vstack([coords_part1, coords_part2])
                child_species = list(species_part1) + list(species_part2)

                # Create structure
                child_struct = Structure(
                    lattice,
                    child_species,
                    child_coords,
                    coords_are_cartesian=False
                )

                # Composition adjustment if needed
                if not variable_composition:
                    child_struct = self._adjust_composition(
                        child_struct,
                        parent_composition
                    )

                return child_struct
            except Exception as e:
                # Fallback to original structure if creation fails
                import logging
                logger = logging.getLogger('opencsp.algorithms.genetic')
                logger.warning(f"Child structure creation failed: {str(e)}")
                return None

        # Attempt to create child structures
        child1_struct = create_child_structure(
            coords1[indices1_part1],
            [species1[i] for i in indices1_part1],
            coords2[indices2_part2],
            [species2[i] for i in indices2_part2],
            struct1.lattice,
            parent1_composition
        )

        child2_struct = create_child_structure(
            coords2[indices2_part1],
            [species2[i] for i in indices2_part1],
            coords1[indices1_part2],
            [species1[i] for i in indices1_part2],
            struct2.lattice,
            parent2_composition
        )

        # Fallback to parents if child creation fails
        if child1_struct is None or child2_struct is None:
            return [struct1, struct2]

        return [child1_struct, child2_struct]

    def _lattice_parameter_crossover(self,
                                     struct1: Structure,
                                     struct2: Structure,
                                     variable_composition: bool) -> List[Structure]:
        """
        Perform lattice parameter crossover between two crystal structures.

        Args:
            struct1: First parent structure
            struct2: Second parent structure
            variable_composition: Allow composition changes

        Returns:
            List of two child structures
        """
        # Random interpolation weight
        w = random.random()

        # Extract lattice parameters
        a1, b1, c1, alpha1, beta1, gamma1 = struct1.lattice.parameters
        a2, b2, c2, alpha2, beta2, gamma2 = struct2.lattice.parameters

        # Interpolate lattice parameters with some bounds
        a_mix1 = a1 * w + a2 * (1 - w)
        b_mix1 = b1 * w + b2 * (1 - w)
        c_mix1 = c1 * w + c2 * (1 - w)
        alpha_mix1 = alpha1 * w + alpha2 * (1 - w)
        beta_mix1 = beta1 * w + beta2 * (1 - w)
        gamma_mix1 = gamma1 * w + gamma2 * (1 - w)

        a_mix2 = a1 * (1 - w) + a2 * w
        b_mix2 = b1 * (1 - w) + b2 * w
        c_mix2 = c1 * (1 - w) + c2 * w
        alpha_mix2 = alpha1 * (1 - w) + alpha2 * w
        beta_mix2 = beta1 * (1 - w) + beta2 * w
        gamma_mix2 = gamma1 * (1 - w) + gamma2 * w

        # Create new lattices
        lattice_mix1 = Lattice.from_parameters(
            a_mix1, b_mix1, c_mix1,
            alpha_mix1, beta_mix1, gamma_mix1
        )

        lattice_mix2 = Lattice.from_parameters(
            a_mix2, b_mix2, c_mix2,
            alpha_mix2, beta_mix2, gamma_mix2
        )

        # Composition tracking
        parent1_composition = dict(struct1.composition.element_composition)
        parent2_composition = dict(struct2.composition.element_composition)

        # Create child structures with original atomic positions
        child1_struct = Structure(
            lattice_mix1,
            struct1.species,
            struct1.frac_coords
        )

        child2_struct = Structure(
            lattice_mix2,
            struct2.species,
            struct2.frac_coords
        )

        # Adjust composition if not variable
        if not variable_composition:
            child1_struct = self._adjust_composition(child1_struct, parent1_composition)
            child2_struct = self._adjust_composition(child2_struct, parent2_composition)

        return [child1_struct, child2_struct]

    def _adjust_composition(self,
                            structure: Structure,
                            target_composition: Dict[str, int]) -> Structure:
        """
        Adjust structure to match target composition.

        Args:
            structure: Structure to adjust
            target_composition: Desired composition

        Returns:
            Adjusted structure
        """
        # Prepare lists for new structure
        new_species = []
        new_coords = []

        # Total number of atoms needed
        total_atoms_needed = sum(target_composition.values())

        # Get all current sites
        all_sites = list(enumerate(structure.sites))

        # Shuffle sites to introduce randomness
        random.shuffle(all_sites)

        # Tracking composition
        current_composition = {}

        # Process sites
        for idx, site in all_sites:
            species = str(site.species_string)

            # Check if this element is still needed
            if species in target_composition:
                current_count = current_composition.get(species, 0)
                target_count = target_composition[species]

                if current_count < target_count:
                    new_species.append(species)
                    new_coords.append(site.frac_coords)
                    current_composition[species] = current_count + 1

        # If we don't have enough atoms, add random new atoms
        while len(new_species) < total_atoms_needed:
            # Choose an element to add
            for element, target_count in target_composition.items():
                current_count = current_composition.get(element, 0)
                if current_count < target_count:
                    # Add a random coordinate
                    new_species.append(element)
                    new_coords.append(np.random.random(3))
                    current_composition[element] = current_count + 1
                    break

        # Create new structure
        return Structure(
            structure.lattice,
            new_species,
            new_coords,
            coords_are_cartesian=False
        )

from typing import Dict, List, Tuple, Set, Optional, Union, Any
import numpy as np
from fractions import Fraction
from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from monty.json import MSONable

class CompositionManager(MSONable):
    """
    Manages compositions for crystal structure prediction with support for two modes:
    1. Fixed-ratio mode: Uses base composition with variable formula units
    2. Variable-composition mode: Uses composition ranges for each element
    """
    
    def __init__(self, 
                 composition: Optional[Dict[str, int]] = None,
                 formula_range: Tuple[int, int] = (1, 1),
                 composition_range: Optional[Dict[str, Tuple[int, int]]] = None,
                 reference_energies: Optional[Dict[str, float]] = None,
                 total_atoms_range: Optional[Tuple[int, int]] = None):
        """
        Initialize the composition manager.
        
        Args:
            composition: Base composition as {element: count}, required for fixed-ratio mode
            formula_range: Range of formula units (min, max) for fixed-ratio mode
            composition_range: Element ranges as {element: (min, max)} for variable-composition mode
            reference_energies: Reference energies for pure elements (eV/atom)
            total_atoms_range: Optional override for total atom count
        
        Note:
            - If composition_range is provided, operates in variable-composition mode
            - Otherwise, operates in fixed-ratio mode using composition and formula_range
        """
        # Determine operating mode
        self.variable_composition = composition_range is not None
        
        # Validate inputs
        if not self.variable_composition and composition is None:
            raise ValueError("Fixed-ratio mode requires 'composition' parameter")
        
        if self.variable_composition and not composition_range:
            raise ValueError("Variable-composition mode requires 'composition_range' parameter")
        
        # Store parameters
        self.base_composition = composition or {}
        self.formula_range = formula_range
        self.composition_range = composition_range or {}
        self.reference_energies = reference_energies or {}
        
        # Determine elements list
        if self.variable_composition:
            self.elements = list(self.composition_range.keys())
        else:
            self.elements = list(self.base_composition.keys())
        
        # Calculate base formula properties (for fixed-ratio mode)
        self.formula_gcd = 1
        self.base_atom_count = 0
        self.reduced_formula = {}
        
        if not self.variable_composition:
            self.base_atom_count = sum(self.base_composition.values())
            if self.base_atom_count > 0:
                self._calculate_base_formula()
        
        # Determine total atoms range
        if total_atoms_range:
            self.total_atoms_range = total_atoms_range
        elif not self.variable_composition:
            # For fixed-ratio mode
            min_atoms = self.base_atom_count * formula_range[0]
            max_atoms = self.base_atom_count * formula_range[1]
            self.total_atoms_range = (min_atoms, max_atoms)
        else:
            # For variable-composition mode
            min_atoms = sum(r[0] for r in self.composition_range.values())
            max_atoms = sum(r[1] for r in self.composition_range.values())
            
            # Apply formula_range to variable composition if specified
            if formula_range != (1, 1):
                min_atoms *= formula_range[0]
                max_atoms *= formula_range[1]
                
            self.total_atoms_range = (min_atoms, max_atoms)
            
        # Initialize phase diagram if reference energies are provided
        self._phase_diagram = None
        if self.reference_energies and len(self.reference_energies) >= len(self.elements):
            self._initialize_phase_diagram()
    
    def _calculate_base_formula(self):
        """Calculate the simplest formula by finding the GCD of atom counts."""
        from math import gcd
        from functools import reduce
        
        counts = list(self.base_composition.values())
        if not counts or all(c == 0 for c in counts):
            self.formula_gcd = 1
            return
            
        # Find GCD of all counts
        self.formula_gcd = reduce(gcd, [c for c in counts if c > 0])
        
        # Create reduced formula
        self.reduced_formula = {
            elem: count // self.formula_gcd 
            for elem, count in self.base_composition.items()
        }
    
    def _initialize_phase_diagram(self):
        """Initialize phase diagram with reference energies for pure elements."""
        entries = []
        for element, energy in self.reference_energies.items():
            entry = PDEntry(Composition(element), energy)
            entries.append(entry)
        self._phase_diagram = PhaseDiagram(entries)
    
    def generate_composition(self) -> Dict[str, int]:
        """
        Generate a composition based on the manager settings.
        
        Returns:
            Dictionary mapping element symbols to counts
        """
        if self.variable_composition:
            return self._generate_variable_composition()
        else:
            return self._generate_fixed_formula_composition()
    
    def _generate_fixed_formula_composition(self) -> Dict[str, int]:
        """Generate a composition with variable formula units but fixed ratios."""
        # Select random number of formula units within range
        min_units, max_units = self.formula_range
        formula_units = np.random.randint(min_units, max_units + 1)
        
        # Calculate composition
        composition = {
            elem: count * formula_units
            for elem, count in self.base_composition.items()
        }
        
        return composition
    
    def _generate_variable_composition(self) -> Dict[str, int]:
        """Generate a random composition within the specified ranges."""
        composition = {}
        for elem, (min_count, max_count) in self.composition_range.items():
            composition[elem] = np.random.randint(min_count, max_count + 1)
        
        # Apply formula scaling if needed
        if self.formula_range != (1, 1):
            min_units, max_units = self.formula_range
            if min_units != 1 or max_units != 1:
                formula_units = np.random.randint(min_units, max_units + 1)
                composition = {
                    elem: count * formula_units
                    for elem, count in composition.items()
                }
        
        return composition
    
    def mutate_composition(self, 
                          current_composition: Dict[str, int], 
                          mutation_strength: float = 0.2) -> Dict[str, int]:
        """
        Mutate a composition based on the manager settings.
        
        Args:
            current_composition: Current composition
            mutation_strength: Strength of mutation (0.0 to 1.0)
            
        Returns:
            Mutated composition
        """
        if self.variable_composition:
            return self._mutate_variable_composition(current_composition, mutation_strength)
        else:
            return self._mutate_formula_units(current_composition, mutation_strength)
    
    def _mutate_formula_units(self, 
                             current_composition: Dict[str, int],
                             mutation_strength: float) -> Dict[str, int]:
        """Mutate by changing the number of formula units."""
        if self.base_atom_count == 0:
            return current_composition.copy()
            
        # Determine current formula units
        total_current = sum(current_composition.values())
        current_units = max(1, total_current // self.base_atom_count)
        
        # Determine new formula units with some randomness
        min_units, max_units = self.formula_range
        range_size = max_units - min_units
        
        # Maximum change is proportional to mutation strength
        max_change = max(1, int(range_size * mutation_strength))
        
        # Random change within the allowed range
        change = np.random.randint(-max_change, max_change + 1)
        new_units = max(min_units, min(max_units, current_units + change))
        
        if new_units == current_units:
            # No change
            return current_composition.copy()
        
        # Calculate new composition based on new formula units
        new_composition = {
            elem: (count * new_units) // current_units
            for elem, count in current_composition.items()
        }
        
        return new_composition
    
    def _mutate_variable_composition(self, 
                                    current_composition: Dict[str, int],
                                    mutation_strength: float) -> Dict[str, int]:
        """Mutate by changing individual element counts within ranges."""
        # Copy current composition
        new_composition = current_composition.copy()
        
        # Get formula units if applicable
        formula_units = 1
        if self.formula_range != (1, 1):
            total_atoms = sum(current_composition.values())
            base_count = sum(r[0] for r in self.composition_range.values())
            if base_count > 0:
                formula_units = max(1, total_atoms // base_count)
        
        # Adjust ranges based on formula units
        effective_ranges = {}
        for elem, (min_count, max_count) in self.composition_range.items():
            effective_ranges[elem] = (
                min_count * formula_units,
                max_count * formula_units
            )
        
        # Number of mutations based on mutation strength
        num_elements = len(self.elements)
        num_mutations = max(1, int(num_elements * mutation_strength))
        
        # Randomly select elements to mutate
        elements_to_change = np.random.choice(
            self.elements, size=min(num_mutations, num_elements), replace=False)
        
        for elem in elements_to_change:
            if elem in effective_ranges:
                min_count, max_count = effective_ranges[elem]
                current = new_composition.get(elem, 0)
                
                # Calculate maximum change based on mutation strength
                range_size = max_count - min_count
                max_change = max(1, int(range_size * mutation_strength))
                
                # Random change within constraints
                change = np.random.randint(-max_change, max_change + 1)
                new_count = max(min_count, min(max_count, current + change))
                
                new_composition[elem] = new_count
        
        return new_composition
    
    def calculate_formation_energy(self, 
                                  composition: Dict[str, int], 
                                  total_energy: float) -> float:
        """
        Calculate formation energy per atom.
        
        Args:
            composition: Atomic composition
            total_energy: Total energy of the structure (eV)
            
        Returns:
            Formation energy per atom (eV/atom)
        """
        if not self.reference_energies:
            return total_energy / sum(composition.values())
        
        # Calculate reference energy
        ref_energy = 0.0
        for elem, count in composition.items():
            if elem in self.reference_energies:
                ref_energy += count * self.reference_energies[elem]
            else:
                return total_energy / sum(composition.values())
        
        total_atoms = sum(composition.values())
        return (total_energy - ref_energy) / total_atoms
    
    def get_e_above_hull(self, 
                        composition: Dict[str, int], 
                        total_energy: float) -> float:
        """
        Calculate energy above hull using pymatgen.
        
        Args:
            composition: Atomic composition
            total_energy: Total energy of the structure (eV)
            
        Returns:
            Energy above hull (eV/atom)
        """
        if self._phase_diagram is None:
            return self.calculate_formation_energy(composition, total_energy)
        
        try:
            # Create pymatgen composition
            comp_dict = {Element(elem): count for elem, count in composition.items()}
            pmg_comp = Composition(comp_dict)
            
            # Total energy (not per atom)
            total_atoms = sum(composition.values())
            
            # Create PDEntry
            entry = PDEntry(pmg_comp, total_energy)
            
            # Calculate energy above hull
            e_above_hull = self._phase_diagram.get_e_above_hull(entry)
            
            return e_above_hull
        except Exception as e:
            print(f"Error calculating energy above hull: {e}")
            return self.calculate_formation_energy(composition, total_energy)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the CompositionManager to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation
        """
        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "variable_composition": self.variable_composition,
            "formula_range": self.formula_range,
            "total_atoms_range": self.total_atoms_range,
        }
        
        # Include mode-specific properties
        if not self.variable_composition:
            d["base_composition"] = self.base_composition
            d["formula_gcd"] = self.formula_gcd
            d["base_atom_count"] = self.base_atom_count
            if hasattr(self, 'reduced_formula'):
                d["reduced_formula"] = self.reduced_formula
        else:
            d["composition_range"] = self.composition_range
            
        # Include reference energies if present
        if self.reference_energies:
            d["reference_energies"] = self.reference_energies
            
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CompositionManager':
        """
        Create a CompositionManager from a dictionary.
        
        Args:
            d: Dictionary containing manager data
            
        Returns:
            Reconstructed CompositionManager
        """
        # Determine mode
        variable_composition = d.get("variable_composition", False)
        
        # Extract common parameters
        formula_range = d.get("formula_range", (1, 1))
        reference_energies = d.get("reference_energies")
        total_atoms_range = d.get("total_atoms_range")
        
        # Extract mode-specific parameters
        composition = d.get("base_composition") if not variable_composition else None
        composition_range = d.get("composition_range") if variable_composition else None
        
        # Create instance
        manager = cls(
            composition=composition,
            formula_range=formula_range,
            composition_range=composition_range,
            reference_energies=reference_energies,
            total_atoms_range=total_atoms_range
        )
        
        # Set additional attributes if needed
        if "formula_gcd" in d:
            manager.formula_gcd = d["formula_gcd"]
            
        if "base_atom_count" in d:
            manager.base_atom_count = d["base_atom_count"]
            
        if "reduced_formula" in d:
            manager.reduced_formula = d["reduced_formula"]
            
        return manager

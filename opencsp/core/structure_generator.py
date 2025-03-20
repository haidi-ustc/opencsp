# opencsp/core/structure_generator.py
"""
This module provides structure generator classes for creating initial atomic structures
in various dimensionalities (clusters, surfaces, crystals) for structure prediction.

The base StructureGenerator class defines the interface for all generators, while specific
implementations like RandomStructureGenerator and SymmetryBasedStructureGenerator provide
different strategies for creating structures.
"""

import random
import traceback
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Type, TypeVar

import numpy as np
from monty.json import MSONable

T = TypeVar('T', bound='StructureGenerator')

class StructureGenerator(ABC, MSONable):
    """
    Abstract base class for structure generators.
    
    This class defines the interface that all structure generators must implement.
    Structure generators are responsible for creating initial structures for
    global optimization algorithms to operate on.
    
    Attributes:
        composition (Dict[str, int]): Chemical composition as element:count pairs
        constraints (List[Any]): Structure constraints to apply during generation
    """
    
    def __init__(self, composition: Dict[str, int], constraints: Optional[List[Any]] = None):
        """
        Initialize a structure generator.
        
        Args:
            composition: Chemical composition, formatted as {element_symbol: count}
            constraints: List of structure constraints to apply during generation
            
        Example:
            >>> from opencsp.core.structure_generator import StructureGenerator
            >>> # Abstract class, must be subclassed
            >>> composition = {'Si': 8, 'O': 16}
            >>> constraints = [MinimumDistanceConstraint({'Si-O': 1.6, 'O-O': 2.0})]
        """
        self.composition = composition
        self.constraints = constraints or []
        
    @abstractmethod
    def generate(self, n: int = 1) -> List[Any]:
        """
        Generate n structures.
        
        Args:
            n: Number of structures to generate
            
        Returns:
            List of generated structures
        """
        pass
        
    def is_valid(self, structure: Any) -> bool:
        """
        Check if a structure satisfies all constraints.
        
        Args:
            structure: Structure to validate
            
        Returns:
            True if the structure satisfies all constraints, False otherwise
            
        Example:
            >>> from opencsp.core.structure_generator import RandomStructureGenerator
            >>> from opencsp.core.constraints import MinimumDistanceConstraint
            >>> min_dist = {'Si-Si': 2.2, 'default': 1.5}
            >>> constraint = MinimumDistanceConstraint(min_dist)
            >>> generator = RandomStructureGenerator({'Si': 8}, volume_range=(100, 200), 
            ...                                     dimensionality=1, constraints=[constraint])
            >>> structure = generator.generate(1)[0]
            >>> is_valid = generator.is_valid(structure)
        """
        for constraint in self.constraints:
            if not constraint.is_satisfied(structure):
                return False
        return True
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the StructureGenerator to a JSON-serializable dictionary.
        
        This method is required by MSONable for serialization.
        
        Returns:
            Dict[str, Any]: JSON-serializable dictionary
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "composition": self.composition,
            "constraints": [constraint.as_dict() if hasattr(constraint, 'as_dict') 
                           else str(constraint) for constraint in self.constraints]
        }
    
    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Create a StructureGenerator from a dictionary.
        
        This method must be implemented by subclasses to properly deserialize
        specific generator types.
        
        Args:
            d: Dictionary containing generator data
            
        Returns:
            StructureGenerator: Reconstructed generator object
        """
        raise NotImplementedError("Subclasses must implement from_dict")


class RandomStructureGenerator(StructureGenerator):
    """
    Generate random structures with specified composition and dimensionality.
    
    This generator creates random structures by placing atoms at random positions
    within a defined volume while respecting constraints such as minimum interatomic
    distances.
    
    Attributes:
        composition (Dict[str, int]): Chemical composition as element:count pairs
        volume_range (Tuple[float, float]): Range of volumes to generate structures in
        dimensionality (int): Structure dimension (1, 2, or 3)
        min_distance (float): Minimum distance between atoms
        constraints (List[Any]): Structure constraints to apply during generation
    """
    
    def __init__(self, composition: Dict[str, int], volume_range: Tuple[float, float] = (100, 500), 
                 dimensionality: int = 3, **kwargs):
        """
        Initialize a random structure generator.
        
        Args:
            composition: Chemical composition, formatted as {element_symbol: count}
            volume_range: Range of volumes (min, max) for the generated structures
            dimensionality: Structure dimension (1: cluster, 2: surface, 3: crystal)
            **kwargs: Additional parameters:
                - min_distance: Minimum allowed distance between atoms (default: 1.5 Å)
                - constraints: List of structure constraints
                
        Example:
            >>> from opencsp.core.structure_generator import RandomStructureGenerator
            >>> # Generate a silicon cluster
            >>> generator = RandomStructureGenerator(
            ...     composition={'Si': 10},
            ...     volume_range=(100, 300),
            ...     dimensionality=1,
            ...     min_distance=2.0
            ... )
            >>> cluster = generator.generate(1)[0]
        """
        super().__init__(composition, kwargs.get('constraints'))
        self.volume_range = volume_range
        self.dimensionality = dimensionality
        self.min_distance = kwargs.get('min_distance', 1.5)  # Default minimum atomic distance
        
    def generate(self, n: int = 1) -> List[Any]:
        """
        Generate n random structures.
        
        Args:
            n: Number of structures to generate
            
        Returns:
            List of generated structures
            
        Example:
            >>> from opencsp.core.structure_generator import RandomStructureGenerator
            >>> generator = RandomStructureGenerator({'Si': 8}, volume_range=(100, 200), dimensionality=3)
            >>> structures = generator.generate(5)  # Generate 5 random silicon structures
            >>> len(structures)  # 5
        """
        structures = []
        attempts = 0
        max_attempts = n * 100  # Maximum attempts per structure
        failed_attempts = 0
        
        print(f"Attempting to generate {n} structures...")
        print(f"Parameters: composition={self.composition}, volume_range={self.volume_range}, "
              f"dimensionality={self.dimensionality}, min_distance={self.min_distance}")
        
        while len(structures) < n and attempts < max_attempts:
            attempts += 1
            
            # Generate structure based on dimensionality
            if self.dimensionality == 1:
                structure = self._generate_1d()
            elif self.dimensionality == 2:
                structure = self._generate_2d()
            else:  # 3D
                structure = self._generate_3d()
                
            # Check if structure generation was successful
            if structure is None:
                failed_attempts += 1
                if failed_attempts % 10 == 0:
                    print(f"Failed to generate structure {failed_attempts} times "
                          f"due to atomic distances being too close")
                continue
                
            # Check if structure satisfies constraints
            if self.is_valid(structure):
                structures.append(structure)
                print(f"Successfully generated structure {len(structures)}/{n}")
            else:
                failed_attempts += 1
                if failed_attempts % 10 == 0:
                    print(f"Failed to validate structure {failed_attempts} times")
        
        print(f"Generated {len(structures)}/{n} structures in {attempts} attempts")
        if len(structures) == 0:
            print("WARNING: Failed to generate any valid structures. Try adjusting parameters:")
            print("- Increase volume_range")
            print("- Decrease min_distance")
            print("- Simplify composition")
        
        return structures
    
    def _generate_1d(self) -> Any:
        """
        Generate a 1D structure (cluster).
        
        Returns:
            A cluster structure
        """
        try:
            from pymatgen.core import Lattice, Structure
            import numpy as np
            
            # Calculate total number of atoms
            num_atoms = sum(self.composition.values())
            
            # Create a sufficiently large box for the cluster
            volume = random.uniform(*self.volume_range)
            a = (volume)**(1/3)
            lattice = Lattice.from_parameters(a*3, a*3, a*3, 90, 90, 90)
            
            # Prepare atom list
            species = []
            for element, count in self.composition.items():
                species.extend([element] * count)
                
            # Randomly place atoms, concentrated near the center
            coords = []
            for _ in range(num_atoms):
                # Place atoms randomly near the box center
                x = 0.5 + (random.random() - 0.5) * 0.5
                y = 0.5 + (random.random() - 0.5) * 0.5
                z = 0.5 + (random.random() - 0.5) * 0.5
                coords.append([x, y, z])
                
            # Create structure
            structure = Structure(lattice, species, coords)
            
            return structure
            
        except ImportError:
            raise ImportError("pymatgen must be installed to generate structures")
        except Exception as e:
            print(f"Error generating 1D structure: {e}")
            return None
    
    def _generate_2d(self) -> Any:
        """
        Generate a 2D structure (surface).
        
        Returns:
            A surface structure
        """
        try:
            from pymatgen.core import Lattice, Structure
            import numpy as np
            
            # Calculate total number of atoms
            num_atoms = sum(self.composition.values())
            
            # Create a cell with extended c parameter for a 2D sheet
            volume = random.uniform(*self.volume_range)
            ab_area = np.sqrt(volume)
            a = b = np.sqrt(ab_area)
            c = 15.0  # Extended c direction with vacuum
            
            lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
            
            # Prepare atom list
            species = []
            for element, count in self.composition.items():
                species.extend([element] * count)
                
            # Place atoms in a layer
            coords = []
            max_attempts = 1000
            for _ in range(num_atoms):
                attempt = 0
                valid_position = False
                
                while not valid_position and attempt < max_attempts:
                    # Random position in xy plane, fixed z near middle
                    x = random.random()
                    y = random.random()
                    z = 0.5  # Middle of the cell
                    
                    # Check distances to existing atoms
                    pos = [x, y, z]
                    valid_position = True
                    
                    for existing_pos in coords:
                        # Calculate distance considering periodicity in xy plane
                        dx = min(abs(x - existing_pos[0]), 1 - abs(x - existing_pos[0])) * a
                        dy = min(abs(y - existing_pos[1]), 1 - abs(y - existing_pos[1])) * b
                        dz = abs(z - existing_pos[2]) * c
                        
                        distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        if distance < self.min_distance:
                            valid_position = False
                            break
                            
                    attempt += 1
                    
                if valid_position:
                    coords.append([x, y, z])
                else:
                    # If we can't place this atom, try increasing the area
                    return None
                
            # Create structure
            structure = Structure(lattice, species, coords)
            
            return structure
            
        except ImportError:
            raise ImportError("pymatgen must be installed to generate structures")
        except Exception as e:
            print(f"Error generating 2D structure: {e}")
            return None
    
    def _generate_3d(self) -> Any:
        """
        Generate a 3D structure (crystal).
        
        Returns:
            A crystal structure
        """
        try:
            from pymatgen.core import Structure, Lattice
            import numpy as np
            
            # Choose random volume
            volume = np.random.uniform(*self.volume_range)
            
            # For cluster structures, use cubic cell
            a = (volume) ** (1/3)
            lattice = Lattice.cubic(a)
            
            # Convert composition to list
            symbols = []
            for element, count in self.composition.items():
                symbols.extend([element] * count)
            
            num_atoms = len(symbols)
            if num_atoms == 0:
                print("Error: No atoms specified in composition")
                return None
                
            # Try different strategies for placing atoms
            # Strategy 1: Random placement near center
            coords = []
            for _ in range(num_atoms):
                x = 0.5 + (np.random.random() - 0.5) * 0.8
                y = 0.5 + (np.random.random() - 0.5) * 0.8
                z = 0.5 + (np.random.random() - 0.5) * 0.8
                coords.append([x, y, z])
                
            # Check interatomic distances
            too_close = True
            max_attempts = 200
            attempts = 0
            
            while too_close and attempts < max_attempts:
                too_close = False
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        # Calculate distance in cartesian coordinates
                        dist = np.linalg.norm(
                            lattice.get_cartesian_coords(coords[i]) - 
                            lattice.get_cartesian_coords(coords[j])
                        )
                        if dist < self.min_distance:
                            # If too close, regenerate position
                            too_close = True
                            x = 0.5 + (np.random.random() - 0.5) * 0.8
                            y = 0.5 + (np.random.random() - 0.5) * 0.8
                            z = 0.5 + (np.random.random() - 0.5) * 0.8
                            coords[i] = [x, y, z]
                            break
                    if too_close:
                        break
                attempts += 1
                
            if too_close:
                # Try Strategy 2: Spherical placement
                print("Trying spherical placement strategy...")
                coords = []
                radius = (3 * volume / (4 * np.pi)) ** (1/3) * 0.4
                
                for i in range(num_atoms):
                    # Random point on sphere
                    phi = np.random.random() * 2 * np.pi
                    costheta = np.random.random() * 2 - 1
                    theta = np.arccos(costheta)
                    
                    # Vary radius to avoid exact sphere
                    r = radius * (0.8 + 0.4 * np.random.random())
                    x = 0.5 + r * np.sin(theta) * np.cos(phi) / a
                    y = 0.5 + r * np.sin(theta) * np.sin(phi) / a
                    z = 0.5 + r * np.cos(theta) / a
                    coords.append([x, y, z])
                    
                # Check distances again
                too_close = False
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.linalg.norm(
                            lattice.get_cartesian_coords(coords[i]) - 
                            lattice.get_cartesian_coords(coords[j])
                        )
                        if dist < self.min_distance:
                            too_close = True
                            break
                    if too_close:
                        break
                
                if too_close:
                    print(f"Unable to place atoms with min_distance={self.min_distance}. "
                          f"Try reducing min_distance or increasing volume_range.")
                    return None
            
            # Create structure
            structure = Structure(lattice, symbols, coords)
            return structure
            
        except ImportError as e:
            print(f"Error importing required modules: {e}")
            raise ImportError("pymatgen must be installed to generate 3D structures")
        except Exception as e:
            print(f"Error generating 3D structure: {e}")
            return None
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the RandomStructureGenerator to a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: JSON-serializable dictionary
        """
        d = super().as_dict()
        d.update({
            "volume_range": self.volume_range,
            "dimensionality": self.dimensionality,
            "min_distance": self.min_distance
        })
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RandomStructureGenerator':
        """
        Create a RandomStructureGenerator from a dictionary.
        
        Args:
            d: Dictionary containing generator data
            
        Returns:
            RandomStructureGenerator: Reconstructed generator
        """
        # Process constraints if present
        constraints = []
        if "constraints" in d:
            from opencsp.core.constraints import Constraint
            for constraint_data in d["constraints"]:
                if isinstance(constraint_data, dict) and "@class" in constraint_data:
                    # Try to reconstruct the constraint object
                    try:
                        constraint_cls = getattr(
                            __import__(constraint_data["@module"], fromlist=[constraint_data["@class"]]),
                            constraint_data["@class"]
                        )
                        constraints.append(constraint_cls.from_dict(constraint_data))
                    except (ImportError, AttributeError):
                        print(f"Warning: Could not reconstruct constraint {constraint_data}")
        
        return cls(
            composition=d["composition"],
            volume_range=d["volume_range"],
            dimensionality=d["dimensionality"],
            min_distance=d["min_distance"],
            constraints=constraints
        )


class SymmetryBasedStructureGenerator(StructureGenerator):
    """
    Generate crystal structures with specified symmetry constraints.
    
    This generator creates structures with specific space groups or symmetry operations,
    useful for generating realistic crystal structures.
    
    Attributes:
        composition (Dict[str, int]): Chemical composition as element:count pairs
        spacegroup (Optional[int]): Space group number (1-230)
        lattice_vectors (Optional[List[List[float]]]): Lattice vectors for the structure
        dimensionality (int): Structure dimension (1, 2, or 3)
        constraints (List[Any]): Structure constraints to apply during generation
    """
    
    def __init__(self, composition: Dict[str, int], spacegroup: Optional[int] = None, 
                 lattice_vectors: Optional[List[List[float]]] = None, dimensionality: int = 3, **kwargs):
        """
        Initialize a symmetry-based structure generator.
        
        Args:
            composition: Chemical composition, formatted as {element_symbol: count}
            spacegroup: Space group number (1-230, None for random)
            lattice_vectors: Lattice vectors (3x3 matrix), None for automatic generation
            dimensionality: Structure dimension (1, 2, or 3)
            **kwargs: Additional parameters:
                - volume_factor: Volume scaling factor (default: 1.0)
                - max_attempts: Maximum generation attempts (default: 500)
                - symprec: Symmetry precision (default: 0.1)
                - constraints: List of structure constraints
                
        Example:
            >>> from opencsp.core.structure_generator import SymmetryBasedStructureGenerator
            >>> # Generate a silicon crystal with space group 227 (diamond cubic)
            >>> generator = SymmetryBasedStructureGenerator(
            ...     composition={'Si': 8},
            ...     spacegroup=227
            ... )
            >>> crystal = generator.generate(1)[0]
        """
        super().__init__(composition, kwargs.get('constraints'))
        self.spacegroup = spacegroup
        self.lattice_vectors = lattice_vectors
        self.dimensionality = dimensionality
        self.volume_factor = kwargs.get('volume_factor', 1.0)
        self.max_attempts = kwargs.get('max_attempts', 500)
        self.symprec = kwargs.get('symprec', 0.1)
        
    def generate(self, n: int = 1) -> List[Any]:
        """
        Generate n symmetry-based structures.
        
        Args:
            n: Number of structures to generate
            
        Returns:
            List of generated structures
            
        Example:
            >>> from opencsp.core.structure_generator import SymmetryBasedStructureGenerator
            >>> generator = SymmetryBasedStructureGenerator(
            ...     composition={'Li': 4, 'O': 4},
            ...     spacegroup=225  # Fm-3m
            ... )
            >>> structures = generator.generate(3)  # Generate 3 structures with Fm-3m symmetry
        """
        structures = []
        
        try:
            # Check if pyxtal is available
            from pyxtal import pyxtal
            from pyxtal.symmetry import get_symbol_and_number
            
            for i in range(n):
                print(f"Generating structure {i+1}/{n}...")
                structure = self._generate_with_pyxtal()
                
                if structure is not None and self.is_valid(structure):
                    structures.append(structure)
                    print(f"Successfully generated structure {len(structures)}/{n}")
                else:
                    print(f"Failed to generate valid structure {i+1}")
        
        except ImportError:
            print("pyxtal is not installed. Falling back to manual symmetry generation.")
            
            # Fallback to basic symmetry generation if pyxtal is not available
            for i in range(n):
                if self.dimensionality == 3:
                    structure = self._generate_3d_symmetry()
                else:
                    print("Only 3D symmetry structures are supported without pyxtal")
                    return []
                    
                if structure is not None and self.is_valid(structure):
                    structures.append(structure)
                    print(f"Successfully generated structure {len(structures)}/{n}")
                else:
                    print(f"Failed to generate valid structure {i+1}")
        
        return structures
    
    def _generate_with_pyxtal(self) -> Any:
        """
        Generate a crystal structure using pyxtal library.
        
        Returns:
            A pymatgen Structure object
        """
        try:
            from pyxtal import pyxtal
            from pyxtal.symmetry import get_symbol_and_number
            
            # Prepare elements and num_ions lists
            elements = []
            num_ions = []
            for element, count in self.composition.items():
                elements.append(element)
                num_ions.append(count)
            
            attempt = 0
            while attempt < self.max_attempts:
                # Choose space group if not specified
                space_group_number = self.spacegroup
                if space_group_number is None:
                    space_group_number = random.randint(1, 230)
                
                # Get space group symbol and number
                symbol, space_group_number = get_symbol_and_number(space_group_number)
                
                try:
                    # Create random crystal
                    rand_crystal = pyxtal()
                    rand_crystal.from_random(3, space_group_number, elements, num_ions, self.volume_factor)
                    
                    # Convert to pymatgen structure
                    structure = rand_crystal.to_pymatgen()
                    
                    print(f"Random crystal generated with space group {space_group_number} ({symbol}).")
                    return structure
                    
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    attempt += 1
            
            print("Max attempts reached. Unable to generate a valid crystal structure.")
            return None
            
        except ImportError:
            raise ImportError("pyxtal must be installed to use symmetry-based generation")
    
    def _generate_3d_symmetry(self) -> Any:
        """
        Generate a 3D structure with symmetry (fallback method).
        
        Returns:
            A pymatgen Structure object
        """
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.symmetry.groups import SpaceGroup
            from pymatgen.core import Structure, Lattice
            
            # Use a basic approach for simpler space groups
            if self.spacegroup is not None:
                # For specific space groups, try using pymatgen's functionality
                try:
                    sg = SpaceGroup.from_int_number(self.spacegroup)
                    
                    # Start with a primitive cell
                    volume = 200.0 * self.volume_factor  # Arbitrary starting volume
                    a = b = c = volume**(1/3)
                    
                    if sg.crystal_system == "cubic":
                        lattice = Lattice.cubic(a)
                    elif sg.crystal_system == "hexagonal":
                        lattice = Lattice.hexagonal(a, c)
                    elif sg.crystal_system == "tetragonal":
                        lattice = Lattice.tetragonal(a, c)
                    elif sg.crystal_system == "orthorhombic":
                        lattice = Lattice.orthorhombic(a, b, c)
                    else:
                        # Default to a reasonable lattice
                        lattice = Lattice.from_parameters(
                            a, a, a, 90, 90, 90
                        )
                    
                    # Place atoms at general positions
                    coords = []
                    species = []
                    
                    # Extract elements and counts
                    elements = list(self.composition.keys())
                    counts = list(self.composition.values())
                    
                    # Start with one atom per element type at general positions
                    for i, (element, count) in enumerate(zip(elements, counts)):
                        # Place first atom at a general position
                        x, y, z = random.random(), random.random(), random.random()
                        
                        coords.append([x, y, z])
                        species.append(element)
                    
                    # Create initial structure
                    structure = Structure(lattice, species, coords)
                    
                    # Apply symmetry operations to get full structure
                    sym_analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)
                    structure = sym_analyzer.get_symmetrized_structure()
                    
                    # If needed, add more atoms to match composition
                    # This is a simplified approach and may not work for all space groups
                    
                    return structure
                    
                except Exception as e:
                    print(f"Error in symmetry generation: {e}")
                    traceback.print_exc()
                    return None
            
            # If no specific space group or symmetry generation failed,
            # fall back to simpler random generation
            random_gen = RandomStructureGenerator(
                self.composition,
                volume_range=(200.0 * self.volume_factor, 300.0 * self.volume_factor),
                dimensionality=3,
                constraints=self.constraints
            )
            
            structures = random_gen.generate(1)
            if structures:
                return structures[0]
            return None
            
        except ImportError:
            raise ImportError("pymatgen must be installed to generate structures")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the SymmetryBasedStructureGenerator to a JSON-serializable dictionary.
        
        Returns:
            Dict[str, Any]: JSON-serializable dictionary
        """
        d = super().as_dict()
        d.update({
            "spacegroup": self.spacegroup,
            "lattice_vectors": self.lattice_vectors,
            "dimensionality": self.dimensionality,
            "volume_factor": self.volume_factor,
            "max_attempts": self.max_attempts,
            "symprec": self.symprec
        })
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SymmetryBasedStructureGenerator':
        """
        Create a SymmetryBasedStructureGenerator from a dictionary.
        
        Args:
            d: Dictionary containing generator data
            
        Returns:
            SymmetryBasedStructureGenerator: Reconstructed generator
        """
        # Process constraints if present
        constraints = []
        if "constraints" in d:
            from opencsp.core.constraints import Constraint
            for constraint_data in d["constraints"]:
                if isinstance(constraint_data, dict) and "@class" in constraint_data:
                    try:
                        constraint_cls = getattr(
                            __import__(constraint_data["@module"], fromlist=[constraint_data["@class"]]),
                            constraint_data["@class"]
                        )
                        constraints.append(constraint_cls.from_dict(constraint_data))
                    except (ImportError, AttributeError):
                        print(f"Warning: Could not reconstruct constraint {constraint_data}")
        
        return cls(
            composition=d["composition"],
            spacegroup=d.get("spacegroup"),
            lattice_vectors=d.get("lattice_vectors"),
            dimensionality=d.get("dimensionality", 3),
            volume_factor=d.get("volume_factor", 1.0),
            max_attempts=d.get("max_attempts", 500),
            symprec=d.get("symprec", 0.1),
            constraints=constraints
        )



if __name__ == '__main__':
   generator = RandomStructureGenerator(composition={'Si': 10},volume_range=(100, 300),dimensionality=1,min_distance=2.)
   generator = SymmetryBasedStructureGenerator(composition={'Li': 4, 'O': 4})

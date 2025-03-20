import uuid
import copy
from typing import Any, Dict, Optional, Union, List, TypeVar, Type
from pymatgen.core import Structure
import numpy as np
from monty.json import MSONable

from opencsp.utils.logging import get_logger
logger = get_logger(__name__)

T = TypeVar('T', bound='Individual')

class Individual(MSONable):
    """
    A class representing a structure unit in the optimization process.
    
    The Individual class encapsulates a structure (ASE Atoms or pymatgen Structure),
    along with its properties such as energy, fitness, and other attributes.
    It serves as the basic unit that genetic algorithms, particle swarm optimization,
    and other optimization methods operate on.
    
    Attributes:
        structure: The atomic structure (ASE Atoms or pymatgen Structure)
        energy: The calculated energy of the structure
        fitness: The fitness value in the optimization context
        dimensionality: The dimensionality of the structure (1=cluster, 2=surface, 3=crystal)
        properties: Dictionary containing additional properties
        id: Unique identifier for the individual
    """
    
    def __init__(self, structure: Any,
                 energy: Optional[float] = None, 
                 fitness: Optional[float] = None,
                 dimensionality: Optional[int] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 id: Optional[str] = None):
        """
        Initialize an Individual instance.
        
        Args:
            structure: The atomic structure (ASE Atoms or pymatgen Structure)
            energy: The calculated energy of the structure (default: None)
            fitness: The fitness value for optimization (default: None)
            dimensionality: The dimensionality of the structure (1=cluster, 2=surface, 3=crystal)
            properties: Dictionary of additional properties (default: empty dict)
            id: Unique identifier (default: auto-generated UUID)
        
        Examples:
            >>> from pymatgen.core import Structure
            >>> from opencsp.core.individual import Individual
            >>> # Create from pymatgen Structure
            >>> lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
            >>> species = ["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"]
            >>> coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            ...           [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]
            >>> structure = Structure(lattice, species, coords)
            >>> individual = Individual(structure, energy=-10.5, fitness=0.95, dimensionality=3)
        """
        self._structure = structure
        self._energy = energy
        self._fitness = fitness
        self._dimensionality = dimensionality
        self._properties = properties or {}
        self._id = id if id is not None else str(uuid.uuid4())[:8]  
    
    @property
    def id(self) -> str:
        """
        Get the individual's unique identifier.
        
        Returns:
            str: Unique identifier string
        """
        return self._id
    
    @id.setter
    def id(self, value: str) -> None:
        """
        Set the individual's unique identifier.
        
        Args:
            value: New identifier string
        """
        self._id = value
    
    @property
    def structure(self) -> Any:
        """
        Get the atomic structure.
        
        Returns:
            Any: The structure object (ASE Atoms or pymatgen Structure)
        """
        return self._structure
    
    @structure.setter
    def structure(self, value: Any) -> None:
        """
        Set the atomic structure.
        
        Args:
            value: New structure object
        """
        self._structure = value
    
    @property
    def energy(self) -> Optional[float]:
        """
        Get the energy value.
        
        Returns:
            Optional[float]: Energy value, or None if not evaluated
        """
        return self._energy
    
    @energy.setter
    def energy(self, value: Optional[float]) -> None:
        """
        Set the energy value.
        
        Args:
            value: New energy value
        """
        self._energy = value
    
    @property
    def fitness(self) -> Optional[float]:
        """
        Get the fitness value.
        
        The fitness value is used by optimization algorithms to determine
        how well the individual performs relative to others in the population.
        
        Returns:
            Optional[float]: Fitness value, or None if not evaluated
        """
        return self._fitness
    
    @fitness.setter
    def fitness(self, value: Optional[float]) -> None:
        """
        Set the fitness value.
        
        Args:
            value: New fitness value
        """
        self._fitness = value
    
    @property
    def dimensionality(self) -> Optional[int]:
        """
        Get the structure dimensionality.
        
        Returns:
            Optional[int]: Dimensionality value (1=cluster, 2=surface, 3=crystal)
        """
        return self._dimensionality
    
    @dimensionality.setter
    def dimensionality(self, value: Optional[int]) -> None:
        """
        Set the structure dimensionality.
        
        Args:
            value: New dimensionality value
        """
        self._dimensionality = value
    
    @property
    def properties(self) -> Dict[str, Any]:
        """
        Get the additional properties dictionary.
        
        Properties can include forces, stresses, or any other attributes
        calculated or assigned to the individual.
        
        Returns:
            Dict[str, Any]: Dictionary of additional properties
        """
        return self._properties
    
    def copy(self) -> 'Individual':
        """
        Create a deep copy of the current individual.
        
        Returns:
            Individual: A new Individual instance with copies of all attributes
            
        Example:
            >>> from pymatgen.core import Structure
            >>> from opencsp.core.individual import Individual
            >>> structure = Structure.from_spacegroup("Fm-3m", [[0, 0, 0]], ["Cu"])
            >>> ind1 = Individual(structure, energy=-5.0, dimensionality=3)
            >>> ind2 = ind1.copy()  # Create a deep copy
            >>> ind1.energy == ind2.energy  # True
            >>> ind1.structure == ind2.structure  # False (different objects)
        """
        # Create new individual and explicitly set all attributes
        new_individual = Individual(
            structure=copy.deepcopy(self.structure),
            energy=self.energy,
            fitness=self.fitness,
            dimensionality=self.dimensionality,
            properties=copy.deepcopy(self.properties),
            id=str(uuid.uuid4())[:8]  # Generate a new ID for the copy
        )
        
        return new_individual
    
    def __str__(self) -> str:
        """
        Return a string representation of the individual.
        
        Returns:
            str: String representation
        """
        return f"Individual(id={self.id}, energy={self.energy}, fitness={self.fitness}, dimensionality={self.dimensionality})"
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the individual.
        
        Returns:
            str: Detailed string representation
        """
        return self.__str__()
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the Individual to a JSON-serializable dictionary.
        
        This method is required by MSONable for serialization.
        
        Returns:
            Dict[str, Any]: JSON-serializable dictionary
            
        Example:
            >>> from pymatgen.core import Structure
            >>> from opencsp.core.individual import Individual
            >>> structure = Structure.from_spacegroup("Fm-3m", [[0, 0, 0]], ["Cu"])
            >>> ind = Individual(structure, energy=-5.0, fitness=0.95, dimensionality=3)
            >>> d = ind.as_dict()
            >>> d['energy']  # -5.0
        """
        from opencsp.utils.serialization import structure_to_dict
        
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "id": self.id,
            "structure": self.structure,
            "energy": self.energy,
            "fitness": self.fitness,
            "dimensionality": self.dimensionality,
            "properties": {k: v for k, v in self.properties.items() 
                          if self._is_serializable(v)}
        }
    
    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """
        Create an Individual from a dictionary.
        
        This method is required by MSONable for deserialization.
        
        Args:
            d: Dictionary containing individual data
            
        Returns:
            Individual: Reconstructed Individual object
            
        Example:
            >>> from opencsp.core.individual import Individual
            >>> d = {'@module': 'opencsp.core.individual', '@class': 'Individual',
            ...      'id': 'abcd1234', 'energy': -5.0, 'fitness': 0.95, 'dimensionality': 3,
            ...      'structure': {'type': 'pymatgen', 'data': {...}},
            ...      'properties': {'forces': [...]}}
            >>> ind = Individual.from_dict(d)
        """
        # from opencsp.utils.serialization import dict_to_structure
        
        # structure = dict_to_structure(d['structure'])
        
        # Create a new Individual
        individual = cls(
            structure=d.get('structure'),
            energy=d.get('energy'),
            fitness=d.get('fitness'),
            dimensionality=d.get('dimensionality'),
            properties=d.get('properties', {}),
            id=d.get('id')
        )
        
        return individual
    
    def _is_serializable(self, obj: Any) -> bool:
        """
        Check if an object is JSON serializable.
        
        Args:
            obj: Object to check
            
        Returns:
            bool: True if the object is serializable, False otherwise
        """
        if obj is None:
            return True
        if isinstance(obj, (bool, int, float, str, list, dict, tuple)):
            return True
        if isinstance(obj, np.ndarray):
            return True  # We'll convert this to list during serialization
        return False
    
    def get_composition(self) -> Dict[str, int]:
        """
        Get the chemical composition of the structure.
        
        Returns:
            Dict[str, int]: Dictionary mapping element symbols to counts
            
        Example:
            >>> from pymatgen.core import Structure
            >>> from opencsp.core.individual import Individual
            >>> structure = Structure.from_spacegroup("Fm-3m", [[0, 0, 0]], ["Cu"])
            >>> ind = Individual(structure)
            >>> comp = ind.get_composition()
            >>> comp  # {'Cu': 4}
        """
        if hasattr(self.structure, 'composition'):  # pymatgen Structure
            return dict(self.structure.composition.element_composition)
        elif hasattr(self.structure, 'get_chemical_symbols'):  # ASE Atoms
            symbols = self.structure.get_chemical_symbols()
            composition = {}
            for symbol in symbols:
                composition[symbol] = composition.get(symbol, 0) + 1
            return composition
        else:
            raise TypeError(f"Unknown structure type: {type(self.structure)}")
    
    def get_formula(self) -> str:
        """
        Get the chemical formula of the structure.
        
        Returns:
            str: Chemical formula string
            
        Example:
            >>> from pymatgen.core import Structure
            >>> from opencsp.core.individual import Individual
            >>> structure = Structure.from_spacegroup("Fm-3m", [[0, 0, 0]], ["Cu"])
            >>> ind = Individual(structure)
            >>> ind.get_formula()  # 'Cu4'
        """
        if hasattr(self.structure, 'composition'):  # pymatgen Structure
            return self.structure.composition.reduced_formula
        elif hasattr(self.structure, 'get_chemical_formula'):  # ASE Atoms
            return self.structure.get_chemical_formula()
        else:
            raise TypeError(f"Unknown structure type: {type(self.structure)}")

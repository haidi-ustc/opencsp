import os
import sys
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type, TypeVar
from contextlib import contextmanager
from monty.json import MSONable
from opencsp.utils.logging import get_logger
"""
Calculator module for energy and property calculations in structure prediction.

This module defines abstract and concrete calculator classes for energy and property
calculations in crystal structure prediction. It provides a common interface
for different types of calculators (ASE, machine learning models, etc.) and
handles conversion between different structure formats.
"""
logger = get_logger(__name__)

# Type variable for Calculator subclasses
C = TypeVar('C', bound='Calculator')

class Calculator(ABC, MSONable):
    """
    Abstract base class for energy and property calculation engines.
    
    This class defines the interface that all calculators must implement,
    providing methods for energy calculation, property extraction, and
    convergence checking.
    
    Attributes:
        parameters (Dict[str, Any]): Dictionary of calculator parameters
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the calculator.
        
        Args:
            **kwargs: Calculator-specific parameters
            
        Example:
            >>> calculator = DerivedCalculator(parameter1=value1, parameter2=value2)
        """
        self.parameters = kwargs
        logger.debug(f"Initialized {self.__class__.__name__} with parameters: {kwargs}")
        
    @abstractmethod
    def calculate(self, structure: Any) -> float:
        """
        Calculate the energy of a given structure.
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            Energy value
            
        Example:
            >>> energy = calculator.calculate(structure)
            >>> print(f"Energy: {energy} eV")
        """
        pass
        
    @abstractmethod
    def get_properties(self, structure: Any) -> Dict[str, Any]:
        """
        Get additional properties of a structure (forces, stress, etc.).
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            Dictionary of properties
            
        Example:
            >>> properties = calculator.get_properties(structure)
            >>> forces = properties.get('forces')
        """
        pass
        
    @abstractmethod
    def is_converged(self, structure: Any) -> bool:
        """
        Check if the calculation has converged.
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            True if converged, False otherwise
            
        Example:
            >>> if calculator.is_converged(structure):
            ...     print("Calculation converged")
        """
        pass
   
    def to_atoms(self, structure: Any) -> Any:
        if hasattr(structure, 'get_positions'):
            return structure

        if hasattr(structure, 'sites'):
            from pymatgen.io.ase import AseAtomsAdaptor
            return AseAtomsAdaptor.get_atoms(structure)

        raise ValueError("Unknown structure format")

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the calculator to a JSON-serializable dictionary.
        
        This method is required by MSONable for serialization.
        
        Returns:
            Dictionary representation of the calculator
            
        Example:
            >>> calculator_dict = calculator.as_dict()
            >>> import json
            >>> json_str = json.dumps(calculator_dict)
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls: Type[C], d: Dict[str, Any]) -> C:
        """
        Create a calculator from a dictionary.
        
        This method must be implemented by subclasses to properly deserialize
        calculator-specific attributes.
        
        Args:
            d: Dictionary containing calculator data
            
        Returns:
            Calculator object
            
        Example:
            >>> calculator = DerivedCalculator.from_dict(calculator_dict)
        """
        # This is a base implementation that should be overridden by subclasses
        parameters = d.get("parameters", {})
        return cls(**parameters)

    @contextmanager
    def suppress_output(self):
        """Context manager to suppress standard output and error"""
        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Create string buffers to capture output
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        try:
            # Redirect stdout and stderr to the buffers
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer            
            yield
        finally:
            # Restore the original stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr



class ASECalculatorWrapper(Calculator):
    """
    Wrapper for ASE calculators with support for structure relaxation.
    
    This class wraps an ASE calculator to provide a consistent interface within
    the openCSP framework. It handles converting between structure formats,
    manages the ASE calculation workflow, and supports structure relaxation
    with constraints.
    
    Attributes:
        ase_calculator: ASE calculator instance
        parameters (Dict[str, Any]): Dictionary of calculator parameters
        relax (bool): Whether to perform structure relaxation
        relax_kwargs (Dict[str, Any]): Parameters for structure relaxation
    """
    
    def __init__(self, ase_calculator: Optional[Any] = None, relax: bool = False, 
                 relax_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize an ASE calculator wrapper.
        
        Args:
            ase_calculator: ASE calculator instance
            relax: Whether to perform structure relaxation (default: False)
            relax_kwargs: Parameters for structure relaxation (optimizer, fmax, steps, etc.)
            **kwargs: Additional parameters
            
        Example:
            >>> from ase.calculators.emt import EMT
            >>> # Basic calculator without relaxation
            >>> calculator = ASECalculatorWrapper(EMT())
            >>> 
            >>> # Calculator with structure relaxation
            >>> calculator = ASECalculatorWrapper(
            ...     EMT(), 
            ...     relax=True, 
            ...     relax_kwargs={'fmax': 0.05, 'steps': 100}
            ... )
        """
        super().__init__(**kwargs)
        self.ase_calculator = ase_calculator
        self.relax = relax
        self.relax_kwargs = relax_kwargs or {}
        logger.info(f"Initialized ASECalculatorWrapper with {ase_calculator.__class__.__name__ if ase_calculator else 'None'} "
                    f"(relax={relax})")
        
    def calculate(self, structure: Any) -> float:
        """
        Calculate energy using the ASE calculator, optionally with relaxation.
        
        This method calculates the energy per atom of the given structure
        using the wrapped ASE calculator. If relaxation is enabled, it will
        first optimize the structure within the given constraints.
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            Energy per atom in eV, or inf if calculation fails
            
        Example:
            >>> energy = calculator.calculate(structure)
        """
        # Convert to ASE Atoms format if needed
        try:
            atoms = self.to_atoms(structure)
            atoms.calc = self.ase_calculator
            
            # Perform relaxation if enabled
            if self.relax:
                with self.suppress_output():
                     self._relax_structure(atoms)
                
            energy = atoms.get_potential_energy()
            energy_per_atom = energy / len(atoms)
            logger.debug(f"Calculated energy: {energy} eV, energy per atom: {energy_per_atom} eV")
            
            # Update the input structure with the relaxed positions if it's an ASE Atoms object
            if self.relax and hasattr(structure, 'set_positions') and id(structure) != id(atoms):
                structure.set_positions(atoms.get_positions())
                logger.debug("Updated input structure with relaxed positions")
                
            return energy_per_atom
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return float('inf')
        
    def get_properties(self, structure: Any) -> Dict[str, Any]:
        """
        Get additional properties using the ASE calculator.
        
        This method extracts properties like forces and stress from the
        structure using the wrapped ASE calculator.
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            Dictionary of properties
            
        Example:
            >>> properties = calculator.get_properties(structure)
            >>> forces = properties.get('forces')
        """
        properties = {}
        try:
            atoms = self.to_atoms(structure)
            atoms.calc = self.ase_calculator
            
            # Try to get forces
            try:
                properties['forces'] = atoms.get_forces()
                logger.debug(f"Retrieved forces for {len(atoms)} atoms")
            except Exception as e:
                logger.warning(f"Could not get forces: {e}")
            
            # Try to get stress
            try:
                properties['stress'] = atoms.get_stress()
                logger.debug(f"Retrieved stress tensor")
            except Exception as e:
                logger.warning(f"Could not get stress: {e}")
                
            # If relaxation was performed, add relaxation info
            if self.relax:
                properties['relaxed'] = True
                
        except Exception as e:
            logger.error(f"Error getting properties: {e}")
            
        return properties
        
    def is_converged(self, structure: Any) -> bool:
        """
        Check if the ASE calculation has converged.
        
        For relaxation, this checks if the optimization has converged to
        the specified force tolerance. For regular calculations, this
        depends on the specific calculator used.
        
        Args:
            structure: Structure object (pymatgen Structure or ASE Atoms)
            
        Returns:
            True if converged, False otherwise
            
        Example:
            >>> converged = calculator.is_converged(structure)
        """
        # For relaxation, we can check if forces are below the threshold
        if self.relax:
            try:
                atoms = self.to_atoms(structure)
                atoms.calc = self.ase_calculator
                forces = atoms.get_forces()
                
                fmax = max(sum(force**2 for force in forces.flatten())**0.5)
                target_fmax = self.relax_kwargs.get('fmax', 0.05)
                
                converged = fmax < target_fmax
                logger.debug(f"Relaxation convergence check: fmax={fmax}, target={target_fmax}, converged={converged}")
                return converged
            except Exception as e:
                logger.warning(f"Could not check convergence: {e}")
                return False
                
        # Default implementation for regular calculations
        logger.debug("Convergence check for regular calculation (default: True)")
        return True
    
    def _relax_structure(self, atoms: Any) -> None:
        """
        Relax atomic structure using ASE optimizers.
        
        This method performs geometry optimization on the given structure
        using the parameters specified in relax_kwargs.
        
        Args:
            atoms: ASE Atoms object to relax
            
        Example:
            >>> atoms = calculator.to_atoms(structure)
            >>> calculator._relax_structure(atoms)
        """
        try:
            # Import ASE optimizers
            from ase.optimize import BFGS, LBFGS, GPMin, MDMin
            
            # Get relaxation parameters
            optimizer_name = self.relax_kwargs.get('optimizer', 'BFGS')
            fmax = self.relax_kwargs.get('fmax', 0.05)
            steps = self.relax_kwargs.get('steps', 80)
            
            # Select optimizer
            optimizer_map = {
                'BFGS': BFGS,
                'LBFGS': LBFGS,
                'GPMin': GPMin,
                'MDMin': MDMin
            }
            
            if optimizer_name not in optimizer_map:
                logger.warning(f"Unknown optimizer: {optimizer_name}, defaulting to BFGS")
                optimizer_name = 'BFGS'
                
            optimizer_cls = optimizer_map[optimizer_name]
            
            # Setup optimizer
            optimizer = optimizer_cls(atoms)
            
            # Apply constraints if provided
            constraints = self.relax_kwargs.get('constraints', None)
            if constraints:
                atoms.set_constraint(constraints)
                logger.debug(f"Applied constraints to structure: {constraints}")
                
            # Run optimization
            logger.debug(f"Starting structure relaxation with {optimizer_name} (fmax={fmax}, steps={steps})")
            optimizer.run(fmax=fmax, steps=steps)
            
            logger.debug(f"Structure relaxation completed: {optimizer.nsteps} steps, converged={optimizer.converged()}")
            
        except ImportError as e:
            logger.error(f"Failed to import ASE optimization modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during structure relaxation: {e}")
            raise
        
    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the ASECalculatorWrapper to a JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of the calculator
            
        Example:
            >>> calculator_dict = calculator.as_dict()
        """
        d = super().as_dict()
        
        # Add ASE calculator info if available
        if self.ase_calculator is not None:
            calculator_type = self.ase_calculator.__class__.__name__
            d["ase_calculator_type"] = calculator_type
            
            # Try to get calculator parameters
            if hasattr(self.ase_calculator, 'parameters'):
                d["ase_calculator_parameters"] = self.ase_calculator.parameters
        
        # Add relaxation parameters
        d["relax"] = self.relax
        d["relax_kwargs"] = self.relax_kwargs
        
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ASECalculatorWrapper':
        """
        Create an ASECalculatorWrapper from a dictionary.
        
        Args:
            d: Dictionary containing ASECalculatorWrapper data
            
        Returns:
            ASECalculatorWrapper object
            
        Example:
            >>> # This creates a wrapper without a calculator
            >>> partial_calculator = ASECalculatorWrapper.from_dict(calculator_dict)
        """
        parameters = d.get("parameters", {})
        relax = d.get("relax", False)
        relax_kwargs = d.get("relax_kwargs", {})
        
        # Cannot fully reconstruct ASE calculator from serialized data
        logger.warning("Reconstructing ASECalculatorWrapper without ASE calculator")
        return cls(ase_calculator=None, relax=relax, relax_kwargs=relax_kwargs, **parameters)

    def set_relax_parameters(self, relax: bool = True, **relax_kwargs) -> None:
        """
        Set parameters for structure relaxation.
        
        Args:
            relax: Whether to enable relaxation
            **relax_kwargs: Parameters for relaxation
                - optimizer: Optimizer to use ('BFGS', 'LBFGS', 'GPMin', 'MDMin')
                - fmax: Maximum force criterion for convergence
                - steps: Maximum number of optimization steps
                - constraints: ASE constraints to apply during relaxation
                
        Example:
            >>> calculator.set_relax_parameters(
            ...     relax=True,
            ...     optimizer='BFGS',
            ...     fmax=0.01,
            ...     steps=200
            ... )
            >>> 
            >>> # With constraints
            >>> from ase.constraints import FixAtoms
            >>> constraint = FixAtoms(indices=[0, 1, 2])  # Fix first three atoms
            >>> calculator.set_relax_parameters(
            ...     relax=True,
            ...     constraints=constraint
            ... )
        """
        self.relax = relax
        if relax_kwargs:
            self.relax_kwargs.update(relax_kwargs)
        logger.info(f"Updated relaxation parameters: relax={relax}, {self.relax_kwargs}")



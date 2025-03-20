#!/usr/bin/env python
"""
Script to export the top N globally optimal structures from population files.

This script finds all popu-*.json files, extracts all individuals, sorts them
by energy, and exports the top N unique structures. It applies symmetry analysis
and can filter structures based on specified space group ranges.
"""

import os
import glob
import re
import numpy as np
from monty.serialization import loadfn, dumpfn
from collections import defaultdict
import argparse
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reconstruct_structure(structure_data):
    """
    Reconstruct a pymatgen Structure or ASE Atoms object from dictionary data.
    
    Args:
        structure_data: Dictionary containing structure data
        
    Returns:
        Reconstructed structure object or None if reconstruction fails
    """
    try:
        # Check if it's a serialized pymatgen Structure
        if isinstance(structure_data, dict) and '@module' in structure_data:
            if 'pymatgen' in structure_data.get('@module', ''):
                from pymatgen.core.structure import Structure
                return Structure.from_dict(structure_data)
        
        # Check if it's just lattice and sites data
        if isinstance(structure_data, dict) and 'lattice' in structure_data and 'sites' in structure_data:
            from pymatgen.core.structure import Structure
            return Structure.from_dict(structure_data)
            
        # Check if it's ASE Atoms format
        if isinstance(structure_data, dict) and 'positions' in structure_data and 'numbers' in structure_data:
            from ase import Atoms
            return Atoms(
                positions=structure_data['positions'],
                numbers=structure_data['numbers'],
                cell=structure_data.get('cell', None),
                pbc=structure_data.get('pbc', None)
            )
            
        logger.warning(f"Unknown structure format: {type(structure_data)}")
        return None
        
    except Exception as e:
        logger.error(f"Error reconstructing structure: {e}")
        return None

def analyze_symmetry(structure, symprec=0.1):
    """
    Analyze structure symmetry and return space group information.
    
    Args:
        structure: Structure object (pymatgen Structure or ASE Atoms) or dict
        symprec: Symmetry precision for spglib
        
    Returns:
        Tuple of (spacegroup number, spacegroup symbol, structure with symmetry applied)
    """
    try:
        # If structure is a dictionary, try to reconstruct it
        if isinstance(structure, dict):
            structure = reconstruct_structure(structure)
            if structure is None:
                return (None, None, structure)
        
        # Try to use pymatgen's symmetry analyzer
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        
        # Convert to pymatgen Structure if needed
        if hasattr(structure, 'get_positions'):  # ASE Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            structure = AseAtomsAdaptor.get_structure(structure)
        
        # Analyze symmetry
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        spg_number = sga.get_space_group_number()
        spg_symbol = sga.get_space_group_symbol()
        
        # Get symmetrized structure
        sym_structure = sga.get_symmetrized_structure()
        
        return (spg_number, spg_symbol, sym_structure)
        
    except ImportError:
        logger.warning("Pymatgen not available for symmetry analysis, using spglib directly")
        try:
            # Try to use spglib directly
            import spglib
            
            # Convert to ASE Atoms if needed
            if hasattr(structure, 'sites'):  # pymatgen Structure
                from pymatgen.io.ase import AseAtomsAdaptor
                structure = AseAtomsAdaptor.get_atoms(structure)
            
            # Get spglib cell
            cell = (structure.get_cell(), structure.get_scaled_positions(), 
                   structure.get_atomic_numbers())
            
            # Get space group data
            spg_data = spglib.get_symmetry_dataset(cell, symprec=symprec)
            spg_number = spg_data['number']
            spg_symbol = spg_data['international']
            
            # No symmetrized structure available with spglib alone
            return (spg_number, spg_symbol, structure)
            
        except ImportError:
            logger.error("Neither pymatgen nor spglib available for symmetry analysis")
            return (None, None, structure)
    except Exception as e:
        logger.error(f"Error analyzing symmetry: {e}")
        return (None, None, structure)

def calculate_structure_similarity(struct1, struct2, tolerance=0.2):
    """
    Calculate similarity between two structures.
    
    Args:
        struct1, struct2: Structure objects or dict
        tolerance: Tolerance for structure matching
        
    Returns:
        True if structures are similar, False otherwise
    """
    try:
        # If structures are dictionaries, try to reconstruct them
        if isinstance(struct1, dict):
            struct1 = reconstruct_structure(struct1)
            if struct1 is None:
                return False
                
        if isinstance(struct2, dict):
            struct2 = reconstruct_structure(struct2)
            if struct2 is None:
                return False
        
        # Try to use pymatgen's structure matcher
        from pymatgen.analysis.structure_matcher import StructureMatcher
        
        # Convert to pymatgen Structure if needed
        if hasattr(struct1, 'get_positions'):  # ASE Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            struct1 = AseAtomsAdaptor.get_structure(struct1)
        
        if hasattr(struct2, 'get_positions'):  # ASE Atoms
            from pymatgen.io.ase import AseAtomsAdaptor
            struct2 = AseAtomsAdaptor.get_structure(struct2)
        
        # Create structure matcher
        matcher = StructureMatcher(ltol=tolerance, stol=tolerance, angle_tol=tolerance)
        
        # Compare structures
        return matcher.fit(struct1, struct2)
        
    except ImportError:
        logger.warning("Pymatgen not available for structure comparison")
        
        # Fall back to simple comparison (energy, number of atoms)
        try:
            # Check number of atoms
            if hasattr(struct1, 'sites') and hasattr(struct2, 'sites'):
                if len(struct1.sites) != len(struct2.sites):
                    return False
            elif hasattr(struct1, 'get_positions') and hasattr(struct2, 'get_positions'):
                if len(struct1) != len(struct2):
                    return False
            else:
                return False
                
            # Without proper structure matching, we can't determine similarity reliably
            # Return False to be safe (consider all structures unique)
            return False
            
        except Exception as e:
            logger.error(f"Error comparing structures: {e}")
            return False
    except Exception as e:
        logger.error(f"Error comparing structures: {e}")
        return False

def save_structure(structure, filename, fmt='cif'):
    """
    Save structure to file.
    
    Args:
        structure: Structure object or dict
        filename: Output filename
        fmt: File format (cif, poscar, xyz, etc.)
    """
    try:
        # If structure is a dictionary, try to reconstruct it
        if isinstance(structure, dict):
            structure = reconstruct_structure(structure)
            if structure is None:
                return False
        
        # Try pymatgen's Structure write methods
        if hasattr(structure, 'to') and callable(structure.to):
            structure.to(filename=filename, fmt=fmt)
            return True
            
        # Try ASE's write method
        elif hasattr(structure, 'get_positions'):
            from ase.io import write
            write(filename, structure)
            return True
            
        else:
            logger.error(f"Unknown structure type: {type(structure)}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving structure to {filename}: {e}")
        return False

def extract_individuals_from_population(pop):
    """
    Extract individual objects from a population, handling different formats.
    
    Args:
        pop: Population object or dictionary
        
    Returns:
        List of individuals
    """
    individuals = []
    
    try:
        # If it's already a list of individuals
        if isinstance(pop, list):
            return pop
            
        # If it has an 'individuals' attribute (normal Population object)
        if hasattr(pop, 'individuals'):
            return pop.individuals
            
        # If it's a dictionary with 'individuals' key
        if isinstance(pop, dict) and 'individuals' in pop:
            return pop['individuals']
            
        logger.warning(f"Unknown population format: {type(pop)}")
        return []
        
    except Exception as e:
        logger.error(f"Error extracting individuals: {e}")
        return []

def get_individual_properties(ind):
    """
    Extract key properties from an individual, handling different formats.
    
    Args:
        ind: Individual object or dictionary
        
    Returns:
        Tuple of (id, energy, fitness, structure, properties)
    """
    try:
        # Initialize with default values
        ind_id = None
        energy = None
        fitness = None
        structure = None
        properties = {}
        
        # Extract properties based on object type
        if hasattr(ind, 'id') and hasattr(ind, 'energy') and hasattr(ind, 'structure'):
            # Standard Individual object
            ind_id = ind.id
            energy = ind.energy
            fitness = ind.fitness
            structure = ind.structure
            properties = getattr(ind, 'properties', {})
            
        elif isinstance(ind, dict):
            # Dictionary representation
            ind_id = ind.get('id')
            energy = ind.get('energy')
            fitness = ind.get('fitness')
            structure = ind.get('structure')
            properties = ind.get('properties', {})
            
        return (ind_id, energy, fitness, structure, properties)
        
    except Exception as e:
        logger.error(f"Error extracting individual properties: {e}")
        return (None, None, None, None, {})

def export_best_structures(
    directory='.', 
    output_dir='best_structures',
    num_structures=10,
    min_spg=1,
    max_spg=230,
    symprec=0.1,
    similarity_tol=0.2
):
    """
    Export the top N globally optimal structures from population files.
    
    Args:
        directory: Directory containing popu-*.json files
        output_dir: Directory to save exported structures
        num_structures: Number of best structures to export
        min_spg: Minimum space group number to include
        max_spg: Maximum space group number to include
        symprec: Symmetry precision for spglib
        similarity_tol: Tolerance for structure similarity comparison
    """
    # Find all population files
    pattern = os.path.join(directory, 'popu-*.json')
    popu_files = glob.glob(pattern)
    
    if not popu_files:
        logger.error(f"No population files (popu-*.json) found in {directory}")
        return
    
    # Sort files by generation number
    popu_files.sort(key=lambda x: int(re.search(r'popu-(\d+)\.json', x).group(1)))
    
    logger.info(f"Found {len(popu_files)} population files")
    
    # Collect all individuals across all generations
    all_individuals = []
    
    # Process each file
    for file_path in popu_files:
        # Extract generation number from filename
        match = re.search(r'popu-(\d+)\.json', file_path)
        if not match:
            continue
            
        gen_num = int(match.group(1))
        
        # Load population data
        try:
            # Load the file
            pop = loadfn(file_path)
            
            # Extract individuals
            individuals = extract_individuals_from_population(pop)
            
            # Process each individual
            for ind in individuals:
                ind_id, energy, fitness, structure, properties = get_individual_properties(ind)
                
                if energy is not None:
                    # Create a standardized individual record
                    individual_record = {
                        'id': ind_id,
                        'energy': energy,
                        'fitness': fitness,
                        'structure': structure,
                        'properties': dict(properties)  # Force to regular dict
                    }
                    
                    # Add generation info
                    individual_record['properties']['generation'] = gen_num
                    
                    # Add to collection
                    all_individuals.append(individual_record)
            
            logger.info(f"Generation {gen_num}: Added {len(individuals)} individuals")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    if not all_individuals:
        logger.error("No valid individuals found in population files")
        return
    
    # Sort all individuals by energy (lowest first)
    all_individuals.sort(key=lambda x: x['energy'])
    
    logger.info(f"Total individuals collected: {len(all_individuals)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare for unique structures
    unique_structures = []
    structure_info = []
    
    # Process each individual, starting from the best energy
    for i, ind in enumerate(all_individuals):
        if len(unique_structures) >= num_structures:
            break
            
        structure = ind['structure']
        
        # Skip if structure is None
        if structure is None:
            continue
            
        # Analyze symmetry
        spg_number, spg_symbol, sym_structure = analyze_symmetry(structure, symprec=symprec)
        
        # Skip if symmetry analysis failed
        if spg_number is None:
            continue
            
        # Skip if space group is not in the specified range
        if spg_number < min_spg or spg_number > max_spg:
            continue
            
        # Check if this structure is similar to any already found
        is_unique = True
        for existing_struct in unique_structures:
            if calculate_structure_similarity(structure, existing_struct, tolerance=similarity_tol):
                is_unique = False
                break
                
        if is_unique:
            # Add to unique structures
            unique_structures.append(structure)
            
            # Get generation info
            gen = ind['properties'].get('generation', 'unknown')
            
            # Create structure info
            info = {
                'rank': len(unique_structures),
                'id': ind['id'],
                'energy': ind['energy'],
                'fitness': ind['fitness'],
                'generation': gen,
                'space_group': spg_number,
                'space_group_symbol': spg_symbol,
                'filename': f"structure_{len(unique_structures):03d}_id{ind['id']}_E{ind['energy']:.6f}_G{gen}_SPG{spg_number}.cif"
            }
            
            structure_info.append(info)
            
            # Save structure
            filename = os.path.join(output_dir, info['filename'])
            save_structure(sym_structure, filename)
            
            logger.info(f"Saved structure {len(unique_structures)}: ID={ind['id']}, Energy={ind['energy']:.6f}, "
                        f"Gen={gen}, Space Group={spg_number} ({spg_symbol})")
    
    # Save summary information
    summary_file = os.path.join(output_dir, 'structure_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(structure_info, f, indent=2)
    
    logger.info(f"Exported {len(unique_structures)} unique best structures to {output_dir}")
    logger.info(f"Summary saved to {summary_file}")
    
    return structure_info

def main():
    parser = argparse.ArgumentParser(description='Export top N globally optimal structures')
    parser.add_argument('--dir', '-d', type=str, default='.',
                        help='Directory containing popu-*.json files (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='best_structures',
                        help='Output directory for best structures (default: best_structures)')
    parser.add_argument('--num', '-n', type=int, default=10,
                        help='Number of best structures to export (default: 10)')
    parser.add_argument('--min-spg', type=int, default=1,
                        help='Minimum space group number to include (default: 1)')
    parser.add_argument('--max-spg', type=int, default=230,
                        help='Maximum space group number to include (default: 230)')
    parser.add_argument('--symprec', type=float, default=0.1,
                        help='Symmetry precision for spglib (default: 0.1)')
    parser.add_argument('--similarity', type=float, default=0.2,
                        help='Tolerance for structure similarity comparison (default: 0.2)')
    
    args = parser.parse_args()
    
    export_best_structures(
        directory=args.dir,
        output_dir=args.output,
        num_structures=args.num,
        min_spg=args.min_spg,
        max_spg=args.max_spg,
        symprec=args.symprec,
        similarity_tol=args.similarity
    )

#if __name__ == '__main__':
#    main()

#!/usr/bin/env python
"""
Simple script to visualize energy data from population files.

This script finds all popu-*.json files, extracts energy data,
and creates a scatter plot with the lowest energy points connected.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from monty.serialization import loadfn

def plot_population_energies(directory='.', output_file='energy_plot.png'):
    """
    Plot energies from all population files in a directory.
    
    Args:
        directory: Directory containing popu-*.json files
        output_file: Output file path for the plot
    """
    # Find all population files
    pattern = os.path.join(directory, 'popu-*.json')
    popu_files = glob.glob(pattern)
    
    if not popu_files:
        print(f"No population files (popu-*.json) found in {directory}")
        return
    
    # Sort files by generation number
    popu_files.sort(key=lambda x: int(re.search(r'popu-(\d+)\.json', x).group(1)))
    
    print(f"Found {len(popu_files)} population files")
    
    # Extract generation numbers and prepare data containers
    generations = []
    all_energies = []
    lowest_energies = []
    
    # Process each file
    for file_path in popu_files:
        # Extract generation number from filename
        match = re.search(r'popu-(\d+)\.json', file_path)
        if not match:
            continue
            
        gen_num = int(match.group(1))
        generations.append(gen_num)
        
        # Load population data
        try:
            pop = loadfn(file_path)
            
            # Extract energies from individuals
            energies = []
            for ind in pop.individuals:
                if hasattr(ind, 'energy') and ind.energy is not None:
                    energies.append(ind.energy)
            
            if energies:
                all_energies.append((gen_num, energies))
                lowest_energy = min(energies)
                lowest_energies.append((gen_num, lowest_energy))
                print(f"Generation {gen_num}: {len(energies)} individuals, lowest energy = {lowest_energy:.6f}")
            else:
                print(f"Generation {gen_num}: No valid energy values found")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_energies:
        print("No valid energy data found in population files")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all energy points as scatter
    for gen_num, energies in all_energies:
        # Create scatter points for this generation
        x = [gen_num] * len(energies)
        plt.scatter(x, energies, alpha=0.5, s=30, color='blue')
    
    # Sort lowest energy points by generation
    lowest_energies.sort(key=lambda x: x[0])
    
    # Extract x and y for the line plot
    x_line = [point[0] for point in lowest_energies]
    y_line = [point[1] for point in lowest_energies]
    
    # Plot the lowest energy line with steps
    plt.step(x_line, y_line, where='post', color='red', linewidth=2, label='Lowest Energy')
    
    # Add markers at each lowest energy point
    plt.scatter(x_line, y_line, color='red', s=50, zorder=5)
    
    # Set plot labels and title
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.title('Energy Evolution by Generation', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Display the plot
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot energies from population files')
    parser.add_argument('--dir', '-d', type=str, default='.',
                        help='Directory containing popu-*.json files (default: current directory)')
    parser.add_argument('--output', '-o', type=str, default='energy_plot.png',
                        help='Output file path (default: energy_plot.png)')
    
    args = parser.parse_args()
    plot_population_energies(args.dir, args.output)

if __name__ == '__main__':
   main()

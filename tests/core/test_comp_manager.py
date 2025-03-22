import unittest
import numpy as np
from opencsp.core.composition_manager import CompositionManager
from pymatgen.core import Composition

class TestCompositionManager(unittest.TestCase):
    def setUp(self):
        # Set a fixed seed for reproducibility of random operations.
        np.random.seed(0)

    def test_calculate_base_formula(self):
        # Test that the base formula is reduced correctly by computing the GCD.
        comp = {'Fe': 2, 'O': 3}
        cm = CompositionManager(comp)
        # For (2, 3), the GCD should be 1; thus, the reduced formula equals the original.
        self.assertEqual(cm.formula_gcd, 1)
        self.assertEqual(cm.reduced_formula, comp)

        comp2 = {'Fe': 4, 'O': 6}
        cm2 = CompositionManager(comp2)
        # For (4, 6), the GCD should be 2, so the reduced formula should be (2, 3).
        self.assertEqual(cm2.formula_gcd, 2)
        self.assertEqual(cm2.reduced_formula, {'Fe': 2, 'O': 3})

    def test_fixed_formula_composition_generation(self):
        # Test composition generation in fixed-formula mode.
        base = {'Fe': 2, 'O': 3}
        # With a formula_range of (1, 1), the generated composition should equal the base composition.
        cm = CompositionManager(base, formula_range=(1, 1))
        comp_generated = cm.generate_composition()
        self.assertEqual(comp_generated, base)

        # With a formula_range of (2, 2), the generated composition should be twice the base composition.
        cm = CompositionManager(base, formula_range=(2, 2))
        comp_generated = cm.generate_composition()
        expected = {elem: count * 2 for elem, count in base.items()}
        self.assertEqual(comp_generated, expected)

    def test_variable_composition_generation(self):
        # Test composition generation in variable-composition mode.
        base = {'Fe': 2, 'O': 3}
        composition_range = {'Fe': (1, 4), 'O': (2, 5)}
        cm = CompositionManager(base, composition_range=composition_range)
        comp_generated = cm.generate_composition()
        # Verify that each element's count is within its specified range.
        self.assertTrue(composition_range['Fe'][0] <= comp_generated['Fe'] <= composition_range['Fe'][1])
        self.assertTrue(composition_range['O'][0] <= comp_generated['O'] <= composition_range['O'][1])

    def test_mutate_formula_units(self):
        # Test mutation in fixed-formula mode (by changing the number of formula units).
        base = {'Fe': 2, 'O': 3}
        formula_range = (1, 5)
        cm = CompositionManager(base, formula_range=formula_range)
        # Create an initial composition corresponding to 2 formula units.
        current = {elem: count * 2 for elem, count in base.items()}
        mutated = cm.mutate_composition(current, mutation_strength=0.5)
        # Ensure that the total number of atoms in the mutated composition is a multiple of the base atom count.
        total_atoms = sum(mutated.values())
        self.assertEqual(total_atoms % cm.base_atom_count, 0)

    def test_mutate_variable_composition(self):
        # Test mutation in variable-composition mode (mutating individual element counts).
        base = {'Fe': 2, 'O': 3}
        composition_range = {'Fe': (1, 4), 'O': (2, 5)}
        cm = CompositionManager(base, composition_range=composition_range)
        current = {'Fe': 2, 'O': 3}  # Initial composition.
        mutated = cm.mutate_composition(current, mutation_strength=0.5)
        # Check that each mutated element count remains within its specified range.
        for elem, (min_val, max_val) in composition_range.items():
            self.assertTrue(min_val <= mutated[elem] <= max_val)

    def test_calculate_formation_energy_without_reference(self):
        # Test formation energy calculation when no reference energies are provided.
        base = {'Fe': 2, 'O': 3}
        cm = CompositionManager(base)
        comp = {elem: count * 2 for elem, count in base.items()}
        total_energy = -20.0
        formation_energy = cm.calculate_formation_energy(comp, total_energy)
        self.assertAlmostEqual(formation_energy, total_energy / sum(comp.values()))

    def test_calculate_formation_energy_with_reference(self):
        # Test formation energy calculation when reference energies are provided.
        base = {'Fe': 2, 'O': 3}
        reference_energies = {'Fe': -1.0, 'O': -2.0}
        cm = CompositionManager(base, formula_range=(1, 1), reference_energies=reference_energies)
        comp = {elem: count for elem, count in base.items()}
        total_energy = -20.0
        # Calculation:
        # Reference energy = 2 * (-1.0) + 3 * (-2.0) = -2 - 6 = -8.0
        # Formation energy = (total_energy - reference_energy) / total atoms = (-20 - (-8)) / 5 = -12/5 = -2.4
        formation_energy = cm.calculate_formation_energy(comp, total_energy)
        self.assertAlmostEqual(formation_energy, -12.0 / 5)

    def test_get_e_above_hull(self):
        # Test the energy-above-hull calculation using pymatgen.
        base = {'Fe': 2, 'O': 3}
        reference_energies = {'Fe': -1.0, 'O': -2.0}
        cm = CompositionManager(base, formula_range=(1, 1), reference_energies=reference_energies)
        comp = {elem: count for elem, count in base.items()}
        total_energy = -20.0
        e_above_hull = cm.get_e_above_hull(comp, total_energy)
        # Since the exact value depends on the phase diagram construction,
        # we only verify that the returned value is a float.
        self.assertIsInstance(e_above_hull, float)

    def test_as_dict_and_from_dict(self):
        # Test serialization (as_dict) and deserialization (from_dict).
        base = {'Fe': 2, 'O': 3}
        composition_range = {'Fe': (1, 4), 'O': (2, 5)}
        reference_energies = {'Fe': -1.0, 'O': -2.0}
        formula_range = (1, 2)
        total_atoms_range = (5, 15)
        
        cm = CompositionManager(
            composition=base,
            formula_range=formula_range,
            composition_range=composition_range,
            reference_energies=reference_energies,
            total_atoms_range=total_atoms_range
        )
        # Convert the instance to a dictionary.
        d = cm.as_dict()
        self.assertIn("base_composition", d)
        self.assertEqual(d["base_composition"], base)
        self.assertEqual(d["formula_range"], formula_range)
        self.assertEqual(d["total_atoms_range"], total_atoms_range)
        self.assertEqual(d["formula_gcd"], cm.formula_gcd)
        if hasattr(cm, "reduced_formula"):
            self.assertEqual(d["reduced_formula"], cm.reduced_formula)
            
        # Create a new instance from the dictionary.
        cm_new = CompositionManager.from_dict(d)
        self.assertEqual(cm_new.base_composition, base)
        self.assertEqual(cm_new.formula_range, formula_range)
        self.assertEqual(cm_new.composition_range, composition_range)
        self.assertEqual(cm_new.reference_energies, reference_energies)
        self.assertEqual(cm_new.total_atoms_range, total_atoms_range)
        self.assertEqual(cm_new.formula_gcd, cm.formula_gcd)
        if hasattr(cm, "reduced_formula"):
            self.assertEqual(cm_new.reduced_formula, cm.reduced_formula)

if __name__ == '__main__':
    unittest.main()


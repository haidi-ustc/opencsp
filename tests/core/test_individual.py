import unittest
import uuid
from pymatgen.core import Structure, Lattice, Element
from opencsp.core.individual import Individual

class TestIndividual(unittest.TestCase):

    def setUp(self):
        lattice = Lattice.cubic(3.0)
        species = ["Si"] * 8
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
        self.structure = Structure(lattice, species, coords)
        self.individual = Individual(
            structure=self.structure,
            energy=-10.5,
            fitness=0.9,
            dimensionality=3,
            properties={"tag": "test"}
        )

    def test_initialization(self):
        self.assertEqual(self.individual.energy, -10.5)
        self.assertEqual(self.individual.fitness, 0.9)
        self.assertEqual(self.individual.dimensionality, 3)
        self.assertEqual(self.individual.properties["tag"], "test")
        self.assertIsInstance(self.individual.id, str)

    def test_id_setter_getter(self):
        new_id = "custom_id"
        self.individual.id = new_id
        self.assertEqual(self.individual.id, new_id)

    def test_structure_getter_setter(self):
        new_structure = self.structure.copy()
        self.individual.structure = new_structure
        self.assertEqual(self.individual.structure, new_structure)

    def test_energy_setter(self):
        self.individual.energy = -12.3
        self.assertEqual(self.individual.energy, -12.3)

    def test_fitness_setter(self):
        self.individual.fitness = 0.88
        self.assertEqual(self.individual.fitness, 0.88)

    def test_dimensionality_setter(self):
        self.individual.dimensionality = 2
        self.assertEqual(self.individual.dimensionality, 2)

    def test_copy(self):
        ind_copy = self.individual.copy()
        self.assertNotEqual(ind_copy.id, self.individual.id)
        self.assertEqual(ind_copy.energy, self.individual.energy)
        self.assertEqual(ind_copy.fitness, self.individual.fitness)
        self.assertEqual(ind_copy.dimensionality, self.individual.dimensionality)
        self.assertEqual(ind_copy.properties, self.individual.properties)
        self.assertNotEqual(id(ind_copy.structure), id(self.individual.structure))  # Deep copy

    def test_get_composition(self):
        composition = self.individual.get_composition()
        self.assertEqual(composition, {Element("Si"): 8.0})

    def test_get_formula(self):
        formula = self.individual.get_formula()
        self.assertEqual(formula, "Si8")

    def test_as_dict_and_from_dict(self):
        d = self.individual.as_dict()
        self.assertEqual(d["energy"], -10.5)
        self.assertEqual(d["fitness"], 0.9)
        self.assertEqual(d["dimensionality"], 3)
        self.assertIn("structure", d)

        ind2 = Individual.from_dict(d)
        self.assertEqual(ind2.energy, -10.5)
        self.assertEqual(ind2.fitness, 0.9)
        self.assertEqual(ind2.dimensionality, 3)

    def test_str_repr(self):
        s = str(self.individual)
        r = repr(self.individual)
        self.assertIn("Individual(id=", s)
        self.assertEqual(s, r)

    def test_is_serializable(self):
        self.assertTrue(self.individual._is_serializable(1))
        self.assertTrue(self.individual._is_serializable([1, 2, 3]))
        self.assertFalse(self.individual._is_serializable(self.structure))  # Not JSON-serializable

if __name__ == '__main__':
    unittest.main()


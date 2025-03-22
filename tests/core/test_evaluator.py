import unittest
from unittest.mock import MagicMock, patch

from opencsp.core.evaluator import Evaluator
from opencsp.core.individual import Individual
from pymatgen.core import Structure, Lattice


def make_dummy_individual(fitness=None, energy=None):
    lattice = Lattice.cubic(3.0)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return Individual(structure=structure, fitness=fitness, energy=energy)


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Mock Calculator with predictable behavior
        self.calculator = MagicMock()
        self.calculator.calculate.return_value = -10.0
        self.calculator.get_properties.return_value = {"dummy_prop": 1.23}

        self.individual = make_dummy_individual()
        self.evaluator = Evaluator(calculator=self.calculator)

    def test_initialization(self):
        self.assertEqual(self.evaluator.evaluation_count, 0)
        self.assertTrue(callable(self.evaluator.fitness_function))

    def test_evaluate_individual(self):
        fitness = self.evaluator.evaluate(self.individual)
        self.assertEqual(fitness, 10.0)
        self.assertEqual(self.individual.energy, -10.0)
        self.assertEqual(self.individual.fitness, 10.0)
        self.assertEqual(self.evaluator.evaluation_count, 1)
        self.assertIn("dummy_prop", self.individual.properties)

    def test_evaluate_with_constraint(self):
        constraint = MagicMock()
        constraint.evaluate.return_value = 2.5
        evaluator = Evaluator(calculator=self.calculator, constraints=[constraint])
        fitness = evaluator.evaluate(self.individual)
        self.assertAlmostEqual(fitness, 10.0 - 2.5)
        constraint.evaluate.assert_called_once_with(self.individual)

    def test_evaluate_with_existing_energy(self):
        self.individual.energy = -5.0
        fitness = self.evaluator.evaluate(self.individual)
        self.assertEqual(fitness, 5.0)
        # Should not call calculator again
        self.calculator.calculate.assert_not_called()

    def test_evaluate_handles_error(self):
        self.calculator.calculate.side_effect = Exception("Calc failed")
        fitness = self.evaluator.evaluate(self.individual)
        self.assertEqual(fitness, float("-inf"))
        self.assertIn("error", self.individual.properties)

    def test_evaluate_population_serial(self):
        individuals = [make_dummy_individual() for _ in range(5)]
        self.evaluator.evaluate_population(individuals, parallel=False)
        self.assertEqual(self.evaluator.evaluation_count, 5)
        for ind in individuals:
            self.assertEqual(ind.fitness, 10.0)

    def test_as_dict(self):
        evaluator_dict = self.evaluator.as_dict()
        self.assertIn("calculator", evaluator_dict)
        self.assertIn("evaluation_count", evaluator_dict)
        self.assertTrue(evaluator_dict["using_default_fitness"])

    def test_from_dict(self):
        d = {"evaluation_count": 5}
        restored = Evaluator.from_dict(d)
        self.assertEqual(restored.evaluation_count, 5)
        self.assertIsNone(restored.calculator)
        self.assertTrue(callable(restored.fitness_function))


if __name__ == "__main__":
    unittest.main()


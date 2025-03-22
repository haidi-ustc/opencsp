import unittest
import random
from pymatgen.core import Structure, Lattice
from opencsp.core.individual import Individual
from opencsp.core.population import Population
from unittest.mock import patch




def generate_individual(fitness=None, energy=None, element="Si"):
    lattice = Lattice.cubic(3.0)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, [element, element], coords)
    return Individual(structure=structure, fitness=fitness, energy=energy)


class TestPopulation(unittest.TestCase):
    
    def setUp(self):
        random.seed(42)  # Deterministic tests
        self.individuals = [generate_individual(fitness=i) for i in range(10)]
        self.population = Population(individuals=self.individuals, max_size=10)

    def test_population_init(self):
        self.assertEqual(self.population.size, 10)
        self.assertEqual(self.population.generation, 0)

    def test_add_individual(self):
        new_ind = generate_individual(fitness=99)
        self.population.add_individual(new_ind)
        self.assertIn(new_ind, self.population.individuals)

    def test_get_best_single(self):
        best = self.population.get_best()
        self.assertEqual(best.fitness, 9)

    def test_get_best_multiple(self):
        top_3 = self.population.get_best(n=3)
        self.assertEqual(len(top_3), 3)
        self.assertEqual([ind.fitness for ind in top_3], [9, 8, 7])

    def test_tournament_selection(self):
        selected = self.population.select_tournament(n=5, tournament_size=3)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_roulette_selection(self):
        selected = self.population.select_roulette(n=5)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(isinstance(ind, Individual) for ind in selected))

    def test_sort_by_fitness(self):
        self.population.sort_by_fitness()
        fitness_values = [ind.fitness for ind in self.population.individuals]
        self.assertEqual(fitness_values, sorted(fitness_values, reverse=True))

    def test_update_with_elitism(self):
        new_inds = [generate_individual(fitness=5), generate_individual(fitness=6)]
        self.population.update(new_individuals=new_inds, elitism=2)
        self.assertEqual(self.population.size, 4)  # 2 elites + 2 new
        self.assertEqual(self.population.generation, 1)

    def test_average_fitness(self):
        avg = self.population.get_average_fitness()
        self.assertAlmostEqual(avg, sum(range(10)) / 10)

    def test_average_fitness_with_none(self):
        self.population.individuals[0].fitness = None
        avg = self.population.get_average_fitness()
        self.assertAlmostEqual(avg, sum(range(1, 10)) / 9)

    def test_as_dict_and_from_dict(self):
        pop_dict = self.population.as_dict()
        new_population = Population.from_dict(pop_dict)
        self.assertEqual(new_population.size, self.population.size)
        self.assertEqual(new_population.generation, self.population.generation)
        self.assertEqual(
            [ind.fitness for ind in new_population.individuals],
            [ind.fitness for ind in self.population.individuals]
        )

    def test_get_diversity_mocked(self):
        with patch("opencsp.utils.structure.calculate_structure_distance", return_value=1.0):
            diversity = self.population.get_diversity()
            n = len(self.population.individuals)
            expected_pairs = n * (n - 1) / 2
            self.assertAlmostEqual(diversity, 1.0)

    def test_get_diversity_insufficient(self):
        pop = Population(individuals=[generate_individual()])
        self.assertEqual(pop.get_diversity(), 0.0)


if __name__ == '__main__':
    unittest.main()


from unittest import TestCase

from feature_selection.genetic_programming.GPFeatureSelection import GPFeatureSelection
from feature_selection.genetic_programming.GPIndividual import GPIndividual

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class TestGPFeatureSelection(TestCase):
    def setUp(self):
        lr = LinearRegression()
        self.X = np.random.rand(40, 8)
        self.y = np.random.randint(0, 2, size=40)
        self.fs = GPFeatureSelection(mean_squared_error, [lr],
                                     init_population_size=10, final_population_size=2,
                                     init_dataset_size=10, final_dataset_size=20,
                                     percent_features_first_selected=0.25,
                                     num_generations=2,
                                     prob_random_mutation=0.1)

        self.fs10 = GPFeatureSelection(mean_squared_error, [lr],
                                     init_population_size=25, final_population_size=5,
                                     init_dataset_size=10, final_dataset_size=30,
                                     percent_features_first_selected=0.25,
                                     num_generations=10,
                                     prob_random_mutation=0.1)

    def test_fit2(self):
        self.fs10.fit(self.X, self.y)

    def test_fit(self):
        self.fs.fit(self.X, self.y)

    def test_update_population(self):
        pop = self.fs.get_init_population(num_features=8)
        self.fs.calc_population_fitnesses(pop, self.X, self.y, train_size=10)

        self.fs.update_population(pop, self.X, self.y, k=0)
        self.assertEqual(len(pop), 10)
        pop = self.fs.update_population(pop, self.X, self.y, k=1)
        self.assertEqual(len(pop), 2)



    def test_pick_pairs_to_reproduce(self):
        pop = self.fs.get_init_population(num_features=8)
        self.fs.calc_population_fitnesses(pop, self.X, self.y, train_size=10)
        pairs = self.fs.pick_pairs_to_reproduce(population=pop, num_pairs=4)
        self.assertEqual(len(pairs), 4)

    def test_calc_population_fitnesses(self):
        pop = self.fs.get_init_population(num_features=8)
        self.fs.calc_population_fitnesses(pop, self.X, self.y, 10)
        for p in pop:
            self.assertTrue(p.fitness is not None)

    def test_do_mutations(self):
        pop = self.fs.get_init_population(num_features=8)
        self.fs.do_mutations(pop)

    def test_get_init_population(self):
        pop = self.fs.get_init_population(num_features=8)
        self.assertEqual(len(pop), 10)
        self.assertEqual(len(pop[0].features), 8)


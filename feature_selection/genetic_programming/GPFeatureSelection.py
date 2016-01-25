__author__ = "anthony"

import numpy as np
import random
from feature_selection.genetic_programming.GPIndividual import GPIndividual
from scipy.stats import rv_discrete
from util.sampling_functions import SamplingFunctions

class GPFeatureSelection:
    def __init__(self, scorer, clfs, use_proba=False,
                 init_population_size=100, final_population_size=10,
                 init_dataset_size=100, final_dataset_size=100000,
                 percent_features_first_selected=0.2,
                 num_generations=50,
                 prob_random_mutation=0.01):
        '''
        Assuming higher is better, it otherwise then invert scorer output.
        '''
        self.scorer = scorer
        self.clfs = clfs
        self.use_proba = use_proba

        self.init_population_size = init_population_size
        self.final_population_size = final_population_size

        self.init_dataset_size = init_population_size
        self.final_dataset_size = final_dataset_size

        self.percent_features_first_selected = percent_features_first_selected
        self.num_generations = num_generations
        self.prob_random_mutation = prob_random_mutation

        self.population_size_schedule = np.linspace(init_population_size, final_population_size, self.num_generations, dtype=int)
        self.dataset_size_schedule = np.linspace(init_dataset_size, final_dataset_size, self.num_generations, dtype=int)


    def fit(self, X, y):
        population = self.get_init_population(X.shape[1])
        for k in range(self.num_generations):
            population = self.update_population(population, X, y, k)
            self.do_mutations(population)

    def update_population(self, population, X, y, k):
        population_size = self.population_size_schedule[k]
        dataset_size = self.dataset_size_schedule[k]
        new_population = []

        self.calc_population_fitnesses(population, X, y, dataset_size)
        pairs = self.pick_pairs_to_reproduce(population, population_size)
        new_population = [a.reproduce(b) for a,b in pairs]

        self.do_mutations(new_population)
        return new_population


    def pick_pairs_to_reproduce(self, population, num_pairs):
        fitnesses = np.array([x.fitness for x in population])
        fitnesses = fitnesses / sum(fitnesses)
        dist = rv_discrete(values=(range(len(fitnesses)), fitnesses))
        pairs_matrix = dist.rvs(size=2*num_pairs).reshape((num_pairs, 2))
        return [(population[a], population[b]) for a,b in pairs_matrix]

    def do_mutations(self, population):
        for individual in population:
            if random.random() <= self.prob_random_mutation:
                individual.random_mutation()

    def get_init_population(self, num_features):

        #all_features = [(np.random.rand(num_features) < self.percent_features_first_selected)*1 for _ in range(self.init_population_size)]

        #all_selected_features_idxs = [set(SamplingFunctions.sample_array(self.init_population_size, self.percent_features_first_selected))
        #                              for _ in range(self.init_population_size)]
        #all_features = [[(i in idxs)*1 for i in range(len(num_features))] for idxs in all_selected_features_idxs]

        feature_idxs = np.array(range(num_features))
        population = []
        for k in range(self.init_population_size):
            selected_features_idxs = SamplingFunctions.sample_array(feature_idxs, self.percent_features_first_selected)
            features = np.zeros(num_features)
            features[selected_features_idxs] = 1
            individual = GPIndividual(random.choice(self.clfs), self.scorer, features, self.use_proba)
            population.append(individual)

        #population = [GPIndividual(random.choice(self.clfs), self.scorer, features, self.use_proba) for features in all_features]
        return population

    def calc_population_fitnesses(self, population, X, y, train_size):
        for p in population:
            p.get_fitness(X, y, train_size)




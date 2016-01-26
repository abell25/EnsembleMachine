__author__ = "anthony"

import numpy as np
import random
from scipy.stats import rv_discrete
import time

from feature_selection.genetic_programming.GPIndividual import GPIndividual
from util.sampling_functions import SamplingFunctions
from util.parallel_functions import ParallelFunctions

import logging
log = logging.getLogger(__name__)

class GPFeatureSelection:
    def __init__(self, scorer, clfs, use_proba=False,
                 init_population_size=100, final_population_size=10,
                 init_dataset_size=100, final_dataset_size=100000,
                 percent_features_first_selected=0.2,
                 num_generations=50,
                 prob_random_mutation=0.01,
                 num_cpus=None):
        '''
        Assuming higher is better, you can use lambda y, y_pred: -scorer(y, y_pred) to invert the scorer if needed.
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

        self.num_cpus = num_cpus
        self.diagnostics = [{} for _ in range(num_generations)]


    def fit(self, X, y):
        t0 = time.time()
        population = self.get_init_population(X.shape[1])
        for k in range(self.num_generations):
            population = self.update_population(population, X, y, k)
            self.do_mutations(population)

        log.info("Finished {0} generations in {1} seconds.".format(self.num_generations, (time.time() - t0)))


    def update_population(self, population, X, y, k):
        t0 = time.time()
        population_size = self.population_size_schedule[k]
        dataset_size = self.dataset_size_schedule[k]
        self.diagnostics[k]["population_size"] = population_size
        self.diagnostics[k]["dataset_size"] = dataset_size

        self.calc_population_fitnesses(population, X, y, dataset_size)
        fitnesses = [x.fitness for x in population]
        self.diagnostics[k]["fitness_mean"] = np.mean(fitnesses)
        self.diagnostics[k]["fitness_std"] = np.std(fitnesses)

        pairs = self.pick_pairs_to_reproduce(population, population_size)
        new_population = [a.reproduce(b) for a,b in pairs]

        num_mutations = self.do_mutations(new_population)

        log.debug("iter: {0}, time: {1}, pop_size: {2}, dataset_size: {3}, fitness: {4:0.5f}/{5:0.5f}, num_mutations: {6}".format(
            k, (time.time() - t0), population_size, dataset_size, np.mean(fitnesses), np.std(fitnesses), num_mutations
        ))
        stats = self.get_aggregated_data(population, X.shape[1], top_k_individuals=10, top_k_used_features=10)
        self.diagnostics[k].update(stats)
        log.debug(stats)
        print("data:{0:5} pop:{1} mean:{2:.5f} std:{3:.5f} max: {4:.5f} top features: {5}".format(
            dataset_size, population_size, stats['mean'], stats['std'], stats['max'], stats['top features']))
        print(" ".join(["({0:.5f}, {1}, {2})".format(x['score'], x['num_features'], x['features'])
          for x in self.diagnostics[k]['best_individuals'][:5]]))

        return new_population

    def pick_pairs_to_reproduce(self, population, num_pairs):
        fitnesses = np.array([x.fitness for x in population])
        fitnesses = fitnesses / sum(fitnesses)
        dist = rv_discrete(values=(range(len(fitnesses)), fitnesses))
        pairs_matrix = dist.rvs(size=2*num_pairs).reshape((num_pairs, 2))
        return [(population[a], population[b]) for a,b in pairs_matrix]

    def do_mutations(self, population):
        num_mutations = 0
        for individual in population:
            if random.random() <= self.prob_random_mutation:
                individual.random_mutation()
                num_mutations += 1
        return num_mutations

    def get_init_population(self, num_features):
        feature_idxs = np.array(range(num_features))
        population = []
        for k in range(self.init_population_size):
            selected_features_idxs = SamplingFunctions.sample_array(feature_idxs, self.percent_features_first_selected)
            features = np.zeros(num_features, dtype=int)
            features[selected_features_idxs] = 1
            individual = GPIndividual(random.choice(self.clfs), self.scorer, features, self.use_proba)
            population.append(individual)

        return population

    def calc_population_fitnesses_parallel(self, population, X, y, train_size):
        if self.num_cpus is 1:
            fitnesses = [p.get_fitness(X, y, train_size) for p in population]
        else:
            def f(p): return p.get_fitness(X, y, train_size)
            fitnesses = ParallelFunctions.pmap(population, f, self.num_cpus)

        for p, fitness in zip(population, fitnesses):
            p.fitness = fitness

    def calc_population_fitnesses(self, population, X, y, train_size):
        for p in population:
            p.get_fitness(X, y, train_size)

    def get_aggregated_data(self, population, num_features, top_k_individuals=10, top_k_used_features=10):
        stats = self.get_averages_for_generation(population, num_features, top_k_used_features)
        best_individuals = self.get_best_stats_for_generation(population, num_features, top_k_individuals)
        stats["best_individuals"] = best_individuals

        log.debug("|-{0:.4f}--({1:.4f}/{2:.4f})--{3:.4f}-|  std: {4:.4f}".format(
                stats["min"], stats["mean"], stats["median"], stats["max"], stats["std"]))
        log.debug("top features: {0}".format(stats["top features"]))
        for ind in best_individuals:
            log.debug("     (ind) score: {0:.5f}, features ({1}): {2}".format(ind["score"], ind["num_features"], sorted(ind["features"])))

        return stats

    def get_averages_for_generation(self, population, num_features, top_k_used_features=10):
        idxs = np.array(range(num_features))

        scores = [p.fitness for p in population]
        score_min = np.min(scores)
        score_max = np.max(scores)
        score_mean = np.mean(scores)
        score_median = np.median(scores)
        score_std = np.std(scores)
        features = [idxs[p.features == 1] for p in population]
        features_idxs = [i for idxs in features for i in idxs]
        log.debug("features: {0}".format([p.features for p in population]))
        log.debug("features_idxs: {0}, max: {1}".format(features_idxs, num_features))
        feature_hist = np.histogram(features_idxs, bins=range(0, num_features))
        log.debug("feature_hist: {0}, num_features: {1}".format(feature_hist, num_features))
        sorted_counts = sorted(list(zip(*feature_hist)), key=lambda x: x[0], reverse=True)
        log.debug("sorted counts: {0}".format(sorted_counts))
        top_features = [x[1] for x in sorted_counts[:top_k_used_features]]
        log.debug("top {0} features: {1}".format(top_k_used_features, top_features))
        return {"min": score_min, "max": score_max, "mean": score_mean, "median": score_median, "std": score_std, "top features": top_features}


    def get_best_stats_for_generation(self, population, num_features, top_k=10):
        idxs = np.array(range(num_features))
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        stats = []
        for i, p in enumerate(sorted_population):
            features, score = p.features, p.fitness
            selected_features = idxs[features == 1]
            num_selected_features = len(selected_features)
            stats.append({"score": score, "num_features": num_selected_features, "features": selected_features})

        return stats










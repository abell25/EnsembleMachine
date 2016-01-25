__author__="anthony"

import random
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

class GPIndividual:
    def __init__(self, clf, scorer, features, use_proba: False):
        self.clf = clf
        self.scorer = scorer
        self.features = features
        self.use_proba = use_proba
        self.fitness = None

    def get_fitness(self, X, y, train_size):
        splits = StratifiedShuffleSplit(y, n_iter=1, train_size=train_size)
        scores = []

        for train_idx, test_idx in splits:
            X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            self.clf.fit(X_tr, y_tr)
            y_pred = self.clf.predict(X_te)
            score = self.scorer(y_te, y_pred)
            scores.append(score)

        mean_score = np.mean(scores)
        self.fitness = mean_score
        return mean_score

    def random_mutation(self):
        k = random.choice(range(len(self.features)))
        self.features[k] = (not self.features[k])*1

    def reproduce(self, other):
        new_features = [random.choice(x) for x in zip(self.features, other.features)]
        return GPIndividual(self.clf, self.scorer, new_features, self.use_proba)


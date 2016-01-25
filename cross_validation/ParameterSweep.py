__author__ = 'anthony'

import random
from time import time
import itertools
import numpy as np
from scipy.stats import mode

from cross_validation.TrialsRecorder import TrialsRecorder

import logging
logger = logging.getLogger(__name__)

class ParameterSweep:
    def __init__(self, X, y, dataset_name, X_val=None, dont_persist_results=False):
        self.trials = []
        self.trials_store = TrialsRecorder(dataset_name)
        self.X = X
        self.y = y
        self.X_val = X_val
        self.dont_persist_results = dont_persist_results

    @staticmethod
    def getParamsGenerator(params):
        params_keys = list(params)
        for params_values in itertools.product(*params.values()):
            yield dict(zip(params_keys, params_values))

    @staticmethod
    def getRandomParams(params):
        arr = list(ParameterSweep.getParamsGenerator(params))
        random.shuffle(arr)
        return arr

    def run_configuration(self, Clf, params, scorer, splits, use_proba=False, save_y_oof=False):
        scores = []
        t0 = time()

        clf = Clf(**params)

        if save_y_oof:
            y_oof = np.zeros(len(self.y))
        if self.X_val is not None:
            y_vals = []

        for i, (train_idx, test_idx) in enumerate(splits):
            t_i = time()
            X_train, X_test, y_train, y_test = self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]

            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_test)[:,1] if use_proba else clf.predict(X_test)

            if save_y_oof:
                y_oof[list(test_idx)] = y_pred
            if self.X_val is not None:
                y_vals.append(clf.predict_proba(self.X_val)[:,1] if use_proba else clf.predict(self.X_val))


            scores.append(scorer(y_test, y_pred))
            logger.debug(" iter {0}: {1:2.2} sec \t score: {2:.4},  running: {3:.4}".format(i, time()-t_i, scores[-1], np.mean(scores)))

        stats = {"score": np.mean(scores), "clf": Clf.__name__, "params": params, "num_folds": len(splits),
                 "train_size": X_train.shape[0], "test_size": X_test.shape[0], "sample_size": X_train.shape[0] + X_test.shape[0]}

        if save_y_oof:
            stats["y_oof"] = y_oof
        if self.X_val is not None:
            stats["y_val"] = np.mean(y_vals, axis=0) if use_proba else mode(y_vals).mode

        logger.info(" time: {0:.2f} sec,  score: {1:.4f} +/- {2:.4f}, stats: {3}".format(time()-t0, np.mean(scores), np.std(scores), stats))

        self.trials.append(stats)

        if not self.dont_persist_results:
            self.trials_store.saveTrial(stats)

        return stats

    def run_configuration_sweep(self, Clf, param_values, scorer, splits, search_plan, use_proba=False, save_y_oof=False, num_samples=10):
        params_samples = []
        if search_plan == "grid":
            params_samples = list(ParameterSweep.getParamsGenerator(param_values))
        if search_plan == "random":
            params_samples = ParameterSweep.getRandomParams(param_values)[:num_samples]

        stats = []
        for params in params_samples:
            stats.append(self.run_configuration(Clf, params, scorer, splits, use_proba, save_y_oof))

        return stats

    def get_trials(self, query=None):
        return [x for x in self.trials_store.getAllTrials(query)]

    def clear_all_trials(self):
        self.trials_store.clear_all_trials()
__author__ = 'anthony'

class ParameterSweep:
    def __init__(self, X, y):
        self.trials = []
        self.X = X
        self.y = y

    @staticmethod
    def getParamsGenerator(params):
        params_keys = list(params)
        for params_values in itertools.product(*params.values()):
            yield dict(zip(params_keys, params_values))

    @staticmethod
    def getRandomParams(params):
        arr = list(ParameterSweep.getParamsGenerator(params))
        shuffle(arr)
        return arr

    def run_configuration(self, Clf, params, scorer, splits, use_proba=False, save_y_oof=False):
        rf = None
        scores = []
        t0 = time()

        clf = Clf(**params)

        if save_y_oof:
            y_oof = np.zeros_like(self.y)
            print("y_oof shape: {0}".format(y_oof.shape))

        for i, (train_idx, test_idx) in enumerate(splits):
            t_i = time()
            X_train, X_test, y_train, y_test = self.X[train_idx], self.X[test_idx], self.y[train_idx], self.y[test_idx]

            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:,1] if use_proba else clf.predict(X_test)

            if save_y_oof:
                print("test_idx: shape: {0}, act: {1}, y_pred: sum: {2}, act: {3}".format(test_idx.shape, test_idx, sum(y_pred), y_pred))
                y_oof[test_idx] = y_pred
                print("y_oof[test_idx] sum: {0}, shape: {1}, act: {2}".format(sum(y_oof[test_idx]), y_oof[test_idx].shape, y_oof[test_idx]))

            scores.append(scorer(y_test, y_pred))
            logger.debug(" iter {0}: {1:2.2} sec \t score: {2:.4},  running: {3:.4}".format(i, time()-t_i, scores[-1], np.mean(scores)))

        stats = {"score": np.mean(scores), "clf": Clf.__name__, "params": params, "num_folds": len(splits),
                 "train_size": X_train.shape[0], "test_size": X_test.shape[0], "sample_size": X_train.shape[0] + X_test.shape[0]}

        if save_y_oof:
            stats["y_oof"] = y_oof

        logger.info(" time: {0:.2f} sec,  score: {1:.4f} +/- {2:.4f}, stats: {3}".format(time()-t0, np.mean(scores), np.std(scores), stats))

        self.trials.append(stats)
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
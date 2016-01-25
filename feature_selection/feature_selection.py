__author__ = 'anthony bell'

from random import shuffle
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split

from problem_type import ProblemType

import logging
log = logging.getLogger(__name__)

class FeatureSelection():
    def __init__(self, lower_is_better = True, method=None, X=None, y=None, X_sub=None, clf=None, score_func=None, problem_type='classification',
                 col_names = None):
        '''
        class for feature selection of raw data.

        :param lower_is_better: whether a
        :param method:
        :param X:
        :param y:
        :param clf:
        :param problem_type:
        :return:
        '''
        self.lower_is_better = lower_is_better
        self.method = method
        self.X = X
        self.y = y
        self.X_sub = X_sub
        self.clf = clf
        self.score_func = score_func
        self.problem_type = problem_type
        self.col_names = col_names if col_names else range(X.shape[1])
    
    def getTransformsList(self, method='all'):
        return {'all': self.allSelection,
                'forwards': self.forwardsSelection,
                'backwards': self.backwardsSelection,
                'importances': self.featureImportancesSelection,
                'random': self.randomSubsetSelection
               }
    
    def allSelection(self):
        """ all selection:
                 returns all features
        """
        return self.X, self.X_sub
    
    def forwardsSelection(self):
        """ forwards selection:
                add features 1-by-1 until score no longer improves

        """
        X, X_sub = FeatureSelection.forwards(self.X, self.y, self.X_sub,
                                   clf=self.clf,
                                   score_func=self.score_func, lower_is_better=self.lower_is_better,
                                   clf_names=self.col_names)
        return X, X_sub
    
    def backwardsSelection(self):
        """ backwards selection:
                remove features 1-by-1 until score no longer improves
        """
        X, X_sub = FeatureSelection.backwards(self.X, self.y, self.X_sub,
                                   clf=self.clf,
                                   score_func=ProblemType.RMSPE, lower_is_better=self.lower_is_better,
                                   clf_names=self.col_names)
        return X, X_sub
    
    def featureImportancesSelection(self, total_importance=0.95):
        """feature Importances selection:
                  uses rf/xgb feature importances to filter useless features
        """
        X, X_sub = FeatureSelection.getFeatureImportanceColumns(self.X, self.y, self.X_sub,
                                                    rf=self.clf, col_names=self.col_names,
                                                    total_importance=total_importance)
        return X, X_sub
    
    def randomSubsetSelection(self, percent=0.5):
        """ random Subset selection
                  returns a random subset of the variables
        """
        N = self.X.shape[1]
        idxs = range(N)
        shuffle(idxs)
        max_idx = int(max(1, np.round(percent*N)))
        rand_idxs = idxs[:max_idx]

        return self.X[:,rand_idxs], self.X_sub[:,rand_idxs]
    

    @staticmethod
    def forwards(X, y, X_sub, clf, score_func, use_proba=False, lower_is_better=False, problem_type=None, clf_names=None, allow_repeats=False, idxs=None, return_idxs=False):

        clf_names = [str(n) for n in range(num_clfs)] if clf_names is None else clf_names

        all_idxs = list(range(num_clfs))
        idxs = [] if idxs is None else idxs
        num_iters = num_clfs
        best_score = 99999999.0 if lower_is_better else -99999999.0
        best_score_val = 0.0
        for iter_i in range(num_iters):
            best_idx = -1

            for i in all_idxs:
                X_latest = X[:,idxs + [i]]
                X_train, X_test_val, y_train, y_test_val = train_test_split(X_latest, y, train_size=0.4) #, test_size=0.20)
                X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, train_size=0.5)

                clf.fit(X_train, y_train)

                if use_proba:
                    y_pred = clf.predict_proba(X_test)
                    y_pred_val = clf.predict_proba(X_val)
                    if problem_type == 'classification':
                        y_pred = y_pred[:,1] #only keep one column
                        y_pred_val = y_pred_val[:,1]
                else:
                    y_pred = clf.predict(X_test)
                    y_pred_val = clf.predict(X_val)

                s = score_func(y_test, y_pred)
                s_val = score_func(y_val, y_pred_val)
                delta = s - best_score

                log.debug('score: {0}, best_score: {1}, delta: {2}, index: {3}'.format(s, best_score, delta, i))

                if lower_is_better:
                    delta = -delta
                if delta > 0:
                    best_score, best_idx = s, i
                    best_score_val = s_val

            if best_idx == -1:
                log.info("no predictor improved performance!  quitting..")
                if return_idxs:
                    return idxs
                else:
                    return X[:,idxs], X_sub[:,idxs]

            if not allow_repeats:
                all_idxs.remove(best_idx)
            idxs.append(best_idx)

            log.info("iter {0}/{1}: predictor: {2} ({3}), score: {4:.4f}, holdout: {5:.4f}".format(iter_i, num_iters, best_idx, clf_names[best_idx], best_score, best_score_val))
            best_idx = -1

        if return_idxs:
            return idxs
        else:
            return X[:,idxs], X_sub[:,idxs]

    # Backwards
    @staticmethod
    def backwards(X, y, X_sub, clf, score_func, use_proba=False, lower_is_better=False, problem_type=None, clf_names=None, idxs=None, return_idxs=False):
        num_clfs = X.shape[1]
        idxs = set(range(num_clfs)) if idxs is None else (set(range(num_clfs)) - set(idxs))
        num_iters = max(num_clfs-1, 0)
        log.info("num features: {0}, num iters: {1}".format(num_clfs, num_iters))
        best_score = 99999999.0 if lower_is_better else -99999999.0
        for iter_i in range(num_iters):
            best_idx = -1

            for i in idxs:

                X_latest = X[:,list(idxs - set([i]))]
                X_train, X_test, y_train, y_test = train_test_split(X_latest, y, train_size=0.20, test_size=0.20)
                clf.fit(X_train, y_train)

                if use_proba:
                    y_pred = clf.predict_proba(X_test)
                    if problem_type == 'classification':
                        y_pred = y_pred[:,1] #only keep one column
                else:
                    y_pred = clf.predict(X_test)

                s = score_func(y_pred, y_test)
                delta = s - best_score

                log.debug('score: {0}, best_score: {1}, delta: {2}, index: {3}'.format(s, best_score, delta, i))

                if lower_is_better:
                    delta = -delta
                if delta > 0:
                    best_score, best_idx = s, i

            if best_idx == -1:
                log.info("no clf is worsening performance!  quitting..")
                return X[:,list(idxs)], X_sub[:,list(idxs)]

            idxs -= set([best_idx])
            log.info("iter {0}/{1}: clf: {2} ({3}), score: {4}".format(iter_i, num_iters, best_idx, clf_names[best_idx], best_score))
            best_idx = -1

        if return_idxs:
            return list(idxs)
        else:
            return X[:,list(idxs)], X_sub[:,list(idxs)]

    @staticmethod
    def getForwardsBackwards(X, y, X_sub, clf, score_func, use_proba=False, lower_is_better=False, problem_type=None, clf_names=None):
        pass

    @staticmethod
    def getFeatureImportanceColumns(X, y, X_sub, rf, col_names=None, total_importance=0.95):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
        rf.fit(X_train, y_train)
        tuples = sorted(list(zip(range(len(rf.feature_importances_)), col_names, rf.feature_importances_)), key=lambda x: x[2], reverse=True)

        sum_imp = 0.0
        trunc_tuples = []
        for idx, col, col_imp in tuples:
            sum_imp += col_imp
            trunc_tuples.append((idx, col, col_imp))
            if sum_imp >= total_importance:
                break

        idxs = [x[0] for x in trunc_tuples]
        return X[:, idxs], X_sub[:, idxs]
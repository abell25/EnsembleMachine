__author__ = 'anthony bell'

import libscores
from scoring_functions import ScoringFunctions

class ChalearnScorer(ScoringFunctions):
    def __init__(self, task):
        self.task = task

        self.scorer_mapping = {
            'bac_metric': {'scorer': self.bac_metric, 'lower_is_better': False},
            'auc_metric': {'scorer': self.auc_metric, 'lower_is_better': False},
            'f1_metric':  {'scorer': self.f1_metric,  'lower_is_better': False},
            'pac_metric': {'scorer': self.pac_metric, 'lower_is_better': False},
            'r2_metric':  {'scorer': self.r2_metric,  'lower_is_better': False},
            'a_metric':   {'scorer': self.a_metric,   'lower_is_better': False}
        }

    def bac_metric(self, y, y_pred):
        return libscores.bac_metric(y, y_pred, self.task)

    def auc_metric(self, y, y_pred):
        return libscores.auc_metric(y, y_pred, self.task)
    
    def f1_metric(self, y, y_pred):
        return libscores.f1_metric(y, y_pred, self.task)
    
    def pac_metric(self, y, y_pred):
        return libscores.pac_metric(y, y_pred, self.task)
    
    def r2_metric(self, y, y_pred):
        return libscores.r2_metric(y, y_pred, self.task)

    def a_metric(self, y, y_pred):
        return libscores.a_metric(y, y_pred, self.task)


    def get_scorer(self, metric):
        if metric not in self.scorer_mapping:
            raise Exception("Metric {0} is not an implemented metric!".format(metric))
        return self.scorer_mapping[metric]
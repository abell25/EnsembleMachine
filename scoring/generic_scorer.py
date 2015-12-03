__author__ = 'anthony bell'

from scoring.scoring_functions import ScoringFunctions

class GenericScorer():
    def __init__(self, metric):
        self.metric = metric

    def getScorer(self):
        metricScorerLookup = {
            'RMSLE': ScoringFunctions.RMSLE,
            'RMSE':  ScoringFunctions.RMSE,
            'RMSPE': ScoringFunctions.RMSPE
        }

        return metricScorerLookup[self.metric]

    def score(self, y, y_pred):
        return self.scorer(y, y_pred)

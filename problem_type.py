__author__ = 'anthony bell'


import numpy as np

class ProblemType():
    """
    object for determining what kind of machine learning problem is being solved.
    """
    ProblemTypes = ['classification', 'regression', 'multiclass classification']
    Metrics = ['F1', 'logloss', 'MSE', 'RMSE', 'AUC', 'RMSPE', 'RMSLE']
    LowerIsBetter = [False, True, True, True, False, True, True]

    def __init__(self, metric, is_classification, is_binary, is_multilabel, is_large_scale=False):
        """
        :param metric:              The Scoring metric.
        :param is_classification:   Whether the problem is classification or not (regression).
        :param is_binary:           Whether the output is binary or not (probabilities).
        :param is_multilabel:       Whether the problem has multiple labels to predict.
        :param is_large_scale:      For large scale problems, we'll need to use linear models instead.  (default: False)
        """



        self.metric = metric
        self.is_classification = is_classification
        self.is_binary = is_binary
        self.is_multilabel = is_multilabel
        self.is_large_scale = is_large_scale


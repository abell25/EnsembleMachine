__author__ = 'anthony bell'


import numpy as np

class ProblemType():
    """
    object for determining what kind of machine learning problem is being solved.
    """

    def __init__(self, metric, scorer, is_classification, is_binary, is_multilabel, is_large_scale=False, time_budget=None):
        """
        :param metric:              (str)  The Scoring metric.
        :param is_classification:   (bool) Whether the problem is classification or not (regression).
        :param is_binary:           (bool) Whether the output is binary or not (probabilities).
        :param is_multilabel:       (bool) Whether the problem has multiple labels to predict.
        :param is_large_scale:      (bool) For large scale problems, we'll need to use linear models instead.  (default: False)
        :param time_budget          (int)  Number of seconds allotted to the algorithm.
        """

        self.metric = metric
        self.scorer = scorer
        self.is_classification = is_classification
        self.is_binary = is_binary
        self.is_multilabel = is_multilabel
        self.is_large_scale = is_large_scale
        self.time_budget = time_budget


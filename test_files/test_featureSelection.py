import numpy as np
from feature_selection import FeatureSelection
from sklearn.ensemble import RandomForestRegressor
from problem_type import ProblemType

from unittest import TestCase

__author__ = 'anthony bell'



class TestFeatureSelection(TestCase):
  def setUp(self):
    self.X = np.array([[1,2,3,4,5],[11,12,13,14,15],[21,22,23,24,25]])
    self.y = np.array([1,11,21])
    self.X_sub = np.array([[31,32,33,34,35],[41,42,43,44,45]])
    self.featureSelection = FeatureSelection(lower_is_better = True, method=None,
                                             X=self.X, y=self.y, X_sub=self.X_sub,
                                             clf=RandomForestRegressor(n_estimators=2), score_func=ProblemType.RMSE,
                                             problem_type='classification',
                                            col_names = ['A', 'B', 'C', 'D', 'E'])

  def test_allSelection(self):
    X, X_sub = self.featureSelection.allSelection()
    self.assertEqual(X.shape[1], 5, 'number of columns of X is not 5!')
    self.assertEqual(X_sub.shape[1], 5, 'number of columns of X_sub is not 5!')

  def test_forwardsSelection(self):
    self.fail()

  def test_backwardsSelection(self):
    self.fail()

  def test_featureImportancesSelection(self):
    self.fail()

  def test_randomSubsetSelection(self):
    X, X_sub = self.featureSelection.randomSubsetSelection(percent=0.4)
    self.assertEqual(X.shape[1], 2, 'number of columns of X is not 2!')
    self.assertEqual(X_sub.shape[1], 2, 'number of columns of X_sub is not 2!')

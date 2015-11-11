import numpy as np
from feature_selection import FeatureSelection
from problem_type import ProblemType
from sklearn.ensemble import RandomForestRegressor
from problem_type import ProblemType
from train_test_data_loader import TrainTestDataLoader
from sklearn.linear_model import LogisticRegressionCV

from unittest import TestCase

__author__ = 'anthony bell'



class TestFeatureSelection(TestCase):
  def setUp(self):
    self.X = np.array([[i,i+1,i+2,i+3, i+4] for i in range(0, 100, 10)])
    self.y = np.array([n*10 + 1 for n in range(10)])
    self.X_sub = np.array([[31,32,33,34,35],[41,42,43,44,45]])
    self.featureSelection = FeatureSelection(lower_is_better = True, method=None,
                                             X=self.X, y=self.y, X_sub=self.X_sub,
                                             clf=RandomForestRegressor(n_estimators=2),
                                             score_func=ProblemType.logloss,
                                             problem_type='classification',
                                            col_names = ['A', 'B', 'C', 'D', 'E'])

  def test_allSelection(self):
    X, X_sub = self.featureSelection.allSelection()
    self.assertEqual(X.shape[1], 5, 'number of columns of X is not 5!')
    self.assertEqual(X_sub.shape[1], 5, 'number of columns of X_sub is not 5!')

  def test_forwardsSelection(self):
    X, X_sub = self.featureSelection.forwardsSelection()
    self.assertTrue(X is not None)
    self.assertTrue(X_sub is not None)

  def test_backwardsSelection(self):
    X, X_sub = self.featureSelection.forwardsSelection()
    self.assertTrue(X is not None)
    self.assertTrue(X_sub is not None)

  def test_featureImportancesSelection(self):
    X, X_sub = self.featureSelection.featureImportancesSelection(total_importance=0.95)
    self.assertTrue(X is not None)
    self.assertTrue(X_sub is not None)

  def test_randomSubsetSelection(self):
    X, X_sub = self.featureSelection.randomSubsetSelection(percent=0.4)
    self.assertEqual(X.shape[1], 2, 'number of columns of X is not 2!')
    self.assertEqual(X_sub.shape[1], 2, 'number of columns of X_sub is not 2!')

  def test_featureExtractionFromActualDataset(self):
    dataLoader = TrainTestDataLoader('../data/rossmann/train_100.csv', '../data/rossmann/test_100.csv', train_labels_column='Sales', test_ids_column='Id')
    dataLoader.cleanData(max_onehot_limit=200)
    X, X_sub, y = dataLoader.getTrainTestData()
    featureSelection = FeatureSelection(lower_is_better=True, method='all', X=X, y=y, clf=LogisticRegressionCV(), problem_type='classification')
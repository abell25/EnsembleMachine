from unittest import TestCase

__author__ = 'anthony bell'

import pandas as pd
from train_test_data_loader import TrainTestDataLoader
from time import time

class TestTrainTestDataLoader(TestCase):
  def setUp(self):
    pass

  def test_rossmannDataLoads(self):
    t0 = time()
    dataLoader = TrainTestDataLoader('../data/rossmann/train_100.csv', '../data/rossmann/test_100.csv', train_labels_column='Sales', test_ids_column='Id')
    dataLoader.cleanData(max_onehot_limit=200)
    X, X_sub, y = dataLoader.getTrainTestData()

    print('completed in {0} seconds!'.format(time()-t0))
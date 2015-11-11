from unittest import TestCase
from chalearn_wrapper import ChalearnWrapper
from os import path

__author__ = 'anthony bell'


class TestChalearnWrapper(TestCase):

  def setUp(self):
    self.dataset_loc = '../data/chalearn_autoML_challenge'
    self.adult_dataset_loc = path.join(self.dataset_loc, 'round0/adult')
    self.chalearnWrapper = ChalearnWrapper(self.dataset_loc)

  def test_getDatasetFiles(self):
    datasetFiles = self.chalearnWrapper.getDatasetFiles(self.adult_dataset_loc, 'adult')

  def test_getAvailableDatasets(self):
    available_datasets = self.chalearnWrapper.getAvailableDatasets()
    self.assertTrue('adult' in available_datasets)

  def test_getDataset(self):
    dataset = self.chalearnWrapper.getDataset('adult')
    self.assertTrue(dataset.has_key('train_data'))

  def test_loadDatasetPropertiesDict(self):
    self.chalearnWrapper.loadDatasetPropertiesDict(path.join(self.dataset_loc, 'round0/adult/adult_public.info'))

  def test_loadNoHeaderDataframe(self):
    df = self.chalearnWrapper.loadNoHeaderDataframe(path.join(self.dataset_loc, 'round0/adult/adult_train.data'))
    self.assertEqual(df.values.shape[0], 34190)
    self.assertEqual(df.values.shape[1], 24)
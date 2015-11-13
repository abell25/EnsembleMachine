from unittest import TestCase
from chalearn_wrapper import ChalearnWrapper
from os import path

from train_test_data_loader import TrainTestDataLoader

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

  def test_loadDataset(self):
    chalearnWrapper = ChalearnWrapper(files_loc='../data/chalearn_autoML_challenge')
    dataset = chalearnWrapper.getDataset('adult')
    dataLoader = TrainTestDataLoader(train=dataset.train_df, test=dataset.test_df, train_labels=dataset.train_labels, try_date_parse=False)
    dataLoader.cleanData(max_onehot_limit=200)
    X, X_sub, y = dataLoader.getTrainTestData()

  def test_loadAllDatasets(self):
    chalearnWrapper = ChalearnWrapper(files_loc='../data/chalearn_autoML_challenge')
    available_datasets = chalearnWrapper.getAvailableDatasets()
    available_datasets = ['dorothea', 'christine', 'jasmine', 'madeline', 'philippine', 'sylvine', 'albert', 'dilbert', 'fabert', 'robert', 'volkert']

    for dataset_name in available_datasets:
        print("loading dataset {0}".format(dataset_name))
        chalearnWrapper.get_train_test_dataset(dataset_name)
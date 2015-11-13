__author__ = 'anthony bell'

from glob import glob
from os import path
import re
import pandas as pd
import numpy as np

from train_test_data_loader import TrainTestDataLoader
from scoring.chalearn_scorer import ChalearnScorer
from domain.dataset import Dataset
from problem_type import ProblemType
from util.sparse_functions import SparseFunctions

import logging
log = logging.getLogger(__name__)

class ChalearnWrapper():
    def __init__(self, files_loc='.'):
        self.files_loc = files_loc


    def get_train_test_dataset(self, dataset_name, dataset_loc = None, max_onehot_limit=2):
        dataset = self.getDataset(dataset_name, dataset_loc)
        test_validation_df = pd.concat([dataset.test_df, dataset.validation_df])
        dataLoader = TrainTestDataLoader(train=dataset.train_df, test=test_validation_df, train_labels=dataset.train_labels, try_date_parse=False)
        dataLoader.cleanData(max_onehot_limit=max_onehot_limit)
        X, X_sub, Y = dataLoader.getTrainTestData()
        return X, X_sub, Y

    def getDatasetFiles(self, dataset_loc, dataset_name, load_dataset=True):
        files = {
            'dataset_properties': '{0}_public.info'.format(dataset_name),
            'train_data': '{0}_train.data'.format(dataset_name),
            'test_data': '{0}_test.data'.format(dataset_name),
            'validation_data': '{0}_valid.data'.format(dataset_name),
            'train_labels': '{0}_train.solution'.format(dataset_name),
            'column_types': '{0}_feat.type'.format(dataset_name)
        }

        for k in files:
            files[k] = path.join(dataset_loc, files[k])

        if load_dataset:
            col_types = self.loadColumnTypes(files['column_types'])
            dataset_features = self.loadDatasetPropertiesDict(files['dataset_properties'])

            files['column_types'] = col_types
            files['dataset_properties'] = dataset_features
            files['train_data'] = self.loadNoHeaderDataframe(files['train_data'], col_types=col_types, is_sparse=dataset_features['is_sparse']),
            files['test_data'] = self.loadNoHeaderDataframe(files['test_data'], col_types=col_types, is_sparse=dataset_features['is_sparse']),
            files['validation_data'] = self.loadNoHeaderDataframe(files['validation_data'], col_types=col_types, is_sparse=dataset_features['is_sparse']),
            files['train_labels'] = self.loadNoHeaderDataframe(files['train_labels'])

            # *dataFrame is being returned in a tuple for some reason
            files['train_data'] = files['train_data'][0] if type(files['train_data']) is tuple else files['train_data']
            files['test_data'] = files['test_data'][0] if type(files['test_data']) is tuple else files['test_data']
            files['validation_data'] = files['validation_data'][0] if type(files['validation_data']) is tuple else files['validation_data']
            files['train_labels'] = files['train_labels'][0] if type(files['train_labels']) is tuple else files['train_labels']

        return files

    def loadColumnTypes(self, column_types_file):
        if path.isfile(column_types_file):
            return np.array([l.strip() for l in open(column_types_file).readlines()])
        else:
            return None

    def getAvailableDatasets(self):
        path_glob = path.join(self.files_loc, 'round*/*')
        paths = glob(path_glob)
        return [x.split('/')[-1] for x in sorted(paths)]

    def getDataset(self, dataset_name, dataset_loc = None):
        files_loc = self.files_loc if dataset_loc == None else dataset_loc
        dataset_location = glob('{0}/*/{1}'.format(files_loc, dataset_name))[0]
        dataset_files = self.getDatasetFiles(dataset_location, dataset_name)

        problem_type = self.loadProblemTypeFromPropertiesDict(dataset_files['dataset_properties'])

        return Dataset(train_df=dataset_files['train_data'],
                       test_df=dataset_files['test_data'],
                       train_labels=dataset_files['train_labels'],
                       problem_type=problem_type,
                       validation_df=dataset_files['validation_data'])

    def loadDatasetPropertiesDict(self, dataset_properties_file):
        file_data = open(dataset_properties_file).read()
        lines = file_data.splitlines()
        tuples = [re.split(' *= *', line) for line in lines]
        d = {x[0]: x[1] for x in tuples}
        for key in d:
            val = d[key]
            if val[0] == val[-1] and val[0] in ['"', "'"]:
                d[key] = val[1:-1]
            else:
                d[key] = int(val)
                if key.startswith('has_') or key.startswith('is_'):
                    d[key] = True if d[key] == 1 else False

        return d

    def loadProblemTypeFromPropertiesDict(self, dataset_properties_dict):
        chalearn_scoring_functions = ChalearnScorer(task=dataset_properties_dict['task'])
        chalearn_scorer = chalearn_scoring_functions.get_scorer(dataset_properties_dict['metric'])

        problem_type = ProblemType(metric = dataset_properties_dict['metric'],
                                   scorer = chalearn_scorer,
                                   is_classification = 'classification' in dataset_properties_dict['task'],
                                   is_binary = 'binary' in dataset_properties_dict['task'],
                                   is_multilabel = 'multilabel' in dataset_properties_dict['task'],
                                   is_large_scale = dataset_properties_dict['is_sparse'],
                                   time_budget = int(dataset_properties_dict['time_budget']))

        return problem_type


    def loadNoHeaderDataframe(self, filename, sep=' ', col_types=None, is_sparse=False):
        if is_sparse:
            return self.loadSparseDataframe(filename, sep)

        df = pd.read_csv(filename, sep=sep, header=None, engine='python')
        df.columns = df.columns.astype(str)

        # The csv files have a stupid space at the end >:0.
        # We are just going to throw away the extra column generated by pandas.
        extra_col = df.columns[-1]
        log.debug("dropping {0}".format(extra_col))
        df.drop(extra_col, axis=1, inplace=True)

        if col_types is not None:
            if len(col_types) != len(df.columns):
                log.error("number of col_types does not match number of columns found!: {0)/{1}, file: {2}".format(len(col_types), len(df.columns), filename))
            else:
                factors_columns = df.columns[col_types == 'Categorical']
                df[factors_columns] = df[factors_columns].astype(str)


        return df

    def loadSparseDataframe(self, filename, sep):
        return SparseFunctions.read_sparse_csv(filename, sep=sep, use_df=True)


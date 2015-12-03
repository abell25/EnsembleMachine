__author__ = 'anthony bell'

import pandas as pd
from categorical_feature_extractor import CategoricalFeatureExtraction
from date_feature_extractor import DateFeatureExtractor
from dateutil import parser
from time import time
from domain.ml_problem import MLproblem
from domain.DataSet import DataSet
from problem_type import ProblemType

import logging
log = logging.getLogger(__name__)

class TrainTestDataLoader():
    def __init__(self, train, test, train_labels=None, train_labels_column=None, test_ids=None, test_ids_column=None, copy=True, sep=',', header='infer', try_date_parse=True):
        ''' Loads the raw data into the autoML pipeline using: imputation, date feature extraction, one-hot and ordinal
            encoding of factors.

            train:               str: train filename, array-like: training data
            test:                str: train filename, array-like: testing data
            train_labels:        str: training labels filename, array-like: training labels as array
            train_labels_column: training labels on train data as column specified
            test_ids:            str: testids filename, array-like test ids as array
            test_ids_column:     test ids on train data as column specified
            copy:                Whether the data should be copied or modified in-place
        '''
        self.train = train
        self.test = test

        self.try_date_parse = try_date_parse

        if type(train) is str:
            log.info('reading train file {0}'.format(train))
            self.train_df = pd.read_csv(train, sep=sep, header=header)
        else:
            self.train_df = train.copy() if copy else train

        if type(test) is str:
            log.info('reading test file {0}'.format(test))
            self.test_df = pd.read_csv(test, sep=sep, header=header)
        else:
            self.test_df = test.copy() if copy else test

        if train_labels is not None:
            if type(train_labels) is str:
                log.info('loading train_labels from {0}'.format(train_labels))
                self.train_labels = pd.read_csv(train_labels, sep=sep, header=header).values[:,0]
            else:
                log.info('loading train_labels from array.')
                self.train_labels = train_labels
        elif train_labels_column:
            log.info('extracting train labels from train_df[{0}]'.format(train_labels_column))
            self.train_labels = self.train_df[train_labels_column]
            self.train_df.drop(train_labels_column, axis=1, inplace=True)
        else:
            raise ValueError("Either train_labels or train_labels_column must be specified!")

        if test_ids:
            self.test_ids = pd.read_csv(test_ids, sep=sep, header=header).values[:,0]
        elif test_ids_column:
            self.test_ids = self.test_df[test_ids_column].values
            self.test_df.drop(test_ids_column, axis=1, inplace=True)
        else:
            self.test_ids = None

    def cleanData(self, max_onehot_limit=100, max_ordinal_limit=10000, numeric_imputed_value=-99999999):
        time_start = time()
        dateFeatureExtractor = DateFeatureExtractor()
        categoricalFeatureExtractor = CategoricalFeatureExtraction()

        train_df, test_df = self.train_df, self.test_df

        shared_columns = categoricalFeatureExtractor.getSharedColumns([train_df, test_df])

        cols, col_types = train_df.columns.values, train_df.dtypes.values
        for col, col_type in zip(cols, col_types):
            logging.debug('processing column: {0} of type {1}'.format(col, col_type))
            if col not in shared_columns:
                pass
            elif col_type == 'object':
                if self.try_date_parse and DateFeatureExtractor.testIfColumnIsDate(train_df[col], num_tries=5):
                    logging.info('loading {0} as a date'.format(col))
                    dateFeatureExtractor.createFeaturesFromDateColumns(train_df, [col])
                    dateFeatureExtractor.createFeaturesFromDateColumns(test_df, [col])
                else:
                    num_categories = len(set(train_df[col].values).union(set(test_df[col].values)))
                    if num_categories < max_onehot_limit and num_categories > 2:
                        logging.info('loading {0} as one-hot (with {1} categories)'.format(col, num_categories))
                        categoricalFeatureExtractor.convertColumnsToOneHot([train_df, test_df], [col])
                    elif num_categories > max_ordinal_limit:
                        log.info('deleting ordinal column {0} with {1} categories'.format(col, num_categories))
                        train_df.drop(col, axis=1, inplace=True)
                        test_df.drop(col, axis=1, inplace=True)
                    else:
                        logging.info('loading {0} as an ordinal variable (with {1} categories)'.format(col, num_categories))
                        categoricalFeatureExtractor.convertColumnsToOrdinal([train_df, test_df], [col])

            elif col_type == 'int' or col_type == 'float':
                train_df[col] = train_df[col].fillna(numeric_imputed_value)
                test_df[col] = test_df[col].fillna(numeric_imputed_value)
            else:
                log.error('pandas type "{0}" is not known!'.format(col_type))

        categoricalFeatureExtractor.removeUnsharedColumns([train_df, test_df])
        log.info('Completed in {0} seconds!'.format(int(time() - time_start)))

    def getTrainTestData(self):
        X = self.train_df.values.astype(float)
        X_sub = self.test_df.values.astype(float)
        y = self.train_labels
        return X, X_sub, y

    def getMLproblem(self, metric, scorer, is_classification, is_binary, is_multilabel=False, is_large_scale=False, time_budget=None):
        problemType = ProblemType(metric=metric,
                                  scorer=scorer,
                                  is_classification=is_classification,
                                  is_binary=is_binary,
                                  is_multilabel=is_multilabel,
                                  is_large_scale=is_large_scale,
                                  time_budget=time_budget
                                  )
        X, X_sub, y = self.getTrainTestData()
        dataset = DataSet(X=X, X_sub=X_sub, y=y)
        return MLproblem(dataset=dataset, problemType=problemType)

                #raise Exception "I don't know what pandas type {0} is!".format(col_type)
    def saveCleanData(self, dataset_name='TEMP'):
        self.train_df.to_csv('{0}_train.csv'.format(dataset_name), index=False)
        self.test_df.to_csv('{0}_test.csv'.format(dataset_name), index=False)
        pd.DataFrame({'y': self.train_labels}).to_csv('{0}_trainLabels.csv'.format(dataset_name), index=False)
        pd.DataFrame({'id': self.test_ids}).to_csv('{0}_testIds'.format(dataset_name), index=False)
__author__ = 'anthony bell'

import pandas as pd
from categorical_feature_extractor import CategoricalFeatureExtraction
from date_feature_extractor import DateFeatureExtractor
from dateutil import parser

import logging
log = logging.getLogger(__name__)

class TrainTestDataLoader():
    def __init__(self, train, test, train_labels=None, train_labels_column=None, test_ids=None, test_ids_column=None, sep=',', header='infer', try_date_parse=True):
        ''' Loads the raw data into the autoML pipeline using: imputation, date feature extraction, one-hot and ordinal
            encoding of factors.

            train: train filename (assumed csv)
            test: test filename   (assumed csv)
            train_labels: training labels filename.  Either `train_labels` or `train_labels_column` must be used.
            train_labels_column: column of train csv that contains the labels, if any.
        '''
        self.train = train
        self.test = test

        if type(train) is str:
            log.info('reading train file {0}'.format(train))
            self.train_df = pd.read_csv(train, sep=sep, header=header)
        else:
            self.train_df = train

        if type(train) is str:
            log.info('reading test file {0}'.format(test))
            self.test_df = pd.read_csv(test, sep=sep, header=header)
        else:
            self.test_df = test

        if train_labels:
            log.info('loading train_labels from {0}'.format(train_labels))
            self.train_labels = pd.read_csv(train_labels, sep=sep, header=header).values[:,0]
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

    def cleanData(self, max_onehot_limit=100, numeric_imputed_value=-99999999):
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
                if DateFeatureExtractor.testIfColumnIsDate(train_df[col], num_tries=5):
                    logging.info('loading {0} as a date'.format(col))
                    dateFeatureExtractor.createFeaturesFromDateColumns(train_df, [col])
                    dateFeatureExtractor.createFeaturesFromDateColumns(test_df, [col])
                else:
                    num_categories = len(set(train_df[col].values).union(set(test_df[col].values)))
                    if num_categories < max_onehot_limit:
                        logging.info('loading {0} as one-hot (with {1} categories)'.format(col, num_categories))
                        categoricalFeatureExtractor.convertColumnsToOneHot([train_df, test_df], [col])
                    else:
                        logging.info('loading {0} as an ordinal variable (with {1} categories)'.format(col, num_categories))
                        categoricalFeatureExtractor.convertColumnsToOrdinal([train_df, test_df], [col])

            elif col_type == 'int' or col_type == 'float':
                train_df[col] = train_df[col].fillna(numeric_imputed_value)
            else:
                log.error('pandas type "{0}" is not known!'.format(col_type))

        categoricalFeatureExtractor.removeUnsharedColumns([train_df, test_df])

                #raise Exception "I don't know what pandas type {0} is!".format(col_type)
    def saveCleanData(self, dataset_name='TEMP'):
        self.train_df.to_csv('{0}_train.csv'.format(dataset_name), index=False)
        self.test_df.to_csv('{0}_test.csv'.format(dataset_name), index=False)
        pd.DataFrame({'y': self.train_labels}).to_csv('{0}_trainLabels.csv'.format(dataset_name), index=False)
        pd.DataFrame({'id': self.test_ids}).to_csv('{0}_testIds'.format(dataset_name), index=False)
__author__ = 'anthony bell'

from unittest import TestCase
from date_feature_extractor import DateFeatureExtractor
import pandas as pd

class TestDateFeatureExtractor(TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame({'A': ['2015-12-28', '2015-12-29', '2015-12-30']})
        self.A_colnames = ['A_%s' % x for x in ['day', 'month', 'year', 'dayofweek', 'dayofyear']]


    def test_createFeaturesFromDateColumns(self):
        dfe = DateFeatureExtractor()
        self.df1['A'] = pd.to_datetime(self.df1['A'])
        dfe.createFeaturesFromDateColumns(self.df1, ['A'])

        self.assertTrue('A' not in self.df1.columns, 'column \'A\' should be deleted by default.')

        for col in self.A_colnames:
            self.assertTrue(col in self.A_colnames, 'column \'{0}\' should have been added!'.format(col))


    def test_createFeaturesFromDateColumns_handlesStringDates(self):
        dfe = DateFeatureExtractor()
        dfe.createFeaturesFromDateColumns(self.df1, ['A'])

        self.assertTrue('A' not in self.df1.columns, 'column \'A\' should be deleted by default.')

        for col in self.A_colnames:
            self.assertTrue(col in self.A_colnames, 'column \'{0}\' should have been added!'.format(col))
from unittest import TestCase
from categorical_feature_extractor import CategoricalFeatureExtraction
import pandas as pd
import numpy as np

__author__ = 'anthony bell'


class TestCategoricalFeatureExtraction(TestCase):
    @classmethod
    def setUp(self):
        self.df1 = pd.DataFrame({'animal': ['cat', 'dog', 'horse','dog'], 'age': [11, 22, 11, 11]})
        self.df2 = pd.DataFrame({'animal': ['horse', 'monkey', 'cow'], 'age': [11, 33, 33]})
        self.animal_onehot_columns = ['animal=cat', 'animal=cow', 'animal=dog', 'animal=horse', 'animal=monkey']

    def test_convertColumnsToOneHot(self):
        cfe = CategoricalFeatureExtraction()
        cfe.convertColumnsToOneHot([self.df1, self.df2], ['animal'])

        for df in [self.df1, self.df2]:
            for onehot_col in self.animal_onehot_columns:
                self.assertTrue(onehot_col in df.columns, 'one-hot column \'{0}\' is missing!'.format(onehot_col))

            self.assertTrue('animal' not in df.columns, 'animal column should have been deleted!')
            self.assertTrue('age' in df.columns, 'age column should not have been deleted!')


    def test_convertColumnsToOrdinal(self):
        cfe = CategoricalFeatureExtraction()
        cfe.convertColumnsToOrdinal([self.df1, self.df2], ['age'])

        for df in [self.df1, self.df2]:
            print df.columns.values
            self.assertTrue('animal' in df.columns, 'animal column should not have been deleted!')
            self.assertTrue('age' in df.columns, 'age column should not have been deleted!')

        self.assertEqual(self.df1['age'][0], 0, '11 should map to 0!')
        self.assertEqual(self.df1['age'][1], 1, '22 should map to 1!')
        self.assertEqual(self.df1['age'][2], 0, '11 should map to 0!')
        self.assertEqual(self.df1['age'][3], 0, '11 should map to 0!')
        self.assertEqual(self.df2['age'][0], 0, '11 should map to 0!')
        self.assertEqual(self.df2['age'][1], 2, '33 should map to 2!')
        self.assertEqual(self.df2['age'][2], 2, '33 should map to 2!')


    def test_fillNAs(self):
        df = pd.DataFrame({'A': [11, 22, np.nan, 44]})
        cfe = CategoricalFeatureExtraction()
        cfe.fillNAs(df, ['A'])
        self.assertEqual(df['A'][0], 11, 'Index 0 should be unchanged!')
        self.assertEqual(df['A'][1], 22, 'Index 1 should be unchanged!')
        self.assertTrue(df['A'][2] <= -99999999, 'Index 2 should be a large negative number!')
        self.assertEqual(df['A'][3], 44, 'Index 3 should be unchanged!')
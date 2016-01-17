from unittest import TestCase
from categorical_feature_extractor import CategoricalFeatureExtraction
import pandas as pd
import numpy as np

__author__ = 'anthony bell'


class TestCategoricalFeatureExtraction(TestCase):
    @classmethod
    def setUp(self):
        self.df1 = pd.DataFrame({'animal': ['cat', 'dog', 'horse','dog'], 'age': [11, 22, 11, 11], 'type': ['house','house','farm','house']})
        self.df2 = pd.DataFrame({'animal': ['horse', 'monkey', 'cow'], 'age': [11, 33, 33], 'type': ['farm', 'wild', 'farm']})
        self.animal_onehot_columns = ['animal=cat', 'animal=cow', 'animal=dog', 'animal=horse', 'animal=monkey']
        self.type_onehot_columns = ['type=house', 'type=farm', 'type=wild']

    def test_convertColumns_thresholdShouldDetermineIfColumnIsOneHotEncoded(self):
        CategoricalFeatureExtraction.convertColumns([self.df1, self.df2], ['animal', 'type'], one_hot_threshold=3)

        #animal should be ordinal, #type should be one-hot
        for df in [self.df1, self.df2]:
            for onehot_col in self.type_onehot_columns:
                self.assertTrue(onehot_col in df.columns)
            self.assertTrue('type' not in df.columns, 'type column should have been deleted')
            self.assertTrue('animal' in df.columns)

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
            print(df.columns.values)
            self.assertTrue('animal' in df.columns, 'animal column should not have been deleted!')
            self.assertTrue('age' in df.columns, 'age column should not have been deleted!')

        self.assertEqual(self.df1['age'][0], 0, '11 should map to 0!')
        self.assertEqual(self.df1['age'][1], 1, '22 should map to 1!')
        self.assertEqual(self.df1['age'][2], 0, '11 should map to 0!')
        self.assertEqual(self.df1['age'][3], 0, '11 should map to 0!')
        self.assertEqual(self.df2['age'][0], 0, '11 should map to 0!')
        self.assertEqual(self.df2['age'][1], 2, '33 should map to 2!')
        self.assertEqual(self.df2['age'][2], 2, '33 should map to 2!')


    def test_removeUnsharedColumns(self):
        cfe = CategoricalFeatureExtraction()
        df1 = pd.DataFrame({'A': [1,2,3], 'B': [2,3,4], 'C': [3,4,5]})
        df2 = pd.DataFrame({'A': [1,2,3], 'B': [2,3,4]})
        df3 = pd.DataFrame({'A': [1,2,3], 'C': [2,3,4]})
        df4 = pd.DataFrame({'A': [1,2,3], 'D': [2,3,4]})
        cfe.removeUnsharedColumns([df1, df2, df3, df4])

        self.assertEqual(len(df1.columns), 1, '{0} columns found for df1 instead of 1!'.format(len(df1.columns)))
        self.assertEqual(len(df2.columns), 1, '{0} columns found for df2! instead of 1'.format(len(df2.columns)))
        self.assertEqual(len(df3.columns), 1, '{0} columns found for df3! instead of 1'.format(len(df3.columns)))
        self.assertEqual(len(df4.columns), 1, '{0} columns found for df4! instead of 1'.format(len(df4.columns)))

        self.assertEqual(df1.columns[0], 'A', 'column for df1 didn\'t contain \'A\'!')
        self.assertEqual(df2.columns[0], 'A', 'column for df2 didn\'t contain \'A\'!')
        self.assertEqual(df3.columns[0], 'A', 'column for df3 didn\'t contain \'A\'!')
        self.assertEqual(df4.columns[0], 'A', 'column for df4 didn\'t contain \'A\'!')


    def test_fillNAs(self):
        df = pd.DataFrame({'A': [11, 22, np.nan, 44]})
        cfe = CategoricalFeatureExtraction()
        cfe.fillNAs(df, ['A'])
        self.assertEqual(df['A'][0], 11, 'Index 0 should be unchanged!')
        self.assertEqual(df['A'][1], 22, 'Index 1 should be unchanged!')
        self.assertTrue(df['A'][2] <= -99999999, 'Index 2 should be a large negative number!')
        self.assertEqual(df['A'][3], 44, 'Index 3 should be unchanged!')
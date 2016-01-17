__author__ = 'anthony bell'

import logging
log = logging.getLogger(__name__)


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

class CategoricalFeatureExtraction:
    def __init__(self):
        pass

    @staticmethod
    def convertColumns(dfs, cols, one_hot_threshold=10):
        if type(cols) is str:
            cols = [cols]

        for col in cols:
            values = set([s for df in dfs for s in df[col].values])

            if len(values) > one_hot_threshold:
                CategoricalFeatureExtraction.convertColumnsToOrdinal(dfs, [col])
            else:
                CategoricalFeatureExtraction.convertColumnsToOneHot(dfs, [col])


    def convertColumnsToOneHot(self, dfs, cols):
        if type(cols) is str:
            cols = [cols]

        CategoricalFeatureExtraction.convertColumnsToOneHot(dfs, cols)


    def convertColumnsToOrdinal(self, dfs, cols):
        if type(cols) is str:
            cols = [cols]

        CategoricalFeatureExtraction.convertColumnsToOrdinal(dfs, cols)

    def getSharedColumns(self, dfs):
        shared_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            shared_columns = shared_columns.intersection(df.columns)
        return shared_columns

    def removeUnsharedColumns(self, dfs):
        shared_columns = self.getSharedColumns(dfs)

        for df in dfs:
            cols_to_drop = list(set(df.columns) - set(shared_columns))
            df.drop(cols_to_drop, axis=1, inplace=True)

    @staticmethod
    def fillNAs(df, cols, numeric_na_value=-99999999, str_na_value='NA'):
        for col in cols:
            if df[col].dtype in ['int', 'float']:
                df[col] = df[col].fillna(numeric_na_value)
            else:
                df[col] = df[col].fillna(str_na_value)


    @staticmethod
    def convertColumnsToOneHot(dfs, cols):
        all_values = []
        for df in dfs:
            CategoricalFeatureExtraction.fillNAs(df, cols)
            all_values += df[cols].to_dict(orient='records')

        vec = DictVectorizer()
        vec.fit(all_values)

        for df in dfs:
            oneHotData = vec.transform(df[cols].to_dict(orient='records')).toarray()
            oneHotColumns = vec.get_feature_names()
            df[oneHotColumns] = pd.DataFrame(data=oneHotData, columns=oneHotColumns).astype(int)
            df.drop(cols, axis=1, inplace=True)

    @staticmethod
    def convertColumnsToOrdinal(dfs, cols):
        all_values = {c:set() for c in cols }
        for df in dfs:
            CategoricalFeatureExtraction.fillNAs(df, cols)
            for col in cols:
                new_values = set(df[col].values)
                all_values[col] = all_values[col].union(new_values)

        for col in cols:
            lbl = LabelEncoder()
            lbl.fit(list(all_values[col]))

            for df in dfs:
                df[col] = lbl.transform(df[col].values).astype(int)



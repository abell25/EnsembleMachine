__author__ = 'anthony bell'

import pandas as pd

class DateFeatureExtractor:
    def __init__(self):
        pass

    def createFeaturesFromDateColumns(self, df, cols, remove_date_column=True):
        if type(cols) is str:
            col = cols
            return DateFeatureExtractor.createFeaturesFromDateColumn(df, col, remove_date_column)

        for col in cols:
            df = DateFeatureExtractor.createFeaturesFromDateColumn(df, col, remove_date_column)

        return df

    @staticmethod
    def createFeaturesFromDateColumn(df, col, remove_date_column=True):
        if col not in df.columns:
            raise 'Column {0} is not in the dataframe!'.format(col)

        col_type = type(df[col].values[0])
        if col_type is str:
            #date column is still a string so we need to parse it!
            df[col] = pd.to_datetime(df[col])

        df['{0}_day'.format(col)] = df[col].apply(lambda x: x.day)
        df['{0}_month'.format(col)] = df[col].apply(lambda x: x.month)
        df['{0}_year'.format(col)] = df[col].apply(lambda x: x.year)
        df['{0}_dayofweek'.format(col)] = df[col].apply(lambda x: x.dayofweek)
        df['{0}_dayofyear'.format(col)] = df[col].apply(lambda x: x.dayofyear)

        if remove_date_column:
            df.drop(col, axis=1, inplace=True)

        return df
__author__ = 'anthony bell'

import pandas as pd
from dateutil import parser

import logging
log = logging.getLogger(__name__)


class DateFeatureExtractor:
    def __init__(self):
        pass

    def createFeaturesFromDateColumns(self, df, cols, remove_date_column=True):
        '''
        Creates the common features extracted from dates (e.g. year, month, day, day of week, day of year, etc.)
        :param df: the dateframe
        :param cols: [str or list of str] columns to parse for dates
        :param remove_date_column: whether to remove the date column after the date features have been extracted.
        :return: the dataframe df
        '''
        if type(cols) is str:
            col = cols
            return DateFeatureExtractor.createFeaturesFromDateColumn(df, col, remove_date_column)

        for col in cols:
            df = DateFeatureExtractor.createFeaturesFromDateColumn(df, col, remove_date_column)

        return df

    def convertDateColumns(self, df, cols=None, create_date_features=True, num_tries=5):
        '''
        Converts all columns on dataframe to datetime that can be converted
        :param df: the dataframe
        :param num_tries: number of elements to try to parse before giving up for each column.
        :return: the dataframe df
        '''
        cols = df.columns if cols is None else cols
        object_cols = cols.values[df.dtypes.values == 'object']
        date_cols = [c for c in object_cols if DateFeatureExtractor.testIfColumnIsDate(df[c], num_tries)]

        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], coerce=True, infer_datetime_format=True)

                if create_date_features:
                    DateFeatureExtractor.createFeaturesFromDateColumn(df, col, remove_date_column=True)

            except ValueError:
                #if we can't parse the date, just let it become a factor
                pass

        return df


    @staticmethod
    def createFeaturesFromDateColumn(df, col, remove_date_column=True):
        if col not in df.columns:
            raise 'Column {0} is not in the dataframe!'.format(col)

        col_type = type(df[col].values[0])
        if col_type is str:
            #date column is still a string so we need to parse it!
            df[col] = pd.to_datetime(df[col], coerce=True, infer_datetime_format=True)

        date_fields = list(zip(*df[col].apply(lambda x: (x.day, x.month, x.year, x.dayofweek, x.dayofyear) if x is not pd.NaT else -99).values))
        date_cols = ['{0}_{1}'.format('Date', s) for s in ['day', 'month', 'year', 'dayofweek', 'dayofyear']]

        for k in range(len(date_cols)):
            df[date_cols[k]] = date_fields[k]

        if remove_date_column:
            df.drop(col, axis=1, inplace=True)

        return df

    @staticmethod
    def testIfColumnIsDate(series, num_tries=4):
        if series.dtype != 'object':
            return False

        try:
            vals = set()
            for val in series:
                vals.add(val)
                if len(vals) > num_tries:
                    break

            for val in list(vals):
                try:
                    if type(val) is not str:
                        continue

                    parser.parse(val)
                    return True
                except:
                    pass

            return False
        except:
            log.info('trying to parse date for testIfColumnIsDate failed, returning false.')
            return False

    @staticmethod
    def partialDateParser(date_str):
        pass

    @staticmethod
    def partialDatetimeIntervalParser(date_str):
        pass
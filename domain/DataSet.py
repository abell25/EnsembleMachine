__author__ = 'anthony bell'

class DataSet():
    def __init__(self, X=None, X_sub=None, y=None, train_df = None, test_df = None, validation_df = None):
        self.X = X
        self.X_sub = X_sub
        self.y = y
        self.train_df = train_df
        self.test_df = test_df

    def getDataset(self):
        if self.X is not None:
            return self.X, self.X_sub, self.y
        else:
            return self.train_df, self.test_df, self.y

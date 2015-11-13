__author__ = 'anthony bell'

import numpy as np
from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score

class ScoringFunctions():


    def __init__(self):
        self.scorer_mapping = {
            'logloss': {'scorer': self.logloss, 'lower_is_better': True}
        }


    @staticmethod
    def F1(y, y_pred):
        return f1_score(y, y_pred)

    @staticmethod
    def logloss(y, y_pred):
        return log_loss(y, y_pred)

    @staticmethod
    def MSE(y, y_pred):
        return mean_squared_error(y, y_pred)

    @staticmethod
    def RMSE(y, y_pred):
        return mean_squared_error(y, y_pred) ** 0.5

    @staticmethod
    def AUC(y, y_pred):
        return roc_auc_score(y, y_pred)

    @staticmethod
    def RMSPE(y, y_pred):
        '''
        source from kaggle comp: https://www.kaggle.com/paso84/rossmann-store-sales/xgboost-in-python-with-rmspe/files
        '''
        w = np.zeros(y.shape, dtype=float)
        ind = y != 0
        w[ind] = 1./(y[ind]**2)

        return np.sqrt(np.mean( w * (y - y_pred)**2 ))

    @staticmethod
    def RMSLE(y, y_pred):
        return ScoringFunctions.RMSE(np.log(y + 1), np.log(y_pred + 1))


    def getScorer(self, scorer_name):
        if scorer_name not in self.scorer_mapping:
            raise NameError('scorer {0} does not exist in the dictionary of available scorers!'.format(scorer_name))
        else:
            return self.scorer_mapping[scorer_name]


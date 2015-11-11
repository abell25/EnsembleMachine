__author__ = 'anthony bell'

import numpy as np
import scipy as sp
from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score

def ScoringFunctions():

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

    scorer_mapping = {
        'auc_metric': AUC,
        'bac_metric': F1,
        'f1_metric': F1,
        'pac_metric': F1,
        'r2_metric': F1,
    }

    @staticmethod
    def getScorer(scorer_name):
        if not scorer_mapping.has_key(scorer_name):
            raise NameError('scorer {0} does not exist in the dictionary of available scorers!'.format(scorer_name))
        else:
            return scorer_mapping[scorer_name]
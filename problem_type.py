__author__ = 'anthony bell'

from sklearn.metrics import f1_score, log_loss, mean_squared_error, roc_auc_score
import numpy as np

class ProblemType():
    """
    object for determining what kind of machine learning problem is being solved.
    """
    ProblemTypes = ['classification', 'regression', 'multiclass classification']
    Metrics = ['F1', 'logloss', 'MSE', 'RMSE', 'AUC', 'RMSPE']
    LowerIsBetter = [False, True, True, True, False, True]

    def __init__(self, problem_type, metric):
        self.problem_type = problem_type
        self.metric = metric


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


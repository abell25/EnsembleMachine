__author__ = 'anthony bell'

#small regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    import xgboost as xgb
except Exception, e:
    print('failed loading xgb module!: {0}'.format(e))

#small classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#large regression
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.svm import LinearSVR

#large classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier, PassiveAggressiveClassifier


class ModelLibrary():
    """ The model libraries used by the model generator
    """
    
    task_types = ['regression', 'classifier']
    model_scale = ['small', 'large']

    model_libraries = {
        'small regression': {
	    'xgb': {
	        'model': xgb.XGBRegression,
		'parameter_ranges': {
                    'max_depth': range(1, 9),
	        }
	    }
	}
    }

    
    def __init__(self, modelType):
       self.modelType = modelType

    def getModelLibrary(self):
    #todo: get model from library
	pass



class Model():
    def __init__(self, name, model, parameter_ranges):
        self.name = name
    #self.model = model
    #self.parameter_ranges = parameter_ranges
    #self.scores = []

    def initModel(self, parameters=None):
        pass
        #model = self.model()
	    #TODO: set parameter valiues for model
	
    
    

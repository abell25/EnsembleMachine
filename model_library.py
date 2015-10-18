__author__ = 'anthony bell'

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
                    'max_depth': range(1, 9)
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
	self.model = model
	self.parameter_ranges = parameter_ranges
	self.scores = []

    def initModel(self, parameters=None):
	model = self.model()
	#TODO: set parameter valiues for model
	
    
    

__author__ = 'anthony bell'


import numpy as np
import random


#small regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


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

    model_libraries = {
        'small regression': {
            'GradientBoostingRegressor': {
                'model': GradientBoostingRegressor,
                'parameter_ranges': {
                    'loss': ['ls'],
                    'learning_rate': np.logspace(-3, 0, 1000),
                    'n_estimators': np.linspace(1, 2000, 1000, dtype=int),
                    'max_depth': np.linspace(1, 12, 12, dtype=int),
                    'subsample': np.linspace(0.5, 1.0, 50)

                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor,
                'parameter_ranges': {
                    'max_depth': [None] + list(np.linspace(1, 80, 80, dtype=int)),
                    'n_estimators': np.linspace(1, 1000, 1000, dtype=int),
                    'min_samples_split': range(2, 21),
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor,
                'parameter_ranges': {
                    'n_neighbors': range(1, 61),
                    'weights': ['uniform', 'distance'],
                    'p': [1,2],
                    'algorithm': ['ball_tree', 'kd_tree', 'auto'],
                    'leaf_size': range(10,51)
                }
            }

	    },
        'large regression': {
            'LinearRegression': {
                'model': LinearRegression,
                'parameter_ranges': {
                    'fit_intercept': [True],
                    'normalize': [False]
                }
            },
            'LinearSVR': {
                'model': LinearSVR,
                'parameter_ranges': {
                    'C': np.logspace(-3, 3, 1000),
                    'fit_intercept': [True]
                }
            },
            'SGDRegressor': {
                'model': SGDRegressor,
                'parameter_ranges': {
                    'penalty': ['l1', 'l2'],
                    'alpha': np.logspace(-6, -1, 1000),
                    'l1_ratio': np.logspace(-3, -0.5, 1000),
                    'loss': ['squared_loss']
                }
            }
        },
        'small classification': {
            'GradientBoostingClassifier': {
                'model': GradientBoostingClassifier,
                'parameter_ranges': {
                    'loss': ['deviance'],
                    'learning_rate': np.logspace(-3, 0, 1000),
                    'n_estimators': np.linspace(1, 2000, 1000, dtype=int),
                    'max_depth': np.linspace(1, 12, 12, dtype=int),
                    'subsample': np.linspace(0.5, 1.0, 50)

                }
            },
            'RandomForestClassifier': {
                'model': RandomForestClassifier,
                'parameter_ranges': {
                    'max_depth': [None] + list(np.linspace(1, 80, 80, dtype=int)),
                    'n_estimators': np.linspace(1, 1000, 1000, dtype=int),
                    'min_samples_split': range(2, 21),
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'KNeighborsClassifier': {
                'model': KNeighborsClassifier,
                'parameter_ranges': {
                    'n_neighbors': range(1, 61),
                    'weights': ['uniform', 'distance'],
                    'p': [1,2],
                    'algorithm': ['ball_tree', 'kd_tree', 'auto'],
                    'leaf_size': range(10,51)
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'parameter_ranges': {
                    'penalty':['l1', 'l2'],
                    'C':np.logspace(-2, 2, 1000),
                    'class_weight':[None],
                    'fit_intercept':[True]
                }
            }
        },
        'large classification': {
            'LogisticRegression': {
                'model': LogisticRegression,
                'parameter_ranges': {
                    'penalty':['l1', 'l2'],
                    'C':np.logspace(-2, 2, 1000),
                    'class_weight':[None],
                    'fit_intercept':[True]
                }
            },
            'SGDClassifier': {
                'model': SGDClassifier,
                'parameter_ranges': {
                    'penalty': ['l1', 'l2'],
                    'alpha': np.logspace(-6, -1, 1000),
                    'l1_ratio': np.logspace(-3, -0.5, 1000),
                    'loss': ['hinge']
                }
            }
        }
    }

    
    def __init__(self, is_classification = False, is_binary = False, is_large_scale = False):
        problem_size = 'large' if is_large_scale else 'small'
        task_type = 'classification' if is_classification else 'regression'
        self.model_library = self.model_libraries['{0} {1}'.format(problem_size, task_type)]

    def getModelLibrary(self):
        return self.model_library

    def generateModel(self, model_selection_method='random', parameter_selection_method='random'):
        model_name = self.pickModel(model_selection_method)
        uninitialized_model = self.model_library[model_name]

        ModelClass, parameter_ranges = uninitialized_model['model'], uninitialized_model['parameter_ranges']
        chosen_parameters = self.pickParameters(parameter_ranges, parameter_selection_method)
        model = ModelClass(**chosen_parameters)
        return model


    def pickModel(self, model_selection_method):
        if model_selection_method is 'random':
            model_name = random.choice(self.model_library.keys())
            return model_name
        else:
            raise Exception('selectionMethod "{0}" is not a valid model selection method!'.format(model_selection_method))

    def pickParameters(self, parameter_ranges, parameter_selection_method):
        chosen_parameter_values = {}
        for param in parameter_ranges:
            if parameter_selection_method is 'random':
                chosen_parameter_values[param] = random.choice(parameter_ranges[param])
            else:
                raise Exception('selectionMethod {0} is not a valid parameter selection method')

        return chosen_parameter_values
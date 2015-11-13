__author__ = 'anthony bell'

class ModelTuner():
    def __init__(self, model_library, X, y, X_sub):
        self.model_library = model_library
        self.X = X
        self.y = y
        self.X_sub = X_sub

    def get_available_methods(self):
        return ['random_configuration', 'randomized_search', 'grid_search']

    def tune_model(self, model_selection_method='random',
                   parameter_selection_method='random',
                   tuning_method='random_configuration',
                   model=None):
        if model is None:
            model = self.model_library.generateModel(model_selection_method=model_selection_method,
                                                     parameter_selection_method=parameter_selection_method)

        if tuning_method == 'random_configuration':
            model.fit(self.X, self.y)

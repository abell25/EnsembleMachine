__author__ = 'anthony bell'

class EnsembleMachine():
    def __init__(self, dataset):
        """ The EnsembleMachine runs on a dataset to generate the best submission in the time limit it can.

        :param dataset: Dataset class that contains data and all the information about the dataset
        :return:
        """
        self.X = dataset.X
        self.y = dataset.y
        self.X_sub = dataset.X_sub

    def run(self):
        pass

    def run_iteration(self, feature_selection, model_library, ensembler, agent):
        feature_options = feature_selection.getOptions()
        model_options = model_library.getOptions()
        ensemble_options = ensembler.getOptions()


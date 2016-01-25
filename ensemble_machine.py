__author__ = 'anthony bell'

from time import time
from train_test_data_loader import TrainTestDataLoader

class EnsembleMachine():
    def __init__(self, mlProblem):
        """ The EnsembleMachine runs on a dataset to generate the best submission in the time limit it can.

        :param dataset: Dataset class that contains data and all the information about the dataset
        :return:
        """
        self.mlProblem = mlProblem
        self.dataset = mlProblem.getDataset()
        self.problemType = mlProblem.getProblemType()
        self.scorer = self.problemType.scorer


    def run(self, max_seconds=None):
        start_time = time()

        X, X_sub, y = self.dataset.getDataset()
        #X_train, X_test, y_train, y_test =

        while (time() - start_time) < max_seconds:
            self.run_iteration()

    def run_iteration(self, feature_selection, model_library, ensembler, agent):
        feature_options = feature_selection.getOptions()
        model_options = model_library.getOptions()
        ensemble_options = ensembler.getOptions()


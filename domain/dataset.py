__author__ = 'anthony bell'

class Dataset():
    def __init__(self, train_df, test_df, train_labels, problem_type, validation_df = None):
        self.train_df = train_df
        self.test_df = test_df
        self.train_labels = train_labels
        self.problem_type = problem_type
        self.validation_df = validation_df
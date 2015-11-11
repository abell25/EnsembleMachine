__author__ = 'anthony bell'

from abc import ABCMeta, abstractmethod, abstractproperty
from problem_type import ProblemType

class DataLoader(metaclass=ABCMeta):
    """ Interface for data loading.

        This interface abstracts the loading of files.
        Implementation specific classes will tell it how to get train,test,validation data and train labels
        as well as scoring function used and expected output.

    """

    @abstractmethod
    def loadData(self):
        """ loadData: should get dataframes for train, test, valid, as well as labels
        :return: a Dataset class, containing the data and the problem specification
        """
        pass


class Dataset():
    def __init__(self, train_df, test_df, validation_df, train_labels, problem_type):
        ''' Object representing the loaded dataset.

        :param train_df:
        :param test_df:
        :param validation_df:
        :param train_labels:
        :param problem_type: ProblemType class
        '''

        if not isinstance(problem_type, ProblemType):
            raise ValueError('problem_type must be of type ProblemType!')

        self.problem_type = problem_type
        self.train_df = train_df
        self.test_df = test_df
        self.validation_df = validation_df
        self.train_labels = train_labels
        self.problem_type = problem_type
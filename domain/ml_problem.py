__author__ = 'anthony bell'

class MLproblem():
    def __init__(self, dataset, problemType):
        self.dataSet = dataset
        self.problemType = problemType

    def getProblemType(self):
        return self.problemType

    def getDataset(self):
        return self.dataSet


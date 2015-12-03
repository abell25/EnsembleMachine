__author__ = 'anthony bell'

class TrainedModel():
    def __init__(self, model, score):
        self.model = model
        self.score = score

    def predict(self, X):
        return self.model.predict(X)

__author__ = 'anthony bell'

from sklearn.cross_validation import train_test_split

class CVtrainer():
    def __init__(self, scorer):
        self.scorer = scorer

    def train_test_split(self, model, X, y, train_size=0.75):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = self.scorer(y_test, y_pred)  bb

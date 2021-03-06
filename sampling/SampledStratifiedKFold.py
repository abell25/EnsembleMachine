__author__ = 'anthony'

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit

import math
import uuid

import logging
logger = logging.getLogger(__name__)

class SampledStratifiedKFold:
    def __init__(self, y, num_folds=5, train_size=None):
        self.y = y
        self.num_folds = num_folds
        self.train_size = train_size

    def getSampledStratifiedKFold(self):
        #y_len = len(self.y)
        #fold_size = math.floor(y_len/self.num_folds)
        #if not self.train_size:
        #    subset_idx = y_len
        #elif self.train_size <= 0.0:
        #    raise("train_size must be greater than 0!")
        #elif self.train_size <= 1.0:
        #    subset_idx = math.floor((y_len - fold_size) * self.train_size)
        #else: #train_size > 1
        #    subset_idx = self.train_size

        skf = StratifiedKFold(self.y, n_folds=self.num_folds)

        for train_idx, test_idx in skf:
            y_train = self.y[train_idx]
            y_train_idxs = SampledStratifiedKFold.getStratifiedSubsample(y_train, self.train_size)
            train_subset_idxs = train_idx[y_train_idxs]
            yield (train_subset_idxs, test_idx)

    @staticmethod
    def getStratifiedSubsample(y, train_size):
        sss = StratifiedShuffleSplit(y, n_iter=1, train_size=train_size)
        idxs = [a for a,b in sss][0]
        return idxs

    def __iter__(self):
        for train_idx, test_idx in self.getSampledStratifiedKFold():
            yield (train_idx, test_idx)

    def __len__(self):
        return self.num_folds
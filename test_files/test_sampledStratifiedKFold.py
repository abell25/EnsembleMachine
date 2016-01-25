from unittest import TestCase

from sampling.SampledStratifiedKFold import SampledStratifiedKFold
import numpy as np

class TestSampledStratifiedKFold(TestCase):
    def test_getSampledStratifiedKFold(self):
        y = np.array([0]*10 + [1]*10)
        skf = SampledStratifiedKFold(y, num_folds=4, train_size=5)
        lists = list(skf)

    def test_getStratifiedSubsample(self):
        y = np.array([0]*10 + [1]*10)
        idxs1 = SampledStratifiedKFold.getStratifiedSubsample(y, train_size=6)
        self.assertTrue(sum(y[idxs1]) == 3)

        idxs2 = SampledStratifiedKFold.getStratifiedSubsample(y, train_size=0.5)
        self.assertTrue(sum(y[idxs2]) == 5)


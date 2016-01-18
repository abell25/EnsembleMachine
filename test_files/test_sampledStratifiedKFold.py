from unittest import TestCase

from sampling.SampledStratifiedKFold import SampledStratifiedKFold
import numpy as np

class TestSampledStratifiedKFold(TestCase):
    def test_getSampledStratifiedKFold(self):
        y = np.array([0]*10 + [1]*10)
        skf = SampledStratifiedKFold(y, num_folds=4, train_size=2)
        lists = list(skf)
from unittest import TestCase

import numpy as np
from util.sampling_functions import SamplingFunctions

class TestSamplingFunctions(TestCase):
    def setUp(self):
        pass

    def test_sample_array__handles_probability(self):
        l = np.array(list(range(3, 33, 3)))
        n = 0.3
        l2 = SamplingFunctions.sample_array(l, n)

        self.assertTrue(len(l2), 3)

    def test_sample_array__handles_int(self):
        l = np.array(list(range(3, 33, 3)))
        n = 3
        l2 = SamplingFunctions.sample_array(l, n)

        self.assertTrue(len(l2), 3)

    def test_sample_array_throws_for_nonpositive_input(self):
        self.assertRaises(ValueError,
                          lambda: SamplingFunctions.sample_array(np.array(list(range(20))), -1))
        self.assertRaises(ValueError,
                          lambda: SamplingFunctions.sample_array(np.array(list(range(20))), 0.0))
        self.assertRaises(ValueError,
                          lambda: SamplingFunctions.sample_array(np.array(list(range(20))), -0.000001))

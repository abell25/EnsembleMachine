__author__ = "anthony"

import random
import numpy as np

class SamplingFunctions:
    @staticmethod
    def sample_array(l, n):
        if n <= 0:
            raise ValueError("n must be positive number (but was n={0})".format(n))

        if isinstance(l, int):
            # if int is supplied, we'll just use a range(l) instead of erroring out
            l = np.array(range(l))

        ar = np.array(range(len(l)))
        random.shuffle(ar)

        # if 0 < n < 1, then it's a probability.
        if n <= 1.0:
            n = round(len(l)*n)
        else:
            n = round(n)


        return l[ar[:n]]
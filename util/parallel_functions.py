__author__ = "Anthony"

import multiprocessing
from joblib import delayed, Parallel

class ParallelFunctions:
    def __init__(self, num_cpus=None):
        if num_cpus is None:
            try:
                num_cpus = multiprocessing.cpu_count()
            except NotImplementedError:
                num_cpus = 2

        self.pool = multiprocessing.Pool(processes=num_cpus)

    @staticmethod
    def getPool(num_cpus=None):
        if num_cpus is None:
            try:
                num_cpus = multiprocessing.cpu_count()
            except NotImplementedError:
                num_cpus = 2

        pool = multiprocessing.Pool(processes=num_cpus)
        return pool


    @staticmethod
    def map(arr, f, num_cpus=None):
        pool = ParallelFunctions.getPool(num_cpus)
        return pool.map(f, arr)

    @staticmethod
    def pmap(arr, f, num_cpus=None):
        return Parallel(n_jobs=num_cpus)(delayed(f)(x) for x in arr)





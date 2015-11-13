__author__ = 'anthony bell'

import  numpy as np
import scipy as sp
import pandas as pd

class SparseFunctions():
    @staticmethod
    def col_lists_to_csr_matrix(lists_of_strs):

        indptr = [0]
        indices = []
        data = []
        vocabulary = {}
        vocab_size = 0
        index = 0
        for line in lists_of_strs:
            for col in line:
                if col not in vocabulary:
                    vocabulary[col] = vocab_size
                    vocab_size += 1

                col_index = vocabulary[col]
                indices.append(col_index)
                data.append(1)
                index += 1
            indptr.append(len(indices))

        return sp.sparse.csr_matrix((data, indices, indptr), dtype=int), list(vocabulary)

    @staticmethod
    def col_lists_to_df(list_of_strs):
        rows = []
        for line in list_of_strs:
            d = {}
            for col in line:
                if col not in d:
                    d[col] = 1
                else:
                    d[col] += 1
            rows.append(d)

        df = pd.DataFrame(rows)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def read_sparse_csv(filename, sep=' ', use_df=False):
        if use_df:
            df = SparseFunctions.col_lists_to_df(l.strip().split(sep) for l in open(filename).readlines())
            return df
        else:
            arr, cols = SparseFunctions.col_lists_to_csr_matrix(l.strip().split(sep) for l in open(filename).readlines())
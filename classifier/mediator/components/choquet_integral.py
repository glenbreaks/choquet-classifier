import numpy as np
from . import helper as h

class ChoquetIntegral:
    def __init__(self, capacity):
        self.capacity = capacity

    def compute_utility_value(self, x, capacity):

        result = 0
        number_of_subsets = h.get_feature_subset(x, 1)

        index_set_of_non_zero_features  = 0
        for j in range(number_of_features):
            pass


    def find_feature_minimum(self, s):
        return np.amin(s)


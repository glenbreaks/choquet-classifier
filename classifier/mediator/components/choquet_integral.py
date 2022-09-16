import numpy as np
from . import helper as h
from math import comb



class ChoquetIntegral:
    def __init__(self, X, y, additivity):
        self.X = X
        self.y = y

        self.number_of_features = np.shape(X)[0]
        self.additivity = additivity

    def compute_utility_value(self, moebius_coefficients, instance):
        choquet_value = 0
        for j in range(len(moebius_coefficients)):
            choquet_value += moebius_coefficients[j] * self.feature_minima_of_instance(instance)[j + 1]

        return choquet_value


    def feature_minima_of_instance(self, instance):
        features = list(range(1, len(instance) + 1))

        powerset_dict = h.get_powerset_dictionary(features, self.additivity)

        minima_dict = {key: np.amin([instance[i - 1] for i in value]) for key, value in powerset_dict.items()}

        return minima_dict

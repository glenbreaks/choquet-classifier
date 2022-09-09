import numpy as np
from . import helper as h
from math import comb

from .parameter_estimation import ParameterEstimation

class ChoquetIntegral:
    def __init__(self, X, y, moebius_transform):
        self.X = X
        self.y = y
        self.moebius_transform = moebius_transform


    def compute_aggregation_value(self, X, additivity):
        """

        Parameters
        -------
        X: array-like of shape (1, n_features)
            instance of data, where n_features is number of features

        Returns
        -------
        integral_value : float
                Integral value computed by the specified method.
        """
        self.number_of_features = np.shape(X)[0]

        number_of_moebius_coefficients = 0

        for i in range(1, additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        result = 0

        #add single criterion values which contribute to aggregation value
        for i in range(self.number_of_features):
            result += self.moebius_transform[i] * X[i]

        # add additivity related values which contribute to aggregation value
        pe = ParameterEstimation(self.X, self.y)
        moebius_matrix = pe.get_subset_matrix(number_of_moebius_coefficients, additivity)

        for i in range(self.number_of_features, len(self.moebius_transform)):
            row = moebius_matrix[i - self.number_of_features - 1]
            criteria = row[:self.number_of_features]
            criteria_in_instance = [X[j] for j in np.where(criteria == 1)[0]]
            result += self.moebius_transform[i] * np.amin(criteria_in_instance)

        return result



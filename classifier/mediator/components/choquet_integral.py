import numpy as np
from . import helper as h
from .parameter_estimation import ParameterEstimation

class ChoquetIntegral:
    def __init__(self, moebius_transform):
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

        result = 0

        #add single criterion values which contribute to aggregation value
        for i in range(self.number_of_features):
            result += self.moebius_transform[i] * X[i]

        # add additivity related values which contribute to aggregation value
        moebius_matrix = ParameterEstimation.get_moebius_matrix(X, additivity)

        for i in range(self.number_of_features + 1, self.moebius_transform):
            row = moebius_matrix[i - self.number_of_features + 1]
            criteria = row[:self.number_of_features]
            criteria_in_instance = [X[j] for j in np.where(criteria == -1)[0]]
            result += self.moebius_transform[i] * np.amin(criteria_in_instance)

        return result



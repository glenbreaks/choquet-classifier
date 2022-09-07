import numpy as np
from . import helper as h

class ChoquetIntegral:
    def __init__(self, moebius_transform):
        self.moebius_transform = moebius_transform


    #TODO:
    def compute_aggregation_value(self, X):
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
        self.number_of_features = np.shape(X)[1]

        result = 0

        #add single criterion values which contribute to aggregation value
        for i in range(self.number_of_features):
            result += self.moebius_transform[i] * X[i]

        # add additivity related values which contribute to aggregation value
        #for i in range(self.number_of_features + 1, self.moebius_transform):
        #    result


   # def _get_

    def find_feature_minimum(self, s):
        return np.amin(s)


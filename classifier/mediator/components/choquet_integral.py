import numpy as np
from . import helper as h
from math import comb



class ChoquetIntegral:
    def __init__(self, X, y, additivity):
        self.X = X
        self.y = y

        self.number_of_features = np.shape(X)[0]
        self.additivity = additivity

    def compute_aggregation_value(self, moebius_coefficient):
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
        number_of_features = self.number_of_features

        number_of_moebius_coefficients = 0

        for i in range(1, self.additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        result = 0

        # add single criterion values which contribute to aggregation value
        # for i in range(self.number_of_features):
        #    result += self.moebius_transform[i] * X[i]

        # add additivity related values which contribute to aggregation value
        # pe = ParameterEstimation(self.X, self.y, self.additivity)
        # subset_matrix = pe.get_subset_matrix(number_of_moebius_coefficients)

        # for row in subset_matrix:
        #    counter = 0
        #    criteria = row[:self.number_of_features]
        #    criteria_in_instance = [X[j] for j in np.where(criteria == 1)[0]]
        #    result += self.moebius_transform[self.number_of_features + counter] * np.amin(criteria_in_instance)
        #    print(self.moebius_transform[self.number_of_features + counter], criteria_in_instance)
        #    counter += 1


        return result

    def feature_minima_of_instance(self, instance):
        features = list(range(1, len(instance) + 1))

        powerset_dict = h.get_powerset_dictionary(features, self.additivity)

        minima_dict = {key: np.amin([instance[i - 1] for i in value]) for key, value in powerset_dict.items()}

        return minima_dict

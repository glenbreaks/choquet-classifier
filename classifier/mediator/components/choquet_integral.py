import numpy as np
from . import helper as h


class ChoquetIntegral:
    """Choquet integral

    Class to compute the Choquet utility value and
    store the moebius coefficients of the fuzzy measure.

    Parameters
    -------
    additivity : int
            Additivity of fuzzy measure.

    moebius_coefficients : list
            List of computed moebius coefficients, with the same internal
            ordering as the moebius_transform dictionary, calculated in
            the parameter estimation, since in Python 3.7+ dictionaries
            are guaranteed to be insertion-ordered. This insertion-order
            is used to easily calculate the terms which, when summed up,
            give the Integral value.
    """

    def __init__(self, additivity, moebius_coefficients):
        self.additivity = additivity
        self.moebius_coefficients = moebius_coefficients

    def compute_utility_value(self, instance):
        """Compute the utility value for a given instance

        Parameters
        -------
        instance:  array-like of shape (1, n_features)
            Instance, where n_features is the number of features.

        Returns
        -------
        choquet_value : float
            Integral value computed by the Choquet Integral.
        """

        choquet_value = 0
        for j in range(len(self.moebius_coefficients)):
            choquet_value += self.moebius_coefficients[j] * self.feature_minima_of_instance(instance)[j + 1]

        return choquet_value

    def feature_minima_of_instance(self, instance):
        """Get minimal feature of every subset and store them
        in a dictionary, with corresponding subset as key.


        Parameters
        -------
        instance : array-like of (1, n_features)
            Instance, where n_features is the number of features.

        Returns
        -------
        minima_dict: dict
            Dictionary, containing all subsets as keys, and their
            corresponding minimal feature of instance. Dictionary
            has the same insertion order as the moebius_coefficients dictionary.
        """

        features = list(range(1, len(instance) + 1))

        powerset_dict = h.get_powerset_dictionary(features, self.additivity)

        minima_dict = {key: np.amin([instance[i - 1] for i in value]) for key, value in powerset_dict.items()}
        return minima_dict

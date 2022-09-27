import numpy as np

from .choquet_integral import ChoquetIntegral


class ChoquisticRegression:
    """Choquistic Regression

    This class implements the Choquistic Regression calculation
    and stores all estimated parameters

    Parameters
    -------
    additivity : int
            Additivity of fuzzy measure.

    parameters : dict
            Dictionary, describing all fitted parameters (scaling, threshold,
            moebius coefficients) in the following manner:
            {'gamma': scaling, 'beta': threshold, frozenset({1}): m({1}),...}

    """
    def __init__(self, additivity, parameters):
        self.additivity = additivity
        self.scaling = parameters['gamma']
        self.threshold = parameters['beta']
        self.moebius_coefficients = list(parameters.values())[2:]

    def compute_regression_value(self, instance):
        """Computes the regression value for given instance
         with fitted parameters.

        Parameters
        -------
        instance:  array-like of shape (1, n_features)
            Instance, where n_features is the number of features.

        Returns
        -------
        regression value: float
            Probability of positive class, computed by the choquistic
            regression model.

        """
        choquet_integral = ChoquetIntegral(self.additivity, self.moebius_coefficients)

        gamma = self.scaling
        beta = self.threshold

        utility_value = choquet_integral.compute_utility_value(instance)

        regression_value = 1 / (1 + np.exp(-gamma * (utility_value - beta)))

        return regression_value

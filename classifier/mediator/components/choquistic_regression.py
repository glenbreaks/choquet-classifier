import numpy as np

from .choquet_integral import ChoquetIntegral


class ChoquisticRegression:
    """Choquistic Regression

    This class implements the Choquistic Regression calculation
    and stores all estimated parameters

    """
    def __init__(self, additivity, parameters):
        self.additivity = additivity
        self.scaling = parameters['gamma']
        self.threshold = parameters['beta']
        self.moebius_coefficients = list(parameters.values())[2:]

    def compute_regression_value(self, instance):
        choquet_integral = ChoquetIntegral(self.additivity, self.moebius_coefficients)

        gamma = self.scaling
        beta = self.threshold

        utility_value = choquet_integral.compute_utility_value(instance)

        regression_value = 1 / (1 + np.exp(-gamma * (utility_value - beta)))

        return regression_value

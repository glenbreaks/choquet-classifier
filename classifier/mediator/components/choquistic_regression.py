import numpy as np

from .choquet_integral import ChoquetIntegral


class ChoquisticRegression:

    def __init__(self, additivity, parameters):
        self.additivity = additivity
        self.parameters = parameters

    def compute_regression_value(self, instance):
        choquet_integral = ChoquetIntegral(self.additivity)

        gamma = self.parameters[0]
        beta = self.parameters[1]
        moebius_coefficients = self.parameters[2:]

        utility_value = choquet_integral.compute_utility_value(moebius_coefficients, instance)

        regression_value = 1 / (1 + np.exp(-gamma * (utility_value - beta)))

        return regression_value

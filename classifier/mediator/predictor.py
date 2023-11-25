
import numpy as np

from .components.choquistic_regression import ChoquisticRegression


class Predictor:
    def __init__(self):
        pass

    @staticmethod
    def get_classes(self, X, additivity, parameters):
        choquistic_regression = ChoquisticRegression(additivity, parameters)

        result = np.where(choquistic_regression.compute_regression_value(X) >= 0.5, 1, 0)

        return result

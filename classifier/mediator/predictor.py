import numpy as np

from .components.choquistic_regression import ChoquisticRegression

 
class Predictor:
    def __init__(self):
        pass

    def get_classes(self, X, additivity, parameters):

        choquistic_regression = ChoquisticRegression(additivity, parameters)

        result = self._get_classes_for_X(X, choquistic_regression)

        return result

    def _get_classes_for_X(self, X, regression_model):

        result = list()

        for x in X:
            regression_value = regression_model.compute_regression_value(x)

            cl = self._get_decision(regression_value)

            result.append(cl)

        return np.array(result)

    def _get_decision(self, regression_value):

        if regression_value >= 0.5:
            return np.array([1])
        else:
            return np.array([0])
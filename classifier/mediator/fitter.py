import numpy as np

from .components.feature_transformation import FeatureTransformation
from .components.parameter_estimation import ParameterEstimation


class Fitter:

    def __init__(self):
        pass

    def fit_feature_transformation(self, X):
        """ Fits the feature transformation for given input data

        :param X: array-like of shape (n_samples, n_features)
                Input data
        :return: FeatureTransformation
                instance of fitted Feature Transformation
        """
        feature_transformation = FeatureTransformation(X)

        return feature_transformation

    def fit_parameters(self, X, y,additivity, regularization_parameter):

        if additivity is None:
            additivity = 1
        else:
            additivity = additivity

        if regularization_parameter is None:
            regularization_parameter = 1
        else:
            regularization_parameter = regularization_parameter

        parameter_estimation = ParameterEstimation(X, y, additivity, regularization_parameter)

        function = parameter_estimation.compute_parameters()

        return function



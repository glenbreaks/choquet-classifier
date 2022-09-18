import numpy as np

from .components.feature_transformation import FeatureTransformation
from .components.parameter_estimation import ParameterEstimation


class Estimator:

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

    def fit_parameters(self, X, y, additivity, regularization_parameter):

        if additivity is None:
            additivity = 1
        else:
            additivity = additivity

        if regularization_parameter is None:
            regularization_parameter = 1
        else:
            regularization_parameter = regularization_parameter

        parameter_estimation = ParameterEstimation(X, y, additivity, regularization_parameter)

        self.parameter_dict = parameter_estimation.compute_parameters()

        return self.parameter_dict

    def get_scaling_factor(self):
        return self.parameter_dict.values()[0]

    def get_threshold(self):
        return self.parameter_dict.values()[1]

    def get_moebius_transform_of_capacity(self):
        return dict(list(self.parameter_dict.items())[2:])






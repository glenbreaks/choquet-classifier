import numpy as np

from .components.feature_transformation import FeatureTransformation


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

    def fit_parameters(self, X, y):
        pass



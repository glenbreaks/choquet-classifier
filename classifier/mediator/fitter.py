import numpy as np

from .components.feature_transformation import FeatureTransformation
from .components.feature_transformation2 import FeatureTransformation2
from .components.parameter_estimation import ParameterEstimation


class Fitter:
    """Fitter

    This class implements the functions to fit the feature transformation
    and the parameters of the Choquet classifier.

    """
    def __init__(self):
        pass

    def fit_feature_transformation(self, X):
        """ Fits the feature transformation for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        feature_transformation : FeatureTransformation
            Instance of the fitted feature transformation.
        """

        feature_transformation = FeatureTransformation(X)

        return feature_transformation

    def fit_parameters(self, X, y, additivity, regularization_parameter):
        """Fits the parameters threshold, scaling and moebius transform
        of the fuzzy measure for given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (1, n_samples)
            Target labels to X.

        additivity : int or None.
            Additivity of fuzzy measure. If None, additivity will
            be set to 2 to estimate parameters.

        regularization_parameter : float or None
            the regularization parameter of the L1-Regularization.
            If None, regularization will be set to 1 to estimate the parameters.

        Returns
        -------
        parameter_dict : dict
            Dictionary, describing all fitted parameters (scaling, threshold,
            moebius coefficients) in the following manner:
            {'gamma': scaling, 'beta': threshold, frozenset({1}): m({1}),...}
        """

        if additivity is None:
            additivity = 2
        else:
            additivity = additivity

        if regularization_parameter is None:
            regularization_parameter = 0.001
        else:
            regularization_parameter = regularization_parameter

        parameter_estimation = ParameterEstimation(X, y, additivity, regularization_parameter)

        self.parameter_dict = parameter_estimation.compute_parameters()

        return self.parameter_dict

    def get_scaling_factor(self):
        return list(self.parameter_dict.values())[0]

    def get_threshold(self):
        return list(self.parameter_dict.values())[1]

    def get_moebius_transform_of_capacity(self):
        return dict(list(self.parameter_dict.items())[2:])






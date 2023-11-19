import numpy as np

from .fitter import Fitter
from .predictor import Predictor
from typing import Tuple
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array


class Mediator:
    """Mediator

    Class to handle estimator and predictor class. Checks the input data, computes the parameters
    of the Choquet classifier and does the classification.
    """

    def __init__(self) -> None:
        self.number_of_features = 0
        self.scaling = None
        self.threshold = None
        self.moebius_transform = None
        self.parameters = None
        self.feature_transformation = None
        self.additivity = None

    def check_train_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Check input data for training

        Compare 'check_X_y" in the documentation of scikit-learn for more information

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (1, n_samples)
            Target labels to X.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Formatted input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (1, n_samples)
            Formatted target labels to X.
        """

        X, y = check_X_y(X, y)

        if not self._check_for_regression_targets(y):
            raise ValueError('Unknown label type: ', y)

        return X, y

    def check_test_data(self, X) -> np.ndarray:
        """Check the input data for classification

        Compare 'check_array' in scikit-learn documentation for more information

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            Formatted input data, where n_samples is the number of samples and
            n_features is the number of features.

        Raises
        -------
        ValueError, if a check should fail.
        """

        X = check_array(X)

        if np.shape(X)[1] != self.number_of_features:
            raise ValueError('Input data does not match number of features')

        return X

    def fit_components(self, X, y, additivity, regularization_parameter) -> bool:
        """Fit the feature transformation and the parameters: threshold, scaling and moebius transform of
        the capacity for a given input data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target labels to X.

        additivity : int
            Additivity(Hyperparameter of the Choquet classifier).

        regularization_parameter : float or None
            the regularization parameter of the L1-Regularization
            in the parameter estimation. If None, regularization
            will be set to 1 to estimate the parameters.

        Returns
        -------
        True, indicating that all parameters have been computed.
        """

        self.number_of_features = np.shape(X)[1]
        self.additivity = additivity

        if additivity is not None and self.number_of_features < additivity:
            raise ValueError('Additivity is greater than number of features')

        fitter = Fitter()

        self.feature_transformation = fitter.fit_feature_transformation(X)

        normalized_X = self._get_normalized_X(X, self.feature_transformation)

        self.parameters = fitter.fit_parameters(normalized_X, y, additivity, regularization_parameter)

        self.scaling = fitter.get_scaling_factor()
        self.threshold = fitter.get_threshold()

        self.moebius_transform = fitter.get_moebius_transform_of_capacity()

        return True

    def predict_classes(self, X) -> np.ndarray:
        """Predict classes for input data.

        Predict classes for input data. The function fit_components
        had to be called in advance to provide the feature transformation
        and the parameters.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        result : ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        predictor = Predictor()

        normalized_X = self._get_normalized_X(X, self.feature_transformation)

        result = predictor.get_classes(normalized_X, self.additivity, self.parameters)

        return result.ravel()

    def _check_for_regression_targets(self, y) -> bool:
        """Check target data for regression targets.

        Regression targets are considered to be real non integer numbers.
        This check is necessary to be compatible with scikit-learn.
        This check is copied from the Sugeno classifier by Sven Meyer:
        https://github.com/smeyer198/sugeno-classifier/blob/main/classifier/mediator/mediator.py

        Parameters
        -------
        y : array-like of shape (1,)
            Target labels.

        Returns
        -------
        False, if there is a regression target, true otherwise.
        """

        for value in y:
            # check for numeric value
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                continue

            if not float(value).is_integer():
                return False

        return True

    def _print_moebius_transform(self):
        moebius_coefficient = self.parameters[2:]
        for key, value in moebius_coefficient.items():
            print(key, '->', value)

    def _get_normalized_X(self, X, f) -> np.ndarray:
        """Normalize the input data using a Feature Transformation.
        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        f : FeatureTransformation
            Feature Transformation.
        Returns
        -------
        result : ndarray of shape (n_samples, n_features)
            Normalized input data, where n_samples is the number of samples
            and n_features the number of features.
        """
        return f(X)

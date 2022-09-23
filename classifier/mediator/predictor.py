import numpy as np

from .components.choquistic_regression import ChoquisticRegression


class Predictor:
    """Predictor

    This class implements the methods for the classification.
    """

    def __init__(self):
        pass

    def get_classes(self, X, additivity, parameters):
        """Get classes for input data.

        Get the classes for given input data. Uses the computed parameters
        (scaling, threshold, moebius coefficients)

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        additivity : int
            Additivity of fuzzy measure.

        parameters : dict
            Dictionary, describing all fitted parameters (scaling, threshold,
            moebius coefficients) in the following manner:
            {'gamma': scaling, 'beta': threshold, frozenset({1}): m({1}),...}

        Returns
        -------
        result: ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        choquistic_regression = ChoquisticRegression(additivity, parameters)

        result = self._get_classes_for_X(X, choquistic_regression)

        return result

    def _get_classes_for_X(self, X, regression_model):
        """Get classes for input data.

        Get classes for input data. Use the regression model
        for the classification.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        regression_model : ChoquisticRegression
            The Choquistic Regression class.

        Returns
        -------
        result : ndarray of shape (1, n_samples)
            Array containing the classes for the corresponding examples.
        """

        result = list()
        for x in X:
            regression_value = regression_model.compute_regression_value(x)
            cl = self._get_decision(regression_value)

            result.append(cl)

        return np.array(result)

    def _get_decision(self, regression_value):
        """Get class for regression value

        Parameters
        -------
        regression_value : float
            calculated regression value
        """

        if regression_value >= 0.5:
            return np.array([1])
        else:
            return np.array([0])

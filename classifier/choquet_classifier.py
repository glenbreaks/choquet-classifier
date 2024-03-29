
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional

from .mediator.mediator import Mediator


class ChoquetClassifier(BaseEstimator, ClassifierMixin):
    """The Choquet classifier

    Implementation of the Choquet classifier,introduced in
    "Learning monotone nonlinear models using the Choquet integral", using the Moebius transform
    presentation for fuzzy measures. This classifier is compatible
    to scikit-learn, i.e. it can be used like any scikit-learn estimator.

    Parameters
    -------
    additivity : int, default=1
        The additivity of the underlying fuzzy measure. Additivity takes interaction between features into account,
        i.e. the additivity value determines the maximum number of interacting features and hence the maximum
        number of moebius coefficients to be estimated. More precisely, with a k-additive measure, sum_i (n over i),
        with n = n_features, coefficients. The value 1 represents a simple additive measure with no interaction
        between features. The default value 2 represents a simple interaction between 2 features.

    regularization: in, default=None
        the regularization parameter of the L1-Regularization in the parameter estimation. Determines the strength of
        regularization of the fitting process.
    """

    def __init__(self, additivity: int = 2, regularization: Optional[float] = None) -> None:
        self.additivity = additivity
        self.regularization = regularization

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChoquetClassifier":
        """Initialize the parameters of the Choquet classifier

        Initialize the feature transformation and the parameters: threshold, scaling and moebius transform of
        the capacity, with the initialized hyperparameter for a given dataset.

        Parameters
        ------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features. The number of features
            has to be more or equal to the additivity.

        y : array-like of shape (n_samples,)
            Target labels to X.

        Returns
        -------
        self: ChoquetClassifier
            Fitted estimator.
        """
        self.mediator_ = Mediator()

        X, y = self.mediator_.check_train_data(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)

        self.mediator_.fit_components(X, y, self.additivity, self.regularization)

        self.n_features_in_ = self.mediator_.number_of_features

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for X

        Predict the class labels for all samples in X, which were
        specified in fit. The number of features have to match the
        number of features from the train data.

        Parameters
        -------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ------
        result : array-like of shape (n_samples,)
                The predicted classes.
        """
        check_is_fitted(self)

        X = self.mediator_.check_test_data(X)

        result = self.mediator_.predict_classes(X)

        return self.classes_[result]

    # ===============================================================
    # Functions for compatibility with scikit-learn. Not for general usage
    # ===============================================================

    def get_params(self, deep: bool = True) -> dict:
        return {"additivity": self.additivity, "regularization": self.regularization}

    def _more_tags(self) -> dict:
        return {
            "binary_only": True,
            "poor_score": True,
            "no_validation": True,
            "_xfail_checks": {"check_classifiers_one_label": "Model does not work on one label yet"},
        }

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the estimator and predict classes for X

        Fit the estimator using the given training data and labels, and predict the class labels for all samples in X.

        Parameters
        ------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features. The number of features
            has to be more or equal to the additivity.

        y : array-like of shape (n_samples,)
            Target labels to X.

        Returns
        ------
        result : array-like of shape (n_samples,)
                The predicted classes.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels

        Parameters
        ------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target labels to X.

        Returns
        ------
        score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        check_is_fitted(self)
        X = self.mediator_.check_test_data(X)
        result = self.mediator_.predict_classes(X)
        return np.mean(result == y)


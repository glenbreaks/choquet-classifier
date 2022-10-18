import numpy as np

from .mediator.mediator import Mediator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ChoquetClassifier(BaseEstimator, ClassifierMixin):
    """The Choquet classifier

    Implementation of the Choquet classifier, which was introduced in
    "Learning monotone nonlinear models using the Choquet integral", using the Moebius transform
    presentation of the underlying fuzzy measure.
    This classifier is compatible to scikit-learn, i.e. it can be used like any scikit-learn estimator.

    Parameters
    -------
    additivity : int, default=1
        The additivity of the underlying fuzzy measure. Additivity takes interaction between features into account,
        i.e. the additivity value determines the maximum number of interacting features and hence the maximum
        number of moebius coefficients to be estimated. More precisely, with a k-additive measure, sum_i (n over i),
        with n = n_features, coefficients. The default value 1 represents a simple additive measure with no interaction
        between features.

    regularization: in, default=None
        the regularization parameter of the L1-Regularization in the parameter estimation. Determines the strength of
        regularization of the fitting process.
    """

    def __init__(self, additivity=1, regularization=None):
        #self.mediator_ = None
        self.additivity = additivity
        self.regularization = regularization

    def fit(self, X, y):
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

    def predict(self, X):
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

    #def predict_proba(self, X):
    #    pass
    # ===============================================================
    # Functions for compatibility with scikit-learn. Not for general usage
    # ===============================================================

    def get_params(self, deep=True):
        return {'additivity': self.additivity,
                'regularization': self.regularization}

    def _more_tags(self):
        return {'binary_only': True, 'poor_score': True, "no_validation": True,
                "_xfail_checks": {"check_classifiers_one_label": "Model does not work on one label yet"}}


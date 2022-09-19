import numpy as np

from .mediator.mediator import Mediator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ChoquetClassifier(BaseEstimator, ClassifierMixin):
    """The Choquet Classifier

    """

    def __init__(self, additivity=1, regularization_parameter=None):
        self.additivity = additivity
        self.regularization_parameter = regularization_parameter

    def fit(self, X, y):
        self.mediator_ = Mediator()

        X, y = self.mediator_.check_train_data(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)



    def predict(self, X):
        check_is_fitted(self)

        X = self.mediator_.check_test_data(X)

        result = self.mediator_.predict_classes(X)

        return result

    def predict_proba(self, X):
        pass

    def score(self, X, y, sample_weight=None):
        pass
    # ===============================================================
    # Functions for compatibility with scikit-learn. Not for general usage
    # ===============================================================

    def get_params(self, deep=True):
        return {'additivity' : self.additivity,
                'scaling' : self.scaling,
                'threshold' : self.threshold}

    def _more_tags(self):
        return {'binary_only': True, 'poor_score': True}

    def _get_threshold(self):
        return self.mediator_.threshold

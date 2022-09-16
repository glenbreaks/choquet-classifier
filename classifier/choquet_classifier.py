from .mediator.mediator import Mediator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class ChoquetClassifier(BaseEstimator, ClassifierMixin):
    """The Choquet Classifier

    """

    def __init__(self, additivity=1, regularization_parameter=None):
        self.additivity = additivity
        self.regularization_parameter = regularization_parameter

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

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
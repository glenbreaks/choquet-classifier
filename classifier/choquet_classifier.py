from .mediator.mediator import Mediator

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class ChoquetClassifier(BaseEstimator, ClassifierMixin):
    """The Choquet Classifier

    """

    def __init__(self, additivity=None, scaling=None, threshold=None):
        self.additivity = additivity
        self.scaling = scaling
        self.threshold = threshold

    def fit(self, X, y):
        pass

    def predict(self,X):
        pass

    # ===============================================================
    # Functions for compatibility with scikit-learn. Not for general usage
    # ===============================================================

    def get_params(self, deep=True):
        return {'additivity' : self.additivity,
                'scaling' : self.scaling,
                'threshold' : self.scaling}

    def _more_tags(self):
        return {'binary_only': True, 'poor_score': True}

    def _get_threshold(self):
        return self.mediator_.threshold
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array

from components.parameter_estimation import ParameterEstimation

class Mediator:
    def __init__(self):
        pass

    def check_train_data(self, X, y):
        """Check input data for training
        'check_X_y' in scikitlearn doc:

        """
        pass

    def check_test_data(self):
        pass

    def fit_components(self, X, y, additivity):

        pass

    def predict_classes(self):
        pass

    def check_for_regression_targets(self, y):
        pass
    
    def _get_normalized_X(self, X, f):
        pass

    def _get_moebius_matrix(self, X, y, additivity):
        return ParameterEstimation(X, y).get_moebius_matrix(additivity)

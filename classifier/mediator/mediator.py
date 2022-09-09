from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
import numpy as np

from components.parameter_estimation import ParameterEstimation

class Mediator:
    def __init__(self):
        self.number_of_features = 0

    def check_train_data(self, X, y):
        """Check input data for training
        'check_X_y' in scikitlearn doc:

        """
        X, y = check_X_y(X, y)

        if not self._check_for_regression_targets(y):
            raise ValueError('Unknown label type: ', y)

        return X, y

    def check_test_data(self):
        pass

    def fit_components(self, X, y, additivity):

        pass

    def predict_classes(self):
        pass

    def _check_for_regression_targets(self, y):
        for value in y:
            # check for numeric value
            if not (isinstance(value, (int, float))
                    and not isinstance(value, bool)):
                continue

            if not float(value).is_integer():
                return False

        return True

    def _get_normalized_X(self, X, f):
        result = list()

        for x in X:
            normalized_x = self._get_normalized_x(x, f)

            result.append(normalized_x)

        return np.array(result)

    def _get_normalized_x(self, x, f):
        normalized_x = list()
        number_of_features = len(x)

        for i in range(number_of_features):
            feature = x[i]
            normalized_feature = f[i](feature)

            normalized_x.append(normalized_feature)

        return np.array(normalized_x)

    #def _get_moebius_matrix(self, X, y, additivity):
    #    return ParameterEstimation(X, y).get_moebius_matrix(additivity)

import numpy as np

from .estimator import Estimator
from .predictor import Predictor

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array


class Mediator:
    def __init__(self):
        self.number_of_features = 0

    def check_train_data(self, X, y):
        """Check input data for training
        'check_X_y' in scikit-learn doc:

        """
        X, y = check_X_y(X, y)

        if not self._check_for_regression_targets(y):
            raise ValueError('Unknown label type: ', y)

        return X, y

    def check_test_data(self, X):

        X = check_array(X)

        if np.shape(X)[1] != self.number_of_features:
            raise ValueError('Input data does not match number of features')

        return X

    def fit_components(self, X, y, additivity, regularization_parameter):

        self.number_of_features = np.shape(X)[1]

        if additivity is not None and self.number_of_features < additivity:
            raise ValueError('Additivity is greater than number of features')

        estimator = Estimator()

        self.additivity = additivity
        self.feature_transformation = estimator.fit_feature_transformation(X)

        normalized_X = self.feature_transformation

        self.parameters = estimator.fit_parameters(normalized_X, y, additivity, regularization_parameter)

        self.scaling = estimator.get_scaling_factor()
        self.threshold = estimator.get_threshold()

        self.moebius_transform = estimator.get_moebius_transform_of_capacity()

        return True

    def predict_classes(self, X):
        predictor = Predictor()

        normalized_data = list()

        for x in X:
            normalized_x = self.feature_transformation(x)
            normalized_data.append(normalized_x)

        result = predictor.get_classes(normalized_data, self.additivity, self.parameters)

        return result

    #Sven Meyer's implementation of the Sugeno Classifier
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
        pass

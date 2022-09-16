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
        'check_X_y' in scikitlearn doc:

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

        self.feature_transformation = estimator.fit_feature_transformation(X)

        normalized_X = self._get_normalized_X(X, self.feature_transformation)

    def predict_classes(self, X):
        predictor = Predictor()
        pass


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
        pass

    def _get_normalized_x(self, x, f):
        normalized_x = list()
        number_of_features = len(x)

        for i in range(number_of_features):
            feature = x[i]
            normalized_feature = f[i](feature)

            normalized_x.append(normalized_feature)

        return np.array(normalized_x)

import unittest
from classifier.mediator.mediator import Mediator
from sklearn.utils.validation import check_array

class TestMediator(unittest.TestCase):

    def test_fit_components(self):
        # test for correct additivity
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]

        additivity = 4

        mediator = Mediator()
        X = check_array(X)

        with self.assertRaises(ValueError):
            mediator.fit_components(X, y, additivity, 0.001)

    def test_for_regression_targets(self):
        # This check is copied from the Sugeno classifier by Sven Meyer:
        # https://github.com/smeyer198/sugeno-classifier/blob/main/classifier/tests/mediator/test_mediator.py
        X = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]

        mediator = Mediator()

        # no regression targets
        y = [2, 5, 2]

        result = mediator._check_for_regression_targets(y)

        self.assertTrue(result)

        # regression targets
        y = [0.1, 0.4, 0.1]

        result = mediator._check_for_regression_targets(y)

        self.assertFalse(result)

        # mixed targets
        y = [3, 6, 0.5]

        result = mediator._check_for_regression_targets(y)

        self.assertFalse(result)

        # ints are allowed to be a floating point
        y = [3.0, 6.0, 5.0]

        result = mediator._check_for_regression_targets(y)

        self.assertTrue(result)

        # strings are considered to be labels and no regression targets
        y = [3, '3.7', 1]

        result = mediator._check_for_regression_targets(y)

        self.assertTrue(result)



if __name__ == '__main__':
    unittest.main()

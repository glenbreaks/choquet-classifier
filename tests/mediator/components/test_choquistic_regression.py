import unittest
from classifier.mediator.components.choquistic_regression import ChoquisticRegression


class TestChoquisticRegression(unittest.TestCase):
    def test_something(self):
        dict = {'gamma': 1, 'beta': 1, frozenset({1}): 1}
        regression = ChoquisticRegression(2, dict)
        print(regression.moebius_coefficients)


if __name__ == '__main__':
    unittest.main()

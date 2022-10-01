import unittest
from classifier.mediator.components.choquistic_regression import ChoquisticRegression


class TestChoquisticRegression(unittest.TestCase):
    def test_regression_calculation(self):
        X = [[0.25, 0.75, 0.25], [0.75, 0.25, 0.75]]
        y = [0, 1]
        parameter_dict = {'gamma': 82.4370341720885, 'beta': 0.48038295029897504, frozenset({1}): 0.32592547605321703,
                frozenset({2}): 0.016041218169457312, frozenset({3}): 0.4061951936952968,
                frozenset({1, 2}): 0.032105832146839916, frozenset({1, 3}): 0.1743782358631544,
                frozenset({2, 3}): 0.0317775846834257, frozenset({1, 2, 3}): 0.013576459388608858}
        regression = ChoquisticRegression(3, parameter_dict)

        self.assertAlmostEqual(regression.compute_regression_value(X[0]), y[0])
        self.assertAlmostEqual(regression.compute_regression_value(X[1]), y[1])


if __name__ == '__main__':
    unittest.main()

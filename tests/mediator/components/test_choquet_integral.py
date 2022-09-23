import unittest
from classifier.mediator.components.choquet_integral import ChoquetIntegral

class TestChoquetIntegral(unittest.TestCase):
    def test_compute_aggregation_value(self):
        X_full = [[.4, .3, .6], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        X = [1, 0.1, 0.01, 0.001]
        y = [0, 1]
        moebius_transform = [0.4, 0.3, 0.2, 0.2, 0.6, 0.1, 0.2, 0.5, 0.6, 0.8]
        choquet_integral = ChoquetIntegral(2, moebius_transform)
        result = choquet_integral.compute_utility_value(X)
        print(result)

    def test_feature_minima(self):
        X_full = [[.4, .3, .6], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        X = [1, 2, 3]
        y = [0, 1]

        moebius_transform = [0.4, 0.3, 0.2, 0.2, 0.6, 0.1, 0.2]
        choquet_integral = ChoquetIntegral(3, moebius_transform)
        result = choquet_integral.feature_minima_of_instance([.4, 0, .6])
        print(result)

if __name__ == '__main__':
    unittest.main()

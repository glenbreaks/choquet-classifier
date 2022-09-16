import unittest
from classifier.mediator.components.choquet_integral import ChoquetIntegral

class TestChoquetIntegral(unittest.TestCase):
    def test_compute_aggregation_value(self):
        X_full = [[.4, .3, .6], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        X = [1, 2, 3]
        y = [0, 1]
        moebius_transform = [0.4, 0.3, 0.2, 0.2, 0.6, 0.1]
        choquet_integral = ChoquetIntegral(2)
        result = choquet_integral.compute_utility_value(moebius_transform, X)
        print(result)

    def test_feature_minima(self):
        X_full = [[.4, .3, .6], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        X = [1, 2, 3]
        y = [0, 1]

        choquet_integral = ChoquetIntegral(2)
        result = choquet_integral.feature_minima_of_instance([.4, 0, .6])
        print(result)

if __name__ == '__main__':
    unittest.main()

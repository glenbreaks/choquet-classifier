import numpy as np

import unittest
from classifier.mediator.components.choquet_integral import ChoquetIntegral


class TestChoquetIntegral(unittest.TestCase):
    def test_compute_aggregation_value(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]

        additivity = 3
        number_of_features = np.shape(X)[1]

        moebius_coefficients = [0.32621127227372915, 0.1096778349966972, 0.14071448801729294, 0.2181001452983398,
                                0.06220493806514601, 0.1071507113468563, 0.035940610001938655]

        capacities = [0.32621127227372915, 0.1096778349966972, 0.14071448801729294, 0.6539892525687662,
                      0.5291306983561681, 0.35754303436084645, 1.0]

        # Calculation of integration values and counter-checking from http://www.isc.senshu-u.ac.jp/~thc0456/Efuzzyweb/
        # by Eiichiro Takahagi, Senshu University, Kawasaki, Japan
        integrated_values = [0.149826, 0.249826, 0.349826, 0.449826]

        choquet_integral = ChoquetIntegral(additivity, moebius_coefficients)
        for i in range(number_of_features):
            self.assertAlmostEqual(choquet_integral.compute_utility_value(X[i]), integrated_values[i], places=6)

    def test_feature_minima(self):
        X_full = [[.4, .3, .6], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        X = [1, 2, 3]
        y = [0, 1]

        moebius_transform = [0.4, 0.3, 0.2, 0.2, 0.6, 0.1, 0.2]
        choquet_integral = ChoquetIntegral(4, moebius_transform)
        result = choquet_integral.feature_minima_of_instance([.4, 0, .6, 0.3])
        print(result)


if __name__ == '__main__':
    unittest.main()

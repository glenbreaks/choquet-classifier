import numpy as np
import unittest

from classifier.mediator.components.feature_transformation import FeatureTransformation


class TestFeatureTransformation(unittest.TestCase):

    def test_no_redundant_data(self):
        data = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]])
        normalized_data = np.array([[0., 0.25], [0.25, 0.], [0.5, 0.75], [0.75, 0.5], [1., 1.]])

        f = FeatureTransformation(data)
        np.testing.assert_almost_equal(f.normalized, normalized_data, 5)

    def test_feature_trans(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]

        f = FeatureTransformation(X)

        instance = np.array([0.3, 0.3, 0.4])
        expected_instance = np.array([0.66666667, 0.33333333, 0.33333333])

        np.testing.assert_almost_equal([expected_instance], f([instance]))

    def test_redundant_data(self):
        X = np.array([[1, 2, 4], [1, 1, 2], [3, 3, 1], [3, 4, 2], [5, 1, 3]])
        normalized_X = np.array([[0., 0.5, 1], [0., 0., 0.375], [0.625, 0.75, 0], [0.625, 1, 0.375], [1., 0, 0.75]])

        f = FeatureTransformation(X)

        np.testing.assert_almost_equal(f.normalized, normalized_X, 5)


if __name__ == '__main__':
    unittest.main()


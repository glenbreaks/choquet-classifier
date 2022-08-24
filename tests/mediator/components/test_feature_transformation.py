import numpy as np
import unittest

from classifier.mediator.components.feature_transformation import FeatureTransformation


class TestFeatureTransformation(unittest.TestCase):

    def test_no_redundant_data(self):
        data = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]])
        normalized_data = np.array([[0., 0.25], [0.25, 0.], [0.5, 0.75], [0.75, 0.5], [1., 1.]])

        f = FeatureTransformation(data)
        np.testing.assert_almost_equal(f.normalized, normalized_data, 5)

    def test_redundant_data(self):
        data = np.array([[1, 2], [1, 1], [3, 4], [3, 3], [5, 5]])
        normalized_data = np.array([[0., 0.25], [0., 0.], [0.625, 0.75], [0.625, 0.5], [1., 1.]])

        f = FeatureTransformation(data)
        np.testing.assert_almost_equal(f.normalized, normalized_data, 5)




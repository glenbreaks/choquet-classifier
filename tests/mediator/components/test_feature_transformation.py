import numpy as np
import unittest

from classifier.mediator.components.feature_transformation import FeatureTransformation


class TestFeatureTransformation(unittest.TestCase):

    def test_no_redundant_data(self):
        data = np.array([[1], [2], [3], [4], [5]])

        f = FeatureTransformation(data)

        self.assertEqual(f[0, 0], 0)
        self.assertEqual(f[1, 0], 0.25)
        self.assertEqual(f[2, 0], 0.5)
        self.assertEqual(f[3, 0], 0.75)
        self.assertEqual(f[4, 0], 1)
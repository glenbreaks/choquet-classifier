import unittest
import numpy as np

from classifier.mediator.components import helper as h


class TestHelper(unittest.TestCase):

    def test_get_feature_subset(self):
        array_1 = np.array([4, 21, 1, 6, 9])
        sorted_array_1 = np.sort(array_1)

        permutation_1 = h._get_permutation_position(array_1, sorted_array_1)

        subset_1_index_1 = h.get_feature_subset(array_1, 1)
        expected_subset_1 = {3, 1, 4, 5, 2}
        self.assertSetEqual(subset_1_index_1, expected_subset_1)

        subset_2_index_2 = h.get_feature_subset(array_1, 2)
        expected_subset_2 = {1, 4, 5, 2}
        self.assertSetEqual(subset_2_index_2, expected_subset_2)

    def test_get_permutation_position(self):
        array_1 = np.array([4, 21, 1, 6, 9])
        sorted_array_1 = np.sort(array_1)

        expected_permutation_1 = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
        permutation_1 = h._get_permutation_position(array_1, sorted_array_1)

        self.assertDictEqual(permutation_1, expected_permutation_1)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from classifier.mediator.components import helper as h


class TestHelper(unittest.TestCase):

    def test_powerset(self):
        s = {2, 4, 6, 8}

        powerset1 = h.get_powerset(s)
        expected_powerset1 = [frozenset(), frozenset([2]), frozenset([4]), frozenset([6]), frozenset([8]),
                             frozenset([2, 4]),  frozenset([2, 6]),  frozenset([2, 8]),  frozenset([4, 6]),
                             frozenset([4, 8]),  frozenset([6, 8]), frozenset([2, 4, 6]),  frozenset([2, 4, 8]),
                             frozenset([2, 6, 8]),  frozenset([4, 6, 8]),  frozenset([2, 4, 6, 8])]

        self.assertCountEqual(list(powerset1), expected_powerset1)

        # test additivity powerset
        powerset2 = h.get_powerset(s, additivity=2)
        expected_powerset2 = [frozenset(), frozenset([2]), frozenset([4]), frozenset([6]), frozenset([8]),
                             frozenset([2, 4]), frozenset([2, 6]), frozenset([2, 8]), frozenset([4, 6]),
                             frozenset([4, 8]), frozenset([6, 8])]

        self.assertCountEqual(list(powerset2), expected_powerset2)


    def test_powerset_dictionary(self):
        s = {1, 2, 3}

        powerset1 = h.get_powerset_dictionary(s)
        print(powerset1)

    def test_subset_dictionary(self):
        s = {1, 2, 3}
        list = h.get_subset_dictionary_list(s, 2)
        print(list)


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

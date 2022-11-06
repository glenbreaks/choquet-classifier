import unittest
import numpy as np
from itertools import chain, combinations

from classifier.mediator.components import helper as h


class TestHelper(unittest.TestCase):

    def test_powerset(self):
        s = {2, 4, 6, 8}

        powerset1 = h.get_powerset(s, 4)
        expected_powerset1 = [frozenset(), frozenset([2]), frozenset([4]), frozenset([6]), frozenset([8]),
                              frozenset([2, 4]), frozenset([2, 6]), frozenset([2, 8]), frozenset([4, 6]),
                              frozenset([4, 8]), frozenset([6, 8]), frozenset([2, 4, 6]), frozenset([2, 4, 8]),
                              frozenset([2, 6, 8]), frozenset([4, 6, 8]), frozenset([2, 4, 6, 8])]

        self.assertCountEqual(list(powerset1), expected_powerset1)

        # test additivity powerset
        powerset2 = h.get_powerset(s, additivity=2)
        expected_powerset2 = [frozenset(), frozenset([2]), frozenset([4]), frozenset([6]), frozenset([8]),
                              frozenset([2, 4]), frozenset([2, 6]), frozenset([2, 8]), frozenset([4, 6]),
                              frozenset([4, 8]), frozenset([6, 8])]

        self.assertCountEqual(list(powerset2), expected_powerset2)

    def test_powerset_dictionary(self):
        s = {1, 2, 3, 4}
        additivity = 2

        powerset_dict = h.get_powerset_dictionary(s, additivity)
        expected_powerset = chain.from_iterable(combinations(s, r) for r in range(1, additivity + 1))

        expected_powerset_list = []
        for item in expected_powerset:
            expected_powerset_list.append(frozenset(item))

        for key, item in powerset_dict.items():
            self.assertEqual(powerset_dict[key], expected_powerset_list[key - 1])

    def test_additivity_powerset(self):
        s = [1, 2, 3]
        additivity = 2

        powerset = h._get_additivity_powerset(s, additivity)
        expected_powerset = [frozenset(), frozenset({1}), frozenset({2}), frozenset({3}),
                             frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3})]

        self.assertListEqual(list(powerset), expected_powerset)

    def test_subset_dictionary_list(self):
        s = {1, 2, 3, 4}
        additivity = 3

        subset_dict_list = h.get_subset_dictionary_list(s, additivity)
        expected_dict_list = [{1: frozenset({1}), 2: frozenset({2}), 5: frozenset({1, 2})},
                              {1: frozenset({1}), 3: frozenset({3}), 6: frozenset({1, 3})},
                              {1: frozenset({1}), 4: frozenset({4}), 7: frozenset({1, 4})},
                              {2: frozenset({2}), 3: frozenset({3}), 8: frozenset({2, 3})},
                              {2: frozenset({2}), 4: frozenset({4}), 9: frozenset({2, 4})},
                              {3: frozenset({3}), 4: frozenset({4}), 10: frozenset({3, 4})},
                              {1: frozenset({1}), 2: frozenset({2}), 3: frozenset({3}), 5: frozenset({1, 2}), 6: frozenset({1, 3}), 8: frozenset({2, 3}), 11: frozenset({1, 2, 3})},
                              {1: frozenset({1}), 2: frozenset({2}), 4: frozenset({4}), 5: frozenset({1, 2}), 7: frozenset({1, 4}), 9: frozenset({2, 4}), 12: frozenset({1, 2, 4})},
                              {1: frozenset({1}), 3: frozenset({3}), 4: frozenset({4}), 6: frozenset({1, 3}), 7: frozenset({1, 4}), 10: frozenset({3, 4}), 13: frozenset({1, 3, 4})},
                              {2: frozenset({2}), 3: frozenset({3}), 4: frozenset({4}), 8: frozenset({2, 3}), 9: frozenset({2, 4}), 10: frozenset({3, 4}), 14: frozenset({2, 3, 4})}]

        self.assertListEqual(subset_dict_list, expected_dict_list)

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

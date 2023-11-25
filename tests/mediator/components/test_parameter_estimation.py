import numpy as np
import numpy.testing as nptest

from classifier.mediator.components import parameter_estimation as est
from classifier.mediator.components import helper as h
import unittest


class TestParameterEstimation(unittest.TestCase):
    def test_constraints_size(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        additivity = 4
        number_of_features = np.shape(X)[1]

        p = est.ParameterEstimation(X, y, additivity)
        number_of_moebius_coefficients = p._get_number_of_moebius_coefficients()

        # scaling (gamma) + threshold (beta) + moebius coefficients
        number_of_parameters = number_of_moebius_coefficients + 2

        number_of_monotonicity_constraints = np.power(2,
                                                      number_of_features - 1) * number_of_features - number_of_features
        number_of_linear_constraints = 1 + number_of_monotonicity_constraints

        bounds, linear_constraints = p._set_constraints()

        self.assertEqual(len(bounds.lb), number_of_parameters)
        self.assertEqual(len(bounds.ub), number_of_parameters)

        linear_constraint_matrix_columns = np.shape(linear_constraints.A)[1]
        linear_constraint_matrix_rows = np.shape(linear_constraints.A)[0]

        self.assertEqual(linear_constraint_matrix_columns, number_of_parameters)
        self.assertEqual(linear_constraint_matrix_rows, number_of_linear_constraints)

    def test_monotonicity_matrix(self):
        X = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
        y = [0, 1]

        additivity = 3
        number_of_features = np.shape(X)[1]
        number_of_monotonicity_constraints = np.power(2,
                                                      number_of_features - 1) * number_of_features - number_of_features

        powersets = h.get_powerset_dictionary(list(range(1, number_of_features + 1)), additivity)
        p = est.ParameterEstimation(X, y, additivity, 1)
        number_of_moebius_coefficients = p._get_number_of_moebius_coefficients()
        monotonicity_matrix = p.get_monotonicity_matrix()

        self.assertEqual(number_of_monotonicity_constraints, np.shape(monotonicity_matrix)[0])

        monotonicity_sets_matrix = []
        for monotonicity_array in monotonicity_matrix:
            monotonicity_check_array = []
            moebius_coefficients_position = np.where(monotonicity_array == 1)[0]
            for index in moebius_coefficients_position:
                monotonicity_check_array.append(powersets[index + 1])
            monotonicity_sets_matrix.append(monotonicity_check_array)

        build_up_monotonicity_matrix = []
        for monotonicity_constraint in monotonicity_sets_matrix:
            arr = np.zeros(number_of_moebius_coefficients)
            indicator_array = [key - 1 for key, value in powersets.items() if value in monotonicity_constraint]
            monotonicity_matrix_row = [1 if index in indicator_array else 0 for index, x in enumerate(arr)]
            build_up_monotonicity_matrix.append(monotonicity_matrix_row)

        nptest.assert_array_equal(monotonicity_matrix, np.array(build_up_monotonicity_matrix))

    def test_linear_constraint_matrix(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [1, 0, 1, 0]

        additivity = 3
        number_of_features = np.shape(X)[1]
        number_of_linear_constraints = np.power(2, number_of_features - 1) * number_of_features - \
                                       number_of_features + 1

        p = est.ParameterEstimation(X, y, additivity, 1)
        number_of_parameters = p._get_number_of_moebius_coefficients() + 2
        linear_constraint_matrix = p._get_linear_constraint_matrix()

        self.assertEqual(number_of_linear_constraints, np.shape(linear_constraint_matrix)[0])
        self.assertEqual(number_of_parameters, np.shape(linear_constraint_matrix)[1])

    def test_compute_parameters(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]

        p = est.ParameterEstimation(X, y, additivity=2, regularization_parameter=0.001)
        parameter_dict = p.compute_parameters()
        self.assertAlmostEqual(parameter_dict['beta'], 0.3764159, places=5)

    def test_monotonicity_from_moebius_coefficients(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]

        additivity = 3
        number_of_features = np.shape(X)[1]
        number_of_monotonicity_constraints = np.power(2,
                                                      number_of_features - 1) * number_of_features - number_of_features

        p = est.ParameterEstimation(X, y, additivity, 1)
        parameter_dict = p.compute_parameters()

        moebius_coefficients = [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({1, 2}),
                                frozenset({1, 3}), frozenset({2, 3}), frozenset({1, 2, 3})]
        moebius_dict = {key: parameter_dict[key] for key in moebius_coefficients}

        coefficient_values = list(moebius_dict.values())
        capacity_of_highest_subset = np.sum(coefficient_values)
        self.assertAlmostEqual(capacity_of_highest_subset, 1)

        capacity_c1 = coefficient_values[0]
        capacity_c2 = coefficient_values[1]
        capacity_c3 = coefficient_values[2]

        capacity_c1_c2 = capacity_c1 + capacity_c2 + coefficient_values[3]
        capacity_c1_c3 = capacity_c1 + capacity_c3 + coefficient_values[4]
        capacity_c2_c3 = capacity_c2 + capacity_c3 + coefficient_values[5]

        self.assertEqual((capacity_c1_c2 >= capacity_c1) & (capacity_c1_c2 >= capacity_c2), True)
        self.assertEqual((capacity_c1_c3 >= capacity_c1) & (capacity_c1_c3 >= capacity_c3), True)
        self.assertEqual((capacity_c2_c3 >= capacity_c2) & (capacity_c2_c3 >= capacity_c3), True)

    def test_l1_regularization(self):
        X = [[.25, .3, .7, .1], [.5, .625, .4, .75], [.35, .4, .5, .4], [.125, .5, .625, .6]]
        y = [1, 1, 1, 0]

        moebius_coefficient = [0.3, 0.3, 0.2, 0.2]
        p = est.ParameterEstimation(X, y, 1, 1)

        self.assertEqual(1.0, p._l1_regularization(moebius_coefficient))


if __name__ == '__main__':
    unittest.main()

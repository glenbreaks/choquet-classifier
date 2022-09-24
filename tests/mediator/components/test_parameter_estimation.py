import numpy as np

from classifier.mediator.components import parameter_estimation as est
import unittest


class TestParameterEstimation(unittest.TestCase):
    def test_constraints_size(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = est.ParameterEstimation(X, y, 4)

        number_of_moebius_coefficients = p._get_number_of_moebius_coefficients()

        # scaling (gamma) + threshold (beta) + moebius coefficients
        number_of_parameters = 1 + 1 + number_of_moebius_coefficients

        number_of_monotonicity_constraints = np.power(2, 3) * 4 - 4
        number_of_linear_constraints = 1 + number_of_monotonicity_constraints

        bounds, linear_constraints = p._set_constraints()

        self.assertEqual(len(bounds.lb), number_of_parameters)
        self.assertEqual(len(bounds.ub), number_of_parameters)

        linear_constraint_matrix_columns = np.shape(linear_constraints.A)[1]
        linear_constraint_matrix_rows = np.shape(linear_constraints.A)[0]

        self.assertEqual(linear_constraint_matrix_columns, number_of_parameters)
        self.assertEqual(linear_constraint_matrix_rows, number_of_linear_constraints)

    def test_monotonicity_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = est.ParameterEstimation(X, y, 2, 1)

        print(p.get_monotonicity_matrix())

    def test_linear_constraint_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [1, 0, 1, 0]

        p = est.ParameterEstimation(X, y, 2, 1)

        print(p._get_linear_constraint_matrix())

    def test_log_likelihood(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [1, 0, 1, 0]

        parameters = np.array([1, 2, 1, 1, 1, 1])

        p = est.ParameterEstimation(X, y, 2, 1)
        print(p._log_likelihood_function(parameters))

    def test_compute_parameters(self):
        X = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
        y = [0, 0, 0, 1]


        p = est.ParameterEstimation(X, y, 3, 1)
        parameter_dict = p.compute_parameters()
        self.assertAlmostEqual(parameter_dict['beta'], 0.4)
        print(p.compute_parameters())

    def test_l1_regularization(self):
        X = [[.25, .3, .7, .1], [.5, .625, .4, .75], [.35, .4, .5, .4], [.125, .5, .625, .6]]
        y = [1, 1, 1, 0]

        moebius_coefficient = [1,1,1,1]
        p = est.ParameterEstimation(X, y, 1, 1)

        print(p._l1_regularization(moebius_coefficient))

if __name__ == '__main__':
    unittest.main()

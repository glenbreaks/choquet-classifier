from classifier.mediator.components import parameter_estimation as pest
import unittest


class TestParameterEstimation(unittest.TestCase):
    def test_set_constraints(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y, 3)

        constraints = p._set_constraints()
        print(constraints)

    def test_subset_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y, 2)

        print(p.get_subset_matrix(10))

    def test_monotonicity_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y, 3)

        print(p.get_monotonicity_matrix())

    def test_linear_constraint_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y, 3)

        print(p._get_linear_constraint_matrix())

if __name__ == '__main__':
    unittest.main()

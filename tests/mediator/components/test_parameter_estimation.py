from classifier.mediator.components import parameter_estimation as pest
import unittest


class TestParameterEstimation(unittest.TestCase):
    def test_set_constraints(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y)

        constraints = p._set_constraints(3)
        print(constraints)

    def test_subset_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y)

        print(p.get_subset_matrix(10, 2))

    def test_monotonicity_matrix(self):
        X = [[1, 2, 3, 5], [2, 3, 4, 1], [3, 4, 5, 2], [4, 5, 6, 6]]
        y = [0, 1]

        p = pest.ParameterEstimation(X, y)

        print(p.get_monotonicity_matrix( 3))
if __name__ == '__main__':
    unittest.main()

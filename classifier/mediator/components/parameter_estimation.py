import numpy as np
from scipy import optimize as opt
from math import comb

from . import moebius_transformation
from .choquet_integral import ChoquetIntegral
from . import helper as h


# TODO: if additivity=None or 1 linear constraints are not to be created
# TODO: mapping of moebius coefficient to subset in dictionary
class ParameterEstimation:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.number_of_features = np.shape(self.X)[1]

    def _log_likelihood_function(self, dataset, parameters):
        pass

    def compute_parameters(self, additivity, regularization_parameter=1):
        bounds, constraints = self._set_constraints(additivity)
        # pack result from opt.minimize in dict (for moebius coefficients)
        # result = opt.minimize()
        pass

    def _set_constraints(self, additivity):
        """ Sets up the bounds and linear constraints for the optimization problem.
            using the Bounds and Constraints classes from scipy.optimize.

        :param additivity: additivity of underlying capacity
        :return: constraints : list
                consisting of the Bounds and Constraints instances
        """
        number_of_moebius_coefficients = 0
        for i in range(1, additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        # set the bounds - order is: gamma, threshold, m(T_1),...,m(T_n) with n being
        lower_bound = [1, 0]
        upper_bound = [np.inf, 1]

        boundary_constraint = np.array([0, 0])

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)

        bounds = opt.Bounds(lower_bound, upper_bound)

        monotonicity_constraints = []
        for j in h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), additivity):
            subsets = j
            arr = np.zeros(number_of_moebius_coefficients)
            for i in range(arr.size):
                if i + 1 in list(subsets.keys())[:-1]:
                    arr[i] = - 1
                elif i + 1 == list(subsets.keys())[-1]:
                    arr[i] = len(list(subsets.keys())[:-1])
            monotonicity_constraints.append(arr)

        linear_constraint_matrix = [np.concatenate((boundary_constraint, np.ones(number_of_moebius_coefficients)))]
        for arr in monotonicity_constraints:
            array = np.concatenate((boundary_constraint, arr))
            linear_constraint_matrix = np.append(linear_constraint_matrix, [array], axis=0)

        lower_limits = np.zeros(linear_constraint_matrix.shape[0])
        upper_limits = np.ones(linear_constraint_matrix.shape[0])

        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, lower_limits, upper_limits)
        return linear_constraint_matrix

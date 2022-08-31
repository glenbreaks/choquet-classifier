import numpy as np
from scipy import optimize as opt
from math import comb

from . import moebius_transformation
from .choquet_integral import ChoquetIntegral
from . import helper as h


# TODO: implement monotonicity condition of moebius implementation
# TODO: mapping of moebius coefficient to subset
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
        for i in range(1, additivity+1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        # set the bounds - order is: gamma, threshold, m(T_1),...,m(T_n) with n being
        lower_bound = [1, 0]
        upper_bound = [np.inf, 1]

        boundary_constraint = np.array([0, 0])

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)
            linear_constraint_matrix = np.append(boundary_constraint, 1)

        monotonicity_constraint = np.array([])
        list2 = []
        for j in range(number_of_moebius_coefficients - self.number_of_features + 1):
            # array for each  boundary condition which needs to be met due
            # to additivity (without gamma, beta positions at beginning)
            # j + 1 is number of current feature
            arr = np.zeros(number_of_moebius_coefficients)
            counter = 0
            for i in range(arr.size):
                # list of number of features
                if i + 1 in h.get_powerset_dictionary(list(range(1, self.number_of_features)), j + 1, additivity):
                    arr[i] = - 1
                    counter += 1
                elif i + 1 == self.number_of_features + j:
                    arr[i] = counter
            #monotonicity_constraint = np.append(monotonicity_constraint, arr, axis=0)
            list2.append(arr)

        bounds = opt.Bounds(lower_bound, upper_bound)
        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, [1], [1])

        return list2

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
        # pack result from opt.minimize in dict (for moebius coeff)
        result = opt.minimize()

    def _set_constraints(self, additivity):
        """ Sets up the bounds and linear constraints for the optimization problem.
            using the Bounds and Constraints classes from scipy.optimize.

        :param additivity: additivity of underlying capacity
        :return: constraints : list
                consisting of the Bounds and Constraints instances
        """
        number_of_moebius_coefficients = comb(self.number_of_features, additivity)

        # set the bounds - order is: regularization_parameter, gamma, threshold, m(T_1),...,m(T_n) with n being
        # number_of_moebius_coefficients
        lower_bound = [1, 1, 0]
        upper_bound = [np.inf, np.inf, 1]

        linear_constraint_matrix = np.array([0, 0, 0])

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)
            linear_constraint_matrix = np.append(linear_constraint_matrix, 1)

        bounds = opt.Bounds(lower_bound, upper_bound)
        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, [1], [1])

        return bounds, linear_constraint


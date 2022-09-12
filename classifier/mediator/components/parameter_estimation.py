import numpy as np
from scipy import optimize as opt
from math import comb

from . import helper as h
from .choquet_integral import ChoquetIntegral


# TODO: if additivity=None or 1 linear constraints are not to be created
class ParameterEstimation:

    def __init__(self, X, y, additivity):
        self.X = X
        self.y = y
        self.additivity = additivity

        self.number_of_features = np.shape(self.X)[1]
        self.number_of_instances = np.shape(self.X)[0]

        # gamma, beta
        self.boundary_constraint = np.array([0, 0])

    def _log_likelihood_function(self, parameters):
        #number_of_moebius_coefficients = self._get_number_of_moebius_coefficients()
        choquet = ChoquetIntegral(self.X, self.y, self.additivity)

        gamma = parameters[0]
        beta = parameters[1]
        moebius_coefficients = parameters[2:]

        result = 0

        for i in range(self.number_of_instances):
            x = self.X[i]
            y = self.y[i]

            choquet_value = 0
            for j in range(len(moebius_coefficients)):
                choquet_value += moebius_coefficients[j] * choquet.feature_minima_of_instance(x)[j + 1]

            result += -gamma*(1 - y)*(choquet_value - beta) - np.log(1 + np.exp(-gamma(choquet_value - beta)))


    def compute_parameters(self):
        """

        :param additivity:
        :param regularization_parameter:

        Returns
        ----------
        function: dict
            Containing keys corresponding to parameters: gamma, beta and subsets of [m]. The values are the
            computed values

        Notes
        ----------
        function has form of: function = {g: x, b: y, 1: m({1}), 2: m({2}),..., }
        """
        bounds, constraints = self._set_constraints()
        # pack result from opt.minimize in dict (for moebius coefficients)
        # result = opt.minimize()
        pass

    def _set_constraints(self):
        """ Sets up the bounds and linear constraints for the optimization problem.
            using the Bounds and Constraints classes from scipy.optimize.

        :param additivity: additivity of underlying capacity
        :return: constraints : list
                consisting of the Bounds and Constraints instances
        """

        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients()

        # set the bounds - order is: gamma, threshold, m(T_1),...,m(T_n) with n being
        lower_bound = [1, 0]
        upper_bound = [np.inf, 1]

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)

        bounds = opt.Bounds(lower_bound, upper_bound)

        linear_constraint_matrix = self._get_linear_constraint_matrix()

        lower_boundary_limit, upper_boundary_limit = ([1], [1])
        lower_limits = np.concatenate((lower_boundary_limit, np.zeros(linear_constraint_matrix.shape[0])), axis=0)
        upper_limits = np.concatenate((upper_boundary_limit, np.ones(linear_constraint_matrix.shape[0])), axis=0)

        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, lower_limits, upper_limits)

        return bounds, linear_constraint

    def get_subset_matrix(self, number_of_moebius_coefficients):
        matrix = []
        for j in h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), self.additivity):
            subsets = j
            arr = np.zeros(number_of_moebius_coefficients)
            for i in range(arr.size):
                if i + 1 in list(subsets.keys())[:-1] or i + 1 == list(subsets.keys())[-1]:
                    arr[i] = len(subsets[i+1])
            matrix.append(arr)

        subset_matrix = np.array(matrix)
        return subset_matrix

    def get_monotonicity_matrix(self):
        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients()
        set_list = h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), self.additivity)

        matrix = []
        for j in set_list:
            max_subset = list(j.values())[-1]
            for criterion in max_subset:
                arr = np.zeros(number_of_moebius_coefficients)
                intersection_sets = [key for key, value in j.items() if len(value.intersection({criterion})) > 0]
                monotonicity_array = [1 if index+1 in intersection_sets else 0 for index, x in enumerate(arr)]
                matrix.append(monotonicity_array)

        monotonicity_matrix = np.array(matrix)

        return monotonicity_matrix

    def _get_linear_constraint_matrix(self):
        monotonicity_matrix = self.get_monotonicity_matrix()
        boundary_constraint = np.ones(self._get_number_of_moebius_coefficients())

        linear_constraint_matrix = np.concatenate(([boundary_constraint], monotonicity_matrix), axis=0)
        linear_constraint_matrix = np.insert(linear_constraint_matrix, 0, values=0, axis=1)
        linear_constraint_matrix = np.insert(linear_constraint_matrix, 0, values=0, axis=1)
        return linear_constraint_matrix

    def _get_number_of_moebius_coefficients(self):
        number_of_moebius_coefficients = 0

        for i in range(1, self.additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        return number_of_moebius_coefficients


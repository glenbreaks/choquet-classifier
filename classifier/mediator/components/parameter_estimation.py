import numpy as np
from scipy import optimize as opt
from math import comb

from . import helper as h


# TODO: if additivity=None or 1 linear constraints are not to be created
class ParameterEstimation:

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.number_of_features = np.shape(self.X)[1]

        # gamma, beta
        self.boundary_constraint = np.array([0, 0])

    def _log_likelihood_function(self, dataset, parameters):
        pass

    def compute_parameters(self, additivity, regularization_parameter=1):
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

        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients(additivity)

        # set the bounds - order is: gamma, threshold, m(T_1),...,m(T_n) with n being
        lower_bound = [1, 0]
        upper_bound = [np.inf, 1]

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)

        bounds = opt.Bounds(lower_bound, upper_bound)

        #linear_constraint_matrix = self._get_linear_constraint_matrix(number_of_moebius_coefficients, additivity)

        #lower_limits = np.zeros(linear_constraint_matrix.shape[0])
        #upper_limits = np.ones(linear_constraint_matrix.shape[0])

        #linear_constraint = opt.LinearConstraint(linear_constraint_matrix, lower_limits, upper_limits)

        return bounds

    def get_subset_matrix(self, number_of_moebius_coefficients, additivity):
        matrix = []
        for j in h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), additivity):
            subsets = j
            arr = np.zeros(number_of_moebius_coefficients)
            for i in range(arr.size):
                if i + 1 in list(subsets.keys())[:-1] or i + 1 == list(subsets.keys())[-1]:
                    arr[i] = len(subsets[i+1])
            matrix.append(arr)

        #linear_constraint_matrix = [np.concatenate((self.boundary_constraint, np.ones(number_of_moebius_coefficients)))]
        #for arr in monotonicity_constraints:
        #    array = np.concatenate((self.boundary_constraint, arr))
        #    linear_constraint_matrix = np.append(linear_constraint_matrix, [array], axis=0)

        subset_matrix = np.array(matrix)
        return subset_matrix

    def get_monotonicity_matrix(self, additivity):
        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients(additivity)
        set_list = h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), additivity)
        #subset_matrix = self.get_subset_matrix(number_of_moebius_coefficients, additivity)

        matrix = []
        #for row in subset_matrix:
        #    index = np.max(np.nonzero(row))
        #    powerset = h.get_powerset_dictionary(list(range(1, self.number_of_features + 1)))[index]
        #    powerset_list = h.get_subset_dictionary_list(powerset, additivity)
        #    max_value = max(row)
            #trfor i in range(max_value):
            #    monotonicity_row = [1 for x in powerset ]


            #for element in powerset_list[1:]
            #    row1 = [1 for x in element]

        for j in set_list:
            counter = 0
            for i in range(len(list(j.values())[-1])):
                arr = np.zeros(number_of_moebius_coefficients)
                row = [x for x in j.values() if len(x.intersection({i+1})) > 0]
                matrix.append(row)



        return matrix


    def _get_linear_constraint_matrix(self, number_of_moebius_coefficients, additivity):
        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients(additivity)

        linear_constraint_matrix = self._get_linear_constraint_matrix(number_of_moebius_coefficients, additivity)

        linear_constraint_matrix = np.delete(linear_constraint_matrix, 0, 0)
        linear_constraint_matrix = np.delete(linear_constraint_matrix, np.s_[:2], 1)

        return linear_constraint_matrix

    def _get_number_of_moebius_coefficients(self, additivity):
        number_of_moebius_coefficients = 0

        for i in range(1, additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        return number_of_moebius_coefficients


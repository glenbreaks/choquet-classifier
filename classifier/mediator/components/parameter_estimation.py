import numpy as np
from scipy import optimize as opt
from math import comb
from typing import Dict, Any, Tuple
from . import helper as h
from .choquet_integral import ChoquetIntegral


class ParameterEstimation:
    def __init__(self, X, y, additivity, regularization_parameter=None) -> None:
        self.X = X
        self.y = y

        self.additivity = additivity

        self.regularization_parameter = regularization_parameter

        self.number_of_features = np.shape(self.X)[1]
        self.number_of_instances = np.shape(self.X)[0]

        # gamma, beta
        self.boundary_constraint = np.array([0, 0])

    def _log_likelihood_function(self, parameters: np.array) -> float:
        gamma = parameters[0]
        beta = parameters[1]
        moebius_coefficients = parameters[2:]

        choquet = ChoquetIntegral(self.additivity, moebius_coefficients)

        result = 0

        for i in range(self.number_of_instances):
            x = self.X[i]
            y = self.y[i]

            choquet_value = choquet.compute_utility_value(x)

            result += gamma * (1 - y) * (choquet_value - beta) + np.log(1 + np.exp(-gamma * (choquet_value - beta)))

        result += self._l1_regularization(moebius_coefficients)

        return result

    def compute_parameters(self) -> Dict[str, Any]:
        bounds, constraints = self._set_constraints()

        # set random normalized starting vector
        x0 = np.concatenate(([1], np.ones(1 + self._get_number_of_moebius_coefficients())), axis=0)
        x0 = x0 / np.sum(x0)

        result = opt.minimize(self._log_likelihood_function, x0, options={'verbose': 1}, bounds=bounds,
                              method='trust-constr', constraints=constraints)

        set_list = h.get_powerset_dictionary(list(range(1, self.number_of_features + 1)), self.additivity)
        parameter_dict = {'gamma': result.x[0], 'beta': result.x[1]}

        moebius_results = result.x[2:]

        # Create a mapping of index to subset
        index_to_subset = {index: subset for index, subset in enumerate(set_list.values())}

        # Efficiently construct moebius_dict using the mapping
        moebius_dict = {index_to_subset[i]: moebius_results[i] for i in range(len(moebius_results))}

        parameter_dict.update(moebius_dict)
        return parameter_dict

    def _set_constraints(self) -> Tuple[opt.Bounds, opt.LinearConstraint]:
        num_moebius_coeff = self._get_number_of_moebius_coefficients()

        # Bounds setup using list comprehension for scalability and readability
        lower_bound = [0.1, 0] + [0] * num_moebius_coeff
        upper_bound = [np.inf, 1] + [1] * num_moebius_coeff

        bounds = opt.Bounds(lower_bound, upper_bound)

        linear_constraint_matrix = self._get_linear_constraint_matrix()

        lower_limits = np.zeros(linear_constraint_matrix.shape[0])
        upper_limits = np.ones(linear_constraint_matrix.shape[0])

        lower_limits[0], upper_limits[0] = 1, 1

        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, lower_limits, upper_limits)

        return bounds, linear_constraint

    def get_monotonicity_matrix(self) -> np.ndarray:
        num_moebius_coeffs = self._get_number_of_moebius_coefficients()
        set_list = h.get_subset_dictionary_list(range(1, self.number_of_features + 1), self.additivity)

        matrix = []
        for subset in set_list:
            max_subset = list(subset.values())[-1]
            for criterion in max_subset:
                row = [(1 if (index + 1) in subset and criterion in subset[index + 1] else 0) for index in
                       range(num_moebius_coeffs)]
                matrix.append(row)

        monotonicity_matrix = np.array(matrix)

        return monotonicity_matrix

    def _get_linear_constraint_matrix(self) -> np.ndarray:
        monotonicity_matrix = self.get_monotonicity_matrix()
        boundary_constraint = np.ones(self._get_number_of_moebius_coefficients())

        linear_constraint_matrix = np.concatenate(([boundary_constraint], monotonicity_matrix), axis=0)
        linear_constraint_matrix = np.insert(np.insert(linear_constraint_matrix, 0, values=0, axis=1), 0, values=0, axis=1)

        return linear_constraint_matrix

    def _get_number_of_moebius_coefficients(self) -> int:
        number_of_moebius_coefficients = 0

        for i in range(1, self.additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        return number_of_moebius_coefficients

    def _l1_regularization(self, moebius_coefficients: np.ndarray) -> float:
        return self.regularization_parameter * sum(abs(coefficient) for coefficient in moebius_coefficients)

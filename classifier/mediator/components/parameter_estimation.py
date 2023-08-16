import numpy as np
from scipy import optimize as opt
from math import comb

from . import helper as h
from .choquet_integral import ChoquetIntegral


class ParameterEstimation:
    """Class to solve the optimization problem for given
    data set X, y.

    Parameters
    -------
    X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features. The number of features
            has to be more or equal to the additivity.

    y : array-like of shape (n_samples,)
            Target labels to X.

    additivity : int, default=1
            The additivity of the underlying fuzzy measure. Additivity takes interaction
            between features into account, i.e. the additivity value determines the maximum
            number of interacting features and hence the maximum number of moebius coefficients
            to be estimated. More precisely, with a k-additive measure, sum_i (n over i),
            with n = n_features, coefficients. The default value 1 represents
            a simple additive measure with no interaction between features.

    regularization_parameter: in, default=None
            the regularization parameter of the L1-Regularization in the parameter estimation.
            Determines the strength of regularization of the fitting process.
    """

    def __init__(self, X, y, additivity, regularization_parameter=None):
        self.X = X
        self.y = y

        self.additivity = additivity

        self.regularization_parameter = regularization_parameter

        self.number_of_features = np.shape(self.X)[1]
        self.number_of_instances = np.shape(self.X)[0]

        # gamma, beta
        self.boundary_constraint = np.array([0, 0])

    def _log_likelihood_function(self, parameters):
        """Log-Likelihood function which is subject to minimization.

        Takes a parameter array and calculates the utility value
        of every instance x of given data set X and employs
        all parameters, instances and corresponding labels
        into the log-likelihood function.

        Parameters
        -------
        parameters: array-like of shape, (1, n_parameters)
            Array containing all parameters to be optimized.
            Order: [scaling, threshold, [moebius_coefficients]]

        Returns
        -------
        result: float
            The calculated value for the log-likelihood.
        """

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

    def compute_parameters(self):
        """Function to solve the minimization problem, using the scipy-package.

        Uses the constructed bounds and linear constraints to build up
        the whole optimization problem and sets a starting vector x0.
        The minimize method of the optimize class of scipy is then
        called with the log-likelihood, starting vector x0 and the bounds
        and linear constraints. used method is "trust-constr"

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

        # set random normalized starting vector
        x0 = np.concatenate(([1], np.ones(1 + self._get_number_of_moebius_coefficients())), axis=0)
        x0 = x0 / np.sum(x0)

        result = opt.minimize(self._log_likelihood_function, x0, options={'verbose': 1}, bounds=bounds,
                              method='trust-constr', constraints=constraints)

        set_list = h.get_powerset_dictionary(list(range(1, self.number_of_features + 1)), self.additivity)
        parameter_dict = {'gamma': result.x[0], 'beta': result.x[1]}

        moebius_results = result.x[2:]
        moebius_dict = {subset: value for key1, subset in set_list.items()
                        for value in moebius_results if key1 == np.where(moebius_results == value)[0] + 1}

        parameter_dict.update(moebius_dict)
        return parameter_dict

    def _set_constraints(self):
        """ Set up the bounds and linear constraints for the optimization problem
            using the Bounds and Constraints classes from scipy.optimize.

        Returns
        -------
        bounds, constraints: Bounds, LinearConstraints
            Returns the datatypes Bounds and LinearConstraints,
            used for the optimization problem.
        """

        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients()

        # set the bounds - order is: gamma, threshold, m(T_1),...,m(T_n) with n being
        lower_bound = [0.1, 0]
        upper_bound = [np.inf, 1]

        for i in range(number_of_moebius_coefficients):
            lower_bound.append(0)
            upper_bound.append(1)

        bounds = opt.Bounds(lower_bound, upper_bound)

        linear_constraint_matrix = self._get_linear_constraint_matrix()

        lower_boundary_limit, upper_boundary_limit = ([1], [1])
        lower_limits = np.concatenate((lower_boundary_limit, np.zeros(linear_constraint_matrix.shape[0] - 1)), axis=0)
        upper_limits = np.concatenate((upper_boundary_limit, np.ones(linear_constraint_matrix.shape[0] - 1)), axis=0)

        lower_limits = lower_limits.flatten()
        upper_limits = upper_limits.flatten()

        linear_constraint = opt.LinearConstraint(linear_constraint_matrix, lower_limits, upper_limits)

        return bounds, linear_constraint

    def get_monotonicity_matrix(self):
        """Set the needed monotonicity constraints for all
        moebius coefficients

        Create all monotonicity constraints for the optimization
        problem and aggregate them in a matrix.

        Returns
        -------
        monotonicity_matrix: array-like of shape (n_monotonicity_constraints, n_moebius_coefficients)
            The matrix representing all monotonicity constraints
            for moebius coefficients.
        """

        number_of_moebius_coefficients = self._get_number_of_moebius_coefficients()
        set_list = h.get_subset_dictionary_list(list(range(1, self.number_of_features + 1)), self.additivity)
        matrix = []
        for set in set_list:
            max_subset = list(set.values())[-1]
            for criterion in max_subset:
                arr = np.zeros(number_of_moebius_coefficients)
                intersection_sets = [key for key, value in set.items() if len(value.intersection({criterion})) > 0]
                monotonicity_array = [1 if index + 1 in intersection_sets else 0 for index, x in enumerate(arr)]
                matrix.append(monotonicity_array)

        monotonicity_matrix = np.array(matrix)

        return monotonicity_matrix

    def _get_linear_constraint_matrix(self):
        """Set the linear constraint matrix needed for the scipy-optimize
        package.

        Use monotonicity matrix and complement according to the
        scipy template for LinearConstraints to use it
        in the minimization problem. Adding boundary condition.

        Returns
        -------
        linear_constraint_matrix: array-like of shape (n_linear_constraints, n_parameters)
            The matrix representing all linear constraints
            for needed parameters.
        """

        monotonicity_matrix = self.get_monotonicity_matrix()
        boundary_constraint = np.ones(self._get_number_of_moebius_coefficients())

        linear_constraint_matrix = np.concatenate(([boundary_constraint], monotonicity_matrix), axis=0)
        linear_constraint_matrix = np.insert(linear_constraint_matrix, 0, values=0, axis=1)
        linear_constraint_matrix = np.insert(linear_constraint_matrix, 0, values=0, axis=1)

        return linear_constraint_matrix

    def _get_number_of_moebius_coefficients(self):
        """Computes the number of needed moebius coefficients
        according to number of features and additivity.

        Returns
        -------
        number_of_moebius_coefficients: int
            Number of needed moebius coefficients for
            given problem
        """

        number_of_moebius_coefficients = 0

        for i in range(1, self.additivity + 1):
            number_of_moebius_coefficients += comb(self.number_of_features, i)

        return number_of_moebius_coefficients

    def _l1_regularization(self, moebius_coefficients):
        """Calculates the l1-Regularizer in the log-likelihood
        function to prevent overfitting.

        Parameters
        -------
        moebius_coefficients: array-like of shape (1, n_moebius_coefficients)
            Array containing all moebius coefficients

        Returns
        -------
        The calculated L1-Regularization for given problem
        """

        return self.regularization_parameter * sum(abs(coefficient) for coefficient in moebius_coefficients)

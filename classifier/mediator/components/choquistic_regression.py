
import numpy as np
from typing import Dict, List
from .choquet_integral import ChoquetIntegral


class ChoquisticRegression:
    """Choquistic Regression

    This class implements the Choquistic Regression calculation
    and stores all estimated parameters

    Attributes
    ----------
    additivity : int
        Additivity of fuzzy measure.

    scaling : float
        Scaling parameter for the regression.

    threshold : float
        Threshold parameter for the regression.

    moebius_coefficients : List[float]
        List of Moebius coefficients.
    """

    def __init__(self, additivity: int, parameters: Dict[str, float]) -> None:
        """
        Parameters
        ----------
        additivity : int
            Additivity of fuzzy measure.

        parameters : Dict[str, float]
            Dictionary, describing all fitted parameters (scaling, threshold,
            moebius coefficients) in the following manner:
            {'gamma': scaling, 'beta': threshold, frozenset({1}): m({1}),...}
        """
        self.additivity = additivity
        self.scaling = parameters['gamma']
        self.threshold = parameters['beta']
        self.moebius_coefficients = list(parameters.values())[2:]

    def compute_regression_value(self, instance: np.ndarray) -> float:
        """Computes the regression value for given instance
         with fitted parameters.

        Parameters
        ----------
        instance: np.ndarray
            Instance, where n_features is the number of features.

        Returns
        -------
        regression_value : float
            Probability of positive class, computed by the choquistic
            regression model.
        """
        choquet_integral = ChoquetIntegral(self.additivity, self.moebius_coefficients)

        gamma = self.scaling
        beta = self.threshold

        utility_value = choquet_integral.compute_utility_value(instance)

        regression_value = 1 / (1 + np.exp(-gamma * (utility_value - beta)))

        return regression_value
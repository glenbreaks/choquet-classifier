import numpy as np
from scipy import optimize as opt

from . import moebius_transformation
from .choquet_integral import ChoquetIntegral
from . import helper as h


    # TODO: function for setting bounds: (eta), gamma, beta
    # TODO: function for setting linear constraints: moebius normalization, moebius monotonicity in relation to additivity
class ParameterEstimation:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.number_of_features = np.shape(self.X)[1]

    def _log_likelihood_function(self,):
        pass

    def optimize_parameters(self, additivity, eta):

        #set bounds
        bounds = opt.Bounds
        #set linear constraints

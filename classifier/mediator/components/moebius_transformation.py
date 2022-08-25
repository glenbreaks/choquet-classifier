import numpy as np



class MoebiusTransformation:

    def __init__(self, additivity, X, y):
        self.additivity = additivity
        self.X = X
        self.y = y
        self.number_of_features = np.shape(self.X)[1]
        pass

    #def moebius_transform(self, additivity, X, y):
    #    result = list()
    #    for i in range(additivity):

    #    return array

    def _subset_coefficient(self, size):
        return pow(-1, self.number_of_features - size)


    #def _update_powerset
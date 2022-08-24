import numpy as np
from itertools import chain, combinations


class MoebiusTransformation:

    def __init__(self, additivity, X, y):
        pass

    def moebius_transform(self, additivity, X, y):
        pass

    def get_powerset(self, s):

        result = list()
        items = list(s)
        powerset = chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))

        for item in powerset:
            result.append(frozenset(item))

        return np.array(result)

    #def _update_powerset
import numpy as np


def get_feature_subset(x, index):
    """Get a subset A_i={p(c_i), ..., p(c_m)} with a permutation p"""

    m = len(x)
    sorted_x = np.sort(x)
    permutation = _get_permutation_position(x, sorted_x)

    result = [permutation[i] for i in range(index, m + 1)]
    return frozenset(result)


def _get_permutation_position(x, sorted_x):
    result = {}

    # to cover duplicate values, create array instead of pure values
    for i in range(len(x)):
        result[i+1] = np.nonzero(x == sorted_x[i])[0][0] + 1

    return result

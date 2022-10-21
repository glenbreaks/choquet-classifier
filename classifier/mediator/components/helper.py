import numpy as np
from itertools import chain, combinations


def get_feature_subset(x, index):
    """Get a subset A_i={p(c_i), ..., p(c_m)} with a permutation p"""

    m = len(x)
    sorted_x = np.sort(x)
    permutation = _get_permutation_position(x, sorted_x)

    result = [permutation[i] for i in range(index, m + 1)]
    return frozenset(result)


def get_powerset(s, additivity):
    result = list()
    items = s
    if additivity is not None:
        powerset = _get_additivity_powerset(items, additivity)
    else:
        powerset = chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))

    for item in powerset:
        result.append(frozenset(item))

    return np.array(result)


def get_powerset_dictionary(s, additivity):
    powerset = get_powerset(s, additivity)
    dict_powerset = dict(enumerate(powerset[1:].flatten(), 1))
    return dict_powerset


def get_subset_dictionary_list(s, additivity):
    powerset_dict = get_powerset_dictionary(s, additivity)
    dict_list = list()

    if additivity == 1:
        subset_list = list(powerset_dict.values())
    else:
        subset_list = list(powerset_dict.values())[len(s):]

    additivity_subset_list = [x for x in subset_list if len(x) <= additivity]

    for i in additivity_subset_list:
        dict_list.append({key: val for key, val in powerset_dict.items() if val <= i})

    return dict_list


def _get_additivity_powerset(items, additivity):
    result = list()

    if len(items) < additivity:
        raise Exception("additivity must be less than or equal to number of features")

    powerset = chain.from_iterable(combinations(items, r) for r in range(additivity + 1))

    for item in powerset:
        result.append(frozenset(item))

    return np.array(result)


def _get_permutation_position(x, sorted_x):
    result = {}

    # to cover duplicate values, create array instead of pure values
    for i in range(len(x)):
        result[i + 1] = np.nonzero(x == sorted_x[i])[0][0] + 1

    return result

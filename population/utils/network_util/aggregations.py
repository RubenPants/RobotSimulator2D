"""
aggregations.py

Each of the supported aggregation functions.
"""
from functools import reduce
from operator import mul

from neat.math_util import mean, median2

from utils.dictionary import D_MAX, D_MAX_ABS, D_MEAN, D_MEDIAN, D_MIN, D_PRODUCT, D_SUM


def product_aggregation(x):  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    return max(x)


def min_aggregation(x):
    return min(x)


def max_abs_aggregation(x):
    return max(x, key=abs)


def median_aggregation(x):
    return median2(x)


def mean_aggregation(x):
    return mean(x)


str_to_aggregation = {
    D_MAX:     max_aggregation,
    D_MAX_ABS: max_abs_aggregation,
    D_MEAN:    mean_aggregation,
    D_MEDIAN:  median_aggregation,
    D_MIN:     min_aggregation,
    D_PRODUCT: product_aggregation,
    D_SUM:     sum_aggregation,
}

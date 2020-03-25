"""
aggregation.py

Node aggregation attribute represented by a string which is used during neuroevolution. The aggregation determines how
to combine multiple input-sources (i.e. average, sum, ...).
"""
from random import choice, random

from configs.genome_config import GenomeConfig


def cross(v1, v2, ratio: float = 0.5):
    """
    Inherit one of the two aggregation attributes from the given parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    return v1 if random() < ratio else v2


def init(cfg: GenomeConfig):
    """Put aggregation to specified aggregation in config, random aggregation chosen if not specified."""
    if cfg.aggregation_default.lower() in ('none', 'random'):
        return choice(cfg.aggregation_options)
    return cfg.aggregation_default


def mutate(v, cfg: GenomeConfig):
    """
    Mutate the aggregation to one of the possible aggregation attributes. Note that mutation *may* change the current
    value, since there is still a chance the same aggregation is chosen again.
    """
    r = random()
    if r < cfg.aggregation_mutate_rate:
        return choice(cfg.aggregation_options)
    return v

"""
bias.py

Bias attribute represented by a float which is used during neuroevolution.
"""
from random import gauss, random

from numpy import clip

from configs.genome_config import GenomeConfig


def cross(v1, v2, ratio: float = 0.5):
    """
    Inherit one of the two bias attributes from the given parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    return v1 if random() < ratio else v2


def init(cfg: GenomeConfig):
    """Random initialized bias value, calculated via a normal distribution."""
    return clip(gauss(cfg.bias_init_mean, cfg.bias_init_stdev),
                a_min=cfg.bias_min_value,
                a_max=cfg.bias_max_value)


def mutate(v, cfg: GenomeConfig):
    """Mutate the given bias-value based on the provided GenomeConfig file."""
    # Check if value is mutated
    r = random()
    if r < cfg.bias_mutate_rate:
        return clip(v + gauss(0.0, cfg.bias_mutate_power), a_min=cfg.bias_min_value, a_max=cfg.bias_max_value)
    
    # Check if value is replaced
    if r < cfg.bias_replace_rate + cfg.bias_mutate_rate:
        return init(cfg)
    
    # No changes, return original value
    return v

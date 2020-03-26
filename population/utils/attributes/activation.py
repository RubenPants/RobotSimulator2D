"""
activation.py

Node activation attribute represented by a string which is used during neuroevolution.
"""
from random import choice, random

from configs.genome_config import GenomeConfig


def cross(v1, v2, ratio: float = 0.5):
    """
    Inherit one of the two activation attributes from the given parents.
    
    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    return v1 if random() < ratio else v2


def init(cfg: GenomeConfig):
    """Put activation to specified activation in config, random activation chosen if not specified."""
    if cfg.activation_default.lower() in ('none', 'random'):
        return choice(list(cfg.activation_options.keys()))
    return cfg.activation_default


def mutate(v, cfg: GenomeConfig):
    """
    Mutate the activation to one of the possible activation attributes. Note that mutation *may* change the current
    value, since there is still a chance the same activation is chosen again.
    """
    r = random()
    if r < cfg.activation_mutate_rate:
        return choice(list(cfg.activation_options.keys()))
    return v

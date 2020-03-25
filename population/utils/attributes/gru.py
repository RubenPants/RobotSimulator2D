"""
gru_bias.py

GRU attribute, used for both biases as weights, represented by a PyTorch tensor which is used during neuroevolution.
Vector are represented by numpy-arrays.
"""
from random import gauss, random

from numpy import clip, zeros

from configs.genome_config import GenomeConfig


def cross_1d(v1, v2, ratio: float = 0.5):
    """
    Cross the two GRU-vector attributes from both parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    assert v1.shape == v2.shape == (3,)
    result = zeros(v1.shape, dtype=float)
    for i in range(len(v1)):
        result[i] = v1[i] if random() < ratio else v2[i]
    return result


def cross_2d(v1, v2, ratio: float = 0.5):
    """
    Cross the two GRU-vector attributes from both parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    assert v1.shape == v2.shape
    assert len(v1.shape) == 2
    result = zeros(v1.shape, dtype=float)
    for col_i in range(result.shape[1]):
        result[:, col_i] = cross_1d(v1=v1[:, col_i], v2=v2[:,col_i], ratio=ratio)
    return result


def init(cfg: GenomeConfig, input_size: int = None):
    """Initialize a GRU-vector"""
    t = zeros((3, input_size), dtype=float) if input_size is not None else zeros((3,), dtype=float)
    
    # Query the FloatAttribute for each initialization of the tensor's parameters
    for t_index in range(len(t)):
        t[t_index] = single_init(cfg)
    return t


def single_init(cfg: GenomeConfig):
    """Random initialized floating GRU value, calculated via a normal distribution."""
    return clip(gauss(cfg.gru_init_mean, cfg.gru_init_stdev), a_min=cfg.gru_min_value, a_max=cfg.gru_max_value)


def mutate_1d(v, cfg: GenomeConfig):
    """Mutate a 1-dimensional GRU-vector."""
    for i, elem in enumerate(v):  # TODO: Clip values!
        result = mutate(elem, cfg=cfg)
        v[i] = result
    return v


def mutate_2d(v, cfg: GenomeConfig, mapping=None):
    """Mutate a 2-dimensional GRU-vector. If mapping is given, it should denote which columns to mutate."""
    for col_i in range(v.shape[1]):
        if mapping and not mapping[col_i]: continue
        v[:, col_i] = mutate_1d(v[:, col_i], cfg=cfg)
    return v


def mutate(v, cfg: GenomeConfig):
    """Mutate the given GRU-value based on the provided GenomeConfig file."""
    # Check if value must mutate
    r = random()
    if r < cfg.gru_mutate_rate:
        return clip(v + gauss(0.0, cfg.gru_mutate_power), a_min=cfg.gru_min_value, a_max=cfg.gru_max_value)
    
    # Check if value must be replaced
    if r < cfg.gru_replace_rate + cfg.gru_mutate_rate:
        return single_init(cfg)
    
    # No changes, return original value
    return v

"""
gru_bias.py

GRU attribute, used for both biases as weights, represented by a PyTorch tensor which is used during neuroevolution.
"""
from random import gauss, random

from numpy import clip, zeros
from torch import float64, tensor

from configs.genome_config import GenomeConfig


def cross_1d(v1, v2, ratio: float = 0.5):
    """
    Cross the two GRU-vector attributes from both parents.

    :param v1: Value of the first parent (self)
    :param v2: Value of the second parent (other)
    :param ratio: Probability that the first parent's attribute is chosen
    """
    assert v1.shape == v2.shape == (3,)
    result = zeros(v1.shape)
    for i in range(len(v1)):
        result[i] = float(v1[i]) if random() < ratio else float(v2[i])
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
    result = zeros(v1.shape)
    for col_i in range(result.shape[1]):
        result[:, col_i] = cross_1d(v1=v1[:, col_i], v2=v2[:col_i], ratio=ratio)
    return result


def init(cfg: GenomeConfig, input_size: int = None):
    """Initialize a GRU-vector"""
    t = tensor(zeros((3, input_size)), dtype=float64) if input_size is not None else tensor(zeros((3,)), dtype=float64)
    
    # Query the FloatAttribute for each initialization of the tensor's parameters
    for t_index in range(len(t)):
        t[t_index] = single_init(cfg)
    return t


def single_init(cfg: GenomeConfig):
    """Random initialized floating GRU value, calculated via a normal distribution."""
    return clip(gauss(cfg.gru_init_mean, cfg.gru_init_stdev), a_min=cfg.gru_min_value, a_max=cfg.gru_max_value)


def mutate_1d(t: tensor, cfg: GenomeConfig):
    """Mutate a 1-dimensional GRU-vector."""
    for i, v in enumerate(t):
        t[i] = mutate(v, cfg=cfg)
    return t


def mutate_2d(t: tensor, cfg: GenomeConfig, mapping=None):
    """Mutate a 2-dimensional GRU-vector. If mapping is given, it should denote which columns to mutate."""
    for col_i in range(t.shape[1]):
        if mapping and not mapping[col_i]: continue
        t[:, col_i] = mutate_1d(t[:, col_i], cfg=cfg)
    return t


def mutate(v, cfg: GenomeConfig):
    """Mutate the given GRU-value based on the provided GenomeConfig file."""
    # Check if value is mutated
    r = random()
    if r < cfg.gru_mutate_rate:
        return clip(v + gauss(0.0, cfg.gru_mutate_power), a_min=cfg.gru_min_value, a_max=cfg.gru_max_value)
    
    # Check if value is replaced
    if r < cfg.gru_replace_rate + cfg.gru_mutate_rate:
        return init(cfg)
    
    # No changes, return original value
    return v

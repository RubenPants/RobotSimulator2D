"""
activations.py

Each of the supported activation functions.
"""
import torch

from utils.dictionary import D_CLAMPED_ABS, D_CLAMPED_LINEAR, D_COS, D_EXP, D_EXP_ABS, D_GAUSS, D_HAT, D_SIGMOID, \
    D_SIN, D_TANH


def clamped_abs_activation(x):
    return min(1.0, torch.abs(x) / 3)


def clamped_linear_activation(x):
    return max(-1.0, min(1.0, x / 3))


def cos_activation(x):
    return torch.cos(x * 3.14)


def exponential_activation(x):
    return 1 - 1 / (2 * torch.exp(torch.abs(3 * x)))


def exponential_abs_activation(x):
    return torch.abs(x) ** (1 / torch.abs(x))


def gauss_activation(x):
    return torch.exp(-x ** 2)


def hat_activation(x):
    return max(0.0, 1 - torch.abs(x / 3))


def sigmoid_activation(x):
    return torch.sigmoid(2 * x)


def sin_activation(x):
    return torch.sin(x * 3.14)


def tanh_activation(x):
    return torch.tanh(x)


str_to_activation = {
    D_CLAMPED_ABS:    clamped_abs_activation,
    D_CLAMPED_LINEAR: clamped_linear_activation,
    D_COS:            cos_activation,
    D_EXP:            exponential_activation,
    D_EXP_ABS:        exponential_abs_activation,
    D_GAUSS:          gauss_activation,
    D_HAT:            hat_activation,
    D_SIGMOID:        sigmoid_activation,
    D_SIN:            sin_activation,
    D_TANH:           tanh_activation,
}

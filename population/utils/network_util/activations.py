"""
activations.py

Each of the supported activation functions.
"""
import torch
import torch.nn.functional as F

from utils.dictionary import D_ABS, D_CUBE, D_GAUSS, D_GELU, D_IDENTITY, D_RELU, D_SIGMOID, D_SIN, D_SQUARE, D_TANH


def sigmoid_activation(x):
    return torch.sigmoid(3 * x)


def tanh_activation(x):
    return torch.tanh(1.5 * x)


def abs_activation(x):
    return torch.abs(x)


def gauss_activation(x):
    return torch.exp(-2.0 * x ** 2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)


def gelu_activation(x):
    return F.gelu(x)


def cube_activation(x):
    return x ** 3


def square_activation(x):
    return x ** 2


str_to_activation = {
    D_SIGMOID:  sigmoid_activation,
    D_TANH:     tanh_activation,
    D_ABS:      abs_activation,
    D_GAUSS:    gauss_activation,
    D_IDENTITY: identity_activation,
    D_SIN:      sin_activation,
    D_RELU:     relu_activation,
    D_GELU:     gelu_activation,
    D_CUBE:     cube_activation,
    D_SQUARE:   square_activation,
}

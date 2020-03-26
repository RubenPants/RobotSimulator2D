"""
activations.py

Each of the supported activation functions.
"""
import torch
import torch.nn.functional as F

from utils.dictionary import D_ABS, D_GAUSS, D_GELU, D_IDENTITY, D_LINEAR, D_RELU, D_SIGMOID, D_SIN, D_TANH


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def abs_activation(x):
    return torch.abs(x)


def linear_activation(x):
    return x


def gauss_activation(x):
    return torch.exp(-5.0 * x ** 2)


def identity_activation(x):
    return x


def sin_activation(x):
    return torch.sin(x)


def relu_activation(x):
    return F.relu(x)


def gelu_activation(x):
    return F.gelu(x)


str_to_activation = {
    D_SIGMOID:  sigmoid_activation,
    D_TANH:     tanh_activation,
    D_ABS:      abs_activation,
    D_LINEAR:   linear_activation,
    D_GAUSS:    gauss_activation,
    D_IDENTITY: identity_activation,
    D_SIN:      sin_activation,
    D_RELU:     relu_activation,
    D_GELU:     gelu_activation,
}

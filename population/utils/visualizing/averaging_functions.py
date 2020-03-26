"""
averaging_functions.py

Functions used to improve visualizations.
"""
import numpy as np


def Forward(values, _):
    """Simply forwarding the values."""
    return values


def EMA(values, window: int = 5):
    """Calculates the exponential moving average over a specified time-window."""
    window = min(window, len(values) - 1)
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(values, weights)[:len(values)]
    ema[:window] = ema[window]
    return ema


def SMA(values, window: int = 5):
    """Calculates the simple moving average."""
    window = min(window, len(values) - 1)
    weights = np.repeat(1., window) / window
    sma = np.convolve(values, weights)[:len(values)]
    sma[:window] = sma[window]
    return sma

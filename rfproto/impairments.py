import numpy as np


def gaussian_noise(A, t):
    """Creates a signal of gaussian noise of t.size with amplitude A"""
    return A * (np.random.randn(t.size) + 1j * np.random.randn(t.size))

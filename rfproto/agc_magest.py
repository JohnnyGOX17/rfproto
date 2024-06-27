import numpy as np
from . import utils


def alphaMax_betaMin(iq_data: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Performs Alpha-Max Beta-Min magnitude estimation

    Parameters
    ----------
    iq_data : ndarray
        Complex (I/Q) sample data vector
    alpha : float
        Alpha coefficient
    beta : float
        Beta coefficient

    Returns
    -------
    y : ndarray
        Magnitude estimates
    """
    alphaMax = alpha * np.maximum(np.abs(np.real(iq_data)), np.abs(np.imag(iq_data)))
    betaMin = beta * np.minimum(np.abs(np.real(iq_data)), np.abs(np.imag(iq_data)))
    return alphaMax + betaMin


def alpha_beta_min_err_coeff() -> tuple[float, float]:
    """Returns tuple (alpha, beta) coefficients that are the closest geometric
    approximation (minimum error)
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Alpha_max_plus_beta_min_algorithm
    """
    a = (2 * np.cos(np.pi / 8)) / (1 + np.cos(np.pi / 8))
    b = (2 * np.sin(np.pi / 8)) / (1 + np.cos(np.pi / 8))
    return a, b


def alpha_beta_min_err_zero_mean_coeff() -> tuple[float, float]:
    """Returns tuple (alpha, beta) coefficients that are the closest geometric
    approximation (minimum error), and then modified to remove mean error/bias."""
    return 0.9481075395270311, 0.39271900146028155


def RSSI(mag_est: np.ndarray, full_scale_val: int = 1) -> np.ndarray:
    """Returns the Received Signal Strength Indicator (RSSI) value in dBFS
    Parameters
    ----------
    mag_est : ndarray
        Array of magnitude values/estimates
    full_scale_val : int, optional, default: 1
        If `mag_est` are FXP/integer values, they are divided by `full_scale_val`
        to normalize output to dB full-scale (dBFS)
    """
    return utils.mag_to_dB(mag_est / full_scale_val)

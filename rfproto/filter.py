import numpy as np


def RaisedCosine(
    sample_rate: float, symbol_rate: float, alpha: float, num_taps: int
) -> np.ndarray:
    """Generates Raised Cosine impulse response

    Parameters
    ----------
    sample_rate : float
        Sample rate of signal (Hz)
    symbol_rate : float
        Symbol rate of signal (Hz)
    alpha : float
        Rolloff (:math: `\\alpha`) of impulse response
    num_taps : int
        Number of filter taps

    References
    ----------
    .. [1] [Raised Cosine Filter - Wikipedia](https://en.wikipedia.org/wiki/Raised-cosine_filter)
    """
    h_rc = np.zeros(num_taps)
    Ts = sample_rate / symbol_rate
    for i in range(num_taps):
        t = (i - (num_taps // 2)) / Ts
        sinc_val = (1.0 / Ts) * np.sinc(t)
        cos_frac = np.cos(np.pi * alpha * t) / (1.0 - ((2.0 * alpha * t) ** 2))
        h_rc[i] = sinc_val * cos_frac
    return h_rc


def RootRaisedCosine(
    sample_rate: float, symbol_rate: float, alpha: float, num_taps: int
) -> np.ndarray:
    """Returns unity-gain, floating-point Root-Raised Cosine (RRC) filter coefficients

    Parameters
    ----------
    sample_rate : float
        Sample rate of signal (Hz)
    symbol_rate : float
        Symbol rate of signal (Hz)
    alpha : float
        Rolloff (:math: `\\alpha`) of impulse response
    num_taps : int
        Number of filter taps. _Should_ be odd for pulse shaping applications, though need not be

    References
    ----------
    .. [1] [Root Raised Cosine (RRC) Filters and Pulse Shaping in Communication Systems - NASA](https://ntrs.nasa.gov/api/citations/20120008631/downloads/20120008631.pdf)
    """
    h_rrc = np.zeros(num_taps)
    weight_sum = 0.0

    if (alpha <= 0.0) or (alpha > 1.0):
        raise ValueError(f"Alpha of {alpha} is not in range (0, 1]")

    for i in range(num_taps):
        idx = round((i - (num_taps / 2.0)) + 0.5)
        t = idx / sample_rate

        if t == 0.0:
            h_rrc[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif abs(t) == 1.0 / (4.0 * alpha * symbol_rate):
            tmp_a = (1.0 + (2.0 / np.pi)) * np.sin(np.pi / (4.0 * alpha))
            tmp_b = (1.0 - (2.0 / np.pi)) * np.cos(np.pi / (4.0 * alpha))
            h_rrc[i] = (alpha / np.sqrt(2.0)) * (tmp_a + tmp_b)
        else:
            tmp_a = np.sin(np.pi * t * (1.0 - alpha) * symbol_rate)
            tmp_b = (
                4.0
                * alpha
                * t
                * symbol_rate
                * np.cos(np.pi * t * (1.0 + alpha) * symbol_rate)
            )
            tmp_c = (
                np.pi * t * (1.0 - (4.0 * alpha * t * symbol_rate) ** 2.0) * symbol_rate
            )
            h_rrc[i] = (tmp_a + tmp_b) / tmp_c

        # filter with unity passband gain has coefficients that sum to 1
        weight_sum += h_rrc[i]
    return h_rrc / weight_sum


def UnityResponse(num_taps: int) -> np.ndarray:
    """Returns unity-gain, passthrough filter coefficients"""
    h_unity = np.zeros(num_taps)
    h_unity[num_taps // 2] = 1.0
    return h_unity

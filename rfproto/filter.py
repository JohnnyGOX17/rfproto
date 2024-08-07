"""Implements filter methods"""

import numpy as np


def RaisedCosine(
    sample_rate: float, symbol_rate: float, alpha: float, num_taps: int
) -> np.ndarray:
    """Generates [Raised Cosine filter](https://en.wikipedia.org/wiki/Raised-cosine_filter) impulse response as:

    ![impulse_response](https://wikimedia.org/api/rest_v1/media/math/render/svg/8b38e84f30fc32db087bddc9570266683691084c)

    Args:
        sample_rate: Sample rate of signal (Hz)
        symbol_rate: Symbol rate of signal (Hz)
        alpha: Roll-off ($\\alpha$) of impulse response
        num_taps: Number of filter taps

    Returns:
        Array of filter coefficients
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
    """Returns unity-gain, floating-point Root-Raised Cosine (RRC) filter coefficients.

    Unity passband gain is achieved by ensuring returned coefficients sum to `1.0`:

    $$ \\hat{h(t)} = h(t) / \\sum h(t) $$

    For more information, see [Root Raised Cosine (RRC) Filters and Pulse Shaping in Communication Systems - NASA](https://ntrs.nasa.gov/api/citations/20120008631/downloads/20120008631.pdf).

    Args:
        sample_rate: Sample rate of signal (Hz)
        symbol_rate: Symbol rate of signal (Hz)
        alpha: Roll-off ($\\alpha$) of impulse response
        num_taps: Number of filter taps

    Returns:
        Array of filter coefficients
    """
    h_rrc = np.zeros(num_taps)
    weight_sum = 0.0

    if (alpha <= 0.0) or (alpha > 1.0):
        raise ValueError(f"Alpha of {alpha} is not in range (0, 1]")

    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd!")

    for i in range(num_taps):
        idx = i - (num_taps / 2.0) + 0.5
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
    """Returns unity-gain, passthrough filter coefficients
    Args:
        num_taps: Number of filter taps

    Returns:
        Array of filter coefficients
    """
    h_unity = np.zeros(num_taps)
    h_unity[num_taps // 2] = 1.0
    return h_unity

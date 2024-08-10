import numpy as np

from . import utils


def awgn(avg_pwr_dB: float, num_samples: int) -> np.ndarray:
    """Creates an [Additive White Gaussian Noise (AWGN)](https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise)
    complex, zero-mean signal with normal distribution. See also [this article on AWGN](https://www.wavewalkerdsp.com/2022/06/01/how-to-create-additive-white-gaussian-noise-awgn/).

    A signal with a given channel SNR can be modeled by adding this noise signal to the desired
    signal. This can be found by `avg_noise_pwr_dB = avg_pwr_dB(desired signal) - SNR` in dB math.

    Args:
        avg_pwr_dB: average power of noise signal (dB)
        num_samples: number of output samples to generate
    """
    avg_pwr = utils.dB_to_power(avg_pwr_dB)
    noise_amp = np.sqrt(2.0 * avg_pwr) / 2.0
    return np.random.normal(0, noise_amp, num_samples) + 1j * np.random.normal(
        0, noise_amp, num_samples
    )

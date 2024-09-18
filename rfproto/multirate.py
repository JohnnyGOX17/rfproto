from . import fxp_int as fp
import numpy as np


def find_rational_frac(denom: int, ratio: float) -> int:
    """Find nearest rational fraction for resampling to 2x sample/symbol:
    https://www.cprogramming.com/tips/tip/rounding-to-fractions"""
    half_round = ratio + (1.0 / (denom * 2.0))
    return int(half_round * denom)


def get_rational_resampling_factors(
    input_fs: float, output_fs: float, denominator: int = 128
) -> tuple[int, int]:
    """Returns tuple of `(upsampling_factor, downsampling_factor)` to hit desired
    output sample rate (`output_fs`, Hz), given input sample rate (`input_fs`, Hz).
    `denominator` determines approximate base number of filter taps to use."""
    resample_ratio = output_fs / input_fs
    upsamp_factor = find_rational_frac(denominator, resample_ratio)
    return (upsamp_factor, denominator)


def interpolate(x: np.ndarray, L: int) -> np.ndarray:
    """Interpolates (aka upsamples/expands) input array L times to output array by zero stuffing

    Args:
        x : Input sample array
        L : Interpolation factor where `len(y) == L * len(x)`

    Returns:
        y : Vector of interpolated output samples
    """
    if L < 1:
        raise ValueError("L must be > 1")
    # Extend w/same datatype as input (e.x. complex)
    y = np.zeros(len(x) * L, dtype=x.dtype)
    y[::L] = x
    return y


def decimate(x: np.ndarray, M: int) -> np.ndarray:
    """Decimates input array by M to output array by keeping every other M sample

    Args:
        x : Input sample array
        M : Decimation factor (output length is 1/M of input)

    Returns:
        y : Vector of decimated output samples
    """
    if M < 1:
        raise ValueError("M must be > 1")
    # simply use Python slice notation of a[start_index:end_index:step]
    return x[::M]


class integrator:
    """Basic class for fixed-point Integrator section"""

    def __init__(self, bit_width: int):
        self.accum = fp.fxp_int(0, bit_width, unsigned=False)

    def step(self, x: int) -> int:
        retval = int(self.accum)
        self.accum += x
        return retval


class comb:
    """Basic class for fixed-point Comb section"""

    def __init__(self, M: int = 1):
        if M < 1:
            raise ValueError("M can't be negative!")
        self.delay_line = [0] * M
        self.output_reg = 0
        self.M = M

    def step(self, x: int) -> int:
        retval = self.output_reg
        self.output_reg = x - self.delay_line[self.M - 1]
        for i in range(self.M):
            if i < self.M - 1:
                self.delay_line[i + 1] = self.delay_line[i]
        self.delay_line[0] = x
        return retval

"""
Utilities & helper functions for things like fixed-point/binary conversion,
 dB/Mag conversion, etc.

**NOTE:** while there are Python packages that can help with arbitrary FXP math
[like this](https://github.com/francof2a/fxpmath), most DSP functions
here can use [numpy integer types](https://numpy.org/doc/stable/reference/arrays.scalars.html#integer-types)
to model common C/HDL fixed-point values. This also removes an external
dependency from this package.
"""

import numpy as np


def fxp_to_dbl(x: int, num_frac_bits: int) -> float:
    """Converts a fixed-point integer `x` into a floating-point value given
    some number of fractional bits `num_frac_bits`. Note that a signed FXP value
    requires 1-bit for sign representation. For example, a 16b signed `short` int
    representing a value with no integer part has `num_frac_bits = 15` (or in
    `Q0.15` in [Q format](https://en.wikipedia.org/wiki/Q_(number_format))"""
    return x / (1 << num_frac_bits)


def dbl_to_fxp(x: float, num_frac_bits: int) -> int:
    """Converts a floating-point value `x` into a fixed-point integer given
    some number of fractional bits `num_frac_bits`. Note that a signed FXP value
    requires 1-bit for sign representation. For example, a 16b signed `short` int
    representing a value with no integer part has `num_frac_bits = 15` (or in
    `Q0.15` in [Q format](https://en.wikipedia.org/wiki/Q_(number_format))"""
    return round(x * (1 << num_frac_bits))


def fxp_truncate(x: int, N: int) -> int:
    """Truncate fixed point value `x` by $N$ bits"""
    return x >> N


def fxp_round_halfup(x: int, N: int) -> int:
    """Round fixed point value `x` by $N$ bits using half-up rounding, like:

    $$ floor(x+0.5) $$
    """
    return ((x >> (N - 1)) + 1) >> 1


def _binlist2int(list_):
    return sum(1 << i for (i, b) in enumerate(list_) if b != 0)


def binlist2int(list_):
    """Converts a bit array to its integer representation."""
    return _binlist2int(list_)


def _int2binlist(int_, width=None):
    if width is None:
        width = max(int_.bit_length(), 1)
    return [(int_ >> i) & 1 for i in range(width)]


def int2binlist(int_, width=None):
    """Converts an integer to its bit array representation."""
    return np.array(_int2binlist(int_, width))


def int_to_bin_str(x: int, width: int = 0) -> str:
    """Convert integer to binary string representation w/o '0b' prefix. If prefix
    is desired, simply use: `bin(x)`
    """
    if width == 0:
        width = int(np.ceil(np.log2(abs(x))))
    return "{0:0{width}b}".format(x, width=width)


def int_to_fixed_width_bin_str(x: int, bit_width: int):
    """Convert integer to 2's complement binary representation, forcing a
    specified bitwidth"""
    # convert to signed-8b twos-complement value
    twos_cmplt = int(x) & ((2 ** int(bit_width)) - 1)
    # write out as padded binary string (always fixed character width)
    return str((bin(twos_cmplt)[2:].zfill(bit_width)))


def int_list_to_bin_str(x: list[int], width: int = 0) -> list[str]:
    """Returns a list of strings based on the binary representation of a list
    of input integers `x`"""
    if width == 0:
        width = int(np.ceil(np.log2(np.max(np.abs(x)))))
    return [int_to_bin_str(i, width) for i in x]


def write_ints_to_file(file_name: str, x: list[int], bit_width: int):
    """Write list of integers in fixed-width, 2's complement binary format to file"""
    fd = open(file_name, "w")
    for val in x:
        fd.write(int_to_fixed_width_bin_str(val, bit_width) + "\n")
    fd.close()


def _safe_np_iter(x: np.ndarray) -> np.ndarray:
    """Replace `0` values with smallest floating point value to avoid $/0$ exceptions"""
    return np.where(x == 0, np.finfo("float").smallest_normal, x)


def power_to_dB(x, ref: float = 1.0):
    """Return the decibel (dB) value of the given linear power value `x` and an
    optional reference value `ref`. This assumes power is already square of an
    energy value (e.g. amplitude or magnitude)"""
    # NOTE: abs() required to get complex value's magnitude
    return 10 * np.log10(np.abs(_safe_np_iter(x)) / ref)


def mag_to_dB(x, ref: float = 1.0):
    """Return the decibel (dB) value of the given linear magnitude value `x` (e.g.
    energy or signal amplitude) and an optional reference value `ref`"""
    # NOTE: abs() required to get complex value's magnitude
    return 20 * np.log10(np.abs(_safe_np_iter(x)) / ref)


def mag_to_dBFS(x, max_mag: float):
    """Converts magnitude value(s) `x` (number of counts for a given integer
    sample type to a dBFS (dB full scale) value, given maximum magnitude `max_mag`.
    Note that `max_mag` is scaled by $\\sqrt{2}$ when input samples are complex to
    account of max magnitude $= \\sqrt{I^{2} + Q^{2}}$, to normalize to 0 dBFS"""
    max_mag_scaled = max_mag
    if np.iscomplexobj(x):
        max_mag_scaled *= np.sqrt(2)
    return mag_to_dB(x, max_mag_scaled)


def dbfs_fft(x, max_mag: float = 1.0):
    """Compute FFT"""
    if np.isrealobj(x):
        return mag_to_dBFS(np.fft.rfft(x), max_mag)
    else:
        return mag_to_dBFS(np.fft.fft(x), max_mag)


def dB_to_power(x):
    """Return the linear power value given the decibel (dB) value `x`"""
    return 10.0 ** (x / 10.0)


def dB_to_mag(x):
    """Return the linear magnitude value given the decibel (dB) value `x`"""
    return 10.0 ** (x / 20.0)


def avg_pwr_dB(x):
    """Returns the average power (in dB) of an input array of linear values (e.x. voltage)"""
    # |x| to give complex magnitude for instantaneous power x**2
    return power_to_dB(np.mean(np.abs(x) ** 2))


def interleave_iq(i: np.ndarray, q: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Interleave separate I/Q arrays into one complex output array"""
    assert len(i) == len(q)
    return np.asarray(i, dtype=dtype) + 1j * np.asarray(q, dtype=dtype)


def deinterleave_iq(iq: np.ndarray, swap_iq: bool = False) -> np.ndarray:
    """De-interleave input I/Q array (e.x. `[I0,Q0,I1,Q1,..]`) to complex output array"""
    if swap_iq:
        return iq[1::2] + 1j * iq[::2]
    else:
        return iq[::2] + 1j * iq[1::2]


def open_iq_file(file, dtype=np.int8, swap_iq: bool = False) -> np.ndarray:
    """Open interleaved binary file and create complex I/Q output array. Note if
    wanting to open a floating point complex file, simply use the NumPy native method:
    `np.fromfile(file, dtype=np.complex64)` for instance for [single precision complex C-type](https://numpy.org/doc/stable/user/basics.types.html#relationship-between-numpy-data-types-and-c-data-types)

    This method is mainly for interleaved complex integer files, [which could be opened via](https://stackoverflow.com/a/32877245)
    `dtype=np.dtype([('re', np.int16), ('im', np.int16)])`, but this still needs conversion to
    floating-point complex type for further math usage."""
    file_iq = np.fromfile(file, dtype=dtype)
    return deinterleave_iq(file_iq, swap_iq)


def write_iq_to_file(
    file,
    iq: np.ndarray,
    dtype=np.int8,
    scale_factor: float = 1.0,
    round_output: bool = True,
):
    """Write complex I/Q array to binary file in interleaved format"""
    output_i = np.real(iq)
    output_q = np.imag(iq)
    # flatten I/Q into interleaved array: I0,Q0,I1,Q1,etc.
    output_iq = [
        iq_out * scale_factor
        for iq_pair in zip(output_i, output_q)
        for iq_out in iq_pair
    ]
    if round_output:
        np.round(output_iq).astype(dtype).tofile(file)
    else:
        np.asarray(output_iq, dtype=dtype).tofile(file)


def gray_code(n: int) -> list[int]:
    """Generate $N$-bit gray code sequence"""
    gray_list = []
    for i in range(0, 1 << n):
        gray_list.append(i ^ (i >> 1))
    return gray_list

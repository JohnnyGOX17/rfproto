import numpy as np
from scipy import signal
from . import filter, modulation, multirate


def cmplx_ct_sinusoid(A: float, W: float, t: np.ndarray, phi: float = 0) -> np.ndarray:
    """Generates a complex, continuous-time sinusoid

    Args:
        A: Amplitude (magnitude of complex phasor)
        W: Frequency (radians/sec)
        t: Vector of time points to evaluate the sinusoid at
        phi: Initial phase of sinusoid (radians)
    """
    return A * np.exp(1j * (W * t + phi))


def cmplx_dt_sinusoid(A: float, f: float, fs: float, num_samples: int) -> np.ndarray:
    """Generates a complex, discrete-time sinusoid. Note `f` and `fs` units must match!

    Args:
        A: Amplitude (magnitude of complex phasor)
        f: Frequency (Hz, or radians/sec)
        fs: Sampling Frequency (Hz, or radians/sec)
        num_samples: Number of samples to output
    """
    t = np.linspace(0, num_samples - 1, num_samples)
    return cmplx_ct_sinusoid(A, 2 * np.pi * f / fs, t, 0)


def cmplx_dt_lfm_chirp(
    A: float, f_start: float, f_end: float, fs: float, num_samples: int
) -> np.ndarray:
    """Generates a complex, discrete-time Linear Frequency Modulated (LFM) Chirp.

    Args:
        A: Amplitude (magnitude of complex phasor)
        f_start: Start Frequency (Hz)
        f_end: End Frequency (Hz)
        fs: Sampling Frequency (Hz)
        num_samples: Number of samples to output
    """
    y = np.zeros(num_samples) + 1j * np.zeros(num_samples)
    chirp_rate = ((f_end - f_start) / fs) / num_samples
    for i in range(num_samples):
        phase = (chirp_rate / 2.0) * (i**2.0)
        phase += (f_start / fs) * i
        y[i] = A * np.exp(1j * (2 * np.pi * phase))
    return y


def real_ct_sinusoid(A: float, W: float, t: np.ndarray, phi: float = 0) -> np.ndarray:
    """Generates a real, continuous-time sinusoid

    Args:
        A: Amplitude
        W: Frequency (radians/sec)
        t: Vector of time points to evaluate the sinusoid at
        phi: Initial phase of sinusoid (radians)
    """
    return A * np.sin(W * t + phi)


def gen_mod_signal(
    modulation_type: str,
    symbols: list[int],
    output_sample_rate: float,
    input_symbol_rate: float,
    tx_pulse_filter: str,
    tx_filter_rolloff: float,
    tx_filter_num_symbols: int = 16,
):
    match modulation_type:
        case "BPSK":
            mod = modulation.MPSKModulation(2)
        case "QPSK":
            mod = modulation.MPSKModulation(4)
        case "8PSK":
            mod = modulation.MPSKModulation(8)
        case _:
            raise ValueError("Unknown modulation type!")
    mapped_iq = mod.modulate(symbols)

    resample_rate = output_sample_rate / input_symbol_rate

    assert (
        resample_rate >= 1.0
    ), "Resampling rate expected to be a positive, upsample factor (L>M)"

    # We can make simplifications when there is an integer resampling factor
    is_int_resamp = round(resample_rate) == resample_rate
    if is_int_resamp:
        (L, M) = int(resample_rate), 1
    else:
        (L, M) = multirate.get_rational_resampling_factors(
            input_symbol_rate, output_sample_rate, 128
        )

    # Since resample_poly wants to run filter at interpolated rate (zero stuffed),
    # multiply intended sample rate by upsample factor. As well, multiply the number
    # of filter taps by the same amount:
    match tx_pulse_filter:
        case "RC":
            tx_filter = filter.RaisedCosine(
                L * input_symbol_rate,
                input_symbol_rate,
                tx_filter_rolloff,
                2 * tx_filter_num_symbols * L + 1,
            )
        case "RRC":
            tx_filter = filter.RootRaisedCosine(
                L * input_symbol_rate,
                input_symbol_rate,
                tx_filter_rolloff,
                2 * tx_filter_num_symbols * L + 1,
            )
        case _:
            raise ValueError("Unknown TX pulse shape filter type!")

    if is_int_resamp:
        tx_resampled_iq = filter.pulse_shape(mapped_iq, L, tx_filter)
    else:
        tx_resampled_iq = signal.resample_poly(
            mapped_iq, L, M, window=tx_filter, padtype="mean"
        )

    return tx_resampled_iq

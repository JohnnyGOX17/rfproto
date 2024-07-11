"""Spectral and baseband signal measurement utilities"""

import numpy as np

from . import utils


# TODO: look at https://www.analog.com/en/technical-articles/how-evm-measurement-improves-system-level-performance.html
def EVM(x: np.ndarray, ref: np.ndarray) -> np.floating:
    """Calculate the Error Vector Magnitude (EVM) of an input sequence.

    Args:
        x: Sample data vector
        ref: Reference decision data vector

    Returns:
        EVM
    """
    return np.std(x - ref) / np.std(ref)


def PSD(
    x: np.ndarray, fs: float, norm: bool = False, max_mag: float = 1.0, fft_shift=False
):
    """Calculates Power Spectral Density (PSD) of a given time signal

    Args:
        x: Sample data vector (time domain)
        fs: Sample frequency of `x` (Hz)
        norm: When True, normalize max frequency bin (e.x. fundamental) to 0.0 dB
        max_mag: maximum input magnitude (or max I or Q value for complex) to calculate dBFS. Only used when `norm == False`
        fft_shift: Shifts the zero-frequency component to the center of the spectrum
    """
    real = np.isrealobj(x)

    psd = utils.dbfs_fft(x, max_mag if not norm else 1.0)
    if norm:
        psd -= psd.max(axis=0)

    # Real PSD is only 0 -> fs/2
    numFreqBins = len(psd) if not real else 2 * len(psd)

    if fft_shift:
        psd = np.fft.fftshift(psd)
        freqBin = np.linspace(-len(psd) // 2, len(psd) // 2, len(psd)) * (
            fs / numFreqBins
        )
    else:
        freqBin = np.linspace(1, len(psd), len(psd)) * (fs / numFreqBins)

    return freqBin, psd


def SFDR(
    x: np.ndarray,
    fs: float,
    norm: bool = False,
    ignore_percent: float = 0.1,
    max_mag: float = 1.0,
):
    """Spurious free dynamic range (SFDR) is the ratio of the RMS value of the
    signal to the RMS value of the worst spurious signal regardless of where it
    falls in the frequency spectrum. The worst spur may or may not be a
    harmonic of the original signal. SFDR is an important specification in
    communications systems because it represents the smallest value of signal
    that can be distinguished from a large interfering signal (blocker). SFDR
    is specified here w.r.t. an actual signal amplitude (dBc). Thus, it's expected
    that the given signal vector `x` has some main frequency component greater than
    any spurs present in the spectrum to return a sensible value.

    References:

    * [Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You Don't Get Lost in the Noise Floor - ADI](https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf)
    * [MonsieurV/py-findpeaks](https://github.com/MonsieurV/py-findpeaks)

    Args:
        x: Sample data vector (time domain)
        fs: Sample frequency of `x` (Hz)
        norm: When True, normalize max frequency bin (e.x. fundamental) to 0.0 dB
        ignore_percent: The fraction of total samples that are ignored around the fundamental for spurs
        max_mag: maximum input magnitude (or max I or Q value for complex) to calculate dBFS. Only used when `norm == False`


    """
    # TODO: really use https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html ?
    freqBin, Y = PSD(x, fs, norm, max_mag)
    idx_fc = np.argmax(Y)  # give index of spectrum fundamental (Fc)
    # +/- percentage from Fc to ignore for SFDR calculation so phase noise or
    # leakage from the main tone doesn't affect these calcs (default +/-10%)
    min_idx = int(idx_fc - int(ignore_percent * len(Y)))
    if min_idx < 0:  # limits check
        min_idx = 0
    max_idx = int(idx_fc + int(ignore_percent * len(Y)))
    if max_idx > len(Y) - 1:
        max_idx = len(Y) - 1
    PSD_non_fc = np.copy(Y)
    PSD_non_fc[min_idx:max_idx] = -10000  # null freq bins we want to ignore
    idx_spur = np.argmax(PSD_non_fc)  # index of largest spur
    d = dict()  # use dictionary for multiple, named return values
    d["fc_dB"] = Y[idx_fc]
    d["fc_Hz"] = freqBin[idx_fc]
    d["spur_dB"] = Y[idx_spur]
    d["spur_Hz"] = freqBin[idx_spur]
    d["SFDR"] = Y[idx_fc] - Y[idx_spur]
    return d


def ideal_SNR(N: int) -> float:
    """Calculate the ideal SNR of an $N$-bit ADC/DAC

    References:

    * [Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You Don't Get Lost in the Noise Floor - ADI](https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf)

    Args:
        N: Number of bits

    Returns:
        y: SNR (dB)
    """
    return (6.02 * N) + 1.76


def FFT_process_gain(M: int) -> float:
    """The theoretical noise floor of the FFT is equal to the theoretical SNR
    plus the FFT process gain, $10\\log{M/2}$. It is important to remember that
    the value for noise used in the SNR calculation is the noise that extends
    over the entire Nyquist bandwidth (DC to $f_{s}/2$), but the FFT acts as a
    narrowband spectrum analyzer with a bandwidth of $f_{s}/M$ that sweeps over the
    spectrum. This has the effect of pushing the noise down by an amount equal
    to the process gain— the same effect as narrowing the bandwidth of an analog
    spectrum analyzer. Thus to find the "real" RMS noise level (which is affected
    by quantization, system or environmental noise), subtract the measured FFT noise
    floor by this processing gain value.

    References:

    * [Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You Don't Get Lost in the Noise Floor - ADI](https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf)

    Args:
        M: Number of FFT bins

    Returns:
        y: FFT processing gain (dB)
    """
    return 10 * np.log(M / 2)

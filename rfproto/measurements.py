import numpy as np

from . import utils


# TODO: look at https://www.analog.com/en/technical-articles/how-evm-measurement-improves-system-level-performance.html
def EVM(x: np.ndarray, ref: np.ndarray) -> np.floating:
    """Returns the Error Vector Magnitude (EVM)

    Parameters
    ----------
    x : ndarray
        Sample data vector
    ref : ndarray
        Reference decision data vector
    """
    return np.std(x - ref) / np.std(ref)


def PSD(x: np.ndarray, fs: float, real: bool = False, norm: bool = False):
    """Calculates Power Spectral Density (PSD) of a given time signal

    Parameters
    ----------
    x : ndarray
        Sample data vector (time domain)
    fs : float
        Sample frequency of `x` (Hz)
    real : bool, default: False
        Whether `x` is real valued or not (complex)
    norm : bool, default: False
        When True, normalize fundamental to 0.0
    """
    if real:
        PSD = utils.mag_to_dB(np.fft.rfft(x))
    else:
        PSD = utils.mag_to_dB(np.fft.fft(x))
    if norm:
        PSD -= PSD.max(axis=0)
    # Real PSD is only 0 -> fs/2
    numFreqBins = PSD.size if not real else 2 * PSD.size
    freqBin = np.linspace(1, PSD.size, PSD.size) * (fs / numFreqBins)
    return freqBin, PSD


def SFDR(
    x: np.ndarray,
    fs: float,
    real: bool = False,
    norm: bool = False,
    ignore_percent: float = 0.1,
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

    Parameters
    ----------
    x : ndarray
        Sample data vector (time domain)
    fs : float
        Sample frequency of `x` (Hz)
    real : bool, default: False
        Whether `x` is real valued or not (complex)
    norm : bool, default: False
        When True, normalize fundamental to 0.0
    ignore_percent : float, default: 0.1
        The fraction of total samples that are ignored around the fundamental for spurs

    Returns
    -------

    References
    ----------
    .. [1] Walt Kester, "Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You
            Don't Get Lost in the Noise Floor", ADI,
            https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf
    .. [2] [MonsieurV/py-findpeaks](https://github.com/MonsieurV/py-findpeaks)
    """
    # TODO: really use https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    freqBin, Y = PSD(x, fs, real, norm)
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
    """Calculate the ideal SNR of an N-bit ADC/DAC

    Parameters
    ----------
    N : int
        Number of bits

    Returns
    -------
    y : float
        SNR (dB)

    References
    ----------
    .. [1] Walt Kester, "Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You
            Don't Get Lost in the Noise Floor", ADI,
            https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf
    """
    return (6.02 * N) + 1.76


def FFT_process_gain(M: int) -> float:
    """The theoretical noise floor of the FFT is equal to the theoretical SNR
    plus the FFT process gain, 10×log(M/2). It is important to remember that
    the value for noise used in the SNR calculation is the noise that extends
    over the entire Nyquist bandwidth (DC to fs/2), but the FFT acts as a
    narrowband spectrum analyzer with a bandwidth of fs/M that sweeps over the
    spectrum. This has the effect of pushing the noise down by an amount equal
    to the process gain— the same effect as narrowing the bandwidth of an analog
    spectrum analyzer. Thus to find the "real" RMS noise level (which is affected
    by quantization, system or environmental noise), subtract the measured FFT noise
    floor by this processing gain value.

    Parameters
    ----------
    M : int
        Number of FFT bins

    Returns
    -------
    y : float
        FFT processing gain (dB)

    References
    ----------
    .. [1] Walt Kester, "Understand SINAD, ENOB, SNR, THD, THD + N, and SFDR so You
            Don't Get Lost in the Noise Floor", ADI,
            https://www.analog.com/media/en/training-seminars/tutorials/MT-003.pdf
    """
    return 10 * np.log(M / 2)

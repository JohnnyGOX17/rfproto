#!/usr/bin/env python

from rfproto import plot, measurements, sig_gen
import numpy as np


def plot_time():
    f = 1100
    fs = 48000
    N = 50000
    test_sine = [np.sin(2 * np.pi * f * x / fs) for x in range(N)]

    # show time signal
    plot.samples(test_sine[0 : int(2 * fs / f)])
    plot.plt.show()

    # show FFT power spectrum of `test_sine`
    freq, Y = measurements.PSD(test_sine, fs, True)
    plot.freq_sig(freq, Y)
    plot.plt.show()


def plot_IQ():
    # show I/Q plot
    N = 1000
    IQ_data = np.array([4 + 4j, -4 + 4j, -4 - 4j, 4 - 4j])
    IQ_data = np.repeat(IQ_data, N // 4)
    # similar to random.shuffle(IQ_data)
    np.random.shuffle(IQ_data)
    # Add AWGN with unity power
    IQ_data += (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    plot.IQ(IQ_data, "I/Q test plot", alpha=0.4)
    plot.plt.show()

    plot.IQ_animated(IQ_data, N // 10, "I/Q test plot")
    plot.plt.show()


def plot_intensity():
    symbol_rate = 7.5e6
    output_fs = 17.22e6
    output_iq = sig_gen.gen_mod_signal(
        "QPSK",
        np.random.randint(0, 4, 256 * 10240).tolist(),
        output_fs,
        symbol_rate,
        "RRC",
        1.0,
    )

    plot.fft_intensity_plot(output_iq, 1024, 2, 300, "plasma")
    plot.plt.show()


def plot_waterfall():
    symbol_rate = 7.5e6
    output_fs = 17.22e6
    output_iq = sig_gen.gen_mod_signal(
        "QPSK",
        np.random.randint(0, 4, 256 * 10240).tolist(),
        output_fs,
        symbol_rate,
        "RRC",
        1.0,
    )

    plot.waterfall(
        output_iq, w=np.hamming(64), fft_len=256, stride_len=32, num_rows=256
    )
    plot.plt.show()


if __name__ == "__main__":
    plot_time()
    plot_IQ()
    plot_intensity()
    plot_waterfall()

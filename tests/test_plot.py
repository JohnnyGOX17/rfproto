from rfproto import plot, measurements
import numpy as np
import os


def test_time():
    if os.environ.get("NO_PLOT") == "true":
        return
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


def test_IQ():
    if os.environ.get("NO_PLOT") == "true":
        return
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


if __name__ == "__main__":
    test_time()
    test_IQ()

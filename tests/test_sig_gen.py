import os
import numpy as np
from rfproto import measurements, plot, sig_gen
import pytest


def test_dt_sine():
    fs = 100.0
    samples = sig_gen.cmplx_dt_sinusoid(8192.0, 10.0, fs, 8192)
    dSFDR = measurements.SFDR(samples, fs)
    assert dSFDR["fc_Hz"] == pytest.approx(10.0, 0.01)
    assert dSFDR["SFDR"] > 70.0


def test_qpsk_gen():
    symbol_rate = 7.5e6
    output_fs = 17.22e6
    output_iq = sig_gen.gen_mod_signal(
        "QPSK",
        np.random.randint(0, 4, 2**16).tolist(),
        output_fs,
        symbol_rate,
        "RRC",
        1.0,
    )

    if os.environ.get("NO_PLOT") == "true":
        return
    plot.spec_an(output_iq, output_fs, fft_shift=True, show_SFDR=False)
    plot.plt.show()

    plot.IQ(output_iq[512:1024], alpha=0.2)
    plot.plt.show()


if __name__ == "__main__":
    test_dt_sine()
    test_qpsk_gen()

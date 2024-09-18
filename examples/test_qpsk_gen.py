#!/usr/bin/env python

from rfproto import plot, sig_gen
import numpy as np


if __name__ == "__main__":
    symbol_rate = 7.5e6
    # output_fs = 17.22e6
    output_fs = 4.0 * symbol_rate
    in_symbols = np.random.randint(0, 4, 2400).tolist()
    output_iq = sig_gen.gen_mod_signal(
        "QPSK",
        in_symbols,
        output_fs,
        symbol_rate,
        "RRC",
        0.25,
    )
    # Add AWGN with unity power

    plot.IQ(output_iq, "I/Q test plot", alpha=0.1)
    plot.plt.show()

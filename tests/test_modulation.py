import os
import numpy as np

from rfproto import modulation, plot


def test_mpsk():
    mpsk_order = [2, 4, 8]

    for m in mpsk_order:
        mpsk_sut = modulation.MPSKModulation(m)
        if os.environ.get("NO_PLOT") == "true":
            return
        plot.IQ(mpsk_sut.constellation, title=f"{m}-PSK Map", label=True)
        plot.plt.show()

        # generate random symbols
        rand_symbols = np.random.randint(0, m, m)
        print(f"{m}-ary PSK input symbols: {rand_symbols}")
        out_iq = mpsk_sut.modulate(rand_symbols.tolist())
        print(f"{m}-ary PSK output sequence: {out_iq}")


def test_qam():
    pass


if __name__ == "__main__":
    test_mpsk()

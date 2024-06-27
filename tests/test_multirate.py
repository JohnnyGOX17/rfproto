from rfproto import multirate
import numpy as np


def test_interpolate():
    x = np.array([1, 2, 3, 4, 5])
    y = multirate.interpolate(x, 3)
    test_interp_out = [
        1.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        3.0,
        0.0,
        0.0,
        4.0,
        0.0,
        0.0,
        5.0,
        0.0,
        0.0,
    ]

    for sample, truth in zip(y, test_interp_out):
        assert sample == truth


def test_decimate():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = multirate.decimate(x, 4)
    test_decim_out = [1, 5, 9]

    for sample, truth in zip(y, test_decim_out):
        assert sample == truth


def test_integrator():
    sut = multirate.integrator(8)
    for _ in range(10):
        print(f"{sut.step(50)}")


def test_resample_factors():
    input_fs = 17.22e6
    output_fs = 15e6
    expected_ratio = output_fs / input_fs
    L, M = multirate.get_rational_resampling_factors(input_fs, output_fs, 128)
    frac_err = 100.0 * ((L / M) - expected_ratio) / expected_ratio
    print(f"Found resample ratio of {L}/{M} = {L/M} [{frac_err:.2f}% error]")


if __name__ == "__main__":
    test_interpolate()
    test_decimate()
    test_integrator()
    test_resample_factors()

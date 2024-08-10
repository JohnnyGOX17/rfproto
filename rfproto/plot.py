import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.ticker import EngFormatter
import scipy.signal as sig
import typing

from . import measurements, frequency, utils


def _plot_common(title: str):
    """Common `rfproto` style plot setup

    Args:
        title: Plot title
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    plt.margins(x=0)
    plt.grid(True, linestyle="--")
    plt.minorticks_on()
    plt.tick_params(labelsize=8)
    return fig, ax


def samples(y, title: str = ""):
    """Plot samples (no time-base)

    Args:
        y: time-series data
        title: Plot title
    """
    fig, ax = _plot_common(title)
    plt.plot(y, linewidth=0.5)
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlabel("Sample", fontsize=12)
    return fig, ax


def time_sig(t, y, title: str = ""):
    """Plot samples over a given time-base

    Args:
        y: time-series data
        t: time vector (same length of `y`)
        title: Plot title
    """
    fig, ax = _plot_common(title)
    plt.plot(t, y, linewidth=0.5)
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlabel("Time (s)", fontsize=12)
    return fig, ax


def filter_coefficients(filter_coef: np.ndarray, title: str = ""):
    """Plot filter coefficients

    Args:
        filter_coef: filter weights (impulse response)
        title: Plot title
    """
    fig, ax = _plot_common(title)
    plt.plot(filter_coef, ".")
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlabel("Index", fontsize=12)
    return fig, ax


def filter_response(filter_coef: np.ndarray, title: str = ""):
    """Plot filter frequency response

    Args:
        filter_coef: filter weights (impulse response)
        title: Plot title
    """
    w, h = sig.freqz(filter_coef)
    fig, ax = _plot_common(title)
    ax.plot(w / np.pi, utils.mag_to_dB(h), "b")
    ax.set_ylabel("Amplitude (dB)", color="b", fontsize=12)
    ax.set_xlabel(r"Normalized Frequency ($\times \pi$ rad/sample)", fontsize=12)
    ax2 = ax.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w / np.pi, angles, "g")
    ax2.set_ylabel("Angle (radians)", color="g", fontsize=12)
    ax2.axis("tight")
    return fig, ax


def freq_sig(freq, y, title: str = "", scale_noise: bool = False, y_unit: str = "dBFS"):
    """Plot frequency-domain input signal

    Args:
        freq: frequency bins
        y: frequency-domain data (same length as number of frequency bins)
        title: Plot title
        scale_noise: don't show full noise floor extent when True
        y_unit: Unit for frequency bin data
    """
    fig, ax = _plot_common(title)
    plt.plot(freq, y, linewidth=0.5)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel(f"Magnitude ({y_unit})", fontsize=12)
    if scale_noise:  # standard pyplot shows full noise extent
        plt.ylim(y.mean(0) - 5, y.max(0) + 5)
    formatter_eng = EngFormatter(unit="Hz")
    ax.xaxis.set_major_formatter(formatter_eng)
    return fig, ax


def spec_an(
    x: np.ndarray,
    fs: float,
    title="",
    scale_noise=False,
    y_unit="dBFS",
    norm: bool = False,
    ignore_percent: float = 0.1,
    fft_shift=False,
    show_SFDR=True,
):
    """Take PSD of time-domain input signal and plot in frequency-domain. Optionally calculate SFDR and show in plot."""
    freq, y_PSD = measurements.PSD(x, fs, norm=norm, fft_shift=fft_shift)

    if show_SFDR:
        dSFDR = measurements.SFDR(x, fs, norm, ignore_percent)
        title += " [SFDR: {:.2f} dB]".format(dSFDR["SFDR"])
        fig, ax = freq_sig(freq, y_PSD, title, scale_noise, y_unit)

        max_freq = max(freq)
        max_dB = max(y_PSD)
        txt_offset_x = 0.05 * max_freq
        txt_offset_y = 0.05 * max_dB if abs(max_dB) > 3 else 5

        spur_f_str = frequency.FreqStr(dSFDR["spur_Hz"], "Hz")
        fund_f_str = frequency.FreqStr(dSFDR["fc_Hz"], "Hz")

        ax.plot(dSFDR["spur_Hz"], dSFDR["spur_dB"], "s", color="orange")
        ax.text(
            dSFDR["spur_Hz"] + txt_offset_x,
            dSFDR["spur_dB"] + txt_offset_y,
            "{:.2f} {} @\n{}".format(dSFDR["spur_dB"], y_unit, spur_f_str),
        )

        ax.plot(dSFDR["fc_Hz"], dSFDR["fc_dB"], "s", color="black")
        ax.text(
            dSFDR["fc_Hz"] + txt_offset_x,
            dSFDR["fc_dB"] - txt_offset_y,
            "{:.2f} {} @\n{}".format(dSFDR["fc_dB"], y_unit, fund_f_str),
        )
    else:
        fig, ax = freq_sig(freq, y_PSD, title, scale_noise, y_unit)

    return fig, ax


def IQ(
    signal: np.ndarray,
    title: str = "",
    alpha: float = 1.0,
    label: bool = False,
):
    """Generate I/Q plot

    Args:
        signal: Complex sample data vector
        title: Plot title
        alpha: Value < 1.0 allows opaque dot points, useful for high sample count clustering visualization
        labels: When True, label each point based on sample/point index
    """
    fig, ax = _plot_common(title)
    plt.plot(np.real(signal), np.imag(signal), ".", alpha=alpha)
    plt.axvline(x=0, color="orange")
    plt.axhline(y=0, color="orange")
    plt.xlabel("In-Phase (I)", fontsize=12)
    plt.ylabel("Quadrature (Q)", fontsize=12)
    plt.axis("equal")
    if label:
        for i in range(len(signal)):
            plt.annotate(
                utils.int_to_bin_str(i, width=(len(signal) - 1).bit_length()),
                xy=(np.real(signal[i]), np.imag(signal[i])),
                xytext=(1.5, 1.5),
                textcoords="offset points",
            )
    return fig, ax


@typing.no_type_check
def IQ_animated(
    signal: np.ndarray,
    num_points_per_frame: int,
    title: str = "IQ Plot",
    file: str = "",
    fps: int = 10,
):
    """Generate animated I/Q plot (shows I/Q over time)
    **NOTE:** can be used in notebook with `%matplotlib widget` macro at top of notebook

    Args:
        signal: Complex I/Q sample data vector
        num_points_per_frame: How many points per frame to plot
        title: Plot title
        file: GIF file to save to when non-empty
        fps: Frames per second to render animated I/Q plit
    """
    num_frames = int(len(signal) / num_points_per_frame)

    fig = plt.figure(constrained_layout=True)
    ax_histxy = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histxy.set(aspect=1)
    ax_histx = ax_histxy.inset_axes([0, 1.05, 1, 0.25], sharex=ax_histxy)
    ax_histy = ax_histxy.inset_axes([1.05, 0, 0.25, 1], sharey=ax_histxy)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    max_mag = max(np.max(np.abs(np.real(signal))), np.max(np.abs(np.imag(signal))))
    max_mag = int(max_mag * 1.05)  # give 5% margin for size

    def update_iq_plot(frame):
        ax_histxy.clear()
        ax_histx.clear()
        ax_histy.clear()
        ax_histxy.set(title=f"{title} (Frame: {frame})")

        start_idx = frame * num_points_per_frame
        end_idx = start_idx + num_points_per_frame
        i_vals = np.real(signal[start_idx:end_idx])
        q_vals = np.imag(signal[start_idx:end_idx])

        ax_histxy.hist2d(i_vals, q_vals, bins=(max_mag, max_mag), cmap=plt.cm.viridis)
        ax_histxy.set_facecolor(mpl.cm.get_cmap("viridis")(0))
        ax_histxy.set(aspect=1)
        ax_histx.hist(i_vals, bins=max_mag)
        ax_histy.hist(i_vals, bins=max_mag, orientation="horizontal")
        ax_histxy.set_xlim([-max_mag, max_mag])
        ax_histxy.set_ylim([-max_mag, max_mag])
        ax_histxy.set_xlabel("I")
        ax_histxy.set_ylabel("Q")

    ani = animation.FuncAnimation(
        fig=fig, func=update_iq_plot, frames=num_frames, interval=fps
    )

    if file:
        writer = animation.PillowWriter(fps=fps)
        ani.save(file, writer=writer)

    plt.show()


# TODO: add jitter/histogram like measurements like https://www.mathworks.com/help/comm/ug/eye-diagram-analysis.html
def eye(
    signal: np.ndarray,
    SPS: int,
    num_disp_sym: int = 2,
    num_sweeps: int = -1,
):
    """Generate eye diagram of time domain input signal

    Args:
        signal: Complex I/Q sample data vector
        SPS: Samples/Symbol ratio (NOTE: must be an integer oversampling ration (OSR) to properly render time-sliced eye
        num_disp_sym: Number of symbols to display in eye diagram
        num_sweeps: Number of eye sweeps to plot, defaults to entire length of input signal
    """
    if num_sweeps < 1:
        num_sweeps = len(signal)

    # resample input data to make somewhat continuous waveform
    resamp = 10
    upsample_by = SPS * resamp
    tx_resamp = sig.resample(signal, len(signal) * resamp)
    samp_per_win = upsample_by * num_disp_sym

    # N is total number of possible windows
    N = len(tx_resamp) // samp_per_win

    tx_eye = np.array(tx_resamp)
    tx_eye.resize(N * samp_per_win)
    grouped = np.reshape(tx_eye, [N, samp_per_win])
    eye = np.real(grouped.T)

    # create an xaxis in samples np.shape(eye) gives the
    # 2 dimensional size of the eye data and the first element
    # is the interpolated number of samples along the x axis
    nsamps = np.shape(eye)[0]
    xaxis = np.arange(nsamps) / resamp

    plt.figure()
    # plot showing continuous trajectory of
    plt.plot(xaxis, eye[:, :num_sweeps])
    # actual sample locations
    plt.plot(xaxis[::resamp], eye[:, :num_sweeps][::resamp], "b.")
    plt.title("Eye Diagram")
    plt.xlabel("Samples")
    plt.grid()
    plt.show()

    return xaxis, eye


def fft_intensity_plot(
    samples: np.ndarray,
    fft_len: int = 256,
    fft_div: int = 2,
    mag_steps: int = 100,
    cmap: str = "plasma",
):
    """Real-Time Spectrum Analyzer like FFT persistence plot
    Based on [tdsepsilon's post](https://teaandtechtime.com/python-intensity-graded-fft-plots/)
    and [notebook code](https://github.com/Tschucker/Python-Intensity-Graded-FFT/blob/main/animated_plots/ig_fft_animation_qpsk.ipynb)"""
    num_ffts = int(np.floor(len(samples) / fft_len))

    fft_array = []
    for i in range(num_ffts):
        temp = np.fft.fftshift(np.fft.fft(samples[i * fft_len : (i + 1) * fft_len]))
        temp_mag = 20.0 * np.log10(np.abs(temp))
        fft_array.append(temp_mag)

    max_mag = np.amax(fft_array)
    min_mag = np.abs(np.amin(fft_array))

    norm_fft_array = fft_array
    for i in range(num_ffts):
        norm_fft_array[i] = (fft_array[i] + (min_mag)) / (max_mag + (min_mag))

    mag_step = 1 / mag_steps

    hitmap_array = np.random.random((mag_steps + 1, int(fft_len / fft_div))) * np.exp(
        -10
    )

    for i in range(num_ffts):
        for m in range(fft_len):
            hit_mag = int(norm_fft_array[i][m] / mag_step)
            hitmap_array[hit_mag][int(m / fft_div)] = (
                hitmap_array[hit_mag][int(m / fft_div)] + 1
            )

    hitmap_array_db = 20.0 * np.log10(hitmap_array + 1)

    figure, axes = plt.subplots()
    axes.imshow(hitmap_array_db, origin="lower", cmap=cmap, interpolation="bilinear")

    return figure


def waterfall(
    x: np.ndarray,
    w: np.ndarray,
    fft_len: int,
    stride_len: int,
    num_rows: int,
    cmap: str = "viridis",
    anim_interval: int = 10,
):
    """Creates animated waterfall spectrogram using Short Time FFTs (STFTs). For static
    image generation of spectrograms, see [SciPy Signal ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html)
    or [matplotlib specgram](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html).

    Args:
        x: input time-domain signal
        w: array of values to use as a window for each sliding window before taking the FFT
        fft_len: length of FFT for each step of the STFT. When > window length, the FFT is zero-padded
        stride_len: same as hop length, how many samples to stride after each step.
        num_rows: number of output rows in plot (e.g. number of history time steps)
        cmap: matplotlib colormap
    """
    Y = np.zeros((num_rows, fft_len))

    fig = plt.figure(frameon=False)
    im = plt.imshow(Y, animated=True, aspect="auto", cmap=cmap)
    plt.tight_layout()
    plt.axis("off")
    # update the color scale min/max on first loop iter
    vmax, vmin = -np.inf, np.inf
    idx = 0

    def update(*args):
        nonlocal idx, vmax, vmin

        # apply window on each data slice at each step
        slice = x[idx : idx + len(w)]
        windowed_slice = slice * w
        Y[0] = utils.mag_to_dB(np.fft.fftshift(np.fft.fft(windowed_slice, n=fft_len)))

        # set dynamic range
        max_val = np.max(Y[0])
        if vmax < max_val:
            vmax = max_val
        # use the first std dev to not make color range so drastic with noise floor
        # TODO: or make this a logarithmic color scale...
        min_val = max_val - np.std(Y[0])
        if vmin > min_val:
            vmin = min_val

        # update row
        im.set_array(Y)
        im.set_clim(vmin, vmax)
        # shift values
        Y[1:] = Y[:-1]
        idx += stride_len
        return (im,)

    # need to save variable or animation never plots
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=(len(x) // stride_len - 3),
        interval=anim_interval,
        blit=True,
        repeat=False,
    )
    plt.show()

import numpy as np
import datetime


class DigIFSignal:
    """
    Generic class for Digital Intermediate Frequency (IF) Signal samples
    & metadata
    """

    def __init__(self, fs: float, fc: float = 0.0, dtype=complex, freq_data=False):
        """Creates a new DigIF class

        Parameters
        ----------
        fs : float
            Sample rate of signal (Hz)
        fc : float, optional, default: 0.0
            Center frequency of IF signal (Hz)
        dtype : data-type, optional, default: 'complex'
            The desired data-type for the signal-data array `self.samples`
        freq_data : boolean, optional, default: 'False'
            `True` when signal-data array `self.samples` represents frequency domain,
            or sprectral, data. `False` indicates time-domain data. Transforms such
            as an FFT/IFFT can toggle this member value.
        """
        self.samples = np.zeros(0, dtype=dtype)
        """ Sample data vector of `dtype` """
        self.fs = fs
        """ Sample rate of IF signal (Hz) """
        self.fc = fc
        """ Center frequency of IF signal (Hz). For example, if signal is to be
        transmitted, this value can represent the mixing frequency (e.x. for a DUC)
        required to upconvert to the right frequency band. """
        self.freq_data = freq_data
        """ `True` when signal-data array `self.samples` represents frequency domain,
        or sprectral, data. `False` indicates time-domain data. Transforms such
        as an FFT/IFFT can toggle this member value. """
        self.timestamp = datetime.datetime.now(datetime.timezone.utc)
        """ Timestamp for the start of the signal (UTC). This is when the signal
        was started to be sampled on receive, or when the signal should start to
        transmit """

    def is_complex(self) -> bool:
        """Returns `True` when signal data is complex-valued, else returns `False`
        to indicate signal data is real-only"""
        return self.samples.dtype == complex

    def __len__(self) -> int:
        """Return the number of samples in the signal data vector"""
        return len(self.samples)

    def __eq__(self, other) -> bool:
        """Test for equivalence of two DigIFSignal objects. **NOTE:** returns `True`
        when the factors required for combining signals are ewquivalent, _not_ if
        the signal samples themselves are equal between the two. These factors are
        sample rate, frequency/time domain, and number of samples"""
        if self.fs != other.fs:
            return False
        if self.freq_data != other.freq_data:
            return False
        if len(self) != len(other):
            return False
        return True

    def __ne__(self, other) -> bool:
        return not self == other


if __name__ == "__main__":
    print("TODO")

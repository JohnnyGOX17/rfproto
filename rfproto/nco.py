#
# NCO implementation
#
import numpy as np
import random

from . import measurements, utils

nco_print_trace = False
use_fxp = True  # else, use floating-point calcs
use_small_angle_approx = False  # use small angle approximation in floating-point Taylor Series (not nearly as good as actual Taylor Series)


# NCO helper functions
def FreqToFcw(f: float, phase_width: int, fs: float) -> int:
    """Convert a desired output frequency `f` (Hz) to NCO Frequency Control Word (FCW)
    based on phase accumulator width `phase_width` and NCO clock frequency `fs`"""
    return np.round(f * (2**phase_width) / fs)


def FcwToFreq(fcw: int, phase_width: int, fs: float) -> float:
    """Convert an NCO's FCW to the associated output frequency (Hz)"""
    return fcw * fs / (2**phase_width)


def factorial(x):
    if x > 1:
        return x * factorial(x - 1)
    else:
        return 1


# Taylor series gives > 100dB SFDR, small angle approx. doesn't help...
# (e.g. cos_err ~= 1 - phase_err and sin_err ~= phase_err
def sin_ts(x):
    p3 = (x**3) / (factorial(3))
    # Beyond 2nd order correction seems to give diminishing/negligible extra precision!
    # p5 = (x**5) / (factorial(5))
    # p7 = (x**7) / (factorial(7))
    # p9 = (x**9) / (factorial(9))
    return x - p3  # + p5 - p7


def sin_ts_fxp(x, num_bits):
    # X assumed already at num_bits width
    first_square = utils.fxp_round_halfup(x * x, num_bits)
    x_cubed = utils.fxp_round_halfup(first_square * x, num_bits)
    fact_3 = utils.dbl_to_fxp(1.0 / 6.0, num_bits - 1)
    return x - utils.fxp_round_halfup(x_cubed * fact_3, num_bits)


def cos_ts(x):
    p2 = (x**2) / (factorial(2))
    # Beyond 2nd order correction seems to give diminishing/negligible extra precision!
    # p4 = (x**4) / (factorial(4))
    # p6 = (x**6) / (factorial(6))
    # p8 = (x**8) / (factorial(8))
    return 1 - p2  # + p4 - p6


def cos_ts_fxp(x, num_bits):
    # X assumed already at num_bits width
    # round/truncate by 1x extra bit to essentially fold in /2!
    x_square = utils.fxp_round_halfup(x * x, num_bits - 1)
    return utils.dbl_to_fxp(1.0, num_bits - 1) - x_square


class Nco:
    def __init__(
        self,
        N: np.uint,
        M: np.uint,
        P: np.uint,
        fs: float,
        quarter_wave: bool = False,
        init_phase: int = 0,
        dither: bool = False,
        dither_bits: int = 4,
        taylor_corr: bool = False,
    ):
        """Create an NCO object which outputs a complex sinusoid. Note the
        real part can just be taken for a real-valued NCO.

        Parameters
        ----------
        N : uint
            Phase accumulator length (number of bits)
        M : uint
            Sine/Cosine LUT sample quantized word length (number of bits)
        P : uint
            LUT table address length (number of bits) for a full sinusoidal wave.
            NOTE: Total depth of full-wave LUT = 2**P, however when `quarter_wave`
             is True, actual total depth of LUT = 2**(P-2)
        fs : float
            Sampling frequency (or clock rate) of NCO (Hz)
        quarter_wave : bool, default: False
            When True, only populate LUT with 1/4 full sinusoid
        init_phase : int, default: 0
            Initial phase (sets initial value in phase accumulator)
        dither : bool, default: False
            When True, enable phase accumulator dithering
        dither_bits : int, default: 4
            Magnitude of phase dithering added to phase accumulator
            Range of: +/- 2**(dither_bits-1)
        taylor_corr: bool, default: False
            When True, enable Taylor series phase and amplitude correction
        """
        if N > 32:
            raise ValueError("Phase accumulator width currently limited to uint32 max!")

        if P > 32:
            raise ValueError("LUT table depth currently limited to 2**32 max!")

        self._N = N
        self._M = M
        self._P = P
        self._fs = fs
        self._quarter_wave = quarter_wave
        self._dither = dither
        self._dither_amp = 2 ** (dither_bits - 1)
        self._taylor_corr = taylor_corr

        # Frequency control word (FCW) can be publicly set/get at any time
        # or set using `SetOutputFreq()`
        self.FCW = 0

        self._phase_acc = init_phase  # internal phase accumulator

        # allocate LUT for complex samples
        LUT_P = P - 2 if quarter_wave else P
        self._LUT = np.zeros(int(2**LUT_P)) * 1j * np.zeros(int(2**LUT_P))

        self._CalcLUTSamples()

        self._sfdr = 6.02 * self._P - 3.92
        self._noise_floor = measurements.ideal_SNR(int(M))
        self._freq_resolution = self._fs / (2.0**N)

    @property
    def SFDR(self):
        """Spurious Free Dynamic Range (SFDR) limited by phase truncation noise
        (no dithering)"""
        return self._sfdr

    @property
    def noise_floor(self):
        """Ideal noise floor due to quantization noise"""
        return self._noise_floor

    @property
    def freq_resolution(self):
        """Frequency resolution based on accumulator width, N (Hz)"""
        return self._freq_resolution

    @property
    def LUT(self):
        """Get the LUT samples of the NCO"""
        return self._LUT

    def _CalcLUTSamples(self):
        """Calculate and fill LUT values (using full-wave, not 1/4)"""

        # Since using signed integers, account for sign bit in quantization
        # Max amplitude -1 full-scale to fit in M-bit vector
        A = (2 ** (self._M - 1)) - 1

        for i in range(len(self._LUT)):
            cos_val = np.floor(A * np.cos((2 * np.pi * i) / (2**self._P)))
            sin_val = np.floor(A * np.sin((2 * np.pi * i) / (2**self._P)))
            self._LUT[i] = cos_val + 1j * sin_val

    def SetOutputFreq(self, f: float):
        """Convert a desired output frequency `f` (Hz) to NCO Frequency Control
        Word (FCW) and set self.FCW"""
        # TODO: do we care to support -fs/2 for signed FCW??
        if not (0.0 <= f <= self._fs / 2):
            raise ValueError(
                f"Given frequency {f} Hz is outside NCO range of 0 Hz -> fs/2"
            )

        self.FCW = FreqToFcw(f, int(self._N), self._fs)

    def Reset(self):
        """Reset internal phase accumulator"""
        self._phase_acc = 0

    def GetCurrentNcoOutput(self) -> complex:
        """Return the current NCO output given current internal phase
        accumulator state"""
        lut_index = 0
        # determine the phase accumulator quad based on it's upper 2 bits
        quad = self._phase_acc // (2 ** (self._N - 2))

        if self._quarter_wave:
            if quad == 0:  # no change
                lut_index = self._phase_acc // (2 ** (self._N - self._P))
            elif quad == 1:  # upper 2'bx1 set
                lut_index = 2 * len(self._LUT) - self._phase_acc // (
                    2 ** (self._N - self._P)
                )
            elif quad == 2:
                lut_index = self._phase_acc // (2 ** (self._N - self._P)) - 2 * len(
                    self._LUT
                )
            elif quad == 3:  # upper 2'bx1 set
                lut_index = 4 * len(self._LUT) - self._phase_acc // (
                    2 ** (self._N - self._P)
                )

        else:
            # take phase accumulator MSBs for LUT index (>> N-P)
            lut_index = self._phase_acc // (2 ** (self._N - self._P))

        # Clamp max index
        lut_index = min(int(lut_index), len(self._LUT) - 1)

        # index LUTs using phase accumulator MSBs to generate complex output sample
        lut_output = 0 + 1j * 0
        if self._quarter_wave and ((quad == 2) or (quad == 3)):  # upper 2'b1x set
            lut_output = -self._LUT[lut_index]
        else:
            lut_output = self._LUT[lut_index]

        if self._taylor_corr:
            # Regardless of full vs quarter wave, these are the phase accumulator
            # MSBs used for the LUT index (phase_acc >> N-P):
            #  phase_acc_MSB = self._phase_acc // (2 ** (self._N - self._P))
            # The "shifted out" bits from the full-width phase accumulator
            # by the phase quantizer gives the truncated (dropped) phase LSBs:
            phase_acc_LSB = int(self._phase_acc % int(2 ** (self._N - self._P)))
            if nco_print_trace:
                print(
                    f"\nN - P: {self._N - self._P} bits | phase_acc_LSB = {phase_acc_LSB}"
                )
            # ^ these dropped bits represent the phase error for a given look-up
            # cycle, as its the leftover phase value not represented by a LUT index.
            # We'll use this phase error value to calculate a sin() & cos() value
            # using a 2nd order Taylor Series expansion.
            # Scale by phase accumulator size to get phase error in radians:
            #  * 2pi since unsigned phase acc range represents phase 0 -> 2pi and
            #  / phase acc width to normalize
            # Floating-Point (normalized radians):
            phase_err_fp = 2.0 * np.pi * phase_acc_LSB / (2.0**self._N)

            # Try 18b math? (DSP mult size optimized...). LSB has a width of:
            # N - P bits, so truncate to desirec
            fxp_num_bits = 18
            phase_err_fxp = utils.fxp_round_halfup(
                phase_acc_LSB, int(self._N - self._P - fxp_num_bits)
            )
            phase_LSB_orig = phase_acc_LSB / (2 ** (self._N - self._P))
            phase_LSB_rounded = phase_err_fxp / (2**fxp_num_bits)
            phase_LSB_err = 0.0
            if phase_LSB_orig != 0.0:
                phase_LSB_err = (
                    100.0 * (phase_LSB_rounded - phase_LSB_orig) / phase_LSB_orig
                )
            if nco_print_trace:
                print(
                    f"Rounding phase_acc_LSB to {fxp_num_bits} bits = {phase_err_fxp} ({phase_LSB_err}% error)"
                )
            # NOTE: -20 was found to give least error (0-mean) against floating point final value
            #  so in HDL implementation/testing, a similar % error/calibration should be done
            fxp_2pi = utils.dbl_to_fxp(2.0 * np.pi / 1000.0, fxp_num_bits - 1) - 20
            if nco_print_trace:
                print(f"2pi fxp ({fxp_num_bits} bits unsigned): {fxp_2pi}")
            phase_err = utils.fxp_round_halfup(
                phase_err_fxp * fxp_2pi, fxp_num_bits + 0
            )
            phase_err_fxp_rounded = phase_err / (2**fxp_num_bits)
            phase_err_fp_fxp_err = 0.0
            if phase_err_fp != 0.0:
                phase_err_fp_fxp_err = (
                    100.0 * (phase_err_fxp_rounded - phase_err_fp) / phase_err_fp
                )

            if nco_print_trace:
                print(
                    f"Phase_err (fp): {phase_err_fp} rads | Phase_err (fxp): {phase_err_fxp_rounded} rads ({phase_err_fp_fxp_err}% error)"
                )

            # cos(b)
            cos_err_fxp = cos_ts_fxp(phase_err, fxp_num_bits + 1)
            # sin(b)
            sin_err_fxp = sin_ts_fxp(phase_err, fxp_num_bits)
            # cos(a)
            lut_real = np.real(lut_output)
            # sin(a)
            lut_imag = np.imag(lut_output)

            cos_err_fp = cos_ts(phase_err_fp)
            sin_err_fp = sin_ts(phase_err_fp)

            # Small angle approximation
            cos_err_fp_small = 1.0 - phase_err_fp
            sin_err_fp_small = phase_err_fp
            if nco_print_trace:
                if cos_err_fp != 0.0:
                    print(
                        f"Cos small-angle approx. error= {100.0*(cos_err_fp_small - cos_err_fp)/cos_err_fp}%"
                    )
                if sin_err_fp != 0.0:
                    print(
                        f"Sin small-angle approx. error= {100.0*(sin_err_fp_small - sin_err_fp)/sin_err_fp}%"
                    )

            if use_small_angle_approx:
                cos_err_fp = cos_err_fp_small
                sin_err_fp = sin_err_fp_small

            cos_round_err = 0.0
            if cos_err_fp != 0.0:
                cos_round_err = (
                    100.0
                    * ((cos_err_fxp / (2**fxp_num_bits)) - cos_err_fp)
                    / cos_err_fp
                )
            sin_round_err = 0.0
            if sin_err_fp != 0.0:
                sin_round_err = (
                    100.0
                    * ((sin_err_fxp / (2**fxp_num_bits)) - sin_err_fp)
                    / sin_err_fp
                )
            if nco_print_trace:
                print(f"Taylor-Series cos_err = {cos_err_fxp} ({cos_round_err}% error)")
                print(f"Taylor-Series sin_err = {sin_err_fxp} ({sin_round_err}% error)")

            cos_err = cos_err_fxp if use_fxp else cos_err_fp
            sin_err = sin_err_fxp if use_fxp else sin_err_fp

            # Use trig identities to combine the LUT values (quantized phase) w/the
            # Taylor series derived values (phase error):
            # cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
            real_corr = (lut_real * cos_err) - (lut_imag * sin_err)
            # sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
            imag_corr = (lut_imag * cos_err) + (lut_real * sin_err)

            phase_corrected = real_corr + 1j * imag_corr
            return phase_corrected
        else:
            return lut_output

    def IncPhaseAcc(self, new_FCW: int):
        """Increment phase accumulator using frequency control word (FCW)"""
        # update phase accumulator for next cycle
        self._phase_acc += new_FCW
        if self._dither:
            self._phase_acc += random.randrange(-self._dither_amp, self._dither_amp)
        # handling wrapping of phase accumulator like in FXP
        if self._phase_acc >= 2**self._N:
            self._phase_acc -= 2**self._N
        if self._phase_acc < 0:
            self._phase_acc += 2**self._N

    def Step(self) -> complex:
        """Step the NCO one clock cycle by converting internal phase accumulator
        -> complex output using LUT and incrementing"""
        retVal = self.GetCurrentNcoOutput()
        self.IncPhaseAcc(self.FCW)
        return retVal

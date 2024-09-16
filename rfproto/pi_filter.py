#
# PI Filter class
#

import numpy as np


class PiFilter:
    def __init__(
        self,
        frac_loop_bw: float,
        detector_gain: float,
        scaling_factor: float = 1.0,
        damp_factor: float = 1 / np.sqrt(2.0),
    ):
        """Create a PI Filter implementation. Updating the loop bandwidth, which
        affects the internal Kp * Ki coefficients, will automatically trigger
        a recalculation of those coefficients.

        Parameters
        ----------
        * frac_loop_bw : float
            The percentage (e.g. 0.0 - 1.0) of total bandwidth to set the 3dB
            bandwidth of the PI loop filter.
        * detector_gain  : float
            Gain of error detector before PI filter, for example:
                abs(max_error) / detector_gain = 1.0
        * scaling_factor : float, optional, default: 1.0
            Scalar value multiplied to Kp & Ki coefficients after derivation
            to scale PI filter output (e.x. INT16_MAX when wanting `short` output)
        * damp_factor : float, optional, default: 1/sqrt(2)
            Damping factor used in calculating coefficients
        """
        self._Ki = 0.0
        self._Kp = 0.0

        self._frac_bandwidth = 0.0  # will use checks in setter property

        self._Kd = detector_gain

        if not (0.0 <= damp_factor <= 3.0):
            raise ValueError(f"Damping factor of {damp_factor} not between 0.0 and 3.0")
        self._zeta = damp_factor

        self._scale_factor = scaling_factor
        self.Kd = detector_gain
        self.zeta = damp_factor

        self.frac_BW = frac_loop_bw  # using setter will trigger Kp/Ki recalculation

        # allow user to publicly get/set accumulator value at any time
        # Can be used as initial value before PI loop use, or for resetting
        # the loop at any time
        self.accumulator = 0.0

    @property
    def frac_BW(self):
        """Fraction (0.0 - 1.0) of total loop Bandwidth the PI loop filter is set to"""
        return self._frac_bandwidth

    @frac_BW.setter
    def frac_BW(self, value):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Fractional bandwidth of {value} not between 0.0 and 1.0")
        self._frac_bandwidth = value
        self._CalcCoeff()  # recalculate Kp * Ki coeffs

    # Kp & Ki coefficients are read-only, calculated within
    @property
    def Kp(self):
        """Proportional gain constant"""
        return self._Kp

    @property
    def Ki(self):
        """Integral gain constant"""
        return self._Ki

    def _CalcCoeff(self):
        """Calculates Kp & Ki coefficients"""
        # natural freq. -> rad/s
        wn = 2.0 * np.pi * self._frac_bandwidth
        # adjust for damping factor (== wn when zeta = 1/sqrt(2))
        alpha = 1.0 - (2.0 * self._zeta * self._zeta)
        wn /= np.sqrt(alpha + np.sqrt(alpha * alpha + 1.0))

        # Here I assume the 2nd pole of our system (the thing responding to this
        # PI loop filter's output) to be z = 1 and to have a gain (`kv`) related
        # to the inverse of symbol period (e.g. an NCO/mixer, or TED). This allows
        # us to use a simplified derivation of discrete time constants.
        self._Kp = 2.0 * self._zeta * wn / self._Kd
        self._Ki = wn * wn / self._Kd

        self._Kp *= self._scale_factor
        self._Ki *= self._scale_factor

    def Step(self, err: float) -> float:
        """Step the PI filter given an input error value `err` and returns the
        next PI filter output"""
        self.accumulator += self._Ki * err
        return self.accumulator + (self._Kp * err)
        ## NOTE: for FPGAs where we want to limit the number of delays from err_in->err_out,
        ##  we could use the below which allows the previous accumulator value to be used.
        ##  It's nearly identical in loop response to the above, but can be done in 2 stages
        ##  (Kp/Ki error products -> out = acc + Kp_prod & acc += Ki_prod) rather than 3 stages
        ##  (Kp/Ki error products -> acc + Kp_prod -> out = acc += Ki_prod)
        # prev_acc = self.accumulator
        # self.accumulator += self._Ki * err
        # return prev_acc + (self._Kp * err)

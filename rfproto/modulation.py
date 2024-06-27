#
# Modulation utilities
#
# Partilally derived from [komm](https://github.com/rwnobrega/komm)
# which is not maintained much...
#
# TODO: https://en.wikipedia.org/wiki/Amplitude_and_phase-shift_keying
# TODO: use techniques like https://www.gaussianwaves.com/2012/10/qam-modulation-simulation-matlab-python/
#
import numpy as np
from . import utils


class Modulation:
    def __init__(self, constellation: np.ndarray):
        self._order = len(constellation)
        if self._order & (self._order - 1):
            raise ValueError("The length of constellation must be a power of two!")
        self._bits_per_symbol = (self._order - 1).bit_length()
        # Gray code the constellation
        gray_codes = utils.gray_code(self._bits_per_symbol)
        self._constellation = np.array([constellation[i] for i in gray_codes])

    @property
    def constellation(self) -> np.ndarray:
        r"""The constellation $S$ of the modulation. This property is read-only."""
        return self._constellation

    @property
    def order(self) -> int:
        r"""The order $M$ of the modulation. This property is read-only."""
        return self._order

    @property
    def bits_per_symbol(self) -> int:
        r"""
        The number bits per symbol of the $M$-ary modulation, equal to $ log2(M) $
        This property is read-only.
        """
        return self._bits_per_symbol

    def modulate(self, symbols: list[int]):
        """Modulates a sequence of symbols to corresponding I/Q constellation points"""
        return self._constellation[symbols]


class MPSKModulation(Modulation):
    """
    M-ary Phase-shift keying (PSK) modulation

    References
    ----------
    .. [1] [Phase-shift keying - Wikipedia](https://en.wikipedia.org/wiki/Phase-shift_keying)
    """

    def __init__(self, order: int, amplitude: float = 1.0):
        r"""
        Parameters
        ----------
        order : int
            The order $M$ of the modulation, must be a power of 2.

        amplitude : float, optional, default = 1.0
            The amplitude $A$ of the modulation.
        """
        constellation = np.zeros(order)
        if order == 2:  # BPSK
            constellation = np.array([1 + 0j, -1 + 0j])
        if order == 4:  # QPSK
            constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
        else:
            constellation = np.exp(2j * np.pi * np.arange(order) / order)
        constellation *= amplitude

        super().__init__(constellation)


# class QAModulation(ComplexModulation):
#    """ Quadratude-amplitude modulation (QAM) """
#
#    def __init__(
#        self, orders, base_amplitudes=1.0, phase_offset=0.0, labeling="reflected_2d"
#    ):
#        """
#        >>> qam = komm.QAModulation(16)
#        >>> qam.constellation  #doctest: +NORMALIZE_WHITESPACE
#        array([-3.-3.j, -1.-3.j,  1.-3.j,  3.-3.j,
#               -3.-1.j, -1.-1.j,  1.-1.j,  3.-1.j,
#               -3.+1.j, -1.+1.j,  1.+1.j,  3.+1.j,
#               -3.+3.j, -1.+3.j,  1.+3.j,  3.+3.j])
#        >>> qam.labeling
#        array([ 0,  1,  3,  2,  4,  5,  7,  6, 12, 13, 15, 14,  8,  9, 11, 10])
#        >>> qam.modulate([0, 0, 1, 1, 0, 0, 1, 0])
#        array([-3.+1.j, -3.-1.j])
#        """
#        if isinstance(orders, (tuple, list)):
#            order_I, order_Q = int(orders[0]), int(orders[1])
#            self._orders = (order_I, order_Q)
#        else:
#            order_I = order_Q = int(np.sqrt(orders))
#            self._orders = int(orders)
#
#        if isinstance(base_amplitudes, (tuple, list)):
#            base_amplitude_I, base_amplitude_Q = float(base_amplitudes[0]), float(
#                base_amplitudes[1]
#            )
#            self._base_amplitudes = (base_amplitude_I, base_amplitude_Q)
#        else:
#            base_amplitude_I = base_amplitude_Q = float(base_amplitudes)
#            self._base_amplitudes = base_amplitude_I
#
#        constellation_I = base_amplitude_I * np.arange(
#            -order_I + 1, order_I, step=2, dtype=int
#        )
#        constellation_Q = base_amplitude_Q * np.arange(
#            -order_Q + 1, order_Q, step=2, dtype=int
#        )
#        constellation = (
#            constellation_I + 1j * constellation_Q[np.newaxis].T
#        ).flatten() * np.exp(1j * phase_offset)
#
#        if isinstance(labeling, str):
#            if labeling == "natural":
#                labeling = Modulation._labeling_natural(order_I * order_Q)
#            elif labeling == "reflected_2d":
#                labeling = Modulation._labeling_reflected_2d(order_I, order_Q)
#            else:
#                raise ValueError(
#                    "Only 'natural' or 'reflected_2d' are supported for {}".format(
#                        self.__class__.__name__
#                    )
#                )
#
#        super().__init__(constellation, labeling)
#
#        self._orders = orders
#        self._base_amplitudes = base_amplitudes

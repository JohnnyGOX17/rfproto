class fxp_int(int):
    """Basic class to model a fixed bit-width integer data type which rolls over
    at maximal/minimal extremes"""

    def __new__(cls, value: int, bit_width: int, unsigned: bool = False):
        return int.__new__(cls, value)

    def __init__(self, value: int, bit_width: int, unsigned: bool = False):
        if bit_width < 2:
            raise ValueError(f"Unsuported bit width of {bit_width}")
        self.bit_width = bit_width
        """ Bit width of FXP integer """
        self.unsigned = unsigned
        # Set the 2's complement minimum and maximum values
        if unsigned:
            self.max_val = (2 ** (bit_width)) - 1
            self.min_val = 0
        else:
            self.max_val = (2 ** (bit_width - 1)) - 1
            self.min_val = -(2 ** (bit_width - 1))

        if (value > self.max_val) or (value < self.min_val):
            raise ValueError(f"Value of {value} out-of-range for bitwidth!")

    def _clamp(self, x: int) -> int:
        delta = 0
        if x > self.max_val:
            delta = x % self.max_val
            return self.min_val + (delta - 1)
        elif x < self.min_val:
            if not self.unsigned:
                delta = abs(x % self.min_val)
            else:  # avoid /0, but still clamp within 2^N
                delta = abs(x) % (2**self.bit_width)
            return self.max_val - (delta - 1)
        else:
            return x

    def __add__(self, other):
        x = int(self)
        if isinstance(other, self.__class__):
            y = int(other)
            return fxp_int(self._clamp(x + y), self.bit_width, self.unsigned)
        elif isinstance(other, int):
            return fxp_int(self._clamp(x + other), self.bit_width, self.unsigned)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{self.__class__}' and '{type(other)}'"
            )

    def __sub__(self, other):
        x = int(self)
        if isinstance(other, self.__class__):
            y = int(other)
            return fxp_int(self._clamp(x - y), self.bit_width, self.unsigned)
        elif isinstance(other, int):
            return fxp_int(self._clamp(x - other), self.bit_width, self.unsigned)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: '{self.__class__}' and '{type(other)}'"
            )

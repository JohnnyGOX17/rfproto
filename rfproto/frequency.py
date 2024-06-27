def FreqStr(value: float, unit: str):
    if value >= 1e9:
        value /= 1e9
        unit = "G" + unit
    elif value >= 1e6:
        value /= 1e6
        unit = "M" + unit
    elif value >= 1e3:
        value /= 1e3
        unit = "k" + unit
    return "%0.2f %s" % (value, unit)


class Frequency(float):
    def __new__(cls, value: float, unit: str):
        return float.__new__(cls, value)

    def __init__(self, value: float, unit: str):
        if (unit != "Hz") and (unit != "radians"):
            raise AttributeError("Frequency unit given is not 'Hz' nor 'radians'")
        self.unit = unit
        """ Frequency unit string (Hz or radians) """

    def __str__(self):
        return FreqStr(self, self.unit)


if __name__ == "__main__":
    try:
        bad_unit = Frequency(57, "kHz")
    except AttributeError:
        print("Caught")
    test = Frequency(46.2, "Hz")
    test_kHz = Frequency(4620, "Hz")
    test_Mrad = Frequency(1000000, "radians")
    test_GHz = Frequency(46e9, "Hz")
    print(test)
    print(test_kHz)
    print(test_Mrad)
    print(test_GHz)

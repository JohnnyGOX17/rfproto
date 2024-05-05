from rfproto import fxp_int as fp


def test_bad_ctor():
    try:
        _ = fp.fxp_int(10, -2)
    except ValueError:
        print("Caught")


def test_unsigned():
    # test ctor
    test_unsigned = fp.fxp_int(2, 8, True)
    assert test_unsigned == 2

    test_unsigned += 254
    # 2 + 254 = 256 -> rolls-over to 0
    assert test_unsigned == 0
    new_unsigned = fp.fxp_int(4, 8, True)
    new_unsigned += test_unsigned
    assert new_unsigned == 4
    new_unsigned = new_unsigned - 6
    # 4 - 6 = -2 -> rolls-over to 254
    assert new_unsigned == 254


def test_signed():
    # test ctor
    test_signed = fp.fxp_int(2, 8, False)
    assert test_signed == 2

    # signed tests
    test_signed += 126
    # 2 + 126 = 128 -> rolls-over to -128
    assert test_signed == -128
    new_signed = fp.fxp_int(4, 8, False)
    new_signed += test_signed
    assert new_signed == -124
    new_signed = new_signed - 6
    # -124 - 6 = -130 -> roll-over back to positive 126
    assert new_signed == 126

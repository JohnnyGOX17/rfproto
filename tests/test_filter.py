import os
from rfproto import filter
from rfproto import plot


def test_rc():
    if os.environ.get("NO_PLOT") == "true":
        return
    plot.samples(filter.RaisedCosine(2.0, 1.0, 0.4, 32))


if __name__ == "__main__":
    test_rc()

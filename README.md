# rfproto

[![CI Pipeline](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml)
[![PyPI - Version](https://badge.fury.io/py/rfproto.svg)](https://badge.fury.io/py/rfproto)
![PyPI - License](https://img.shields.io/pypi/l/rfproto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python library for RF and SDR prototyping. Provides reusable methods for RF measurements, communication systems, radar, antenna arrays, and related signal processing tasks. The library supports both floating-point and fixed-point arithmetic for modeling hardware implementations (FPGA/ASIC).

## Features & Modules

The library is organized into focused modules for different aspects of RF/SDR signal processing:

### Core Signal Processing
- **`nco.py`**: Numerically Controlled Oscillator (NCO) with both floating-point and fixed-point support, including frequency-to-FCW conversion and Taylor series approximations for sine/cosine generation
- **`filter.py`**: Filter design including Raised Cosine, Root Raised Cosine, and polyphase filtering
- **`multirate.py`**: Multirate signal processing (interpolation, decimation, rational resampling with polyphase filters)
- **`sig_gen.py`**: Signal generation utilities for test waveforms and modulated signals
- **`modulation.py`**: Modulation/demodulation schemes for digital communications

### Measurements & Analysis
- **`measurements.py`**: Signal quality metrics (EVM, PSD, SFDR, SNR, and spectral analysis)
- **`plot.py`**: Visualization utilities (time-domain, frequency-domain, constellation diagrams, eye diagrams)
- **`impairments.py`**: Channel impairment modeling (noise, phase offset, timing errors)

### Fixed-Point Arithmetic
- **`fxp_int.py`**: Fixed-point integer class modeling hardware bit-width constraints with overflow/wrapping behavior
- **`utils.py`**: Fixed-point/floating-point conversion, binary representation, rounding methods, and dB/magnitude conversions

### Control & Processing
- **`agc_magest.py`**: Automatic Gain Control (AGC) using magnitude estimation
- **`pi_filter.py`**: Proportional-Integral filter implementations
- **`signal.py`**: Basic signal operations and transformations
- **`frequency.py`**: Frequency-related utilities and conversions

## Architecture

### Fixed-Point Arithmetic Support

A key feature of `rfproto` is dual floating-point and fixed-point implementations for hardware prototyping:

1. **`fxp_int` class**: Models fixed bit-width integers with 2's complement overflow/wrapping behavior
2. **Conversion utilities**: `dbl_to_fxp()` / `fxp_to_dbl()` for converting between representations
3. **Rounding operations**: `fxp_round_halfup()` / `fxp_truncate()` for proper fixed-point arithmetic
4. **Q-format notation**: Uses standard Q-format (e.g., Q0.15 for 16-bit signed with 15 fractional bits)
5. **Switchable implementations**: Some modules like `nco.py` have `use_fxp` flags to toggle between floating-point and fixed-point

### Design Patterns

- **Normalized filters**: Filter functions return normalized coefficients (sum to 1.0 for unity passband gain)
- **Rational resampling**: Use `multirate.get_rational_resampling_factors()` for optimal interpolation/decimation ratios
- **Type hints**: All functions include type hints for parameters and return values
- **Doctests**: Inline documentation examples that serve as both docs and tests

## Developing

### Building & CI

* Install editable local version (preferably within a [venv](https://john-gentile.com/kb/programming_languages/python.html#virtual-environments-venv)) with all optional packages for testing with `$ pip install --upgrade -e .[docs,test]` (add `--user` if not in `venv`).
* Install pre-commit checks with `$ ln -sf ../../scripts/pre-commit ./.git/hooks/pre-commit`
* Trigger GitHub action to publish to PyPI with a tagged commit (e.x. `git tag -am "test auto versioning" 0.0.2`) on `main` branch. Note versioning is also inferred from the git tag value, and this will only run on push on tag.

### Testing

Run test suite with `$ ./scripts/run-tests.sh`

Additional testing options:
```bash
# Run tests with plots displayed (default suppresses plots)
./scripts/run-tests.sh --show_plots

# Run specific test file
pytest tests/test_filter.py

# Run with coverage report
pytest --doctest-modules --cov-report=term-missing:skip-covered --cov=rfproto/ tests/

# Run mypy static type checking
mypy rfproto/
```

The test suite includes:
- pytest with doctest support for inline documentation examples
- Tests in both `tests/` directory and doctests embedded in module docstrings
- Coverage reporting via pytest-cov
- Full mypy type checking (enforced in CI)
- Pre-commit hook that runs full test suite and auto-generates TODO.md

### Documentation

Documentation uses [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), preview with `$ mkdocs serve -a localhost:8888`. Publishes with GitHub action as well.

### Code Formatting

The project uses `black` for code formatting. The pre-commit hook automatically runs the full test suite (including formatting checks) before commits are allowed.

## TODO

See auto-generated [TODO.md](./TODO.md).

- [ ] Look at https://github.com/veeresht/CommPy since its unmaintained
- [ ] Filtering/convolution kernels like https://joht.github.io/johtizen/algorithm/2022/10/22/a-different-approach-to-convolution.html and https://dsp.stackexchange.com/questions/15412/fir-filters-direct-form-transposed-fir


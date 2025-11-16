# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`rfproto` is a Python library for RF (Radio Frequency) and SDR (Software Defined Radio) prototyping. It provides reusable methods for RF measurements, communication systems, radar, antenna arrays, and related signal processing tasks. The library supports both floating-point and fixed-point arithmetic for modeling hardware implementations.

## Development Commands

### Setup
```bash
# Install editable version with all optional dependencies (preferably in a venv)
pip install --upgrade -e .[docs,test]

# Install pre-commit hook
ln -sf ../../scripts/pre-commit ./.git/hooks/pre-commit
```

### Testing
```bash
# Run full test suite (mypy + pytest with coverage)
./scripts/run-tests.sh

# Run tests with plots displayed (default suppresses plots)
./scripts/run-tests.sh --show_plots

# Run specific test file
pytest tests/test_filter.py

# Run with coverage report
pytest --doctest-modules --cov-report=term-missing:skip-covered --cov=rfproto/ tests/
```

### Type Checking
```bash
# Run mypy static type checking
mypy rfproto/
```

### Code Formatting
The project uses `black` for code formatting. The pre-commit hook automatically runs tests before commits.

### Documentation
```bash
# Preview documentation locally
mkdocs serve -a localhost:8888
```

Documentation uses mkdocs-material and is auto-published via GitHub Actions.

### Publishing
Versioning is inferred from git tags using `setuptools-git-versioning`. To trigger a PyPI publish:
```bash
# Create and push a tagged commit on main branch
git tag -am "version description" X.Y.Z
git push origin main --tags
```

## Architecture

### Module Organization

The library is organized into focused modules, each handling a specific aspect of RF/SDR signal processing:

- **`nco.py`**: Numerically Controlled Oscillator implementation with both floating-point and fixed-point (FXP) support. Includes frequency-to-FCW conversion helpers and Taylor series approximations for sine/cosine generation.

- **`filter.py`**: Filter design and implementation including Raised Cosine, Root Raised Cosine, and polyphase filtering. Depends on `multirate.py` for resampling operations.

- **`multirate.py`**: Multirate signal processing including interpolation, decimation, and rational resampling with polyphase filters. Provides `find_rational_frac()` to determine optimal upsampling/downsampling factors.

- **`fxp_int.py`**: Fixed-point integer class that models hardware bit-width constraints with overflow/underflow wrapping behavior. Used throughout the library for modeling FPGA/ASIC implementations.

- **`utils.py`**: Core utilities for fixed-point/floating-point conversion (`fxp_to_dbl`, `dbl_to_fxp`), binary representation, rounding methods (`fxp_round_halfup`, `fxp_truncate`), and dB/magnitude conversions.

- **`measurements.py`**: Signal quality metrics including EVM (Error Vector Magnitude), PSD (Power Spectral Density), SFDR, SNR, and other spectral analysis functions.

- **`plot.py`**: Visualization utilities for time-domain signals, frequency-domain analysis, constellation diagrams, and eye diagrams.

- **`sig_gen.py`**: Signal generation utilities for test waveforms and modulated signals.

- **`modulation.py`**: Modulation/demodulation schemes for digital communications.

- **`impairments.py`**: Channel impairment modeling (noise, phase offset, timing errors, etc.).

- **`agc_magest.py`**: Automatic Gain Control (AGC) using magnitude estimation.

- **`pi_filter.py`**: Proportional-Integral filter implementations.

- **`signal.py`**: Basic signal operations and transformations.

- **`frequency.py`**: Frequency-related utilities and conversions.

### Fixed-Point Arithmetic

The library has dual floating-point and fixed-point implementations for many algorithms to support hardware prototyping:

1. **`fxp_int` class** in `fxp_int.py`: Models fixed bit-width integers with 2's complement overflow/wrapping.

2. **Utility functions** in `utils.py`:
   - `dbl_to_fxp()` / `fxp_to_dbl()`: Convert between floating-point and fixed-point representations
   - `fxp_round_halfup()` / `fxp_truncate()`: Fixed-point rounding operations
   - Uses Q-format notation (e.g., Q0.15 for 16-bit signed values with 15 fractional bits)

3. **FXP flags**: Some modules like `nco.py` have `use_fxp` flags to switch between floating-point and fixed-point implementations.

### Testing Strategy

- Uses pytest with doctest support for inline documentation examples
- Tests include both discrete test files in `tests/` and doctests embedded in module docstrings
- Coverage reporting via `pytest-cov`
- Tests can optionally display plots using `NO_PLOT` environment variable
- Pre-commit hook runs full test suite and auto-generates TODO.md from code comments

### Type Checking

- Full mypy type checking is enforced in CI
- `mypy.ini` ignores missing imports for matplotlib and scipy
- All functions should include type hints for parameters and return values

## Common Patterns

### Working with Fixed-Point Values

When implementing or modifying fixed-point algorithms:
- Use numpy integer types (int8, int16, int32, etc.) or the `fxp_int` class
- Specify fractional bit width explicitly (e.g., `num_frac_bits = 15` for Q0.15)
- Use `fxp_round_halfup()` for proper fixed-point rounding (not Python's `round()`)
- Model bit-growth in multiplications and accumulations

### Filter Design

- Filters return normalized coefficients (sum to 1.0 for unity passband gain)
- Use `multirate.get_rational_resampling_factors()` to find optimal interpolation/decimation ratios
- Polyphase filters are preferred for efficient multirate processing

### Signal Measurements

- Most measurement functions operate on numpy arrays
- PSD functions support both real and complex signals
- Use `norm=True` for normalized plots, or specify `max_mag` for dBFS calculations

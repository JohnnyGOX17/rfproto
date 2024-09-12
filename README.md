# rfproto

[![CI Pipeline](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml)
[![PyPI - Version](https://badge.fury.io/py/rfproto.svg)](https://badge.fury.io/py/rfproto)
![PyPI - License](https://img.shields.io/pypi/l/rfproto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python library for RF and SDR prototyping. Helpful reuse methods for RF measurements, as well as experimenting with topics like communication systems, radar, antenna arrays, etc.


## Developing

### Building & CI

* Install editable local version (preferably within a [venv](https://john-gentile.com/kb/programming_languages/python.html#virtual-environments-venv)) with all optional packages for testing with `$ pip install --upgrade -e .[docs,test]` (add `--user` if not in `venv`).
* Install pre-commit checks with `$ ln -sf ../../scripts/pre-commit ./.git/hooks/pre-commit`
* Trigger GitHub action to publish to PyPI with a tagged commit (e.x. `git tag -am "test auto versioning" 0.0.2`) on `main` branch. Note versioning is also inferred from the git tag value, and this will only run on push on tag.

### Testing

Run test suite with `$ ./scripts/run-tests.sh`

### Documentation

Documentation uses [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), preview with `$ mkdocs serve -a localhost:8888`. Publishes with GitHub action as well.

## TODO

See auto-generated [TODO.md](./TODO.md).

- [ ] Look at https://github.com/veeresht/CommPy since its unmaintained
- [ ] Filtering/convolution kernels like https://joht.github.io/johtizen/algorithm/2022/10/22/a-different-approach-to-convolution.html and https://dsp.stackexchange.com/questions/15412/fir-filters-direct-form-transposed-fir


# rfproto

[![CI Pipeline](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml)
[![PyPI - Version](https://badge.fury.io/py/rfproto.svg)](https://badge.fury.io/py/rfproto)
![PyPI - License](https://img.shields.io/pypi/l/rfproto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python for RF and SDR prototyping.


## Building & CI

* Trigger GitHub action to publish to PyPI with a tagged commit (e.x. `git tag -am "test auto versioning" 0.0.2`) on `main` branch. Note versioning is also inferred from the git tag value, and this will only run on push on tag.
* Documentation uses [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), preview with `$ mkdocs serve`. Publishes with GitHub action as well.


# rfproto

[![CI Pipeline](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyGOX17/rfproto/actions/workflows/ci.yml)

Python for RF and SDR prototyping.


## Building & CI

* Trigger GitHub action to publish to PyPI with a tagged commit (e.x. `git tag -am "test auto versioning" 0.0.2`) on `main` branch. Note versioning is also inferred from the git tag value, and this will only run on push on tag.
* Documentation uses [mkdocs-material](https://squidfunk.github.io/mkdocs-material/), preview with `$ mkdocs serve`. Publishes with GitHub action as well.


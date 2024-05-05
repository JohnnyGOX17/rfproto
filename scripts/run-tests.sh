#!/usr/bin/env bash

echo "Running tests..."

# start at top of Git repo
pushd "$(git rev-parse --show-toplevel)" > /dev/null || exit

echo "    > [mypy] Static type checking"
if ! mypy rfproto/; then
  echo "mypy failed!"
  exit 1
fi

echo "    > [pytest] Running unit tests (discrete + doctests) w/coverage"
if ! pytest --doctest-modules --cov-report=term-missing:skip-covered --cov=. .; then
  echo "pytest failed!"
  exit 1
fi

echo "Tests successfully completed!"

popd > /dev/null || exit

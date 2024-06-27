#!/usr/bin/env bash

echo "Running tests..."
src_folder="rfproto/"
tst_folder="tests/"

if [[ "$1" == "--show_plots" ]]; then
  export NO_PLOT="false"
else
  export NO_PLOT="true"
fi

# start at top of Git repo
pushd "$(git rev-parse --show-toplevel)" > /dev/null || exit

echo "    > [mypy] Static type checking"
if ! mypy $src_folder; then
  echo "mypy failed!"
  exit 1
fi

echo "    > [pytest] Running unit tests (discrete + doctests) w/coverage"
if ! pytest --doctest-modules --cov-report=term-missing:skip-covered --cov=$src_folder $tst_folder; then
  echo "pytest failed!"
  exit 1
fi

echo "Tests successfully completed!"

popd > /dev/null || exit

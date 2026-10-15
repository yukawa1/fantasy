#!/bin/sh
# FANTASY app launcher for macOS and Linux.
#
# Runs the app with the recommended Python version and the tested package
# versions (uv.lock), in an environment of its own (~/.fantasy_agn/env-py3.12),
# so it does not depend on (or change) any other Python installation.
# The first start downloads Python and the packages; later starts are instant.
set -e
PYTHON_VERSION="3.12"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${FANTASY_ENV:-$HOME/.fantasy_agn/env-py$PYTHON_VERSION}"

if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
  PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "FANTASY uses uv (https://docs.astral.sh/uv/) to install Python $PYTHON_VERSION and its packages."
  printf "Install uv now with its official installer (https://astral.sh/uv/install.sh)? [y/N] "
  read -r answer
  case "$answer" in
    [yY]*) curl -LsSf https://astral.sh/uv/install.sh | sh
           PATH="$HOME/.local/bin:$PATH" ;;
    *) echo "Install uv (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
       exit 1 ;;
  esac
fi

cd "$HERE"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" exec uv run --locked --python "$PYTHON_VERSION" fantasy-app "$@"

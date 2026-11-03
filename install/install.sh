#!/bin/sh
# Install FANTASY on macOS or Linux and check that it works.
#
#   sh install.sh                    # the wheel next to this script (tester bundle)
#   sh install.sh path/to/fantasy_agn-X.whl
#   sh install.sh "git+https://github.com/yukawa1/fantasy@v2"
#
# FANTASY is installed with uv (https://docs.astral.sh/uv/) as a tool: Python 3.12
# and the tested package versions in an environment of its own, so nothing else on
# the computer is changed. Afterwards the commands  fantasy  and  fantasy-app  exist.
set -e
PYTHON_VERSION="3.12"
HERE="$(cd "$(dirname "$0")" && pwd)"

SOURCE="$1"
if [ -z "$SOURCE" ]; then
  SOURCE="$(ls "$HERE"/fantasy_agn-*.whl 2>/dev/null | sort | tail -n 1)"
fi
if [ -z "$SOURCE" ]; then
  echo "No FANTASY wheel found next to this script; give one (or a git URL) as argument."
  exit 1
fi

# uv: use it if present, otherwise offer its official installer
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
  PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "FANTASY is installed with uv (https://docs.astral.sh/uv/), which also provides Python $PYTHON_VERSION."
  printf "Install uv now with its official installer (https://astral.sh/uv/install.sh)? [y/N] "
  read -r answer
  case "$answer" in
    [yY]*) curl -LsSf https://astral.sh/uv/install.sh | sh
           PATH="$HOME/.local/bin:$PATH" ;;
    *) echo "Install uv (https://docs.astral.sh/uv/getting-started/installation/) and run this again."
       exit 1 ;;
  esac
fi

CONSTRAINTS=""
if [ -f "$HERE/constraints.txt" ]; then
  CONSTRAINTS="--constraints $HERE/constraints.txt"      # the package versions FANTASY was tested with
fi

echo "Installing FANTASY from $(basename "$SOURCE") (Python $PYTHON_VERSION) ..."
# shellcheck disable=SC2086
uv tool install --force --reinstall-package fantasy_agn --python "$PYTHON_VERSION" $CONSTRAINTS "$SOURCE"   # never a cached older build
BIN="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
PATH="$BIN:$PATH"
[ -n "$FANTASY_NO_PATH" ] || uv tool update-shell >/dev/null 2>&1 || true   # tool folder on the PATH of new terminals

echo
echo "Checking the installation (downloads the stellar templates, 13 MB, once) ..."
if fantasy test --full; then
  echo
  echo "Installed. In a new terminal:  fantasy app   (or  fantasy --help)"
  printf "Start the FANTASY app now? [Y/n] "
  read -r answer
  case "$answer" in
    [nN]*) ;;
    *) exec fantasy app ;;
  esac
else
  echo
  echo "Some checks failed. Please send the output above (or of  fantasy test --full --verbose)"
  echo "to the FANTASY authors."
  exit 1
fi

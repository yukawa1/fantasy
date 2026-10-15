#!/bin/sh
# Double-click in the macOS Finder to start the FANTASY app.
exec "$(cd "$(dirname "$0")" && pwd)/fantasy-app.sh" "$@"

#!/bin/sh
# macOS: double-click to install FANTASY (opens a Terminal window).
cd "$(dirname "$0")" && sh ./install.sh
echo
printf "Press Return to close this window."
read -r _

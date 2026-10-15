#!/bin/sh
# Build the tester bundle dist/FANTASY-<version>.zip: the wheel, the tested package
# versions (from uv.lock) and the installers. Run from anywhere:  sh install/make_bundle.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -n 1)"
NAME="FANTASY-$VERSION"
OUT="dist/$NAME"

rm -rf "$OUT" "dist/$NAME.zip" dist/fantasy_agn-*.whl
uv build --wheel --out-dir dist -q
mkdir -p "$OUT"
cp dist/fantasy_agn-"$VERSION"-*.whl "$OUT/"
# exact versions of the dependencies FANTASY is tested with (not FANTASY itself)
uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt \
  | grep -v '^#' | grep -v '^ *-e ' | sed '/^$/d' > "$OUT/constraints.txt"
cp install/install.sh install/install.bat "install/Install FANTASY.command" "$OUT/"
cp install/README-INSTALL.txt "$OUT/README.txt"
chmod +x "$OUT/install.sh" "$OUT/Install FANTASY.command"
(cd dist && zip -qr "$NAME.zip" "$NAME")
echo "dist/$NAME.zip:"
unzip -l "dist/$NAME.zip" | sed -n '4,$p' | sed '$d' | sed '$d'

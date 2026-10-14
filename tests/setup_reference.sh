#!/bin/sh
# Fetches Fish's PyTorch reference (fish-speech at the commit the parity tests
# were written against) into tests/.ref/fish-speech.
set -eu
COMMIT=1ff3c3ed2f80d3106a83909dd25b83129c3d816f
DEST="$(cd "$(dirname "$0")" && pwd)/.ref/fish-speech"
if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" ]; then
  echo "fish-speech $COMMIT already in $DEST"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin https://github.com/fishaudio/fish-speech.git
git -C "$DEST" fetch -q --depth 1 origin "$COMMIT"
git -C "$DEST" checkout -q FETCH_HEAD
echo "fish-speech $COMMIT in $DEST"

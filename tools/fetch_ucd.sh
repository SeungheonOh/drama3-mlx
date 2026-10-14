#!/bin/sh
# Downloads the Unicode Character Database files tools/gen_unicode.c reads.
set -eu
dest="${1:-tools/ucd}"
for version in 16.0.0 9.0.0; do
  mkdir -p "$dest/$version"
  for file in UnicodeData.txt PropList.txt DerivedNormalizationProps.txt; do
    if [ ! -s "$dest/$version/$file" ]; then
      curl -sSfL "https://www.unicode.org/Public/$version/ucd/$file" -o "$dest/$version/$file"
    fi
  done
done
echo "UCD files in $dest"

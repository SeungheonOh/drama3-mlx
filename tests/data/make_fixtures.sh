#!/bin/sh
# Generates the reference-audio fixtures with macOS `say` (the voices may not be
# redistributed, so the WAVs are made locally instead of being committed).
# Formats are varied on purpose: rates, bit depths, mono / true stereo.
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-../../.venv/bin/python}"
LONG="$(cat long.txt)"
SHORT="$(cat short.txt)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

say -v Samantha -o "$tmp/s_long.aiff" "$LONG"
say -v Samantha -o "$tmp/s_short.aiff" "$SHORT"
say -v Daniel -o "$tmp/d_long.aiff" "$LONG"
say -v Daniel -o "$tmp/d_short.aiff" "$SHORT"

afconvert -f WAVE -d LEI16@24000 "$tmp/s_short.aiff" ref_samantha.wav
afconvert -f WAVE -d LEI16@44100 "$tmp/s_long.aiff" samantha_long_44k_i16.wav
afconvert -f WAVE -d LEI16@16000 "$tmp/d_long.aiff" daniel_long_16k_i16.wav
afconvert -f WAVE -d LEI24@22050 "$tmp/d_short.aiff" daniel_short_22k_i24.wav
afconvert -f WAVE -c 2 -d LEF32@48000 "$tmp/s_long.aiff" "$tmp/stereo.wav"

# true stereo: the right channel is the left one delayed 5 ms and scaled by 0.6
"$PYTHON" - "$tmp/stereo.wav" samantha_long_48k_stereo_f32.wav <<'PY'
import sys
import numpy as np
import soundfile as sf
audio, sr = sf.read(sys.argv[1], dtype="float32")
left = audio[:, 0]
right = np.concatenate([np.zeros(240, np.float32), 0.6 * left[:-240]])
sf.write(sys.argv[2], np.stack([left, right], 1), sr, subtype="FLOAT")
PY
echo "fixtures written to $(pwd)"

"""End-to-end CLI runs (each loads the model in its own process; this module never
loads one in-process)."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

import d3
from voice import median_f0

DATA = Path(__file__).parent / "data"
BINARY = d3.ROOT / "build" / "drama3-tts"
SAMANTHA = DATA / "samantha_long_44k_i16.wav"
LONG = (DATA / "long.txt").read_text().strip()
TEXT = "Get the trust fund to the bank early, before the storm arrives tonight."


def reference_f0(path):
    audio, sr = sf.read(path, dtype="float32")
    return median_f0(audio.mean(axis=1) if audio.ndim > 1 else audio, sr)


def run_cli(*args):
    result = subprocess.run([str(BINARY), *map(str, args)], capture_output=True, text=True,
                            cwd=d3.ROOT)
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_cli_ref_audio_and_ref_codes_agree(tmp_path):
    """The CLI clones from a 48 kHz stereo float WAV; feeding the same reference as
    precomputed codes (from `drama3-tts encode`) must give the identical output."""
    stereo = DATA / "samantha_long_48k_stereo_f32.wav"
    ref_codes = tmp_path / "ref.npy"
    run_cli("encode", "--input", stereo, "--output", ref_codes)
    common = ["--text", TEXT, "--seed", "7", "--temperature", "1.0", "--top-p", "0.9",
              "--model", d3.MODEL_DIR]
    out_a, codes_a = tmp_path / "a.wav", tmp_path / "a.npy"
    out_b, codes_b = tmp_path / "b.wav", tmp_path / "b.npy"
    run_cli("tts", *common, "--ref-audio", stereo, "--ref-text", LONG, "--output", out_a,
            "--save-codes", codes_a)
    run_cli("tts", *common, "--ref-codes", ref_codes, "--ref-text", LONG, "--output", out_b,
            "--save-codes", codes_b)
    a, b = np.load(codes_a), np.load(codes_b)
    assert a.shape[0] == 10 and a.shape[1] > 40
    assert np.array_equal(a, b)
    audio, sr = sf.read(out_a, dtype="float32")
    assert sr == 44100 and len(audio) == a.shape[1] * 2048
    target = reference_f0(stereo)
    assert abs(median_f0(audio, sr) - target) <= 0.15 * target


def test_cli_missing_model_is_a_clean_error(tmp_path):
    result = subprocess.run([str(BINARY), "decode", "--input", str(tmp_path / "x.npy"),
                             "--output", str(tmp_path / "x.wav"), "--model",
                             str(tmp_path / "nowhere")], capture_output=True, text=True)
    assert result.returncode == 1, (result.returncode, result.stderr)
    assert "cannot open" in result.stderr and "mlx error" not in result.stderr


def test_cli_rejects_mismatched_reference_args(tmp_path):
    result = subprocess.run(
        [str(BINARY), "tts", "--text", "hi", "--output", str(tmp_path / "x.wav"),
         "--ref-audio", str(SAMANTHA)],
        capture_output=True, text=True, cwd=d3.ROOT)
    assert result.returncode != 0
    assert "--ref-text" in result.stderr

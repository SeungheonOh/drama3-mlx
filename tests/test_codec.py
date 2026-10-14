"""Codec decode and resampling against fish-speech's DAC and torchaudio."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

import d3
import ref
from fish_speech.models.text2semantic.inference import load_codec_model

torch.set_grad_enabled(False)
DATA = Path(__file__).parent / "data"


def sample_codes(frames=120, seed=0):
    """Real codes: the encoded reference clip, tiled to the requested length."""
    codec = d3.Codec("float32")
    codes = d3.encode_reference(codec, DATA / "samantha_long_44k_i16.wav")
    return np.ascontiguousarray(np.tile(codes, (1, 1 + frames // codes.shape[1]))[:, :frames])


@pytest.fixture(scope="module")
def codes():
    return sample_codes()


@pytest.mark.parametrize("dtype,bound", [("float32", 1e-4), ("bfloat16", 5e-2)])
def test_decode_matches_fish(codes, dtype, bound):
    fish = load_codec_model(str(d3.CHECKPOINT_DIR / "codec.pth"), "cpu",
                            getattr(torch, dtype))
    theirs = fish.from_indices(torch.from_numpy(codes.astype(np.int64))[None])[0, 0]
    ours = d3.Codec(dtype).decode(codes)
    assert len(ours) == codes.shape[1] * 2048 == len(theirs)
    err = ref.rel_err(ours, theirs.float().numpy())
    print(f"{dtype} decode rel err {err:.2e}")
    assert err < bound


@pytest.mark.parametrize("orig", [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000])
def test_resample_matches_torchaudio(orig):
    rng = np.random.default_rng(orig)
    audio = (rng.standard_normal(orig // 3) * 0.1).astype(np.float32)
    theirs = torchaudio.functional.resample(torch.from_numpy(audio)[None], orig, 44100)[0]
    ours = d3.resample(audio, orig, 44100)
    assert len(ours) == len(theirs)
    assert ref.rel_err(ours, theirs.numpy()) < 1e-5


def test_codes_roundtrip_file(tmp_path, codes):
    """drama3-tts decode writes PCM16 like fish-speech's sf.write."""
    import subprocess

    path = tmp_path / "codes.npy"
    np.save(path, codes.astype(np.int64))
    out = tmp_path / "out.wav"
    result = subprocess.run([str(d3.ROOT / "build" / "drama3-tts"), "decode", "--input",
                             str(path), "--output", str(out), "--codec-dtype", "float32",
                             "--model", str(d3.MODEL_DIR)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    info = sf.info(out)
    assert info.samplerate == 44100 and info.subtype == "PCM_16" and info.channels == 1
    audio, _ = sf.read(out, dtype="float32")
    expected = d3.Codec("float32").decode(codes)
    assert np.abs(audio - expected).max() < 2 / 32767

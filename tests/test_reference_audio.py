"""Reference audio -> codes must match fish-speech's encode_audio."""

from pathlib import Path

import numpy as np
import pytest
import torch

import d3
import ref  # noqa: F401  (puts fish-speech on sys.path)
from fish_speech.models.text2semantic.inference import encode_audio, load_codec_model

torch.set_grad_enabled(False)
DATA = Path(__file__).parent / "data"
FIXTURES = sorted(DATA.glob("*.wav"))


@pytest.fixture(scope="module")
def fish_codec():
    return load_codec_model(str(d3.CHECKPOINT_DIR / "codec.pth"), "cpu", torch.float32)


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_encode_matches_fish(fish_codec, codec, path):
    theirs = encode_audio(path, fish_codec, "cpu").numpy()
    ours = d3.encode_reference(codec, path)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs), (ours != theirs).mean(axis=1)


def test_unsupported_input_rejected(codec, tmp_path):
    bad = tmp_path / "not_a.wav"
    bad.write_bytes(b"ID3\x03\x00 this is an mp3 header, not RIFF")
    with pytest.raises(ValueError, match="RIFF"):
        d3.encode_reference(codec, bad)
    with pytest.raises(ValueError, match="cannot open"):
        d3.encode_reference(codec, tmp_path / "missing.wav")

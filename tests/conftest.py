"""Shared fixtures. The converted model is 39 GB: each test module holds at most one
instance and frees it when the module finishes, so tests that run the CLI (another
39 GB process) never overlap with an in-process model."""

import ctypes
from pathlib import Path

import pytest

import d3

HERE = Path(__file__).parent
FIXTURES = ["ref_samantha.wav", "samantha_long_44k_i16.wav", "samantha_long_48k_stereo_f32.wav",
            "daniel_long_16k_i16.wav", "daniel_short_22k_i24.wav"]


def pytest_sessionstart(session):
    missing = []
    if not (HERE / ".ref" / "fish-speech" / "fish_speech").is_dir():
        missing.append("Fish's reference: run tests/setup_reference.sh")
    if not all((HERE / "data" / f).exists() for f in FIXTURES):
        missing.append("audio fixtures: run tests/data/make_fixtures.sh")
    if not d3.LIB_PATH.exists():
        missing.append(f"{d3.LIB_PATH}: build with cmake (see README)")
    if not (d3.MODEL_DIR / "config.json").exists():
        missing.append(f"{d3.MODEL_DIR}: run tools/convert.py (see README)")
    if missing:
        pytest.exit("missing test prerequisites:\n  " + "\n  ".join(missing), returncode=2)


@pytest.fixture(scope="module")
def model():
    m = d3.Model()
    yield m
    d3.lib().d3_model_free.argtypes = [ctypes.c_void_p]
    d3.lib().d3_model_free(m.handle)


@pytest.fixture(scope="module")
def codec():
    c = d3.Codec("float32")
    yield c
    d3.lib().d3_codec_free.argtypes = [ctypes.c_void_p]
    d3.lib().d3_codec_free(c.handle)

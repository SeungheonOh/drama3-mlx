"""Voice cloning behaviour: generations follow the reference voice.

Without a speaker-embedding model offline, pitch is the check: a female reference
(Samantha, ~180 Hz) and a male one (Daniel, ~115 Hz) must yield clones near their
reference's median F0, dialogues must switch voice with the speaker tag, and an
edited take must keep the voice of the take it continues.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import d3
from voice import f0_track, median_f0

DATA = Path(__file__).parent / "data"
SAMANTHA = DATA / "samantha_long_44k_i16.wav"
DANIEL = DATA / "daniel_long_16k_i16.wav"
LONG = (DATA / "long.txt").read_text().strip()
SHORT = (DATA / "short.txt").read_text().strip()
TEXT = "Get the trust fund to the bank early, before the storm arrives tonight."
SAMPLING = dict(temperature=1.0, top_k=30, top_p=0.9)


def reference_f0(path):
    audio, sr = sf.read(path, dtype="float32")
    return median_f0(audio.mean(axis=1) if audio.ndim > 1 else audio, sr)


@pytest.mark.parametrize("wav", [SAMANTHA, DANIEL], ids=["samantha", "daniel"])
def test_clone_follows_reference_pitch(model, codec, wav):
    target = reference_f0(wav)
    rows = model.build_prompt(TEXT, refs=[(LONG, d3.encode_reference(codec, wav))])
    for seed in range(3):
        codes, eos = d3.generate(model, rows, 600, seed=seed, **SAMPLING)
        assert eos, seed
        assert 60 <= codes.shape[1] <= 140, codes.shape  # ~3-6 s for this sentence
        f0 = median_f0(codec.decode(codes), 44100)
        print(f"{wav.name} seed {seed}: reference {target:.0f} Hz, clone {f0:.0f} Hz")
        assert abs(f0 - target) <= 0.15 * target, (seed, f0, target)


def test_dialogue_switches_voice(model, codec):
    refs = [("<|speaker:0|> " + LONG, d3.encode_reference(codec, SAMANTHA)),
            ("<|speaker:1|> " + SHORT, d3.encode_reference(codec, DATA / "daniel_short_22k_i24.wav"))]
    text = ("<|speaker:0|> Did you remember to lock the front door before we left the house? "
            "<|speaker:1|> Of course I did, I checked it twice on the way out.")
    rows = model.build_prompt(text, refs=refs)
    gaps = []
    for seed in range(4):
        codes, eos = d3.generate(model, rows, 800, seed=seed, **SAMPLING)
        assert eos, seed
        f0 = f0_track(codec.decode(codes), 44100)
        third = len(f0) // 3
        gaps.append(np.median(f0[:third]) - np.median(f0[-third:]))
        print(f"seed {seed}: {codes.shape[1]} frames, first third - last third = {gaps[-1]:.0f} Hz")
    assert np.median(gaps) > 40, gaps
    assert min(gaps) > 0, gaps


def test_previous_take_keeps_voice(model, codec):
    ref_codes = d3.encode_reference(codec, DANIEL)
    take1_text = "The quarterly report is due on Wednesday, so please finish it by noon."
    take2_text = "The quarterly report is due on Thursday, so please finish it by noon."
    take1, eos1 = d3.generate(model, model.build_prompt(take1_text, refs=[(LONG, ref_codes)]),
                              600, seed=0, **SAMPLING)
    rows = model.build_prompt(take2_text, refs=[(LONG, ref_codes)], history=[(take1_text, take1)])
    take2, eos2 = d3.generate(model, rows, 600, seed=0, **SAMPLING)
    assert eos1 and eos2
    f1 = median_f0(codec.decode(take1), 44100)
    f2 = median_f0(codec.decode(take2), 44100)
    print(f"take 1 {take1.shape[1]} frames {f1:.0f} Hz, edited take {take2.shape[1]} frames "
          f"{f2:.0f} Hz")
    assert abs(f2 - f1) <= 0.15 * f1
    assert abs(take2.shape[1] - take1.shape[1]) <= 0.3 * take1.shape[1]

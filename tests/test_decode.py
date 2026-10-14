"""Decoding: incremental backbone steps, the tower decode step and the sampler."""

import ctypes

import numpy as np
import pytest
import torch

import d3
import ref
from fish_speech.models.text2semantic.fish_omni.tower import sample_top_k_top_p

torch.set_grad_enabled(False)
TEXT = "Hello! This is a quick test of Fish Audio Drama three, running natively on Apple silicon."


def test_incremental_layers_match_prefill(model):
    """Every layer: one row at a time after a prefill equals a one-shot pass
    (relative to the layer's update, which the residual stream would hide)."""
    rows = model.build_prompt(TEXT)
    R, split = len(rows), 30
    x = model.embed_frames(rows)
    for i in range(40):
        model.reset(R + 8)
        full = model.layer(i, x, 0)
        model.reset(R + 8)
        inc = np.concatenate([model.layer(i, x[:split], 0)]
                             + [model.layer(i, x[p:p + 1], p) for p in range(split, R)])
        err = ref.rel_err(inc[split:] - x[split:], full[split:] - x[split:])
        assert err < 3e-2, (i, err)
        x = full


def tower_decode(model, logits, hidden, pos, frame, history, temperature, top_k, top_p, seed):
    L = d3._setup_model_api()
    L.d3_tower_decode.argtypes = [ctypes.c_void_p, d3._f32p, d3._f32p, ctypes.c_int, d3._i32p,
                                  d3._i32p, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                  ctypes.c_float, ctypes.c_ulonglong, d3._i32p]
    out = np.zeros(11, np.int32)
    history = np.ascontiguousarray(history, np.int32)
    L.d3_tower_decode(model.handle, d3._fptr(np.ascontiguousarray(logits, np.float32)),
                      d3._fptr(np.ascontiguousarray(hidden, np.float32)), pos,
                      d3._iptr(np.ascontiguousarray(frame, np.int32)), d3._iptr(history),
                      len(history), temperature, top_k, top_p, seed, d3._iptr(out))
    return out


def test_decode_step_matches_fish_greedy(model):
    """With top_k = 1 the first decode step is deterministic: same frame as Fish."""
    cfg = ref.config()
    rows = model.build_prompt(TEXT)
    R = len(rows)
    hidden = d3.prefill(model, rows)
    logits = model.audio_logits(hidden[-1:])
    tower = ref.audio_decoder(torch.float32)
    tower.setup_caches(R + 8)
    frames = torch.from_numpy(rows.astype(np.int64))
    positions = torch.arange(R + 8)
    h = torch.from_numpy(hidden)
    tower.prefill(h[:R - 1], positions[:R - 1], frames[:R - 1], frames[1:R])
    candidates = torch.cat((torch.tensor([cfg.eos_token_id]),
                            torch.arange(cfg.semantic_start_token_id,
                                         cfg.semantic_end_token_id + 1)))
    full = torch.full((1, cfg.text.vocab_size), -1e30)
    full[0, candidates] = torch.from_numpy(logits[0])
    theirs = tower.decode_step(full, h[-1:], positions[R - 1:R], frames[R - 1:R],
                               frames[:R, 0], temperature=0.7, top_k=1, top_p=0.7)[0].numpy()
    ours = tower_decode(model, logits, hidden[-1:], R - 1, rows[R - 1], rows[-16:, 0], 0.7, 1,
                        0.7, 0)
    assert ours.tolist() == theirs.tolist()


@pytest.mark.parametrize("temperature,top_k,top_p",
                         [(0.7, 30, 0.7), (1.0, 30, 0.9), (1.0, 5, 1.0), (0.3, 30, 0.95)])
def test_sampler_distribution(temperature, top_k, top_p):
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((1, 4097)) * 3).astype(np.float32)
    n = 4000
    ours = np.array([d3.sample(logits, temperature, top_k, top_p, s)[0] for s in range(n)])
    torch.manual_seed(0)
    theirs = np.array([int(sample_top_k_top_p(torch.from_numpy(logits), temperature, top_k,
                                              top_p)[0]) for _ in range(n)])
    assert set(np.unique(ours)) <= set(np.unique(theirs)) | set(np.argsort(-logits[0])[:top_k])
    # the exact kept set and probabilities, from the definition
    values = np.sort(logits[0] / temperature)[::-1].astype(np.float64)
    order = np.argsort(-logits[0] / temperature, kind="stable")
    p = np.exp(values - values.max())
    p /= p.sum()
    keep = ~((np.cumsum(p) > top_p) | (np.arange(len(p)) >= top_k))
    keep[0] = True
    probs = np.where(keep, p, 0)
    probs /= probs.sum()
    expected = dict(zip(order[keep], probs[keep]))
    for token, prob in expected.items():
        freq_ours = np.mean(ours == token)
        tol = 4 * np.sqrt(prob * (1 - prob) / n) + 1e-3
        assert abs(freq_ours - prob) < tol, (token, freq_ours, prob)
    assert set(np.unique(ours)) <= set(expected)

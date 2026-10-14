"""Long prompts (voice references, > 128 rows) through the production chunked prefill,
against Fish's reference run over the whole sequence in fp32.

The C runtime computes in bf16, so its error is judged against a bf16 noise floor:
the same C code run over the prompt in one shot, without 128-row chunks.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

import d3
import ref

torch.set_grad_enabled(False)
DATA = Path(__file__).parent / "data"
CACHE = Path(__file__).parent / ".cache"
TEXT = "<|speaker:1|> The quarterly report is due on Thursday, so please finish it by noon. " \
       "<|speaker:0|> Understood, I'll send it over this afternoon."
REFS = [
    ("samantha_long_44k_i16.wav", "long.txt", "<|speaker:0|> "),
    ("daniel_long_16k_i16.wav", "long.txt", "<|speaker:1|> "),
]
CHUNK = 128


def reference_pass(rows: np.ndarray):
    """Fish's backbone over all rows in fp32, layer by layer: (inputs, outputs, final)."""
    key = hashlib.sha256(rows.tobytes()).hexdigest()[:16]
    path = CACHE / f"ref_backbone_{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["inputs"], data["outputs"], data["final"]
    cfg = ref.config()
    text = cfg.text
    tower = ref.audio_decoder(torch.float32)
    frames = torch.from_numpy(rows.astype(np.int64))
    emb = ref.embed_tokens(torch.float32)
    x = tower.embed_backbone(frames, emb[frames[:, 0]])[None]
    del emb, tower
    ctx = ref.layer_context(torch.arange(len(rows)), text.rotary_dim, text.rope_theta)
    inputs, outputs = [], []
    for i in range(text.num_hidden_layers):
        layer = ref.decoder_layer(i, torch.float32)
        y = layer(x, ctx)
        inputs.append(x[0].numpy().astype(np.float32))
        outputs.append(y[0].numpy().astype(np.float32))
        x = y
        del layer
        print(f"  reference layer {i} done", flush=True)
    final = ref.final_norm(torch.float32)(x)[0].numpy()
    CACHE.mkdir(exist_ok=True)
    np.savez(path, inputs=np.stack(inputs), outputs=np.stack(outputs), final=final)
    return np.stack(inputs), np.stack(outputs), final


def fish_bf16_chain(rows: np.ndarray) -> np.ndarray:
    """Fish's backbone run end to end in bf16 (its production dtype): final hidden."""
    key = hashlib.sha256(rows.tobytes()).hexdigest()[:16]
    path = CACHE / f"fish_bf16_final_{key}.npy"
    if path.exists():
        return np.load(path)
    cfg = ref.config()
    text = cfg.text
    tower = ref.audio_decoder(torch.bfloat16)
    frames = torch.from_numpy(rows.astype(np.int64))
    emb = ref.embed_tokens(torch.bfloat16)
    x = tower.embed_backbone(frames, emb[frames[:, 0]])[None]
    del emb, tower
    ctx = ref.layer_context(torch.arange(len(rows)), text.rotary_dim, text.rope_theta)
    for i in range(text.num_hidden_layers):
        layer = ref.decoder_layer(i, torch.bfloat16)
        x = layer(x, ctx)
        del layer
        print(f"  fish bf16 layer {i} done", flush=True)
    final = ref.final_norm(torch.bfloat16)(x)[0].float().numpy()
    CACHE.mkdir(exist_ok=True)
    np.save(path, final)
    return final


def row_errors(ours, theirs):
    return np.array([ref.rel_err(a, b) for a, b in zip(ours, theirs)])


@pytest.fixture(scope="module")
def setup(model, codec):
    refs = []
    for wav, txt, tag in REFS:
        refs.append((tag + (DATA / txt).read_text().strip(), d3.encode_reference(codec, DATA / wav)))
    rows = model.build_prompt(TEXT, refs=refs)
    inputs, outputs, final = reference_pass(rows)
    return model, rows, inputs, outputs, final


def test_prompt_is_long(setup):
    _, rows, *_ = setup
    assert len(rows) > 4 * CHUNK, len(rows)


def calibration(inputs: np.ndarray, key: str):
    """Per layer, on the bf16-rounded reference input: Fish's fp32 output and the
    relative error of Fish's own bf16 run against it (the bf16 noise floor)."""
    path = CACHE / f"calibration_{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["r32"], data["fish_err"]
    cfg = ref.config().text
    R = inputs.shape[1]
    ctx = ref.layer_context(torch.arange(R), cfg.rotary_dim, cfg.rope_theta)
    r32_all, fish_err = [], []
    for i in range(len(inputs)):
        xb = torch.from_numpy(inputs[i]).to(torch.bfloat16)[None]
        xr = xb[0].float().numpy()
        layer = ref.decoder_layer(i, torch.float32)
        r32 = layer(xb.float(), ctx)[0].numpy()
        del layer
        layer = ref.decoder_layer(i, torch.bfloat16)
        f16 = layer(xb, ctx)[0].float().numpy()
        del layer
        r32_all.append(r32)
        fish_err.append(np.linalg.norm(f16 - r32) / np.linalg.norm(r32 - xr))
        print(f"  calibration layer {i} done", flush=True)
    np.savez(path, r32=np.stack(r32_all), fish_err=np.array(fish_err))
    return np.stack(r32_all), np.array(fish_err)


def test_layers_chunked_vs_reference(setup):
    """Each layer on the reference input, run in 128-row chunks with caches, judged
    against Fish's own bf16 run of the same layer on the same input."""
    model, rows, inputs, _, _ = setup
    R = len(rows)
    r32, fish_err = calibration(inputs, hashlib.sha256(rows.tobytes()).hexdigest()[:16])
    for i in range(len(inputs)):
        xr = torch.from_numpy(inputs[i]).to(torch.bfloat16).float().numpy()
        model.reset(R + 8)
        chunked = np.concatenate([model.layer(i, xr[s:s + CHUNK], s) for s in range(0, R, CHUNK)])
        model.reset(R + 8)
        single = model.layer(i, xr, 0)
        update = np.linalg.norm(r32[i] - xr)
        e_chunked = np.linalg.norm(chunked - r32[i]) / update
        e_single = np.linalg.norm(single - r32[i]) / update
        print(f"layer {i:2d}: rel err chunked {e_chunked:.2e} single {e_single:.2e} "
              f"fish-bf16 {fish_err[i]:.2e}")
        # chunked prefill adds nothing over a one-shot pass
        assert e_chunked <= 1.02 * e_single + 1e-4, (i, e_chunked, e_single)
        # and the C runtime is as accurate as Fish's own bf16 path
        assert e_chunked <= 1.15 * fish_err[i] + 1e-3, (i, e_chunked, fish_err[i])


def test_production_prefill_vs_reference(setup):
    """Final hidden states through all 40 layers, against Fish fp32; the noise floor
    is Fish's own bf16 run chained end to end."""
    model, rows, _, _, final = setup
    R = len(rows)
    chunked = d3.prefill(model, rows)
    model.reset(R + 8)
    single = model.backbone(model.embed_frames(rows), 0)
    fish = fish_bf16_chain(rows)
    e_chunked = row_errors(chunked, final)
    e_single = row_errors(single, final)
    e_fish = row_errors(fish, final)
    print(f"final hidden row rel err (median / 90th pct): C chunked "
          f"{np.median(e_chunked):.3e} / {np.percentile(e_chunked, 90):.3e}, C one-shot "
          f"{np.median(e_single):.3e} / {np.percentile(e_single, 90):.3e}, Fish bf16 "
          f"{np.median(e_fish):.3e} / {np.percentile(e_fish, 90):.3e}")
    assert np.median(e_chunked) <= 1.15 * np.median(e_fish) + 1e-3
    assert np.percentile(e_chunked, 90) <= 1.15 * np.percentile(e_fish, 90) + 1e-3


def test_audio_logits_last_rows(setup):
    """Decode logits (eos + semantic band) of the last prompt rows."""
    model, rows, _, _, final = setup
    hidden = d3.prefill(model, rows)
    fish = fish_bf16_chain(rows)
    head = ref.lm_head_rows(torch.float32)
    theirs = (torch.from_numpy(final[-8:]) @ head.T).numpy()
    fish_logits = (torch.from_numpy(fish[-8:]) @ head.T).numpy()
    ours = model.audio_logits(hidden[-8:])
    e_ours = ref.rel_err(ours, theirs)
    e_fish = ref.rel_err(fish_logits, theirs)
    overlap = np.mean([len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10]))
                       for a, b in zip(ours, theirs)])
    print(f"last-8-row logits rel err: C {e_ours:.3e}, Fish bf16 {e_fish:.3e}; "
          f"top-10 overlap {overlap:.1f}/10; argmax agreement "
          f"{np.mean(ours.argmax(-1) == theirs.argmax(-1)):.2f}")
    assert e_ours <= 1.15 * e_fish + 1e-3
    assert overlap >= 8


def reference_tower_forced(tower, hidden, rows, p):
    """Fish's tower: chunked teacher-forced prefill of rows < p, then the forced pass at p."""
    R = len(rows)
    tower.setup_caches(R + 8)
    frames = torch.from_numpy(rows.astype(np.int64))
    positions = torch.arange(R + 8)
    h = torch.from_numpy(hidden)
    for start in range(0, p, CHUNK):
        end = min(start + CHUNK, p)
        tower.prefill(h[start:end], positions[start:end], frames[start:end],
                      frames[start + 1:end + 1])
    nxt = frames[p + 1:p + 2]
    semantic = tower.is_semantic(nxt[:, 0])
    x = tower.run_shared(h[p:p + 1], nxt[:, 1], semantic, positions[p:p + 1], frames[p:p + 1])
    logits = []
    for j, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
        x = layer(x, nxt[:, j + 1], semantic, positions[p:p + 1])
        logits.append(head(x)[0])
    return torch.stack(logits).numpy()


def test_tower_long_prompt(setup):
    """Tower teacher-forced through a chunked prefill, on reference hidden states."""
    model, rows, _, _, final = setup
    R = len(rows)
    tower = ref.audio_decoder(torch.float32)
    semantic = np.nonzero(rows[:, 0] >= ref.config().semantic_start_token_id)[0]
    probes = sorted({int(semantic[-1]) - 1, CHUNK * 2, CHUNK * 3 + 5, R - 2 - 0})
    for p in probes:
        if rows[p + 1, 0] < ref.config().semantic_start_token_id and p != R - 2:
            continue
        model.reset(R + 8)
        for start in range(0, p, CHUNK):
            end = min(start + CHUNK, p)
            model.tower_prefill(final[start:end], start, rows[start:end],
                                rows[start + 1:end + 1])
        ours = model.tower_forced(final[p], p, rows[p:p + 1], rows[p + 1:p + 2])
        theirs = reference_tower_forced(tower, final, rows, p)
        errs = [ref.rel_err(a, b) for a, b in zip(ours, theirs)]
        agree = (ours.argmax(-1) == theirs.argmax(-1)).mean()
        print(f"row {p}: per-codebook rel err max {max(errs):.3e}, argmax agreement {agree:.2f}")
        assert max(errs) < 2e-2, (p, errs)
        for j in np.nonzero(ours.argmax(-1) != theirs.argmax(-1))[0]:
            # a flip is only acceptable between near-tied reference logits
            a, b = ours[j].argmax(), theirs[j].argmax()
            gap = theirs[j][b] - theirs[j][a]
            noise = np.abs(ours[j] - theirs[j]).max()
            print(f"  codebook {j + 1}: {a} vs {b}, reference gap {gap:.4f}, max abs diff "
                  f"{noise:.4f}")
            assert gap <= 2 * noise, (p, j, gap, noise)

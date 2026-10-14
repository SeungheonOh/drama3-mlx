"""Fish's PyTorch reference (fish-speech 1ff3c3e) wired to the converted weights.

The reference gets the same expert values the C runtime uses (the MLX 8-bit
weights, dequantized), so comparisons measure implementation error only.
Every other tensor comes straight from the original checkpoint.
"""

import json
import sys
from functools import lru_cache

import mlx.core as mx
import numpy as np
import torch

import d3

REF_DIR = d3.ROOT / "tests" / ".ref" / "fish-speech"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.layers import (  # noqa: E402
    LayerContext,
    ZeroCenteredRMSNorm,
)
from fish_speech.models.text2semantic.fish_omni.tower import (  # noqa: E402
    FishOmniAudioDecoder,
    load_audio_decoder_weights,
)

SRC = "model.language_model."


@lru_cache(maxsize=1)
def config() -> FishOmniConfig:
    return FishOmniConfig.from_dict(json.loads((d3.CHECKPOINT_DIR / "config.json").read_text()))


@lru_cache(maxsize=1)
def _weight_maps():
    orig = json.loads((d3.CHECKPOINT_DIR / "model.safetensors.index.json").read_text())
    conv = json.loads((d3.MODEL_DIR / "model.safetensors.index.json").read_text())
    return orig["weight_map"], conv["weight_map"]


_shards: dict[str, dict] = {}


def _load(path) -> dict:
    key = str(path)
    if key not in _shards:
        if len(_shards) > 3:
            _shards.clear()
        _shards[key] = mx.load(key)
    return _shards[key]


def orig_tensor(name: str) -> torch.Tensor:
    orig, _ = _weight_maps()
    value = _load(d3.CHECKPOINT_DIR / orig[name])[name]
    return torch.from_numpy(np.array(value.astype(mx.float32)))


def converted_tensor(name: str) -> mx.array:
    _, conv = _weight_maps()
    return _load(d3.MODEL_DIR / conv[name])[name]


def dequantized_experts(layer: int, proj: str) -> torch.Tensor:
    base = f"layers.{layer}.mlp.experts.{proj}"
    w = mx.dequantize(
        converted_tensor(base + ".weight"),
        converted_tensor(base + ".scales"),
        converted_tensor(base + ".biases"),
        group_size=64,
        bits=8,
    )
    return torch.from_numpy(np.array(w.astype(mx.float32)))


def decoder_layer(index: int, dtype: torch.dtype) -> DecoderLayer:
    """Fish's DecoderLayer with the checkpoint weights (experts as the C runtime has them)."""
    cfg = config()
    layer = DecoderLayer(cfg.text, index, fp8=False, dtype=dtype)
    prefix = f"{SRC}layers.{index}."
    orig, _ = _weight_maps()
    with torch.no_grad():
        params = dict(layer.named_parameters())
        for name in orig:
            if not name.startswith(prefix) or ".mlp.experts." in name:
                continue
            local = name[len(prefix):]
            params[local].copy_(orig_tensor(name).to(dtype))
        layer.mlp.experts.w13_weight.copy_(dequantized_experts(index, "gate_up_proj").to(dtype))
        layer.mlp.experts.w2_weight.copy_(dequantized_experts(index, "down_proj").to(dtype))
    return layer.eval()


def layer_context(positions: torch.Tensor, rotary_dim: int, theta: float) -> LayerContext:
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (theta ** (exponents / rotary_dim))
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return LayerContext(
        input_pos=positions,
        cos=angles.cos(),
        sin=angles.sin(),
        kv_len=int(positions[-1]) + 1,
        continue_state=False,
        use_cache=False,
    )


def final_norm(dtype: torch.dtype) -> ZeroCenteredRMSNorm:
    cfg = config().text
    norm = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(orig_tensor(f"{SRC}norm.weight").to(dtype))
    return norm.eval()


def embed_tokens(dtype: torch.dtype) -> torch.Tensor:
    return orig_tensor(f"{SRC}embed_tokens.weight").to(dtype)


def lm_head_rows(dtype: torch.dtype) -> torch.Tensor:
    """lm_head rows of the decode candidates: eos then the semantic band."""
    cfg = config()
    full = orig_tensor("lm_head.weight")
    rows = torch.cat(
        (full[cfg.eos_token_id: cfg.eos_token_id + 1],
         full[cfg.semantic_start_token_id: cfg.semantic_end_token_id + 1])
    )
    return rows.to(dtype)


def audio_decoder(dtype: torch.dtype) -> FishOmniAudioDecoder:
    cfg = config()
    tower = FishOmniAudioDecoder(cfg, dtype=dtype)
    orig, _ = _weight_maps()
    names = [n for n in orig if n.startswith("audio_decoder")]
    load_audio_decoder_weights(tower, ((n, orig_tensor(n).to(dtype)) for n in names))
    return tower.eval()


def rel_err(ours: np.ndarray, ref: np.ndarray) -> float:
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(ours - ref) / max(np.linalg.norm(ref), 1e-30))

"""Convert the fishaudio/drama-3 checkpoint into the layout the C runtime loads.

Backbone experts are block-FP8 in the checkpoint; they are dequantized exactly as
Fish's reference does on non-CUDA devices (fp32 block product, then bf16) and
re-quantized to MLX 8-bit affine (group 64). Every other model weight stays bf16,
except zero-centered RMSNorm weights, which are stored as fp32 (1 + w).

The lm_head keeps only the rows the decoder ever samples: eos and the semantic
band, in ascending id order.

The codec (S2-Pro modded DAC) is folded out of weight norm in fp32 and written
with MLX convolution layouts.
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

FP8_BLOCK = 128
EXPERT_BITS = 8
EXPERT_GROUP = 64
SHARD_BYTES = 4 << 30

ZERO_CENTERED_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "enorm.weight",
    "hnorm.weight",
    "final_layernorm.weight",
    "attention_norm.weight",
    "ffn_norm.weight",
)


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = out_dir
        self.prefix = prefix
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.files: list[tuple[str, list[str]]] = []

    def add(self, name: str, value: mx.array) -> None:
        if name in self.pending or any(name in keys for _, keys in self.files):
            raise ValueError(f"duplicate output tensor {name}")
        mx.eval(value)
        self.pending[name] = value
        self.pending_bytes += value.nbytes
        if self.pending_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        name = f"{self.prefix}-{len(self.files) + 1:05d}.safetensors"
        mx.save_safetensors(str(self.out_dir / name), self.pending)
        self.files.append((name, list(self.pending)))
        print(f"  wrote {name}: {len(self.pending)} tensors, "
              f"{self.pending_bytes / 2**30:.2f} GiB", flush=True)
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> dict[str, str]:
        self.flush()
        return {key: file for file, keys in self.files for key in keys}


def dequantize_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block-FP8 [out, in] (raw e4m3 bytes) with fp32 scales -> bf16, as Fish does."""
    rows, cols = weight.shape
    values = mx.from_fp8(weight, dtype=mx.float32)
    blocks = values.reshape(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
    scaled = blocks * scale_inv.astype(mx.float32)[:, None, :, None]
    return scaled.reshape(rows, cols).astype(mx.bfloat16)


def norm_weight(value: mx.array) -> mx.array:
    return value.astype(mx.float32) + 1.0


class Checkpoint:
    """Lazy access to checkpoint tensors by name, one open shard at a time."""

    def __init__(self, path: Path):
        self.path = path
        index = json.loads((path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.cache_file: str | None = None
        self.cache: dict[str, mx.array] = {}

    def get(self, name: str) -> mx.array:
        file = self.weight_map[name]
        if file != self.cache_file:
            self.cache = mx.load(str(self.path / file))
            self.cache_file = file
        return self.cache[name]

    def names(self) -> list[str]:
        return list(self.weight_map)


def convert_backbone(ckpt: Checkpoint, config: dict, writer: ShardWriter) -> int:
    text = config["text_config"]
    num_layers = text["num_hidden_layers"]
    num_experts = text["num_experts"]
    src = "model.language_model."
    handled = set()

    def take(name: str) -> mx.array:
        handled.add(name)
        return ckpt.get(name)

    # Group names by shard to keep shard switches rare for plain tensors.
    plain = [
        n for n in ckpt.names()
        if n.startswith(src) and ".mlp.experts." not in n
    ]
    plain.sort(key=lambda n: (ckpt.weight_map[n], n))
    for name in plain:
        out = name[len(src):]
        value = take(name)
        if out.endswith(ZERO_CENTERED_SUFFIXES) or out == "norm.weight":
            value = norm_weight(value)
        elif out.endswith("linear_attn.conv1d.weight"):
            value = value.reshape(value.shape[0], value.shape[-1])
        writer.add(out, value)

    for layer in range(num_layers):
        started = time.perf_counter()
        prefix = f"{src}layers.{layer}.mlp.experts"
        gate_up, down = [], []
        for expert in range(num_experts):
            parts = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                weight = take(f"{prefix}.{expert}.{proj}.weight")
                scale = take(f"{prefix}.{expert}.{proj}.weight_scale_inv")
                parts[proj] = dequantize_fp8(weight, scale)
            gate_up.append(mx.concatenate([parts["gate_proj"], parts["up_proj"]], 0))
            down.append(parts["down_proj"])
        for proj, stack in (("gate_up_proj", gate_up), ("down_proj", down)):
            stacked = mx.stack(stack)
            weight, scales, biases = mx.quantize(
                stacked, group_size=EXPERT_GROUP, bits=EXPERT_BITS
            )
            base = f"layers.{layer}.mlp.experts.{proj}"
            writer.add(f"{base}.weight", weight)
            writer.add(f"{base}.scales", scales)
            writer.add(f"{base}.biases", biases)
        print(f"  layer {layer}: experts quantized in "
              f"{time.perf_counter() - started:.1f}s", flush=True)

    # Audio rows of lm_head: eos then the semantic band (ascending ids).
    eos = config["eos_token_id"]
    start = config["semantic_start_token_id"]
    end = config["semantic_end_token_id"]
    assert eos < start, "candidate order assumes eos precedes the semantic band"
    lm_head = take("lm_head.weight")
    rows = mx.concatenate([lm_head[eos:eos + 1], lm_head[start:end + 1]], 0)
    writer.add("lm_head_audio.weight", rows)
    return len(handled)


TOWER_PREFIXES = {
    "audio_decoder.codebook_embeddings.": "tower.codebook_embeddings.",
    "audio_decoder_mtp_shared.layers.": "tower.shared.",
    "audio_decoder_mtp.layers.": "tower.residual.",
    "audio_decoder_mtp.heads.": "tower.heads.",
}


def convert_tower(ckpt: Checkpoint, writer: ShardWriter) -> int:
    count = 0
    names = [n for n in ckpt.names() if n.startswith("audio_decoder")]
    names.sort(key=lambda n: (ckpt.weight_map[n], n))
    for name in names:
        out = next(
            new + name[len(old):]
            for old, new in TOWER_PREFIXES.items()
            if name.startswith(old)
        )
        out = out.replace(".semantic_embedding.", ".embedding.")
        out = out.replace(".codebook_embedding.", ".embedding.")
        out = out.replace(".transformer_layer.", ".block.")
        value = ckpt.get(name)
        if out.endswith(ZERO_CENTERED_SUFFIXES):
            value = norm_weight(value)
        writer.add(out, value)
        count += 1
    return count


def convert_codec(src: Path, dst: Path) -> None:
    import torch

    state = torch.load(str(src), map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("generator.") for k in state):
        state = {
            k[len("generator."):]: v for k, v in state.items()
            if k.startswith("generator.")
        }
    out: dict[str, np.ndarray] = {}
    folded = set()
    for key in state:
        if key.endswith((".causal_mask", ".freqs_cis")):
            continue  # non-persistent buffers, rebuilt by the module
        if key.endswith(".parametrizations.weight.original0"):
            base = key[: -len(".parametrizations.weight.original0")]
            g = state[key].float()
            v = state[base + ".parametrizations.weight.original1"].float()
            out[base + ".weight"] = torch._weight_norm(v, g, 0)
            folded.add(base)
            continue
        if key.endswith(".parametrizations.weight.original1"):
            continue
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            g = state[key].float()
            v = state[base + ".weight_v"].float()
            out[base + ".weight"] = torch._weight_norm(v, g, 0)
            folded.add(base)
            continue
        if key.endswith(".weight_v"):
            continue
        out[key] = state[key].float()

    converted: dict[str, np.ndarray] = {}
    for key, value in out.items():
        value = value.detach().contiguous()
        if key.endswith(".alpha"):
            value = value.reshape(-1)  # Snake [1, C, 1] -> [C]
        elif key.endswith(".weight") and value.ndim == 3:
            if is_transposed_conv(key):
                value = value.permute(1, 2, 0)  # [in, out, k] -> [out, k, in]
            elif key.startswith("quantizer.") and (
                ".in_proj." in key or ".out_proj." in key
            ):
                value = value[..., 0]  # 1x1 conv -> linear [out, in]
            else:
                value = value.permute(0, 2, 1)  # [out, in, k] -> [out, k, in]
        converted[key] = value.contiguous().numpy()
    mx.save_safetensors(
        str(dst), {k: mx.array(v) for k, v in converted.items()}
    )
    print(f"  codec: {len(converted)} tensors ({len(folded)} weight-norm folds)")


TRANSPOSED_CONV_KEYS = (
    # DecoderBlock: block.0 Snake, block.1 the upsampling ConvTranspose1d.
    re.compile(r"decoder\.model\.\d+\.block\.1\.conv\.weight"),
    re.compile(r"quantizer\.upsample\.\d+\.0\.conv\.weight"),
)


def is_transposed_conv(key: str) -> bool:
    return any(pattern.fullmatch(key) for pattern in TRANSPOSED_CONV_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/drama-3"))
    parser.add_argument("--output", type=Path, default=Path("mlx_model"))
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-codec", action="store_true")
    args = parser.parse_args()

    config = json.loads((args.checkpoint / "config.json").read_text())
    if config.get("model_type") != "fish_omni":
        raise SystemExit(f"not a Drama 3 checkpoint: {config.get('model_type')!r}")
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "bsgemm-moe" or quant.get(
        "weight_block_size"
    ) != [FP8_BLOCK, FP8_BLOCK]:
        raise SystemExit(f"unsupported quantization_config {quant!r}")

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_model:
        ckpt = Checkpoint(args.checkpoint)
        writer = ShardWriter(args.output, "model")
        started = time.perf_counter()
        handled = convert_backbone(ckpt, config, writer)
        handled += convert_tower(ckpt, writer)
        weight_map = writer.finish()
        missing = len(ckpt.names()) - handled
        if missing:
            raise SystemExit(f"{missing} checkpoint tensors were not converted")
        (args.output / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}, indent=1)
        )
        out_config = dict(config)
        out_config["mlx_conversion"] = {
            "experts": {"bits": EXPERT_BITS, "group_size": EXPERT_GROUP,
                        "mode": "affine"},
            "norms": "zero-centered norms stored as fp32 (1 + w)",
            "lm_head_audio_rows": [config["eos_token_id"],
                                   config["semantic_start_token_id"],
                                   config["semantic_end_token_id"]],
        }
        (args.output / "config.json").write_text(json.dumps(out_config, indent=2))
        for name in ("tokenizer.json", "LICENSE"):
            shutil.copy(args.checkpoint / name, args.output / name)
        print(f"model converted in {time.perf_counter() - started:.0f}s")
    if not args.skip_codec:
        convert_codec(args.checkpoint / "codec.pth", args.output / "codec.safetensors")


if __name__ == "__main__":
    main()

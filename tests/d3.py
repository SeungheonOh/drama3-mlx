"""ctypes bindings to libdrama3 for the parity tests."""

import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = Path(os.environ.get("D3_LIB", ROOT / "build" / "libdrama3.dylib"))
MODEL_DIR = Path(os.environ.get("D3_MODEL", ROOT / "mlx_model"))
CHECKPOINT_DIR = Path(os.environ.get("D3_CHECKPOINT", ROOT / "checkpoints" / "drama-3"))

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(LIB_PATH))
        _lib.d3_tok_load.restype = ctypes.c_void_p
        _lib.d3_tok_load.argtypes = [ctypes.c_char_p]
        _lib.d3_tok_encode.restype = ctypes.c_long
        _lib.d3_tok_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        ]
        _lib.d3_tok_token_to_id.restype = ctypes.c_long
        _lib.d3_tok_token_to_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.d3_pretokenize.restype = ctypes.c_long
        _lib.d3_pretokenize.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        _lib.d3_nfc.restype = ctypes.c_long
        _lib.d3_nfc.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ]
    return _lib


class Tokenizer:
    def __init__(self, path: Path):
        self.handle = lib().d3_tok_load(str(path).encode())
        if not self.handle:
            raise RuntimeError(f"cannot load {path}")

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        cap = len(data) + 16
        out = (ctypes.c_int * cap)()
        n = lib().d3_tok_encode(self.handle, data, len(data), out, cap)
        if n < 0:
            raise ValueError("invalid UTF-8")
        return list(out[:n])

    def token_to_id(self, token: str) -> int:
        return lib().d3_tok_token_to_id(self.handle, token.encode())


def pretokenize(text: str) -> list[tuple[int, int]]:
    data = text.encode("utf-8")
    cap = 2 * len(text) + 4
    out = (ctypes.c_size_t * cap)()
    n = lib().d3_pretokenize(data, len(data), out, cap)
    return [(out[2 * i], out[2 * i + 1]) for i in range(n)]


def nfc(text: str) -> str:
    data = text.encode("utf-8")
    cap = len(data) * 4 + 16
    out = ctypes.create_string_buffer(cap)
    n = lib().d3_nfc(data, len(data), out, cap)
    return out.raw[:n].decode("utf-8")


# ---- model / codec bindings (numpy in, numpy out) ----

import numpy as np

_f32p = ctypes.POINTER(ctypes.c_float)
_i32p = ctypes.POINTER(ctypes.c_int)


def _fptr(a: np.ndarray):
    return a.ctypes.data_as(_f32p)


def _iptr(a: np.ndarray):
    return a.ctypes.data_as(_i32p)


def _setup_model_api():
    L = lib()
    if getattr(L, "_model_api", False):
        return L
    L.d3_model_load.restype = ctypes.c_void_p
    L.d3_model_load.argtypes = [ctypes.c_char_p]
    L.d3_model_load_light.restype = ctypes.c_void_p
    L.d3_model_load_light.argtypes = [ctypes.c_char_p]
    L.d3_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.d3_embed_frames.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int, _f32p, ctypes.c_size_t]
    L.d3_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, _f32p, ctypes.c_int,
                           ctypes.c_int, _f32p, ctypes.c_size_t]
    L.d3_backbone.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int, ctypes.c_int, _f32p,
                              ctypes.c_size_t]
    L.d3_audio_logits.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int, _f32p, ctypes.c_size_t]
    L.d3_tower_prefill.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int, ctypes.c_int, _i32p,
                                   _i32p]
    L.d3_tower_forced.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int, _i32p, _i32p, _f32p,
                                  ctypes.c_size_t]
    L.d3_sample.argtypes = [_f32p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                            ctypes.c_float, ctypes.c_ulonglong, _i32p]
    L.d3_build_prompt.restype = ctypes.c_long
    L.d3_build_prompt.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
        _i32p, _i32p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), _i32p, _i32p, _i32p,
        ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
    ]
    L.d3_split_text.restype = ctypes.c_long
    L.d3_split_text.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_size_t]
    L.d3_codec_load.restype = ctypes.c_void_p
    L.d3_codec_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
    L.d3_codec_decode.restype = ctypes.c_long
    L.d3_codec_decode.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int, _f32p, ctypes.c_size_t]
    L.d3_codec_encode.restype = ctypes.c_long
    L.d3_codec_encode.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_size_t, _i32p,
                                  ctypes.c_size_t]
    L.d3_resample.restype = ctypes.c_long
    L.d3_resample.argtypes = [_f32p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, _f32p,
                              ctypes.c_size_t]
    L._model_api = True
    return L


HIDDEN = 2048
WIDTH = 11
CANDIDATES = 4097
CODE_VOCAB = 4096


class Model:
    def __init__(self, path=None, light=False):
        L = _setup_model_api()
        path = str(path or MODEL_DIR).encode()
        self.handle = L.d3_model_load_light(path) if light else L.d3_model_load(path)

    def reset(self, capacity: int):
        lib().d3_reset(self.handle, capacity)

    def embed_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = np.ascontiguousarray(frames, dtype=np.int32)
        out = np.zeros((len(frames), HIDDEN), np.float32)
        lib().d3_embed_frames(self.handle, _iptr(frames), len(frames), _fptr(out), out.size)
        return out

    def layer(self, index: int, x: np.ndarray, pos: int, which: int = 0) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros_like(x)
        lib().d3_layer(self.handle, index, which, _fptr(x), len(x), pos, _fptr(out), out.size)
        return out

    def backbone(self, x: np.ndarray, pos: int) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros_like(x)
        lib().d3_backbone(self.handle, _fptr(x), len(x), pos, _fptr(out), out.size)
        return out

    def audio_logits(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        out = np.zeros((len(hidden), CANDIDATES), np.float32)
        lib().d3_audio_logits(self.handle, _fptr(hidden), len(hidden), _fptr(out), out.size)
        return out

    def tower_prefill(self, hidden, pos, frames, next_frames):
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        frames = np.ascontiguousarray(frames, dtype=np.int32)
        next_frames = np.ascontiguousarray(next_frames, dtype=np.int32)
        lib().d3_tower_prefill(self.handle, _fptr(hidden), len(hidden), pos, _iptr(frames),
                               _iptr(next_frames))

    def tower_forced(self, hidden, pos, frame, next_frame) -> np.ndarray:
        hidden = np.ascontiguousarray(hidden, dtype=np.float32).reshape(1, HIDDEN)
        frame = np.ascontiguousarray(frame, dtype=np.int32)
        next_frame = np.ascontiguousarray(next_frame, dtype=np.int32)
        out = np.zeros((WIDTH - 2, CODE_VOCAB), np.float32)
        lib().d3_tower_forced(self.handle, _fptr(hidden), pos, _iptr(frame), _iptr(next_frame),
                              _fptr(out), out.size)
        return out

    def build_prompt(self, text, refs=(), history=()):
        def pack(turns):
            texts = (ctypes.c_char_p * max(len(turns), 1))(*[t.encode() for t, _ in turns])
            codes = np.concatenate([np.ascontiguousarray(c, np.int32).ravel() for _, c in turns]) \
                if turns else np.zeros(1, np.int32)
            frames = np.array([c.shape[1] for _, c in turns] or [0], np.int32)
            return texts, np.ascontiguousarray(codes, np.int32), frames

        rt, rc, rf = pack(list(refs))
        ht, hc, hf = pack(list(history))
        cap = 20000 * WIDTH
        out = np.zeros(cap, np.int32)
        err = ctypes.create_string_buffer(512)
        n = lib().d3_build_prompt(self.handle, text.encode(), len(refs), rt, _iptr(rc), _iptr(rf),
                                  len(history), ht, _iptr(hc), _iptr(hf), _iptr(out), cap, err,
                                  512)
        if n < 0:
            raise ValueError(err.value.decode())
        return out[: n * WIDTH].reshape(n, WIDTH).copy()


def sample(logits: np.ndarray, temperature, top_k, top_p, seed) -> np.ndarray:
    L = _setup_model_api()
    logits = np.ascontiguousarray(logits, np.float32)
    out = np.zeros(len(logits), np.int32)
    L.d3_sample(_fptr(logits), logits.shape[0], logits.shape[1], temperature, top_k, top_p,
                seed, _iptr(out))
    return out


def split_text(text: str, max_bytes=300, max_speakers=3) -> list[str]:
    L = _setup_model_api()
    cap = len(text.encode()) * 4 + 4096
    out = ctypes.create_string_buffer(cap)
    n = L.d3_split_text(text.encode(), max_bytes, max_speakers, out, cap)
    if n == 0:
        return []
    return out.value.decode().split("\x1e")


class Codec:
    def __init__(self, dtype="float32", path=None):
        L = _setup_model_api()
        code = {"float32": 0, "bfloat16": 1, "float16": 2}[dtype]
        self.handle = L.d3_codec_load(str(path or MODEL_DIR / "codec.safetensors").encode(), code)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.ascontiguousarray(codes, np.int32)
        cap = codes.shape[1] * 2048
        out = np.zeros(cap, np.float32)
        n = lib().d3_codec_decode(self.handle, _iptr(codes), codes.shape[1], _fptr(out), cap)
        return out[:n]

    def encode(self, audio: np.ndarray) -> np.ndarray:
        audio = np.ascontiguousarray(audio, np.float32)
        frames_cap = len(audio) // 2048 + 2
        out = np.zeros(10 * frames_cap, np.int32)
        n = lib().d3_codec_encode(self.handle, _fptr(audio), len(audio), _iptr(out), out.size)
        return out[: 10 * n].reshape(10, n).copy()


def resample(audio: np.ndarray, orig: int, new: int) -> np.ndarray:
    L = _setup_model_api()
    audio = np.ascontiguousarray(audio, np.float32)
    cap = int(np.ceil(len(audio) * new / orig)) + 16
    out = np.zeros(cap, np.float32)
    n = L.d3_resample(_fptr(audio), len(audio), orig, new, _fptr(out), cap)
    return out[:n]


def _setup_extra():
    L = _setup_model_api()
    if getattr(L, "_extra_api", False):
        return L
    L.d3_prefill.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int, ctypes.c_int, _f32p]
    L.d3_generate.restype = ctypes.c_long
    L.d3_generate.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_float, ctypes.c_int, ctypes.c_float,
                              ctypes.c_ulonglong, _i32p, ctypes.c_size_t]
    L.d3_encode_reference.restype = ctypes.c_long
    L.d3_encode_reference.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _i32p, ctypes.c_size_t,
                                      ctypes.c_char_p, ctypes.c_size_t]
    L._extra_api = True
    return L


def prefill(model: "Model", rows: np.ndarray, capacity: int | None = None) -> np.ndarray:
    """Production chunked prefill; returns every row's backbone hidden [R, H]."""
    L = _setup_extra()
    rows = np.ascontiguousarray(rows, np.int32)
    out = np.zeros((len(rows), HIDDEN), np.float32)
    L.d3_prefill(model.handle, _iptr(rows), len(rows), capacity or len(rows) + 8, _fptr(out))
    return out


def generate(model: "Model", rows: np.ndarray, max_new: int, temperature=0.7, top_k=30,
             top_p=0.7, seed=0) -> tuple[np.ndarray, bool]:
    """generate_frames: (codes [10, frames], reached eos)."""
    L = _setup_extra()
    rows = np.ascontiguousarray(rows, np.int32)
    out = np.zeros(10 * max_new, np.int32)
    n = L.d3_generate(model.handle, _iptr(rows), len(rows), max_new, temperature, top_k, top_p,
                      seed, _iptr(out), out.size)
    frames = abs(n)
    return out[: 10 * frames].reshape(10, frames).copy(), n > 0


def encode_reference(codec: "Codec", path) -> np.ndarray:
    L = _setup_extra()
    cap = 10 * 200000
    out = np.zeros(cap, np.int32)
    err = ctypes.create_string_buffer(512)
    n = L.d3_encode_reference(codec.handle, str(path).encode(), _iptr(out), cap, err, 512)
    if n < 0:
        raise ValueError(err.value.decode())
    return out[: 10 * n].reshape(10, n).copy()

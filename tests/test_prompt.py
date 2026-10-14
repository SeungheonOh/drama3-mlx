"""Prompt rows and long-text chunking must match fish-speech exactly."""

import random

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

import d3
import ref  # noqa: F401  (puts fish-speech on sys.path)
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


@pytest.fixture(scope="module")
def light_model():
    return d3.Model(light=True)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(d3.CHECKPOINT_DIR / "tokenizer.json"))


def codes(frames, seed):
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 1024, size=(10, frames))
    c[0] = rng.integers(0, 4096, size=frames)
    return c.astype(np.int32)


def fish_rows(tok, text, refs=(), history=()):
    return build_drama3_prompt(
        text, tok, ref.config(),
        [t for t, _ in refs], [torch.from_numpy(c.astype(np.int64)) for _, c in refs],
        history=[(t, torch.from_numpy(c.astype(np.int64))) for t, c in history],
    ).numpy()


CASES = [
    ("Hello there.", [], []),
    ("Get the trust fund to the bank early.", [("We asked over twenty people.", codes(75, 0))], []),
    ("<|speaker:1|> Two refs, tagged text.",
     [("  <|speaker:1|> leading space moves behind the tag", codes(40, 1)),
      ("　全角スペースと話者タグなし", codes(33, 2))], []),
    ("<|speaker:0|>Hi! <|speaker:3|>Hello! <|speaker:0|>Bye.",
     [("reference one", codes(12, 3))],
     [("The quarterly report is due on Wednesday.", codes(90, 4)),
      ("<|speaker:2|> second take", codes(5, 5))]),
    ("  \n leading white space and [laughs] tags <whisper>quiet</whisper>", [],
     [("history only", codes(7, 6))]),
    ("中文测试，带有标点符号。", [("参考文本", codes(64, 7))], []),
]


@pytest.mark.parametrize("text,refs,history", CASES)
def test_rows_match(light_model, tok, text, refs, history):
    ours = light_model.build_prompt(text, refs=refs, history=history)
    theirs = fish_rows(tok, text, refs, history)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("text", [
    "", "   \n\t ", "　", "<|im_start|>x", "a<|im_end|>", "<|voice|>", "<|semantic:3|>",
    "<|speaker:16|> x", "<|speaker:01|> x", "<|speaker:|> x", "<|speaker x", "<|text|>",
])
def test_invalid_text_rejected(light_model, tok, text):
    with pytest.raises(InvalidInputError):
        fish_rows(tok, text)
    with pytest.raises(ValueError):
        light_model.build_prompt(text)


def test_invalid_reference_rejected(light_model, tok):
    bad = codes(10, 0)
    bad[3, 4] = 4096
    with pytest.raises(InvalidInputError):
        fish_rows(tok, "ok", [("ref", bad)])
    with pytest.raises(ValueError):
        light_model.build_prompt("ok", refs=[("ref", bad)])
    with pytest.raises(ValueError):
        light_model.build_prompt("ok", refs=[("", codes(10, 0))])


def test_too_long(light_model):
    with pytest.raises(ValueError, match="rows"):
        light_model.build_prompt("ok", refs=[("long reference", codes(8200, 0))])


CHUNK_TEXTS = [
    "Short text.",
    "First sentence is here. Second sentence follows! Is this the third? Yes; and a fourth… "
    * 12,
    "No punctuation at all just words separated by spaces " * 20,
    "x" * 1000,
    "价格是3.14元。这是第二句话！第三句话？" * 20,
    "<|speaker:0|> Hello there, how are you doing today? "
    "<|speaker:1|> I'm fine, thanks. [laughs] And you? <|speaker:2|> Me too. "
    "<|speaker:3|> Fourth speaker joins. <|speaker:0|> Back to zero. " * 6,
    "Before tags. <|speaker:5|> " + "Long monologue sentence number one. " * 15
    + "<|speaker:6|> reply",
    "<whisper>this is a long protected span that should never be split even if it is long "
    "and keeps going and going and going and going and going and going and going and going "
    "and going and going and going and going and going and going and going and going and "
    "going and going and going and going</whisper> then normal text. " * 2,
    "Brackets [a very long cue inside brackets that goes on and on and on and on and on and "
    "on and on and on and on and on and on and on and on and on and on and on and on and on "
    "and on and on and on and on and on and on and on and on] after. " * 2,
    "<|speaker:1|>", "   ", "<|speaker:2|>   <|speaker:3|> only the last speaks",
]


@pytest.mark.parametrize("text", CHUNK_TEXTS)
def test_chunking_matches(text):
    assert d3.split_text(text) == split_drama3_text(text)


def test_chunking_fuzz():
    rng = random.Random(0)
    pieces = ["Hello", "world", ",", ".", "!", "?", ";", "。", "，", "、", "…", " ", "  ", "\n",
              "3.14", "<|speaker:0|>", "<|speaker:1|>", "<|speaker:7|>", "[laughs]",
              "<b>bold text</b>", "<|im_end|>", "中文", "ąęść", "😀", "word-with-dash", ":"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 120)))
        for max_bytes in (40, 300):
            assert d3.split_text(text, max_bytes, 3) == split_drama3_text(text, max_bytes, 3), \
                repr(text)

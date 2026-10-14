"""The C tokenizer must reproduce Hugging Face tokenizers id for id."""

import random
import sys
import unicodedata

import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import normalizers

import d3

TOKENIZER_JSON = d3.MODEL_DIR / "tokenizer.json"
if not TOKENIZER_JSON.exists():
    TOKENIZER_JSON = d3.CHECKPOINT_DIR / "tokenizer.json"

CORPUS = [
    "",
    "Hello, world!",
    "We asked over twenty different people, and they all said it was his.",
    "I'm sure they'll say it's fine, but we've seen what'd happen. DON'T! 'S 'T 'Re",
    "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.",
    "<|speaker:0|> Hello there.\n<|speaker:1|> Hi! <|speaker:15|>ok<|speaker:16|>",
    "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|><|semantic:0|><|semantic:4095|>",
    "你好，欢迎使用语音合成服务，今天的天气非常不错。",
    "他停顿了一下，低声说道：其实，我早就知道这件事了，只是一直没说出口。",
    "こんにちは、世界！カタカナとひらがな、漢字。",
    "안녕하세요. 한국어 텍스트입니다; 조합형 각",
    "Ça va? Œuvre naïve — façade, déjà vu, Ångström, ﬁ ligature, Ｆｕｌｌｗｉｄｔｈ",
    "é à̖ ȫ ḍ̇ q̣̇",
    "Numbers: 1234567890 3.14159 1,000,000 ٣٤٥ ①②③ ½ Ⅻ",
    "Emoji 👍🏽 👨‍👩‍👧‍👦 🇯🇵 ❤️ 😀😃",
    "tabs\tand\t\ttabs   spaces    \n\n\n  newlines\r\n\r\nend   ",
    "    leading spaces and trailing    ",
    "\n",
    "  \n  \n",
    "x y　z w\u0085v",
    "!!!??? ... --- *** ### $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}}",
    " ' '' ''' 's 'S 'll 'LL 'lL 've 'VE 're 'Re 'd 'D 'm 'M 'ſ 'x",
    "مرحبا بالعالم. هذا نص عربي.",
    "Привет, мир! Это русский текст.",
    "नमस्ते दुनिया। यह हिंदी है।",
    "สวัสดีชาวโลก นี่คือภาษาไทย",
    "Γειά σου Κόσμε. ΆΈΉΊΌΎΏ ΐΰ",
    "<think>\n\n</think>\n\n<tool_call>x</tool_call>",
    "<|endoftext|><|im_end|><|im_end|>abc<|im_end",
    "​‌‍﻿ zero width",
    "\U0001d400\U0001d401 math bold, \U00020000 cjk ext b",
]


@pytest.fixture(scope="module")
def tokenizers():
    return d3.Tokenizer(TOKENIZER_JSON), HFTokenizer.from_file(str(TOKENIZER_JSON))


def test_corpus(tokenizers):
    ours, ref = tokenizers
    for text in CORPUS:
        assert ours.encode(text) == ref.encode(text).ids, text


def test_token_to_id(tokenizers):
    ours, ref = tokenizers
    for token in ["<|im_end|>", "<|semantic:0|>", "<|semantic:4095|>", "<|voice|>",
                  "<|speaker:0|>", "<|speaker:15|>", "Ġthe", "hello", "<|nope|>"]:
        expected = ref.token_to_id(token)
        assert ours.token_to_id(token) == (-1 if expected is None else expected), token


def all_code_points():
    return [c for c in range(sys.maxunicode + 1) if not 0xD800 <= c <= 0xDFFF]


def test_nfc_every_code_point():
    nfc = normalizers.NFC()
    mismatches = []
    for cp in all_code_points():
        s = chr(cp)
        if d3.nfc(s) != nfc.normalize_str(s):
            mismatches.append(cp)
    assert not mismatches, [hex(c) for c in mismatches[:20]]


def test_nfc_random_combining_sequences():
    nfc = normalizers.NFC()
    rng = random.Random(0)
    marks = [c for c in range(0x300, 0x370)] + [0x1AB0, 0x1DC0, 0x20D0, 0x302A, 0x3099]
    bases = [c for c in all_code_points() if unicodedata.decomposition(chr(c))
             and not unicodedata.decomposition(chr(c)).startswith("<")]
    bases += [ord(c) for c in "aeiouAEIOUnNcCsSzZ"] + list(range(0x1100, 0x1113))
    bases += list(range(0x1161, 0x1176)) + list(range(0x11A8, 0x11C3))
    bases += [0xAC00, 0xAC01, 0xD7A3]
    for _ in range(20000):
        s = "".join(
            chr(rng.choice(bases if rng.random() < 0.5 else marks))
            for _ in range(rng.randint(1, 6))
        )
        assert d3.nfc(s) == nfc.normalize_str(s), [hex(ord(c)) for c in s]


def test_every_code_point_in_context(tokenizers):
    ours, ref = tokenizers
    cps = all_code_points()
    templates = ["{}", "a{}b", " {}x", "'{}", "{}́", "1{} ", " {}{} ", "\n{}\n"]
    for template in templates:
        texts = [template.format(chr(c), chr(c)) for c in cps]
        for start in range(0, len(texts), 20000):
            batch = texts[start:start + 20000]
            expected = ref.encode_batch(batch)
            for text, enc in zip(batch, expected):
                assert ours.encode(text) == enc.ids, (template, [hex(ord(c)) for c in text])


def test_random_fuzz(tokenizers):
    ours, ref = tokenizers
    rng = random.Random(1)
    alphabet = (
        list(" \t\n\r 　'\".,!?;:-_()[]<>|/\\0123456789")
        + list("abcXYZéüßſİıǅﬁ") + list("中文日本語한국어ㄱ") + ["́", "̣", "ि"]
        + ["<|im_end|>", "<|speaker:3|>", "<|semantic:7|>", "'s", "'LL", "\r\n", "👍🏽"]
    )
    for _ in range(30000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        assert ours.encode(text) == ref.encode(text).ids, repr(text)

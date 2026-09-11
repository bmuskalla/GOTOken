#!/usr/bin/env python3
"""
Reference tokenizer that uses nothing but tokenizer.json and Python's
Unicode database.

This is the algorithm src/tokenizer.bm implements, written first in Python so
it could be validated against HuggingFace on thousands of strings before
being ported to BASIC. Anything HuggingFace does that this file does not is
listed under "Known differences" below.

The pipeline for encode(text):

  1. pre-tokenize: split the text into chunks with the GPT-2 regex
         's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
     implemented as a hand-written scanner (no regex engine in BASIC).
     Merges never cross chunk boundaries.
  2. for each chunk: start from one token per byte, then repeatedly merge the
     adjacent pair whose concatenation is a vocab token with the best score
     (llama2.c's loop; score = -merge_rank, so lowest rank merges first).
  3. bytes with no token (21 of them, never in valid UTF-8 text except a few
     control characters) are dropped, as HuggingFace does.

Known differences from HuggingFace: special tokens such as <|endoftext|> in
the *input text* are treated as ordinary text here.

Usage:
    python bpe_ref.py            # validate against HuggingFace on the test corpus
"""

import json
import random
import sys
import unicodedata
from bisect import bisect_right
from pathlib import Path

TOKENIZER_JSON = Path(__file__).resolve().parent.parent / "model" / "tokenizer.json"
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
NO_MERGE_SCORE = -1e30


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map, in which tokenizer.json
    writes its vocab (a space shows up as 'Ġ')."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def unicode_ranges(pred) -> list[tuple[int, int]]:
    out, start = [], None
    for cp in range(0x110000):
        if pred(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


WHITE_SPACE = set(range(0x09, 0x0E)) | {0x20, 0x85, 0xA0, 0x1680} | set(range(0x2000, 0x200B)) | \
    {0x2028, 0x2029, 0x202F, 0x205F, 0x3000}


class Ranges:
    """Sorted inclusive [lo, hi] codepoint ranges with a binary-search test."""

    def __init__(self, pairs):
        self.lo = [a for a, _ in pairs]
        self.hi = [b for _, b in pairs]

    def __contains__(self, cp: int) -> bool:
        i = bisect_right(self.lo, cp) - 1
        return i >= 0 and cp <= self.hi[i]


class Tokenizer:
    def __init__(self, path=TOKENIZER_JSON):
        tj = json.loads(Path(path).read_text())
        assert tj["model"]["type"] == "BPE"
        vocab: dict[str, int] = tj["model"]["vocab"]
        self.vocab_size = len(vocab)
        u2b = {ch: b for b, ch in bytes_to_unicode().items()}
        self.tokens: list[bytes] = [b""] * self.vocab_size
        for s, i in vocab.items():
            self.tokens[i] = bytes(u2b[ch] for ch in s)
        for a in tj["added_tokens"]:  # specials are literal text, not byte-level encoded
            self.tokens[a["id"]] = a["content"].encode("utf-8")
        self.id_of: dict[bytes, int] = {t: i for i, t in enumerate(self.tokens)}
        # score = -rank of the merge that produces a token; the first rule wins
        self.scores: list[float] = [NO_MERGE_SCORE] * self.vocab_size
        for r, m in enumerate(tj["model"]["merges"]):
            a, b = m.split(" ") if isinstance(m, str) else m
            i = self.id_of[bytes(u2b[ch] for ch in a + b)]
            if self.scores[i] == NO_MERGE_SCORE:
                self.scores[i] = -r
        self.max_token_length = max(len(t) for t in self.tokens)
        self.byte_id: list[int | None] = [self.id_of.get(bytes([b])) for b in range(256)]
        self.letters = Ranges(unicode_ranges(lambda cp: unicodedata.category(chr(cp)).startswith("L")))
        self.numbers = Ranges(unicode_ranges(lambda cp: unicodedata.category(chr(cp)).startswith("N")))
        self.spaces = Ranges(unicode_ranges(lambda cp: cp in WHITE_SPACE))

    # -- 1. pre-tokenizer -------------------------------------------------- #
    def _cls(self, ch: str) -> str:
        cp = ord(ch)
        if cp in self.spaces:
            return "S"
        if cp in self.letters:
            return "L"
        if cp in self.numbers:
            return "N"
        return "P"  # anything else: punctuation, symbols, marks, controls

    def pretokenize(self, text: str) -> list[str]:
        chunks, i, n = [], 0, len(text)
        while i < n:
            # 's|'t|'re|'ve|'m|'ll|'d   (lowercase only, in this order)
            if text[i] == "'":
                for c in CONTRACTIONS:
                    if text.startswith(c, i):
                        chunks.append(c)
                        i += len(c)
                        break
                else:
                    c = None
                if c:
                    continue
            #  ?\p{L}+  |   ?\p{N}+  |   ?[^\s\p{L}\p{N}]+     (optional literal space)
            j = i + 1 if text[i] == " " else i
            if j < n:
                k = self._cls(text[j])
                if k != "S":
                    e = j
                    while e < n and self._cls(text[e]) == k:
                        e += 1
                    chunks.append(text[i:e])
                    i = e
                    continue
            # \s+(?!\S)  |  \s+
            if self._cls(text[i]) == "S":
                e = i
                while e < n and self._cls(text[e]) == "S":
                    e += 1
                if e < n and e - i > 1:  # followed by non-space: leave the last space for it
                    e -= 1
                chunks.append(text[i:e])
                i = e
                continue
            raise AssertionError(f"scanner stuck at {i}: {text[i]!r}")
        return chunks

    # -- 2. byte-pair merging ---------------------------------------------- #
    def encode_chunk(self, chunk: bytes) -> list[int]:
        ids = [self.byte_id[b] for b in chunk if self.byte_id[b] is not None]
        while len(ids) > 1:
            best, best_id, best_at = -1e31, -1, -1
            for i in range(len(ids) - 1):
                merged = self.id_of.get(self.tokens[ids[i]] + self.tokens[ids[i + 1]])
                if merged is not None and self.scores[merged] > best:
                    best, best_id, best_at = self.scores[merged], merged, i
            if best_at < 0:
                break
            ids[best_at:best_at + 2] = [best_id]
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in self.pretokenize(text) for i in self.encode_chunk(chunk.encode("utf-8"))]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.tokens[i] for i in ids).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# test corpus: hand-written edge cases plus random soup over many scripts
# --------------------------------------------------------------------------- #
HAND = [
    "Hello world", "The quick brown fox jumps over the lazy dog.", "  leading spaces and 12345 digits",
    "Ünïcödé ✓ 日本語 🚀 naïve café", "def main():\n    return 0\n", "Hello\n\nWorld", "Hello\n\n\nWorld",
    "a  b", "a   b", "end  ", "I'm you're they'd it's We'RE", "x'S", "3.14", "١٢٣", "Ⅻ", "a\xa0b", "a\xa0\xa0b",
    "a\x1cb", "a\x04b", "tab\there", "日本語テキスト", "hello,world!!  ...", "ⓐ①", "a​b", "áb",
    "_x_ x_y", "1st 2nd", "e.g. i.e.", "a\r\nb", " \n", "x \n y", "", " ", "\n", "'", "''", "don't",
    "Größe straße", "Ελληνικά", "Кириллица", "العربية", "עברית", "हिन्दी", "ไทย", "한국어", "😀😃😄", "👨‍👩‍👧",
    "The cat sat on the mat.", "Once upon a time, there was a", "x=1; y=2 // comment", "<html><body>",
    "🇩🇪", "é", "\t\t", "a\tb", "100%", "$5.99", "C++", "C#", "#include <stdio.h>",
]
POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'\"-_()[]{}<>/\\@#$%^&*+=~`|\n\t") \
    + list("äöüßéèêàçñøåæ") + list("日本語中文한국어") + list("αβγδΩ") + list("абвгд") + list("🚀✓😀❤️") \
    + ["\xa0", "　", "​", "́", "'s", "'re", "  ", "\n\n", "\r\n"]


def corpus(n_random: int = 2000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out = list(HAND)
    for _ in range(n_random):
        out.append("".join(rng.choice(POOL) for _ in range(rng.randint(1, 40))))
    return out


def validate(n_random: int = 2000) -> int:
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    ours = Tokenizer()
    bad = 0
    tests = corpus(n_random)
    for s in tests:
        a, b = ours.encode(s), hf.encode(s, add_special_tokens=False)
        if a != b:
            bad += 1
            if bad <= 10:
                print(f"MISMATCH {s!r}\n  ours {a}\n  hf   {b}")
        if ours.decode(a) != s.encode("utf-8").decode("utf-8", "replace") and not any(
                ours.byte_id[x] is None for x in s.encode("utf-8")):
            bad += 1
            print(f"DECODE MISMATCH {s!r} -> {ours.decode(a)!r}")
    print(f"{len(tests) - bad}/{len(tests)} strings match HuggingFace ({len(HAND)} hand-written, {n_random} random)")
    return bad


if __name__ == "__main__":
    sys.exit(1 if validate() else 0)

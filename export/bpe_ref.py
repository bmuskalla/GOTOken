#!/usr/bin/env python3
"""
Reference tokenizer that uses nothing but tokenizer.bin.

This is the algorithm src/tokenizer.bm implements, written first in Python so
it can be validated against HuggingFace on thousands of strings before being
ported to BASIC. Anything HuggingFace does that this file does not is either
handled at export time or listed under "Known differences" below.

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

import random
import struct
import sys
from bisect import bisect_right
from pathlib import Path

TOKENIZER_BIN = Path(__file__).resolve().parent.parent / "model" / "tokenizer.bin"
CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


class Ranges:
    """Sorted inclusive [lo, hi] codepoint ranges with a binary-search test."""

    def __init__(self, pairs):
        self.lo = [a for a, _ in pairs]
        self.hi = [b for _, b in pairs]

    def __contains__(self, cp: int) -> bool:
        i = bisect_right(self.lo, cp) - 1
        return i >= 0 and cp <= self.hi[i]


class Tokenizer:
    def __init__(self, path=TOKENIZER_BIN):
        with open(path, "rb") as f:
            self.vocab_size, self.max_token_length = struct.unpack("<ii", f.read(8))
            self.tokens: list[bytes] = []
            self.scores: list[float] = []
            for _ in range(self.vocab_size):
                score, n = struct.unpack("<fi", f.read(8))
                self.tokens.append(f.read(n))
                self.scores.append(score)
            classes = {}
            for name in ("L", "N", "WS"):
                (n,) = struct.unpack("<i", f.read(4))
                classes[name] = Ranges([struct.unpack("<ii", f.read(8)) for _ in range(n)])
            assert not f.read(1), "trailing bytes in tokenizer.bin"
        self.letters, self.numbers, self.spaces = classes["L"], classes["N"], classes["WS"]
        self.id_of: dict[bytes, int] = {t: i for i, t in enumerate(self.tokens)}
        self.byte_id: list[int | None] = [self.id_of.get(bytes([b])) for b in range(256)]

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

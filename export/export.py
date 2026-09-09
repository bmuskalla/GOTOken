#!/usr/bin/env python3
"""
Step 0: export SmolLM2-135M into two flat binaries the BASIC engine can read.

    weights.bin    fixed header + fp32 little-endian blobs in a documented order
    tokenizer.bin  vocab as (score, byte length, raw bytes) records, llama2.c style

All "format intelligence" lives here: bf16 -> fp32 conversion, reading the
config instead of hardcoding it, undoing the GPT-2 byte-to-unicode trick in the
tokenizer vocab, and re-ordering the Q/K projection rows so the BASIC RoPE can
use llama2.c's interleaved pair layout.  The BASIC side only ever does
`GET #1` on a stream of INT32s and SINGLEs.

See ../FORMAT.md for the exact layout.

Usage:
    python export.py                      # writes ../model/weights.bin + tokenizer.bin
    python export.py --out /some/dir      # somewhere else
    python export.py --check-only         # re-read existing binaries and print the checkpoint
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MAGIC = 0x4B4F5447  # the bytes b"GTOK" read as a little-endian INT32
VERSION = 1
HEADER_FMT = "<10i2f"  # 10 INT32 then 2 float32, all little-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 48 bytes
NO_MERGE_SCORE = -1e30  # tokens that no BPE merge produces (byte tokens, specials)
F32 = np.dtype("<f4")  # explicit little-endian fp32, never rely on host order


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["model_type"] == "llama", cfg["model_type"]
    assert cfg["hidden_act"] == "silu", cfg["hidden_act"]
    assert cfg.get("rope_scaling") is None, "rope scaling not supported"
    assert not cfg.get("attention_bias", False), "attention bias not supported"
    return {
        "dim": cfg["hidden_size"],
        "hidden_dim": cfg["intermediate_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_heads": cfg["num_attention_heads"],
        "n_kv_heads": cfg["num_key_value_heads"],
        "vocab_size": cfg["vocab_size"],
        "seq_len": cfg["max_position_embeddings"],
        "tied": 1 if cfg["tie_word_embeddings"] else 0,
        "rope_theta": float(cfg["rope_theta"]),
        "rms_norm_eps": float(cfg["rms_norm_eps"]),
    }


def pack_header(c: dict) -> bytes:
    return struct.pack(
        HEADER_FMT,
        MAGIC, VERSION,
        c["dim"], c["hidden_dim"], c["n_layers"], c["n_heads"], c["n_kv_heads"],
        c["vocab_size"], c["seq_len"], c["tied"],
        c["rope_theta"], c["rms_norm_eps"],
    )


def unpack_header(raw: bytes) -> dict:
    fields = struct.unpack(HEADER_FMT, raw)
    names = ["magic", "version", "dim", "hidden_dim", "n_layers", "n_heads",
             "n_kv_heads", "vocab_size", "seq_len", "tied", "rope_theta", "rms_norm_eps"]
    return dict(zip(names, fields))


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
def permute_for_interleaved_rope(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Reorder the output rows of a Q or K projection so that, per head, the
    rotary pair (j, j + head_dim/2) used by HuggingFace's rotate_half becomes
    the adjacent pair (2j, 2j+1) used by llama2.c's run.c.

    Same trick as llama2.c's export.py `permute_reverse`.  Attention scores are
    dot products within a head, so permuting q and k rows identically leaves
    every layer output bit-for-bit unchanged; only the internal q/k vectors
    look different from a HF forward hook."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // n_heads
    assert head_dim * n_heads == out_dim and head_dim % 2 == 0
    return (
        w.reshape(n_heads, 2, head_dim // 2, in_dim)
        .transpose(0, 2, 1, 3)
        .reshape(out_dim, in_dim)
    )


def layer_tensors(i: int, c: dict) -> list[tuple[str, str, int | None]]:
    """(HF name, role, n_heads-for-permute or None) in the order they are written."""
    p = f"model.layers.{i}."
    return [
        (p + "input_layernorm.weight", "rms_att", None),
        (p + "self_attn.q_proj.weight", "wq", c["n_heads"]),
        (p + "self_attn.k_proj.weight", "wk", c["n_kv_heads"]),
        (p + "self_attn.v_proj.weight", "wv", None),
        (p + "self_attn.o_proj.weight", "wo", None),
        (p + "post_attention_layernorm.weight", "rms_ffn", None),
        (p + "mlp.gate_proj.weight", "w1", None),
        (p + "mlp.down_proj.weight", "w2", None),
        (p + "mlp.up_proj.weight", "w3", None),
    ]


def to_f32(t: torch.Tensor) -> np.ndarray:
    # bf16 -> fp32 is exact: same 8-bit exponent, mantissa padded with 16 zero bits.
    return np.ascontiguousarray(t.to(torch.float32).numpy().astype(F32))


def export_weights(model_dir: Path, c: dict, out_path: Path) -> np.ndarray:
    """Write weights.bin.  Returns the fp32 embedding table for the checkpoint print."""
    written = 0
    with safe_open(model_dir / "model.safetensors", framework="pt") as f, open(out_path, "wb") as out:
        out.write(pack_header(c))

        def put(name: str, arr: np.ndarray, expect: tuple):
            nonlocal written
            assert arr.shape == expect, f"{name}: {arr.shape} != {expect}"
            assert arr.dtype == F32 and arr.flags.c_contiguous
            out.write(arr.tobytes())
            written += arr.size

        emb = to_f32(f.get_tensor("model.embed_tokens.weight"))
        put("token_embedding", emb, (c["vocab_size"], c["dim"]))

        dim, hd, kv = c["dim"], c["hidden_dim"], c["n_kv_heads"] * (c["dim"] // c["n_heads"])
        shapes = {"rms_att": (dim,), "wq": (dim, dim), "wk": (kv, dim), "wv": (kv, dim),
                  "wo": (dim, dim), "rms_ffn": (dim,), "w1": (hd, dim), "w2": (dim, hd), "w3": (hd, dim)}
        for i in range(c["n_layers"]):
            for name, role, permute_heads in layer_tensors(i, c):
                arr = to_f32(f.get_tensor(name))
                if permute_heads:
                    arr = np.ascontiguousarray(permute_for_interleaved_rope(arr, permute_heads))
                put(name, arr, shapes[role])
            print(f"  layer {i:2d} written", end="\r", flush=True)
        print()

        put("model.norm.weight", to_f32(f.get_tensor("model.norm.weight")), (dim,))
        if not c["tied"]:
            put("lm_head.weight", to_f32(f.get_tensor("lm_head.weight")), (c["vocab_size"], dim))

        leftover = sorted(k for k in f.keys() if k not in _expected_keys(c))
        assert not leftover, f"unexported tensors: {leftover}"

    print(f"  {written:,} parameters -> {out_path} ({out_path.stat().st_size:,} bytes)")
    return emb


def _expected_keys(c: dict) -> set[str]:
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(c["n_layers"]):
        keys.update(name for name, _, _ in layer_tensors(i, c))
    if not c["tied"]:
        keys.add("lm_head.weight")
    return keys


def check_rope_permutation(model_dir: Path, c: dict) -> None:
    """Numerically prove the Q/K row permutation is harmless: HF-style RoPE on the
    original projections and llama2.c-style RoPE on the permuted ones must give
    identical q.k attention scores for every head."""
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        wq = to_f32(f.get_tensor("model.layers.0.self_attn.q_proj.weight")).astype(np.float64)
        wk = to_f32(f.get_tensor("model.layers.0.self_attn.k_proj.weight")).astype(np.float64)
    n_heads, n_kv = c["n_heads"], c["n_kv_heads"]
    head_dim = c["dim"] // n_heads
    rng = np.random.default_rng(0)
    x_q, x_k = rng.standard_normal(c["dim"]), rng.standard_normal(c["dim"])
    pos = 7
    freqs = 1.0 / (c["rope_theta"] ** (np.arange(0, head_dim, 2) / head_dim))
    ang = pos * freqs
    cos, sin = np.cos(ang), np.sin(ang)

    def rope_hf(v):  # rotate_half: pairs (j, j + head_dim/2)
        a, b = v[: head_dim // 2], v[head_dim // 2:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin])

    def rope_interleaved(v):  # run.c: pairs (2j, 2j+1)
        a, b = v[0::2], v[1::2]
        o = np.empty_like(v)
        o[0::2], o[1::2] = a * cos - b * sin, b * cos + a * sin
        return o

    q_hf, k_hf = wq @ x_q, wk @ x_k
    q_il = permute_for_interleaved_rope(wq, n_heads) @ x_q
    k_il = permute_for_interleaved_rope(wk, n_kv) @ x_k
    for h in range(n_heads):
        kvh = h // (n_heads // n_kv)
        qs, ks = slice(h * head_dim, (h + 1) * head_dim), slice(kvh * head_dim, (kvh + 1) * head_dim)
        s_hf = rope_hf(q_hf[qs]) @ rope_hf(k_hf[ks])
        s_il = rope_interleaved(q_il[qs]) @ rope_interleaved(k_il[ks])
        assert abs(s_hf - s_il) < 1e-9 * max(1.0, abs(s_hf)), (h, s_hf, s_il)
    print(f"  RoPE permutation check: {n_heads} heads, HF vs interleaved scores identical")


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map.  The vocab in
    tokenizer.json is stored through this map (a space shows up as 'Ġ'); we
    invert it so tokenizer.bin holds the actual bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def unicode_class_ranges() -> dict[str, list[tuple[int, int]]]:
    """Codepoint ranges for the three character classes the GPT-2 regex uses:
    letters (L*), numbers (N*), and whitespace (the Unicode White_Space
    property, which is what the Rust regex crate's \s means)."""
    import unicodedata

    def ranges_where(pred) -> list[tuple[int, int]]:
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

    white_space = set(range(0x09, 0x0E)) | {0x20, 0x85, 0xA0, 0x1680} | set(range(0x2000, 0x200B)) | \
        {0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
    return {
        "L": ranges_where(lambda cp: unicodedata.category(chr(cp)).startswith("L")),
        "N": ranges_where(lambda cp: unicodedata.category(chr(cp)).startswith("N")),
        "WS": ranges_where(lambda cp: cp in white_space),
    }


def export_tokenizer(model_dir: Path, c: dict, out_path: Path) -> list[bytes]:
    tj = json.loads((model_dir / "tokenizer.json").read_text())
    assert tj["model"]["type"] == "BPE"
    vocab: dict[str, int] = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    assert len(vocab) == c["vocab_size"], (len(vocab), c["vocab_size"])

    u2b = {ch: b for b, ch in bytes_to_unicode().items()}
    tokens: list[bytes | None] = [None] * c["vocab_size"]
    for s, i in vocab.items():
        tokens[i] = bytes(u2b[ch] for ch in s)
    for a in tj["added_tokens"]:  # specials are literal text, not byte-level encoded
        tokens[a["id"]] = a["content"].encode("utf-8")
    assert all(t is not None for t in tokens)

    # llama2.c's encoder picks the adjacent pair whose merged string has the
    # highest score.  GPT-2 BPE picks the merge with the lowest rank.  So:
    # score = -rank.  Tokens no merge produces get a sentinel far below any rank.
    rank: dict[str, int] = {}
    for r, m in enumerate(merges):
        a, b = m.split(" ") if isinstance(m, str) else m
        rank.setdefault(a + b, r)
    scores = np.full(c["vocab_size"], NO_MERGE_SCORE, dtype=F32)
    n_merged = 0
    for s, i in vocab.items():
        if s in rank:
            scores[i] = -rank[s]
            n_merged += 1

    max_len = max(len(t) for t in tokens)
    with open(out_path, "wb") as out:
        out.write(struct.pack("<ii", c["vocab_size"], max_len))
        for t, sc in zip(tokens, scores):
            out.write(struct.pack("<fi", float(sc), len(t)))
            out.write(t)
        # The pre-tokenizer regex needs \p{L}, \p{N} and \s over all of Unicode.
        # Ship them as sorted inclusive codepoint ranges so BASIC does a binary
        # search instead of carrying a Unicode database.
        for name, ranges in unicode_class_ranges().items():
            out.write(struct.pack("<i", len(ranges)))
            for lo, hi in ranges:
                out.write(struct.pack("<ii", lo, hi))
            print(f"  unicode class {name}: {len(ranges)} ranges")

    # Not every byte has a token: this vocab was trained on UTF-8 text, so bytes
    # that never occur in valid UTF-8 (0xC0, 0xC1, 0xF5..0xFF) and a few control
    # characters are absent.  HF drops such bytes silently; the BASIC tokenizer's
    # byte fallback (step 7) has to do the same.
    single = {t[0] for t in tokens if len(t) == 1}
    missing = sorted(set(range(256)) - single)
    print(f"  {c['vocab_size']} tokens, max {max_len} bytes, {n_merged} merge-produced, "
          f"{len(merges)} merge rules, {len(single)} byte tokens, {len(tj['added_tokens'])} specials "
          f"-> {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  bytes with no token ({len(missing)}): {[f'{b:#04x}' for b in missing]}")
    return tokens


def check_tokenizer_roundtrip(model_dir: Path, tokens: list[bytes]) -> None:
    """Byte-level BPE is lossless: HF ids -> our bytes must rebuild the text."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    samples = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "  leading spaces and 12345 digits",
        "Ünïcödé ✓ 日本語 🚀 naïve café",
        "def main():\n    return 0\n",
    ]
    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        rebuilt = b"".join(tokens[i] for i in ids).decode("utf-8")
        assert rebuilt == s, (s, ids, rebuilt)
    print(f"  tokenizer round-trip: {len(samples)} strings rebuilt byte-exact from HF ids")


# --------------------------------------------------------------------------- #
# checkpoint: read the files back the way BASIC will
# --------------------------------------------------------------------------- #
def print_checkpoint(weights_path: Path, tokenizer_path: Path) -> None:
    with open(weights_path, "rb") as f:
        h = unpack_header(f.read(HEADER_SIZE))
        first5 = np.frombuffer(f.read(5 * 4), dtype=F32)
    print("\nweights.bin header")
    for k, v in h.items():
        extra = f"  (b'{struct.pack('<i', v).decode()}')" if k == "magic" else ""
        print(f"  {k:<13} {v}{extra}")
    print("\nfirst 5 floats of token_embedding (row 0 = token id 0):")
    for i, v in enumerate(first5):
        print(f"  [{i}] {float(v)!r:<16} hex {v.tobytes().hex()}   (bytes as stored, little-endian)")

    n_params = (weights_path.stat().st_size - HEADER_SIZE) // 4
    print(f"\n  total fp32 params in file: {n_params:,}")
    with open(weights_path, "rb") as f:
        f.seek(-5 * 4, 2)
        last5 = np.frombuffer(f.read(5 * 4), dtype=F32)
    print("\nlast 5 floats of the file (tail of rms_final; proves the blob-size arithmetic):")
    for i, v in enumerate(last5):
        print(f"  [{i}] {float(v)!r:<16} hex {v.tobytes().hex()}")

    with open(tokenizer_path, "rb") as f:
        vocab_size, max_len = struct.unpack("<ii", f.read(8))
        print(f"\ntokenizer.bin: vocab_size={vocab_size} max_token_length={max_len}")
        print("  first 20 tokens (id: score, bytes):")
        for i in range(20):
            score, n = struct.unpack("<fi", f.read(8))
            raw = f.read(n)
            print(f"  {i:5d}: {score:>8.6g}  {raw!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "model")
    ap.add_argument("--check-only", action="store_true", help="skip export, just re-read the binaries")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    weights_path, tokenizer_path = args.out / "weights.bin", args.out / "tokenizer.bin"

    if not args.check_only:
        print(f"downloading/locating {args.model}")
        model_dir = Path(snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"]))
        c = load_config(model_dir)
        print("config:", json.dumps(c))

        print("exporting weights")
        export_weights(model_dir, c, weights_path)
        check_rope_permutation(model_dir, c)

        print("exporting tokenizer")
        tokens = export_tokenizer(model_dir, c, tokenizer_path)
        check_tokenizer_roundtrip(model_dir, tokens)

    print_checkpoint(weights_path, tokenizer_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

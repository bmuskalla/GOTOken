#!/usr/bin/env python3
"""
The Python oracle: reference numbers from the HuggingFace model for every
step's checkpoint, and a comparison against the BASIC engine's output.

Kept in sync with gotoken.bas for the whole project. oracle.ipynb is a thin
notebook front-end over the functions in this file.

Usage:
    python oracle.py encode "some text"          # token ids, to pick test tokens
    python oracle.py step2 --token 9690           # reference logits for one token
    python oracle.py step2 --token 9690 --basic   # ...and compare with build/gotoken
    python oracle.py step3 --token 9690 --basic   # RMSNorm + MatMul in isolation vs BASIC
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from export import permute_for_interleaved_rope

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
REPO = Path(__file__).resolve().parent.parent
BASIC_EXE = REPO / "build" / "gotoken"


def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


# --------------------------------------------------------------------------- #
# step 2: embedding row -> tied output head, nothing in between
# --------------------------------------------------------------------------- #
def step2_logits(model, token_id: int) -> np.ndarray:
    """logits[v] = dot(embedding[v], embedding[token_id]), in float64.

    float64 so the reference is the 'true' value; the BASIC side sums 576
    products in fp32 and lands within ~1e-5 of it.  Also asserts the tie:
    lm_head and embed_tokens are literally the same tensor."""
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight
    assert head.data_ptr() == emb.data_ptr(), "expected tied embeddings"
    emb64 = emb.detach().double()
    return (emb64 @ emb64[token_id]).numpy()


def print_step2(model, tok, token_id: int, k: int = 5) -> dict:
    logits = step2_logits(model, token_id)
    top = np.argsort(-logits)[:k]
    print(f"token {token_id} = {tok.convert_ids_to_tokens(token_id)!r}")
    for i in range(5):
        print(f"logit {i} {logits[i]:.7g}")
    for r, v in enumerate(top, 1):
        print(f"top {r} id {v} logit {logits[v]:.7g}   {tok.convert_ids_to_tokens(int(v))!r}")
    return {"logits": logits, "top": top}


# --------------------------------------------------------------------------- #
# running and parsing the BASIC engine
# --------------------------------------------------------------------------- #
def run_basic(*args: str) -> str:
    if not BASIC_EXE.exists():
        sys.exit(f"{BASIC_EXE} not found; run ./build.sh first")
    out = subprocess.run([str(BASIC_EXE), *map(str, args)], capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0 or "error:" in out.stdout:
        sys.exit(f"BASIC engine failed:\n{out.stdout}{out.stderr}")
    return out.stdout


def parse_step2(text: str) -> dict:
    first = {int(m[1]): float(m[2]) for m in re.finditer(r"^logit\s+(\d+)\s+(\S+)", text, re.M)}
    top = [(int(m[1]), int(m[2]), float(m[3]))
           for m in re.finditer(r"^top\s+(\d+)\s+id\s+(\d+)\s+logit\s+(\S+)", text, re.M)]
    assert len(first) == 5 and top, f"could not parse BASIC output:\n{text}"
    return {"first": first, "top": top}


def compare_step2(model, tok, token_id: int, rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    ref = step2_logits(model, token_id)
    got = parse_step2(run_basic("logits", token_id))
    ok = True
    print("\ncompare BASIC vs oracle")
    for i, v in got["first"].items():
        d = abs(v - ref[i])
        flag = "ok" if d <= atol + rtol * abs(ref[i]) else "MISMATCH"
        ok &= flag == "ok"
        print(f"  logit {i}: basic {v:.7g}  oracle {ref[i]:.7g}  diff {d:.2e}  {flag}")
    ref_top = np.argsort(-ref)[: len(got["top"])]
    for (r, vid, val), rid in zip(got["top"], ref_top):
        d = abs(val - ref[vid])
        flag = "ok" if vid == rid and d <= atol + rtol * abs(ref[vid]) else "MISMATCH"
        ok &= flag == "ok"
        print(f"  top {r}: basic id {vid} {val:.7g}  oracle id {rid} {ref[rid]:.7g}  diff {d:.2e}  {flag}")
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# step 3: the kernels in isolation. Real layer-0 weights, a real embedding row.
# --------------------------------------------------------------------------- #
F32 = np.dtype("<f4")


def step3_refs(model, token_id: int) -> tuple[dict, dict]:
    """float64 reference outputs and the fp32 weight matrices as BASIC sees them
    (Q/K rows permuted exactly like the exporter does)."""
    cfg = model.config
    l0 = model.model.layers[0]
    f32 = lambda t: t.detach().float().numpy().astype(F32)
    x = f32(model.get_input_embeddings().weight[token_id]).astype(np.float64)
    g = f32(l0.input_layernorm.weight).astype(np.float64)
    xb = x / np.sqrt((x * x).mean() + cfg.rms_norm_eps) * g
    weights = {
        "wq": permute_for_interleaved_rope(f32(l0.self_attn.q_proj.weight), cfg.num_attention_heads),
        "wk": permute_for_interleaved_rope(f32(l0.self_attn.k_proj.weight), cfg.num_key_value_heads),
        "w1": f32(l0.mlp.gate_proj.weight),
    }
    refs = {"x": x, "rmsnorm": xb}
    for name, W in weights.items():
        refs[name] = W.astype(np.float64) @ xb
    return refs, weights


def parse_vectors(text: str) -> dict[str, np.ndarray]:
    """'vec <name> <n>' followed by n lines '<i> <value> <hex>'. Values are
    rebuilt from the hex, so they are the exact fp32 bits BASIC holds."""
    out, name, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^vec\s+(\w+)\s+(\d+)", line)
        if m:
            if name:
                out[name] = np.array(buf, dtype=F32)
            name, buf = m[1], []
            continue
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+([0-9a-f]{8})$", line)
        if m and name:
            buf.append(np.frombuffer(bytes.fromhex(m[3]), dtype=F32)[0])
    if name:
        out[name] = np.array(buf, dtype=F32)
    return out


def emulate_matmul_fp32(W: np.ndarray, xb: np.ndarray, fused: bool) -> np.ndarray:
    """Replay BASIC's loop: acc = acc + W[i,j]*xb[j] for j = 0..n-1, in fp32.
    fused=False rounds the product and then the sum (two roundings).
    fused=True rounds once, like an FMA instruction: product exact in fp64."""
    d, n = W.shape
    acc = np.zeros(d, dtype=F32)
    if fused:
        W64, x64 = W.astype(np.float64), xb.astype(np.float64)
        for j in range(n):
            acc = (acc.astype(np.float64) + W64[:, j] * x64[j]).astype(F32)
    else:
        for j in range(n):
            acc = (acc + W[:, j] * xb[j]).astype(F32)
    return acc


def compare_vec(name: str, got: np.ndarray, ref: np.ndarray, rtol: float, atol: float) -> bool:
    err = np.abs(got.astype(np.float64) - ref)
    bound = atol + rtol * np.abs(ref)
    big = np.abs(ref) > 1e-2
    rel = (err[big] / np.abs(ref[big])).max() if big.any() else 0.0
    exact = int((got.astype(np.float64) == ref).sum())
    ok = bool((err <= bound).all())
    print(f"  {name:<8} n={len(ref):4d}  max abs err {err.max():.2e}  max rel err {rel:.2e}"
          f"  bit-exact vs float64 {exact}/{len(ref)}  {'ok' if ok else 'MISMATCH'}")
    return ok


def compare_step3(model, tok, token_id: int, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    refs, weights = step3_refs(model, token_id)
    got = parse_vectors(run_basic("kernels", token_id))
    assert set(got) == set(refs), (set(got), set(refs))

    print(f"\ncompare BASIC vs float64 oracle, token {token_id} = {tok.convert_ids_to_tokens(token_id)!r}")
    print(f"  criterion: |basic - ref| <= {atol:g} + {rtol:g} * |ref|")
    ok = all(compare_vec(n, got[n], refs[n], rtol, atol) for n in ["x", "rmsnorm", "wq", "wk", "w1"])

    # The matmul input BASIC actually used is its own fp32 rmsnorm output.
    # Feed that to three fp32 computations and count exact bit matches.
    xb = got["rmsnorm"]
    W = weights["wq"]
    plain = emulate_matmul_fp32(W, xb, fused=False)
    fused = emulate_matmul_fp32(W, xb, fused=True)
    blas = (torch.from_numpy(W) @ torch.from_numpy(xb)).numpy()
    print("\nsame fp32 matmul (wq), different accumulation: bits identical to BASIC")
    for label, v in [("sequential fp32, round mul then add", plain),
                     ("sequential fp32, fused multiply-add", fused),
                     ("torch fp32 matmul (BLAS order)", blas),
                     ("float64 reference rounded to fp32", refs["wq"].astype(F32))]:
        print(f"  {label:<38} {int((v == got['wq']).sum()):3d}/{len(xb)}  max diff {np.abs(v - got['wq']).max():.2e}")

    # Where the FLOPs go: every parameter is one multiply-add per token.
    n_params = sum(p.numel() for p in model.parameters())
    d, L = model.config.hidden_size, model.config.num_hidden_layers
    norm_flops = 3 * d * (2 * L + 1)
    print(f"\nFLOPs per token: matmul ~{2 * n_params:.3g}, rmsnorm ~{norm_flops:.3g}"
          f" -> matmul share {100 * 2 * n_params / (2 * n_params + norm_flops):.4f}%")
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode", help="show token ids for a string")
    p.add_argument("text")
    p = sub.add_parser("step2", help="reference logits from embedding row -> tied head")
    p.add_argument("--token", type=int, required=True)
    p.add_argument("--basic", action="store_true", help="run build/gotoken and compare")
    p = sub.add_parser("step3", help="RMSNorm and MatMul on layer 0, in isolation")
    p.add_argument("--token", type=int, required=True)
    p.add_argument("--basic", action="store_true", help="run build/gotoken kernels and compare")
    args = ap.parse_args()

    tok = load_tokenizer()
    if args.cmd == "encode":
        ids = tok.encode(args.text, add_special_tokens=False)
        for i in ids:
            print(f"{i:6d}  {tok.convert_ids_to_tokens(i)!r}")
        return 0

    model = load_model()
    if args.cmd == "step2":
        print_step2(model, tok, args.token)
        if args.basic:
            return 0 if compare_step2(model, tok, args.token) else 1
    if args.cmd == "step3":
        refs, _ = step3_refs(model, args.token)
        for name, v in refs.items():
            print(f"{name:<8} n={len(v):4d}  first 4: {np.array2string(v[:4], precision=7)}")
        if args.basic:
            return 0 if compare_step3(model, tok, args.token) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

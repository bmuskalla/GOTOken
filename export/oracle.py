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
    python oracle.py step4 --tokens 504 2644 2643 --basic  # layer 0 over a sequence vs a forward hook
    python oracle.py step5 --tokens 504 2644 2643 --steps 8 --basic  # full forward + greedy decode
    python oracle.py step6 --tokens 504 2644 2643 --steps 8 --basic  # KV cache: same bits, timing curve
    python oracle.py step7 --basic                        # tokenizer round trip vs HF on the corpus
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
    first = {int(m[1]): float(m[2]) for m in re.finditer(r"^logit\s+(\d+)\s*(\S+)", text, re.M)}
    top = [(int(m[1]), int(m[2]), float(m[3]))
           for m in re.finditer(r"^top\s+(\d+)\s+id\s+(\d+)\s+logit\s*(\S+)", text, re.M)]
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
# step 4: one transformer layer on one token, checked against a forward hook
# --------------------------------------------------------------------------- #
def rope_interleaved(vec: np.ndarray, position: int, head_dim: int, theta: float) -> np.ndarray:
    """RoPE the way run.c and the BASIC engine apply it: adjacent pairs
    (2j, 2j+1) of every head rotated by position * theta^(-2j/head_dim)."""
    out = vec.copy()
    n_heads = len(vec) // head_dim
    j = np.arange(0, head_dim, 2)
    ang = position / theta ** (j / head_dim)
    cos, sin = np.cos(ang), np.sin(ang)
    for h in range(n_heads):
        seg = vec[h * head_dim:(h + 1) * head_dim]
        a, b = seg[0::2], seg[1::2]
        out[h * head_dim:(h + 1) * head_dim:2] = a * cos - b * sin
        out[h * head_dim + 1:(h + 1) * head_dim:2] = a * sin + b * cos
    return out


def rope_theta(cfg) -> float:
    """transformers >= 5 keeps it in rope_parameters, older versions on the config."""
    rp = getattr(cfg, "rope_parameters", None)
    return float(rp["rope_theta"] if rp else cfg.rope_theta)


def step4_refs(model, tokens: list[int]) -> dict:
    """Layer-0 output at the last position from a forward hook on the real
    model run over the whole sequence, plus float64 references for the last
    token's q and k after RoPE (built from step 3's pieces)."""
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    last, position = tokens[-1], len(tokens) - 1
    refs3, weights = step3_refs(model, last)
    xb = refs3["rmsnorm"]
    refs = {
        "q_rope": rope_interleaved(weights["wq"].astype(np.float64) @ xb, position, head_dim, rope_theta(cfg)),
        "k_rope": rope_interleaved(weights["wk"].astype(np.float64) @ xb, position, head_dim, rope_theta(cfg)),
    }

    captured = {}

    def hook(module, inputs, output):
        captured["out"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([tokens]))
    finally:
        handle.remove()
    refs["layer0"] = captured["out"][0, -1].double().numpy()
    return refs


def compare_step4(model, tok, tokens: list[int], rtol: float = 1e-4, atol: float = 1e-4) -> bool:
    refs = step4_refs(model, tokens)
    got = parse_vectors(run_basic("layer", *tokens))
    print(f"\ncompare BASIC layer 0 vs HF forward hook over {len(tokens)} tokens "
          f"{[tok.convert_ids_to_tokens(t) for t in tokens]}, output at the last position")
    print(f"  criterion: |basic - ref| <= {atol:g} + {rtol:g} * |ref|")
    ok = all(compare_vec(n, got[n], refs[n], rtol, atol) for n in ["q_rope", "k_rope", "layer0"])
    x_in = step3_refs(model, tokens[-1])[0]["x"]
    print(f"  residual stream at the last position: |x_in| rms {np.sqrt((x_in ** 2).mean()):.4f} -> "
          f"|x_out| rms {np.sqrt((refs['layer0'] ** 2).mean()):.4f}")
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# step 5: full forward and greedy decoding
# --------------------------------------------------------------------------- #
def step5_logits(model, tokens: list[int]) -> np.ndarray:
    """Next-token logits after the whole model, at the last position."""
    with torch.no_grad():
        return model(input_ids=torch.tensor([tokens])).logits[0, -1].double().numpy()


def step5_greedy(model, tokens: list[int], steps: int) -> list[dict]:
    """Greedy decoding done the slow, obvious way (whole sequence re-forwarded
    per token, like the BASIC engine in step 5) so we can record the margin
    between the best and second-best logit at every step. Cross-checked
    against model.generate."""
    seq = list(tokens)
    out = []
    for _ in range(steps):
        logits = step5_logits(model, seq)
        top2 = np.argsort(-logits)[:2]
        out.append({"id": int(top2[0]), "logit": logits[top2[0]], "margin": logits[top2[0]] - logits[top2[1]]})
        seq.append(int(top2[0]))
    with torch.no_grad():
        gen = model.generate(torch.tensor([tokens]), max_new_tokens=steps, do_sample=False,
                             pad_token_id=model.config.eos_token_id)
    hf_ids = gen[0, len(tokens):].tolist()
    ours = [o["id"] for o in out][: len(hf_ids)]
    assert hf_ids == ours, f"manual greedy {ours} != model.generate {hf_ids}"
    return out


def parse_step5_gen(text: str) -> list[dict]:
    pat = (r"^gen\s+(\d+)\s+seqlen\s+(\d+)\s+id\s+(\d+)\s+logit\s*(\S+)\s+hex\s+([0-9a-f]{8})"
           r"\s+margin\s*(\S+)\s+forwards\s+(\d+)\s+secs\s*(\S+)")
    return [{"seqlen": int(m[2]), "id": int(m[3]), "logit": float(m[4]),
             "bits": np.frombuffer(bytes.fromhex(m[5]), dtype=F32)[0],
             "margin": float(m[6]), "forwards": int(m[7]), "secs": float(m[8].replace("D", "E"))}
            for m in re.finditer(pat, text, re.M)]


def compare_step5(model, tok, tokens: list[int], steps: int, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
    ok = True
    # 1. logits after the full forward
    ref = step5_logits(model, tokens)
    got = parse_step2(run_basic("forward", *tokens))
    print(f"\nfull forward over {[tok.convert_ids_to_tokens(t) for t in tokens]}: logits at the last position")
    print(f"  criterion: |basic - ref| <= {atol:g} + {rtol:g} * |ref|")
    for i, v in got["first"].items():
        d = abs(v - ref[i])
        flag = "ok" if d <= atol + rtol * abs(ref[i]) else "MISMATCH"
        ok &= flag == "ok"
        print(f"  logit {i}: basic {v:.6f}  oracle {ref[i]:.6f}  diff {d:.1e}  {flag}")
    ref_top = np.argsort(-ref)[: len(got["top"])]
    for (r, vid, val), rid in zip(got["top"], ref_top):
        d = abs(val - ref[vid])
        flag = "ok" if vid == rid and d <= atol + rtol * abs(ref[vid]) else "MISMATCH"
        ok &= flag == "ok"
        print(f"  top {r}: basic id {vid} {val:.6f}  oracle id {rid} {ref[rid]:.6f}  diff {d:.1e}  {tok.convert_ids_to_tokens(int(rid))!r}  {flag}")

    # 2. greedy decoding, token for token
    ref_gen = step5_greedy(model, tokens, steps)
    text = run_basic("generate-nocache", steps, *tokens)
    got_gen = parse_step5_gen(text)
    print(f"\ngreedy decode without KV cache, {steps} tokens")
    for s, (g, r) in enumerate(zip(got_gen, ref_gen)):
        flag = "ok" if g["id"] == r["id"] else "MISMATCH"
        ok &= flag == "ok"
        print(f"  step {s}: basic {g['id']:6d} {tok.convert_ids_to_tokens(g['id'])!r:<14} oracle {r['id']:6d}"
              f"  logit diff {abs(g['logit'] - r['logit']):.1e}  margin {r['margin']:.3f}  {g['secs']:.1f}s  {flag}")
    ok &= len(got_gen) == len(ref_gen)
    ids = tokens + [r["id"] for r in ref_gen]
    print(f"  smallest margin {min(r['margin'] for r in ref_gen):.3f}")
    print(f"  text: {tok.decode(tokens)!r} -> {tok.decode(ids[len(tokens):])!r}")
    m = re.search(r"generated\s+\d+\s+tokens with\s+(\d+)\s+forwards in\s+(\S+)", text)
    if m:
        print(f"  BASIC: {m[1]} forwards in {float(m[2]):.1f}s, {steps / float(m[2]):.2f} tok/s")
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# step 6: the KV cache. Same bits as step 5, one forward per token.
# --------------------------------------------------------------------------- #
def compare_step6(model, tok, tokens: list[int], steps: int, long_steps: int = 24) -> bool:
    ok = True
    ref_gen = step5_greedy(model, tokens, steps)
    nocache = parse_step5_gen(run_basic("generate-nocache", steps, *tokens))
    cache = parse_step5_gen(run_basic("generate", steps, *tokens))
    assert len(nocache) == len(cache) == steps, (len(nocache), len(cache))

    print(f"\nKV cache vs re-forwarding the prefix, prompt {[tok.convert_ids_to_tokens(t) for t in tokens]}, {steps} tokens")
    print("  step  seqlen  token            HF  same id  same bits   no-cache fwds  secs   cache fwds  secs   speedup")
    for s_, (a, b, r) in enumerate(zip(nocache, cache, ref_gen)):
        same_id = a["id"] == b["id"] == r["id"]
        same_bits = a["bits"] == b["bits"]
        ok &= same_id and same_bits
        print(f"  {s_:4d}  {b['seqlen']:6d}  {tok.convert_ids_to_tokens(b['id'])!r:<14} {r['id']:6d}  "
              f"{'yes' if same_id else 'NO ':>7}  {'yes' if same_bits else 'NO ':>9}   "
              f"{a['forwards']:13d}  {a['secs']:5.1f}   {b['forwards']:10d}  {b['secs']:5.2f}   {a['secs'] / b['secs']:5.1f}x")
    t_no, t_c = sum(a["secs"] for a in nocache), sum(b["secs"] for b in cache)
    f_no, f_c = sum(a["forwards"] for a in nocache), sum(b["forwards"] for b in cache)
    print(f"  total: no-cache {f_no} forwards {t_no:.1f}s ({steps / t_no:.2f} tok/s), "
          f"cache {f_c} forwards {t_c:.1f}s ({steps / t_c:.2f} tok/s), {t_no / t_c:.1f}x")

    # A longer cached run: per-token time should stay flat. Attention grows with
    # the sequence length, but it is a sliver next to the fixed matmul cost.
    long = parse_step5_gen(run_basic("generate", long_steps, *tokens))
    secs = [g["secs"] for g in long[1:]]
    print(f"\ncached run of {long_steps} tokens: per-token secs after prefill "
          f"first {secs[0]:.2f}, middle {secs[len(secs) // 2]:.2f}, last {secs[-1]:.2f} at seqlen {long[-1]['seqlen']}")
    ids = tokens + [g["id"] for g in long]
    print(f"  text: {tok.decode(tokens)!r} -> {tok.decode(ids[len(tokens):])!r}")
    hf = model.generate(torch.tensor([tokens]), max_new_tokens=long_steps, do_sample=False,
                        pad_token_id=model.config.eos_token_id)[0, len(tokens):].tolist()
    n_match = next((i for i, (a, b) in enumerate(zip(ids[len(tokens):], hf)) if a != b), long_steps)
    print(f"  matches HF greedy for {n_match}/{long_steps} tokens")
    ok &= n_match == long_steps
    print("PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# step 7: the tokenizer, against HuggingFace on the whole test corpus
# --------------------------------------------------------------------------- #
def compare_step7(tok, n_random: int = 2000) -> bool:
    import struct
    import tempfile
    from bpe_ref import Tokenizer as RefTokenizer, corpus, HAND

    tests = corpus(n_random)
    with tempfile.NamedTemporaryFile("wb", suffix=".bin", delete=False) as f:
        f.write(struct.pack("<i", len(tests)))
        for s_ in tests:
            b = s_.encode("utf-8")
            f.write(struct.pack("<i", len(b)) + b)
        batch = f.name
    text = run_basic("encode-batch", batch)
    got = [[int(x) for x in line[4:].split()] for line in text.splitlines() if line.startswith("ids:")]
    assert len(got) == len(tests), (len(got), len(tests))

    ref = RefTokenizer()
    bad = 0
    for s_, ids in zip(tests, got):
        hf = tok.encode(s_, add_special_tokens=False)
        if ids != hf:
            bad += 1
            if bad <= 10:
                print(f"  MISMATCH {s_!r}\n    basic {ids}\n    hf    {hf}\n    ref   {ref.encode(s_)}")
    m = re.search(r"encoded\s+(\d+)\s+strings in\s+(\S+)", text)
    print(f"\ntokenizer: BASIC vs HuggingFace on {len(tests)} strings "
          f"({len(HAND)} hand-written, {n_random} random): {len(tests) - bad} match")
    if m:
        print(f"  BASIC encoded {m[1]} strings in {float(m[2].replace('D', 'E')):.2f}s")

    # decode round trip through BASIC for a few strings with full byte coverage
    ok = bad == 0
    for s_ in HAND[:6]:
        ids = tok.encode(s_, add_special_tokens=False)
        out = run_basic("decode", *ids)
        # everything after the label, minus PRINT's own newline (the text may contain newlines)
        line = out.split("text: ", 1)[1] if "text: " in out else None
        if line is not None and line.endswith("\n"):
            line = line[:-1]
        same = line == s_
        ok &= same
        print(f"  decode {s_!r:<50} {'ok' if same else 'MISMATCH: ' + repr(line)}")
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
    p = sub.add_parser("step4", help="transformer layer 0 over a token sequence vs a forward hook")
    p.add_argument("--tokens", type=int, nargs="+", required=True)
    p.add_argument("--basic", action="store_true", help="run build/gotoken layer and compare")
    p = sub.add_parser("step5", help="full forward + greedy decode vs transformers")
    p.add_argument("--tokens", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--basic", action="store_true", help="run build/gotoken forward/generate and compare")
    p = sub.add_parser("step6", help="KV cache: same output as step 5, timing curve")
    p.add_argument("--tokens", type=int, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--long", type=int, default=24, help="length of the cached-only run")
    p.add_argument("--basic", action="store_true", help="run both generate modes and compare")
    p = sub.add_parser("step7", help="tokenizer: BASIC encode/decode vs HuggingFace on the corpus")
    p.add_argument("--random", type=int, default=2000)
    p.add_argument("--basic", action="store_true", help="run build/gotoken encode-batch and compare")
    args = ap.parse_args()

    tok = load_tokenizer()
    if args.cmd == "step7":
        if not args.basic:
            sys.exit("step7 compares the BASIC tokenizer; pass --basic (bpe_ref.py validates the Python reference)")
        return 0 if compare_step7(tok, args.random) else 1
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
    if args.cmd == "step4":
        refs = step4_refs(model, args.tokens)
        for name, v in refs.items():
            print(f"{name:<8} n={len(v):4d}  first 4: {np.array2string(v[:4], precision=7)}")
        if args.basic:
            return 0 if compare_step4(model, tok, args.tokens) else 1
    if args.cmd == "step5":
        ref = step5_logits(model, args.tokens)
        for r, v in enumerate(np.argsort(-ref)[:5], 1):
            print(f"top {r} id {v} logit {ref[v]:.6f}   {tok.convert_ids_to_tokens(int(v))!r}")
        for s_, g in enumerate(step5_greedy(model, args.tokens, args.steps)):
            print(f"gen {s_} id {g['id']} logit {g['logit']:.6f} margin {g['margin']:.3f}   {tok.convert_ids_to_tokens(g['id'])!r}")
        if args.basic:
            return 0 if compare_step5(model, tok, args.tokens, args.steps) else 1
    if args.cmd == "step6":
        if not args.basic:
            sys.exit("step6 is a BASIC measurement; pass --basic")
        return 0 if compare_step6(model, tok, args.tokens, args.steps, args.long) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    got = parse_step2(run_basic(token_id))
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
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encode", help="show token ids for a string")
    p.add_argument("text")
    p = sub.add_parser("step2", help="reference logits from embedding row -> tied head")
    p.add_argument("--token", type=int, required=True)
    p.add_argument("--basic", action="store_true", help="run build/gotoken and compare")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

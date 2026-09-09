# GOTOken

An LLM inference engine written in BASIC (QB64), running
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

This is a learning project, built one verified step at a time. The BASIC code
follows karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `run.c`
almost 1:1, and every step is checked against the HuggingFace model in Python
before the next one starts.

## Layout

    export/        Python: turns the HF checkpoint into flat binaries
    model/         weights.bin + tokenizer.bin (generated, not committed)
    FORMAT.md      exact byte layout of both binaries
    *.bas          the engine (coming step by step)

## Step 0: export the model

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd export
uv venv .venv
uv pip install --python .venv/bin/python torch transformers safetensors numpy huggingface_hub
.venv/bin/python export.py
```

(A recent uv can replace the two install lines with `uv sync` in `export/`.)

```bash
.venv/bin/python export.py
```

This downloads the checkpoint (~270 MB bf16), writes `model/weights.bin`
(538 MB fp32) and `model/tokenizer.bin`, and prints the header plus the first
five floats of the embedding table. Those values are the reference for the
step 1 loader; they are also recorded in `FORMAT.md`.

The script also verifies two things on every run: that the Q/K row permutation
it applies for RoPE leaves attention scores unchanged, and that token bytes it
exports rebuild the original text from HuggingFace token ids.

## Plan

| step | what | checkpoint |
|-----:|------|------------|
| 0 | Python export script | header + first 5 embedding floats printed |
| 1 | BASIC loader | first 5 floats match byte-exact |
| 2 | embedding lookup + tied output head | logits match Python for one token |
| 3 | matmul + RMSNorm kernels | match Python within ~1e-5 |
| 4 | one transformer layer at position 0 | layer-0 output matches a forward hook |
| 5 | full forward + greedy decode | token-for-token match with `transformers` |
| 6 | KV cache | same output, measurably faster |
| 7 | BPE tokenizer | round-trip matches Python tokenizer |
| 8 | sampler + REPL | temp 0 reproduces greedy |
| 9 | int8 quantization (optional) | quality holds, faster |

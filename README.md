# GOTOken

An LLM inference engine written in BASIC (QB64), running
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

This is a learning project, built one verified step at a time. The BASIC code
follows karpathy's [llama2.c](https://github.com/karpathy/llama2.c) `run.c`
almost 1:1, and every step is checked against the HuggingFace model in Python
before the next one starts.

## Layout

    export/export.py    Python: turns the HF checkpoint into flat binaries
    export/oracle.py    Python: reference values per step + comparison with the engine
    export/oracle.ipynb notebook front-end over oracle.py
    model/              weights.bin + tokenizer.bin (generated, not committed)
    FORMAT.md           exact byte layout of both binaries
    gotoken.bas         the engine's main program
    src/*.bi            declarations (TYPEs, CONSTs, shared arrays)
    src/*.bm            subs and functions
    build.sh            compiles gotoken.bas with QB64 into build/gotoken

QB64 has no modules, only textual `$INCLUDE`. Declarations must precede the
main program and subs must follow it, hence the `.bi` includes at the top of
`gotoken.bas` and the `.bm` includes at the bottom. Include paths resolve
relative to the including file.

    src/model.bi    Config TYPE, the flat weight array w(), per-tensor offsets
    src/model.bm    LoadModel, MapWeights
    src/state.bi    activations of the current forward pass (run.c RunState)
    src/kernels.bm  MatMul, RmsNorm, Softmax
    src/forward.bm  Embed, Rope, TransformerLayer, Forward, Classify
    src/generate.bm Generate (greedy decoding), BestTwo
    src/checks.bm   checkpoint printers in a format oracle.py parses
    src/util.bm     FloatHex$, Fail

## Step 0: export the model

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd export
uv venv .venv
uv pip install --python .venv/bin/python torch transformers safetensors numpy huggingface_hub
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

## Step 1: load the weights in BASIC

Uses classic [QB64 2.1](https://qb64.com). On macOS the release tarball
ships as source and builds itself with clang (needs Xcode command line tools):

```bash
curl -sSL -o /tmp/qb64.tar.gz https://github.com/QB64Official/qb64/releases/download/v2.1/qb64_dev_2022-09-08-07-14-00_47f5044_osx.tar.gz
tar xzf /tmp/qb64.tar.gz -C /tmp && mv /tmp/qb64_*_osx ~/qb64
cd ~/qb64 && find . -name "*.command" -exec chmod +x {} \;
(cd internal/c/libqb/os/osx && ./setup_build.command)
(cd internal/c/parts/video/font/ttf/os/osx && ./setup_build.command)
cp internal/source/* internal/temp/
(cd internal/c && clang++ -w qbx.cpp libqb/os/osx/libqb_setup.o parts/video/font/ttf/os/osx/src.o -framework GLUT -framework OpenGL -framework Cocoa -o ../../qb64)
```

(That is `setup_osx.command` minus the step that launches the IDE.) Then:

```bash
./build.sh && ./build/gotoken
```

The program reads the 48-byte header into a `TYPE`, computes where every
tensor starts, loads all 134.5M floats into one `SINGLE` array with a single
`GET`, and prints the first and last five floats with their raw bytes. Both
must match the `FORMAT.md` tables exactly. An optional argument overrides the
checkpoint path, like `run.c`.

Why one array: `run.c` mmaps the file and keeps `float*` pointers into it.
BASIC has no pointers, so the engine keeps one array `w()` and a `LONG`
offset per tensor. Element `(o, i)` of a `[out, in]` matrix at offset `p` is
`w(p + o * in + i)`, which is the layout the matmul in step 3 reads
contiguously.

## Step 2: embedding row to logits, no transformer

```bash
./build.sh && ./build/gotoken logits 2644
cd export && .venv/bin/python oracle.py step2 --token 2644 --basic
```

`gotoken logits <token_id>` copies that token's embedding row into `x()` and runs the
output head: `logits = wcls * x`, one dot product per vocab entry. The header
says `tied = 1`, so `wcls` is the embedding table itself, and every logit is
"how similar is this vocab row to the input row". Without a single
transformer layer the top-5 for ` cat` (id 2644) is ` cat`, `cat`, `cats`,
`Cat`, `<filename>`. The transformer's whole job, from step 4 on, is to turn
`x` from "this token" into "the next token".

The oracle computes the same dot products in float64 and compares. BASIC
accumulates 576 fp32 products in sequence and lands within about 1e-6 of the
float64 answer; what tolerance is acceptable and why exact match is impossible
is the topic of step 3, where the matmul gets tested in isolation.

`oracle.py encode "some text"` prints token ids to pick test tokens with.

## Step 3: MatMul and RMSNorm in isolation

```bash
./build.sh && ./build/gotoken kernels 2644
cd export && .venv/bin/python oracle.py step3 --token 2644 --basic
```

`gotoken kernels <token_id>` runs each kernel once on real layer-0 weights
with the token's embedding row as input, and prints every output element
with its raw fp32 bits. The oracle recomputes the same thing in float64 and
checks `|basic - ref| <= 1e-5 + 1e-5 * |ref|`.

| vector  | n    | max abs err | max rel err |
|---------|-----:|------------:|------------:|
| rmsnorm | 576  | 4.8e-7      | 5.4e-7      |
| wq      | 576  | 3.9e-6      | 8.5e-6      |
| wk      | 192  | 5.5e-6      | 9.8e-6      |
| w1      | 1536 | 1.0e-6      | 1.8e-5      |

Three things to take from this step:

**Inference is matmul.** Every parameter is one multiply-add per token, so a
forward pass is about 2 x 134.5M = 269 MFLOP of matmul against 0.1 MFLOP of
RMSNorm. The matmul share is 99.96%. Whatever is done to make this fast later
(step 9) is done to the matmul inner loop, nothing else matters.

**Exact match is impossible, and that is fine.** fp32 addition is not
associative, so summing 576 products gives a different result for every
summation order. The oracle demonstrates this by computing the identical fp32
matmul four ways and counting outputs that match BASIC bit for bit:

| computation                              | bit-identical to BASIC |
|------------------------------------------|-----------------------:|
| sequential fp32, round mul then add      | 576 / 576              |
| sequential fp32, fused multiply-add      | 243 / 576              |
| torch fp32 matmul (BLAS blocked order)   |  61 / 576              |
| float64 reference rounded to fp32        |  26 / 576              |

The first row is the proof that the BASIC loop does exactly what it says:
replay its accumulation order and you get its bits. The other rows are all
"correct" too. Correctness is a tolerance, not an equality.

**Why the tolerance has two parts.** Relative error alone fails on outputs
near zero, where terms of size 0.01 cancel to a result of size 0.0001 and
the fp32 rounding noise is large compared to the result (w1's 1.8e-5 above
is such an element). Absolute error alone would be meaningless for large
outputs. `atol + rtol * |ref|` covers both. 1e-5 is comfortable here: the
expected error of a 576-term fp32 sum is around sqrt(576) x 6e-8 x (typical
partial sum), and the observed errors sit a decade below the bound.

## Step 4: one transformer layer

```bash
./build.sh && ./build/gotoken layer 504 2644 2643 335 260
cd export && .venv/bin/python oracle.py step4 --tokens 504 2644 2643 335 260 --basic
```

`gotoken layer <id> [<id> ...]` runs layer 0 over a token sequence, token
*i* at position *i*, and prints the last token's q and k after RoPE and the
residual stream after the layer. The oracle registers a forward hook on
`model.model.layers[0]`, runs the real model on the same ids, and compares
the last position. q and k are checked against an independent numpy RoPE.

| sequence            | q after RoPE | k after RoPE | layer output |
|---------------------|-------------:|-------------:|-------------:|
| ` cat`              | 3.9e-6       | 5.5e-6       | 1.5e-5       |
| `The cat sat on the`| 4.0e-6       | 4.5e-6       | 5.7e-6       |
| 14 tokens           | 6.5e-6       | 5.9e-6       | 1.5e-5       |

(max abs error vs the reference; criterion `1e-4 + 1e-4 * |ref|`, passed
with a decade to spare.)

`TransformerLayer` in `src/forward.bm` is the body of the layer loop in
run.c's `forward()`, and reads top to bottom as the layer's data flow:

    xb  = rmsnorm(x)                  pre-norm: the sublayer sees unit-scale input
    q, k, v = wq xb, wk xb, wv xb     projections
    rope(q, k, position)              encode position as a rotation
    cache[layer][position] = k, v     keep this position's k and v
    for each head h:
        scores[t] = q_h . k_{h/3}[t] / sqrt(64)   over t = 0..position
        softmax(scores)
        xb_h = sum_t scores[t] * v_{h/3}[t]
    x += wo xb                        residual add
    xb  = rmsnorm(x)                  pre-norm again
    x += w2( silu(w1 xb) * (w3 xb) )  SwiGLU FFN, residual add

**RoPE is a rotation, not an addition.** Each head's 64 floats are 32 pairs;
pair *j* is rotated by the angle `position * theta^(-2j/64)`. Pair 0 turns
fastest, pair 31 barely moves. Because a dot product of two rotated vectors
depends only on the angle between them, `q(pos) . k(t)` depends on
`pos - t`: relative position falls out of the geometry for free. This is why
the position enters through q and k only, never through v or x. At position 0
every angle is 0 and RoPE is the identity, which is why the single-token
check is not enough and the check runs over a sequence.

**GQA.** 9 query heads, 3 key/value heads. Query head *h* reads KV head
`h \ 3`, so heads 0,1,2 share one k and v. That is the entire mechanism:
`(h \ kvMul) * headDim` in the cache index. It cuts k and v to a third of the
size, which matters because from step 6 on the KV cache is what fills memory
as the sequence grows, and the quality cost of sharing is small.

**The residual stream.** `x` is never overwritten inside a layer, only added
to. Each sublayer reads a normalized copy, computes something, and adds it
back. The norm before each sublayer (pre-norm) is what lets 30 of these
stack without the scale of `x` running away: for ` cat` the rms of `x` goes
from 0.08 to 2.5 through layer 0 alone.

**Why the check takes a sequence.** The plan said one token at position 0.
That passed, but with one token attention has one score, softmax makes it
1.0, and the head output is just v, so the attention loop, softmax, the GQA
mapping and RoPE were all running without being tested. A first attempt at
a lone token at position 7 instead read seven never-written cache rows and
failed against HuggingFace, which attends only to what exists. Feeding a
real sequence makes every line of the layer count, while the reference is
still just a hook on layer 0.

## Step 5: full forward and greedy decoding

```bash
./build.sh
./build/gotoken forward 504 2644 2643 335 260        # logits after all 30 layers
./build/gotoken generate 8 504 2644 2643 335 260     # greedy, 8 new tokens
cd export && .venv/bin/python oracle.py step5 --tokens 504 2644 2643 335 260 --steps 8 --basic
```

`Forward` in `src/forward.bm` is now run.c's `forward()`: embed, 30 layers,
final RMSNorm, classifier. `Generate` in `src/generate.bm` is the
autoregressive loop: forward the sequence, take the argmax, append it,
repeat. Prompts are token ids for now (`oracle.py encode "text"` gives
them); the tokenizer is step 7.

The oracle checks two things. The logits at the last prompt position
against the real model after all 30 layers:

| prompt               | max logit diff | top-5 ids |
|----------------------|---------------:|-----------|
| `The cat sat on the` | 1.7e-5         | identical |

And greedy decoding token for token against `model.generate(do_sample=False)`:

| prompt                            | continuation                       | steps | smallest margin |
|-----------------------------------|------------------------------------|------:|----------------:|
| `The cat sat on the`              | ` bed, and the cat sat on the`     | 8/8   | 0.189           |
| `Once upon a time, there was a`   | ` little girl named Lily. She`     | 6/6   | 0.317           |

**Greedy is the oracle because it is deterministic.** Sampling would make
two correct engines disagree by design. Argmax makes them agree exactly, as
long as the fp32 drift (about 1e-5 on the logits after 30 layers) stays
smaller than the gap between the best and second-best logit. That gap is
the `margin` column: the smallest was 0.19, four orders of magnitude above
the drift. A near-tie could flip a token in a perfectly correct engine, so
a mismatch is a bug only if the margin was comfortable, and the oracle
prints it for every step.

**Autoregression is the whole loop.** The model never produces text; it
produces one distribution over the next token, and `Generate` feeds its own
argmax back in. Note the first prompt: after ` bed,` the model walks
straight back into ` and the cat sat on the`. That is what a 135M model's
greedy decoding looks like, and step 8's sampler exists to do better.

**No KV cache yet, and it shows.** Every new token re-forwards the entire
sequence from position 0, so step *s* costs `n_prompt + s` forwards. The
per-step time in the oracle output climbs linearly, 4.8 s to 11.5 s over 8
steps, for 68 forwards and 0.12 tokens/s. A single forward is 0.95 s in
unoptimized QB64 output (no `-O` flags, bounds checks on). Step 6 removes
the quadratic term; the linear per-forward cost is step 9's problem.

**A bug that was not one.** The second prompt first reported a mismatch at
` Lily`. The engine had produced it; its logit was negative, and BASIC's
`PRINT` puts no space before a negative number, so the line read
`logit-.4843422` and the parser dropped it. Labels now end in an explicit
space. Worth knowing: the greedy pick can have a negative logit, logits are
scores, not probabilities.

## Plan

| step | what | checkpoint |
|-----:|------|------------|
| 0 | Python export script | header + first 5 embedding floats printed (done) |
| 1 | BASIC loader | first 5 floats match byte-exact (done) |
| 2 | embedding lookup + tied output head | logits match Python for one token (done) |
| 3 | matmul + RMSNorm kernels | match Python within ~1e-5 (done) |
| 4 | one transformer layer | layer-0 output matches a forward hook (done) |
| 5 | full forward + greedy decode | token-for-token match with `transformers` (done) |
| 6 | KV cache | same output, measurably faster |
| 7 | BPE tokenizer | round-trip matches Python tokenizer |
| 8 | sampler + REPL | temp 0 reproduces greedy |
| 9 | int8 quantization (optional) | quality holds, faster |

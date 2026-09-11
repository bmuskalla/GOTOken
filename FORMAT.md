# Model loading and memory layout

The engine reads a HuggingFace checkpoint directly. A model directory holds
three files in the HuggingFace snapshot layout:

    config.json          the shape and the hyperparameters
    model.safetensors    the weights
    tokenizer.json       the byte-level BPE vocab and merge rules

This document describes what the loader (`src/model.bm`, `src/tokenizer.bm`)
requires of them and how the result is laid out in memory, which is what
the rest of the engine sees.

## What the loader supports

The engine implements the Llama architecture as run.c does: pre-norm
attention with RoPE and grouped-query heads, a SwiGLU feed-forward, RMSNorm,
and an output head that may be tied to the embedding table. `config.json`
must therefore say `model_type: llama`, `hidden_act: silu`, and have no
`rope_scaling`. The loader reads the dimensions, layer and head counts, the
vocab size, the context length, `tie_word_embeddings`, `rope_theta` and
`rms_norm_eps` from it; nothing about the shape is hardcoded.

Weights must be in one `model.safetensors` file with the standard Llama
tensor names, as bf16 or fp32. The tokenizer must be a byte-level BPE with
the GPT-2 pre-tokenizer regex, which is what `tokenizer.json` describes for
GPT-2-style and Llama-3-style tokenizers.

## Weights

A safetensors file is an 8-byte little-endian header length, a JSON header
mapping each tensor name to `{dtype, shape, data_offsets}`, then the raw
tensor bytes. The loader parses the header with the engine's own JSON
parser and reads each tensor with a `GET` at its offset.

**bf16 to fp32** is exact: a bf16 is the top 16 bits of an fp32 (same sign
and 8-bit exponent, 7 of the 23 mantissa bits). The loader reads the tensor
as 16-bit integers, shifts each left by 16 into an `_UNSIGNED LONG` array,
and copies the bytes into the `SINGLE` weight array with `_MEMCOPY`.

### Memory layout

All weights live in one flat `SINGLE` array `w()`, and each tensor has a
`LONG` start offset (`offWq(l)` and so on). This is run.c's
`memory_map_weights` with pointers replaced by integers. Tensors are placed
in this order, interleaved per layer:

| tensor           | HF name                         | shape          |
|------------------|---------------------------------|----------------|
| token_embedding  | model.embed_tokens.weight       | [vocab, dim]   |
| per layer:       |                                 |                |
| &nbsp;&nbsp;rms_att | input_layernorm.weight       | [dim]          |
| &nbsp;&nbsp;wq      | self_attn.q_proj.weight      | [dim, dim]     |
| &nbsp;&nbsp;wk      | self_attn.k_proj.weight      | [kv_dim, dim]  |
| &nbsp;&nbsp;wv      | self_attn.v_proj.weight      | [kv_dim, dim]  |
| &nbsp;&nbsp;wo      | self_attn.o_proj.weight      | [dim, dim]     |
| &nbsp;&nbsp;rms_ffn | post_attention_layernorm.weight | [dim]       |
| &nbsp;&nbsp;w1 (gate) | mlp.gate_proj.weight       | [hidden, dim]  |
| &nbsp;&nbsp;w2 (down) | mlp.down_proj.weight       | [dim, hidden]  |
| &nbsp;&nbsp;w3 (up)   | mlp.up_proj.weight         | [hidden, dim]  |
| rms_final        | model.norm.weight               | [dim]          |
| wcls             | lm_head.weight                  | [vocab, dim], only when not tied |

with `head_dim = dim / n_heads` and `kv_dim = n_kv_heads * head_dim`. When
the embeddings are tied, `wcls` is the embedding table itself. The w1/w2/w3
naming follows llama2.c so the FFN reads like `run.c`:
`w2( silu(w1 x) * (w3 x) )`.

### Matrix layout

Every matrix is stored **row-major with shape [out, in]**, exactly as
HuggingFace stores `nn.Linear` weights and exactly as `run.c` expects:

    y[o] = sum over i of  w(p + o*in + i) * x[i]

for a matrix starting at offset `p`. The inner loop walks memory
contiguously. No transposition is needed.

### Q/K row permutation (RoPE convention)

HuggingFace applies RoPE with `rotate_half`: within each head the rotated
pair is `(j, j + head_dim/2)`. llama2.c rotates adjacent pairs `(2j, 2j+1)`.
To keep the BASIC RoPE identical to `run.c`, the loader reorders the rows of
`wq` and `wk` per head as it converts them (llama2.c's `permute_reverse`):

    new row 2j     = old row j
    new row 2j + 1 = old row j + head_dim/2

Attention scores are per-head dot products, invariant under a permutation
applied to both q and k, so every layer output and every logit stays
identical to HuggingFace. Only the raw q and k vectors differ from a
`q_proj` / `k_proj` forward hook by this permutation.

## Tokenizer

`model.vocab` in `tokenizer.json` maps token strings to ids and
`model.merges` lists the merge rules in rank order. Both are written in
GPT-2's byte-level alphabet: each character stands for one byte, printable
Latin-1 characters for themselves and the remaining 68 bytes for codepoints
U+0100 and up, so a space appears as `Ġ`. The loader inverts that map to get
raw bytes, which is what the engine tokenizes.

In memory:

* `tokStr(id)`: the token's bytes. `added_tokens` (the special tokens) are
  literal text, not byte-level encoded.
* `tokScore(id)`: `-rank` of the merge rule that produces the token, so
  llama2.c's "merge the best-scoring adjacent pair" loop reproduces the
  tokenizer's rank order. Tokens no merge produces (the single-byte tokens
  and the specials) get a score far below any rank and are never merged
  into.
* `byteId(b)`: the id of the single-byte token for byte `b`, or -1. A vocab
  trained on UTF-8 text may lack tokens for bytes that never occur in it;
  such bytes are dropped, as HuggingFace does.
* a hash table from bytes to id.
* the codepoint ranges for `\p{L}`, `\p{N}` and `\s` (the Unicode
  White_Space property) from `src/unicode.bm`, for the pre-tokenizer regex.

The pre-tokenizer is the GPT-2 regex. Other steps listed in
`tokenizer.json`'s `pre_tokenizer` are not implemented; a `Digits` step is
harmless when no merge rule involves a digit, which the loader does not
check.

## Sanity check

`gotoken info` prints the shape read from `config.json` and the first and
last five weights in memory with their raw bytes. For a bf16 checkpoint the
low two bytes of every weight are zero. The head shows the first tensor
landed at offset 0; the tail shows every tensor size in between was added
up correctly. Compare against the same values read from the checkpoint with
any other tool.

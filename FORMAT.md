# Model loading and memory layout

The engine reads the HuggingFace checkpoint directly. A model directory
holds three files in the HuggingFace snapshot layout:

    config.json          the shape: dims, layers, heads, vocab, rope_theta, eps
    model.safetensors    the weights, bf16, 269 MB
    tokenizer.json       the byte-level BPE vocab and merge rules

`fetch-model.sh` downloads them. This document describes what the loader
(`src/model.bm`, `src/tokenizer.bm`) makes of them and how the result is
laid out in memory, which is what the rest of the engine sees.

## config.json

| key                      | field           | SmolLM2-135M |
|--------------------------|-----------------|-------------:|
| hidden_size              | dim             | 576 |
| intermediate_size        | hidden_dim      | 1536 |
| num_hidden_layers        | n_layers        | 30 |
| num_attention_heads      | n_heads         | 9 |
| num_key_value_heads      | n_kv_heads      | 3 |
| vocab_size               | vocab_size      | 49152 |
| max_position_embeddings  | seq_len         | 8192 |
| tie_word_embeddings      | tied            | true |
| rope_theta               | rope_theta      | 100000 |
| rms_norm_eps             | rms_norm_eps    | 1e-5 |

`model_type` must be `llama`, `hidden_act` must be `silu`, `rope_scaling`
must be null. Note `rope_theta`: llama2.c hardcodes 10000; using that here
would produce plausible-looking garbage.

Derived values:

    head_dim = dim / n_heads              = 64
    kv_dim   = n_kv_heads * head_dim      = 192
    kv_mul   = n_heads / n_kv_heads       = 3   (query heads per KV head)

`seq_len` is the model's trained context. The engine allocates a smaller
KV cache (`maxSeq` in `src/forward.bm`, 1024).

## model.safetensors

An 8-byte little-endian header length, a JSON header mapping each tensor
name to `{dtype, shape, data_offsets}`, then the raw tensor bytes. The
loader parses the header with the engine's own JSON parser and reads each
tensor with a `GET` at its offset.

**bf16 to fp32** is exact: a bf16 is the top 16 bits of an fp32 (same sign
and 8-bit exponent, 7 of the 23 mantissa bits). The loader reads the tensor
as 16-bit integers, shifts each left by 16 into an `_UNSIGNED LONG` array,
and copies the bytes into the `SINGLE` weight array with `_MEMCOPY`. The
low two bytes of every weight are zero.

### Memory layout

All weights live in one flat `SINGLE` array `w()`, and each tensor has a
`LONG` start offset (`offWq(l)` and so on). This is run.c's
`memory_map_weights` with pointers replaced by integers. Tensors are placed
in this order, interleaved per layer:

| tensor           | HF name                    | shape            | floats     |
|------------------|----------------------------|------------------|-----------:|
| token_embedding  | model.embed_tokens.weight  | [vocab, dim]     | 28,311,552 |
| per layer, 30 times: |                        |                  |            |
| &nbsp;&nbsp;rms_att | input_layernorm.weight  | [dim]            | 576 |
| &nbsp;&nbsp;wq      | self_attn.q_proj.weight | [dim, dim]       | 331,776 |
| &nbsp;&nbsp;wk      | self_attn.k_proj.weight | [kv_dim, dim]    | 110,592 |
| &nbsp;&nbsp;wv      | self_attn.v_proj.weight | [kv_dim, dim]    | 110,592 |
| &nbsp;&nbsp;wo      | self_attn.o_proj.weight | [dim, dim]       | 331,776 |
| &nbsp;&nbsp;rms_ffn | post_attention_layernorm.weight | [dim]    | 576 |
| &nbsp;&nbsp;w1 (gate) | mlp.gate_proj.weight  | [hidden, dim]    | 884,736 |
| &nbsp;&nbsp;w2 (down) | mlp.down_proj.weight  | [dim, hidden]    | 884,736 |
| &nbsp;&nbsp;w3 (up)   | mlp.up_proj.weight    | [hidden, dim]    | 884,736 |
| rms_final        | model.norm.weight          | [dim]            | 576 |
| wcls             | lm_head.weight             | [vocab, dim]     | only if not tied; here wcls = token_embedding |

Total: 134,515,008 floats, 538 MB. The w1/w2/w3 naming follows llama2.c so
the FFN reads like `run.c`: `w2( silu(w1 x) * (w3 x) )`.

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

## tokenizer.json

The `model.vocab` object maps token strings to ids and `model.merges` lists
the merge rules in rank order. Both are written in GPT-2's byte-level
alphabet: each character stands for one byte, printable Latin-1 characters
for themselves and the 68 others (controls, space, 0x7f to 0xa0, 0xad) for
codepoints U+0100 and up, so a space appears as `Ġ`. The loader inverts
that map to get raw bytes, which is what the engine tokenizes.

In memory:

* `tokStr(id)`: the token's bytes. `added_tokens` (the 17 specials, ids 0
  to 16; `<|endoftext|>` is id 0 and serves as BOS and EOS) are literal
  text, not byte-level encoded.
* `tokScore(id)`: `-rank` of the merge rule that produces the token, so
  llama2.c's "merge the best-scoring adjacent pair" loop reproduces GPT-2's
  rank order; `-1e30` for the 235 byte tokens and the specials, which no
  merge produces. Every one of the 48900 rules maps to exactly one token.
* `byteId(b)`: the id of the single-byte token for byte `b`, or -1. 21 byte
  values have no token (`0x04 0x06 0x13 0x14 0x16 0x1d 0xc0 0xc1 0xf1 0xf2
  0xf5..0xff`); they cannot appear in UTF-8 training text. HuggingFace drops
  such bytes silently and so does the engine.
* a hash table from bytes to id (FNV-1a, linear probing).
* the codepoint ranges for `\p{L}` (648), `\p{N}` (134) and `\s` (10, the
  Unicode White_Space property) from `src/unicode.bm`, for the GPT-2
  pre-tokenizer regex.

The pre-tokenizer is the GPT-2 regex alone. `tokenizer.json` also lists a
`Digits(individual_digits)` step, but no merge rule and no multi-character
token contains a numeric character, so digits come out one token each
either way, and transformers 5 drops the step when loading.

## Reference values

For a quick sanity check, `gotoken info` prints the first and last five
weights in memory. Expected, as fp32 with the bytes in memory order:

| weight | value | bytes |
|---|---|---|
| w(0), embedding row 0 | -0.11767578125 | `00 00 f1 bd` |
| w(1) | 0.02783203125 | `00 00 e4 3c` |
| w(2) | 0.048095703125 | `00 00 45 3d` |
| w(3) | -0.0079345703125 | `00 00 02 bc` |
| w(4) | -0.05615234375 | `00 00 66 bd` |
| w(n-5) .. w(n-1), tail of rms_final | 2.28125, 1.90625, 1.8671875, 1.90625, 1.984375 | `00 00 12 40` ... `00 00 fe 3f` |

The head proves the first tensor landed at offset 0; the tail proves every
tensor size in between was added up correctly.

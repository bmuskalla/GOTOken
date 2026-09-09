# Binary formats

Both files are produced by `export/export.py`. Everything is little-endian.
The BASIC side never parses anything cleverer than a stream of 4-byte
INT32/SINGLE values followed by raw bytes.

## weights.bin

### Header (48 bytes)

| offset | field          | type    | SmolLM2-135M |
|-------:|----------------|---------|-------------:|
|      0 | magic          | INT32   | 1263490119 (`b"GTOK"`) |
|      4 | version        | INT32   | 1 |
|      8 | dim            | INT32   | 576 |
|     12 | hidden_dim     | INT32   | 1536 |
|     16 | n_layers       | INT32   | 30 |
|     20 | n_heads        | INT32   | 9 |
|     24 | n_kv_heads     | INT32   | 3 |
|     28 | vocab_size     | INT32   | 49152 |
|     32 | seq_len        | INT32   | 8192 |
|     36 | tied           | INT32   | 1 |
|     40 | rope_theta     | SINGLE  | 100000.0 |
|     44 | rms_norm_eps   | SINGLE  | 1e-5 |

Every value is read from `config.json` at export time. The two SINGLE fields
exist because SmolLM2 does not use llama2.c's defaults: `rope_theta` is
100000, not 10000. Hardcoding 10000 would produce plausible-looking garbage
from step 4 onward.

`seq_len` is the model's trained context. The engine may allocate a smaller
KV cache.

Derived values the loader will need:

    head_dim = dim / n_heads              = 64
    kv_dim   = n_kv_heads * head_dim      = 192
    kv_mul   = n_heads / n_kv_heads       = 3   (query heads per KV head)

### Weight blobs (fp32, in this order, no padding)

| blob             | shape            | floats     |
|------------------|------------------|-----------:|
| token_embedding  | [vocab, dim]     | 28,311,552 |
| per layer, 30 times: |              |            |
| &nbsp;&nbsp;rms_att | [dim]         | 576 |
| &nbsp;&nbsp;wq      | [dim, dim]    | 331,776 |
| &nbsp;&nbsp;wk      | [kv_dim, dim] | 110,592 |
| &nbsp;&nbsp;wv      | [kv_dim, dim] | 110,592 |
| &nbsp;&nbsp;wo      | [dim, dim]    | 331,776 |
| &nbsp;&nbsp;rms_ffn | [dim]         | 576 |
| &nbsp;&nbsp;w1 (gate) | [hidden, dim] | 884,736 |
| &nbsp;&nbsp;w2 (down) | [dim, hidden] | 884,736 |
| &nbsp;&nbsp;w3 (up)   | [hidden, dim] | 884,736 |
| rms_final        | [dim]            | 576 |
| wcls             | [vocab, dim]     | only if tied == 0 (absent here) |

Total: 134,515,008 floats, 538,060,032 bytes of weights, file size 538,060,080.

Mapping from HuggingFace names: `input_layernorm` = rms_att, `q/k/v/o_proj` =
wq/wk/wv/wo, `post_attention_layernorm` = rms_ffn, `gate_proj` = w1,
`down_proj` = w2, `up_proj` = w3, `model.norm` = rms_final. The w1/w2/w3
naming follows llama2.c so the BASIC FFN reads the same as `run.c`:
`w2( silu(w1 x) * (w3 x) )`.

Unlike llama2.c, blobs are interleaved per layer rather than grouped by kind.
The loader reads layer 0 completely, then layer 1, and so on.

### Matrix layout

Every matrix is stored **row-major with shape [out, in]**, exactly as
HuggingFace stores `nn.Linear` weights and exactly as `run.c` expects:

    y[o] = sum over i of  W[o*in + i] * x[i]

The inner loop over `i` walks memory contiguously. No transposition is applied
at export; the "pre-transposed" requirement is met by the fact that the HF
layout already is the layout a dot-product-per-output-row matmul wants.

For a 1D BASIC array this is a direct `GET`. For a 2D array note that QB64
stores the **first** subscript fastest, so the natural declaration is
`DIM w(0 TO in - 1, 0 TO out - 1)` and the element is `w(i, o)`. Step 1
verifies whichever choice is made.

### Q/K row permutation (RoPE convention)

HuggingFace applies RoPE with `rotate_half`: within each head the rotated pair
is `(j, j + head_dim/2)`. llama2.c rotates adjacent pairs `(2j, 2j+1)`. To keep
the BASIC RoPE identical to `run.c`, the exporter reorders the output rows of
`wq` and `wk` per head (llama2.c's `permute_reverse`):

    new row 2j     = old row j
    new row 2j + 1 = old row j + head_dim/2

Attention scores are per-head dot products, which are invariant under a
permutation applied to both q and k, so every layer output and every logit
stays bit-identical to HuggingFace. Only the raw `q_proj` / `k_proj` outputs
differ from a forward hook by this permutation. The exporter checks this
numerically on layer 0 every run.

## tokenizer.bin

    INT32  vocab_size            49152
    INT32  max_token_length      81   (bytes; size your read buffer with this)
    repeated vocab_size times, in token-id order:
        SINGLE score
        INT32  length
        BYTE[length] raw UTF-8 bytes of the token

This is the llama2.c `tokenizer.bin` scheme with the vocab size prepended.

* Bytes are real bytes. The GPT-2 byte-to-unicode trick used inside
  `tokenizer.json` (space shown as `Ġ`, etc.) is undone at export.
* Scores emulate GPT-2 merge ranks for llama2.c's "merge the best-scoring
  adjacent pair" loop: a token produced by merge rule number `r` has
  `score = -r`, so lower rank wins as the highest score. Every one of the
  48900 merge rules maps to exactly one token.
* Tokens that no merge produces (the 235 single-byte tokens and the 17
  special tokens, ids 0 to 16) have `score = -1e30`. The encoder must never
  merge into them.
* `<|endoftext|>` is id 0 and serves as both BOS and EOS.
* 21 byte values have no token at all: `0x04 0x06 0x13 0x14 0x16 0x1d 0xc0
  0xc1 0xf1 0xf2 0xf5..0xff`. They cannot appear in the UTF-8 training data.
  HuggingFace drops such bytes silently; the byte fallback in step 7 does the
  same.

## Step 1 checkpoint values

First 5 floats of `token_embedding` (row 0), as stored:

| index | value            | bytes on disk |
|------:|------------------|---------------|
| 0     | -0.11767578125   | `00 00 f1 bd` |
| 1     |  0.02783203125   | `00 00 e4 3c` |
| 2     |  0.048095703125  | `00 00 45 3d` |
| 3     | -0.0079345703125 | `00 00 02 bc` |
| 4     | -0.05615234375   | `00 00 66 bd` |

The low two bytes are always zero: the checkpoint is bf16, and bf16 to fp32
just pads the mantissa with 16 zero bits, so the conversion is exact.

' state.bi - the activations of the current forward pass (run.c's RunState).
' Sized by AllocRunState once the config is known.

REDIM SHARED x(0 TO 0) AS SINGLE '      the residual stream, [dim]
REDIM SHARED xb(0 TO 0) AS SINGLE '     x after a norm, input to a sublayer, [dim]
REDIM SHARED xb2(0 TO 0) AS SINGLE '    a sublayer's output before the residual add, [dim]
REDIM SHARED q(0 TO 0) AS SINGLE '      query, [dim]
REDIM SHARED k(0 TO 0) AS SINGLE '      key, [kv_dim]
REDIM SHARED v(0 TO 0) AS SINGLE '      value, [kv_dim]
REDIM SHARED att(0 TO 0) AS SINGLE '    attention scores, [n_heads * max_seq]
REDIM SHARED hb(0 TO 0) AS SINGLE '     FFN hidden buffer, [hidden_dim]
REDIM SHARED hb2(0 TO 0) AS SINGLE '    FFN hidden buffer, [hidden_dim]
REDIM SHARED logits(0 TO 0) AS SINGLE ' one score per vocab entry, [vocab]

' Keys and values of every position seen so far, per layer:
' [n_layers][max_seq][kv_dim], flat, like run.c's key_cache / value_cache.
' Attention at position pos reads rows 0..pos of the current layer. Until
' step 6 the rows are recomputed every forward pass; step 6 keeps them.
DIM SHARED maxSeq AS LONG
REDIM SHARED keyCache(0 TO 0) AS SINGLE
REDIM SHARED valueCache(0 TO 0) AS SINGLE

' The token sequence of the last Generate call (prompt + generated).
REDIM SHARED genSeq(0 TO 0) AS LONG
DIM SHARED genLen AS LONG

' Generate output mode: 0 = one "gen ..." line per token (for the oracle),
' 1 = stream the decoded text and stop at <|endoftext|> (the REPL).
DIM SHARED genStream AS LONG

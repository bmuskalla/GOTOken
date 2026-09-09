' state.bi - the activations of the current forward pass (run.c's RunState).
' Sized by AllocRunState once the config is known.

REDIM SHARED x(0 TO 0) AS SINGLE '      the residual stream, [dim]
REDIM SHARED xb(0 TO 0) AS SINGLE '     x after a norm, input to a sublayer, [dim]
REDIM SHARED q(0 TO 0) AS SINGLE '      query, [dim]
REDIM SHARED k(0 TO 0) AS SINGLE '      key, [kv_dim]
REDIM SHARED hb(0 TO 0) AS SINGLE '     FFN hidden buffer, [hidden_dim]
REDIM SHARED logits(0 TO 0) AS SINGLE ' one score per vocab entry, [vocab]

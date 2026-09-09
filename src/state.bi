' state.bi - the activations of the current forward pass (run.c's RunState).
' Sized by AllocRunState once the config is known.

REDIM SHARED x(0 TO 0) AS SINGLE '      the residual stream, [dim]
REDIM SHARED logits(0 TO 0) AS SINGLE ' one score per vocab entry, [vocab]

' sampler.bi - how a token is chosen from logits (run.c's Sampler struct).

DIM SHARED samplerTemperature AS SINGLE ' 0 = greedy (argmax); higher = flatter distribution
DIM SHARED samplerTopP AS SINGLE '        nucleus: sample only from the smallest set whose probs sum to >= topP (1 = off)
DIM SHARED samplerTopK AS LONG '          sample only from the k most likely tokens (0 = off)
DIM SHARED rngState AS _UNSIGNED _INTEGER64 ' xorshift64* state, set by the seed

REDIM SHARED probs(0 TO 0) AS SINGLE '    softmax(logits / temperature), [vocab]
REDIM SHARED candIdx(0 TO 0) AS LONG '    candidate token ids after cutoff, sorted by prob desc
REDIM SHARED candProb(0 TO 0) AS SINGLE

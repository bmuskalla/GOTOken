' tokenizer.bi - the vocabulary and the tables the tokenizer needs.
' Loaded from tokenizer.bin by LoadTokenizer (tokenizer.bm).

DIM SHARED tokVocabSize AS LONG, tokMaxLen AS LONG
REDIM SHARED tokStr(0 TO 0) AS STRING '   raw bytes of each token, by id
REDIM SHARED tokScore(0 TO 0) AS SINGLE ' -merge_rank, or -1e30 if no merge produces it
DIM SHARED byteId(0 TO 255) AS LONG '     id of the single-byte token for each byte, -1 if none

' string -> id, open addressing with linear probing. Power of two, > 2x vocab.
CONST HASH_SIZE = 131072
CONST HASH_MASK = HASH_SIZE - 1
REDIM SHARED hashId(0 TO HASH_MASK) AS LONG ' -1 = empty slot

' Codepoint ranges (inclusive, sorted) for the three character classes of
' the GPT-2 pre-tokenizer regex: \p{L}, \p{N}, \s.
DIM SHARED nLetterRanges AS LONG, nNumberRanges AS LONG, nSpaceRanges AS LONG
REDIM SHARED letterLo(0 TO 0) AS LONG, letterHi(0 TO 0) AS LONG
REDIM SHARED numberLo(0 TO 0) AS LONG, numberHi(0 TO 0) AS LONG
REDIM SHARED spaceLo(0 TO 0) AS LONG, spaceHi(0 TO 0) AS LONG

' Character classes as returned by CharClass
CONST CLS_SPACE = 0, CLS_LETTER = 1, CLS_NUMBER = 2, CLS_OTHER = 3

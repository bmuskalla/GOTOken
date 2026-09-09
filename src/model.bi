' model.bi - the model's shape and its weights.
' Included at the top of gotoken.bas. Declarations only; code is in model.bm.

' ---------------------------------------------------------------------------
' Config: the 48-byte header of weights.bin, laid out exactly as on disk.
' One GET fills the whole record. This is run.c's `Config` struct plus the two
' floats SmolLM2 needs that llama2.c hardcodes (rope_theta, rms_norm_eps).
' ---------------------------------------------------------------------------
TYPE Config
    magic AS LONG
    version AS LONG
    modelDim AS LONG ' "dim" is a reserved word in BASIC
    hiddenDim AS LONG
    nLayers AS LONG
    nHeads AS LONG
    nKvHeads AS LONG
    vocabSize AS LONG
    seqLen AS LONG
    tied AS LONG
    ropeTheta AS SINGLE
    rmsNormEps AS SINGLE
END TYPE

CONST MAGIC = 1263490119 ' the bytes "GTOK" read as a little-endian INT32
CONST VERSION = 1
CONST HEADER_BYTES = 48

DIM SHARED cfg AS Config
DIM SHARED headDim AS LONG, kvDim AS LONG, kvMul AS LONG
DIM SHARED nParams AS LONG

' ---------------------------------------------------------------------------
' The model is one flat SINGLE array plus named offsets into it.
'
' run.c mmaps the file and sets float* pointers into the mapping
' (memory_map_weights). BASIC has no pointers, so the analogue is a single
' array w() holding every parameter, and LONG offsets that say where each
' named tensor starts. Element (o, i) of a matrix [out, in] that starts at
' offset p lives at w(p + o * in + i).
' ---------------------------------------------------------------------------
REDIM SHARED w(0 TO 0) AS SINGLE

DIM SHARED offTokenEmbedding AS LONG
REDIM SHARED offRmsAtt(0 TO 0) AS LONG, offWq(0 TO 0) AS LONG, offWk(0 TO 0) AS LONG
REDIM SHARED offWv(0 TO 0) AS LONG, offWo(0 TO 0) AS LONG, offRmsFfn(0 TO 0) AS LONG
REDIM SHARED offW1(0 TO 0) AS LONG, offW2(0 TO 0) AS LONG, offW3(0 TO 0) AS LONG
DIM SHARED offRmsFinal AS LONG, offWcls AS LONG

' GOTOken - an LLM inference engine in BASIC, running SmolLM2-135M.
' Follows karpathy's llama2.c run.c as closely as the language allows.
'
' Step 1: load weights.bin and prove the bytes arrived intact.
'
' Build (see build.sh):  ~/qb64/qb64 -x -c gotoken.bas -o build/gotoken

$CONSOLE:ONLY
OPTION _EXPLICIT

' ---------------------------------------------------------------------------
' Config: the 48-byte header of weights.bin, laid out exactly as on disk.
' One GET fills the whole record. This is run.c's `Config` struct plus the two
' floats SmolLM2 needs that llama2.c hardcodes (rope_theta, rms_norm_eps).
' ---------------------------------------------------------------------------
TYPE Config
    magic AS LONG
    version AS LONG
    modelDim AS LONG
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

' ---------------------------------------------------------------------------
' The model is one flat SINGLE array plus named offsets into it.
'
' run.c mmaps the file and sets float* pointers into the mapping
' (memory_map_weights). BASIC has no pointers, so the analogue is a single
' array w() holding every parameter, and LONG offsets that say where each
' named tensor starts. Element (o, i) of a matrix [out, in] that starts at
' offset p lives at w(p + o * in + i).
' ---------------------------------------------------------------------------
DIM SHARED cfg AS Config
DIM SHARED headDim AS LONG, kvDim AS LONG, kvMul AS LONG
DIM SHARED nParams AS LONG

REDIM SHARED w(0 TO 0) AS SINGLE

DIM SHARED offTokenEmbedding AS LONG
REDIM SHARED offRmsAtt(0 TO 0) AS LONG, offWq(0 TO 0) AS LONG, offWk(0 TO 0) AS LONG
REDIM SHARED offWv(0 TO 0) AS LONG, offWo(0 TO 0) AS LONG, offRmsFfn(0 TO 0) AS LONG
REDIM SHARED offW1(0 TO 0) AS LONG, offW2(0 TO 0) AS LONG, offW3(0 TO 0) AS LONG
DIM SHARED offRmsFinal AS LONG, offWcls AS LONG

' ---------------------------------------------------------------------------
' main
' ---------------------------------------------------------------------------
DIM t0 AS DOUBLE, secs AS DOUBLE, i AS LONG, path AS STRING

' Like run.c: argv[1] is the checkpoint. QB64 chdirs into the executable's
' folder at startup, so the default is resolved from the launch directory.
path = COMMAND$
IF path = "" THEN path = _STARTDIR$ + "/model/weights.bin"

t0 = TIMER(0.001)
LoadModel path
secs = TIMER(0.001) - t0

PRINT "weights.bin header"
PRINT "  magic        "; cfg.magic
PRINT "  version      "; cfg.version
PRINT "  dim          "; cfg.modelDim
PRINT "  hidden_dim   "; cfg.hiddenDim
PRINT "  n_layers     "; cfg.nLayers
PRINT "  n_heads      "; cfg.nHeads
PRINT "  n_kv_heads   "; cfg.nKvHeads
PRINT "  vocab_size   "; cfg.vocabSize
PRINT "  seq_len      "; cfg.seqLen
PRINT "  tied         "; cfg.tied
PRINT "  rope_theta   "; cfg.ropeTheta
PRINT "  rms_norm_eps "; cfg.rmsNormEps
PRINT "  derived: head_dim ="; headDim; " kv_dim ="; kvDim; " kv_mul ="; kvMul
PRINT
PRINT "loaded"; nParams; "floats ("; nParams * 4 \ 1048576; "MB ) in"; secs; "s"
PRINT
PRINT "first 5 floats of token_embedding (w(offTokenEmbedding + i)):"
FOR i = 0 TO 4
    PRINT "  ["; i; "] "; CDBL(w(offTokenEmbedding + i)); "  hex "; FloatHex$(w(offTokenEmbedding + i))
NEXT
PRINT
PRINT "last 5 floats of the file (tail of rms_final, w(nParams - 5 + i)):"
FOR i = 0 TO 4
    PRINT "  ["; i; "] "; CDBL(w(nParams - 5 + i)); "  hex "; FloatHex$(w(nParams - 5 + i))
NEXT
SYSTEM

' ---------------------------------------------------------------------------
' LoadModel: read the header, size the array, slurp every float.
' Equivalent to run.c read_checkpoint + memory_map_weights.
' ---------------------------------------------------------------------------
SUB LoadModel (path AS STRING)
    DIM f AS LONG, fileBytes AS _INTEGER64
    f = FREEFILE
    OPEN path FOR BINARY AS #f
    fileBytes = LOF(f)
    IF fileBytes < HEADER_BYTES THEN Fail "file too small: " + path

    GET #f, , cfg ' the whole 48-byte record in one read
    IF cfg.magic <> MAGIC THEN Fail "bad magic " + STR$(cfg.magic) + " (not a GOTOken weights file, or wrong endianness)"
    IF cfg.version <> VERSION THEN Fail "unsupported version " + STR$(cfg.version)

    headDim = cfg.modelDim \ cfg.nHeads
    kvDim = cfg.nKvHeads * headDim
    kvMul = cfg.nHeads \ cfg.nKvHeads

    MapWeights
    IF fileBytes <> HEADER_BYTES + 4 * CDBL(nParams) THEN
        Fail "size mismatch: header implies " + STR$(HEADER_BYTES + 4 * CDBL(nParams)) + " bytes, file has " + STR$(fileBytes)
    END IF

    ' One GET on an array reads the whole array: nParams * 4 bytes straight
    ' from the file into memory. SINGLE is IEEE 754 fp32 and the file is
    ' little-endian fp32, so on any little-endian CPU this is byte-exact.
    REDIM w(0 TO nParams - 1) AS SINGLE
    GET #f, HEADER_BYTES + 1, w()
    IF LOC(f) <> fileBytes THEN Fail "short read: " + STR$(LOC(f)) + " of " + STR$(fileBytes)
    CLOSE #f
END SUB

' ---------------------------------------------------------------------------
' MapWeights: walk the blob order from FORMAT.md, assigning each tensor its
' start offset. This is memory_map_weights from run.c with pointer bumps
' replaced by integer bumps. Blobs are interleaved per layer.
' ---------------------------------------------------------------------------
SUB MapWeights
    DIM p AS LONG, l AS LONG
    DIM d AS LONG, hd AS LONG, kv AS LONG
    d = cfg.modelDim: hd = cfg.hiddenDim: kv = kvDim

    REDIM offRmsAtt(0 TO cfg.nLayers - 1) AS LONG, offWq(0 TO cfg.nLayers - 1) AS LONG
    REDIM offWk(0 TO cfg.nLayers - 1) AS LONG, offWv(0 TO cfg.nLayers - 1) AS LONG
    REDIM offWo(0 TO cfg.nLayers - 1) AS LONG, offRmsFfn(0 TO cfg.nLayers - 1) AS LONG
    REDIM offW1(0 TO cfg.nLayers - 1) AS LONG, offW2(0 TO cfg.nLayers - 1) AS LONG
    REDIM offW3(0 TO cfg.nLayers - 1) AS LONG

    p = 0
    offTokenEmbedding = p: p = p + cfg.vocabSize * d
    FOR l = 0 TO cfg.nLayers - 1
        offRmsAtt(l) = p: p = p + d
        offWq(l) = p: p = p + d * d '       [dim, dim]
        offWk(l) = p: p = p + kv * d '      [kv_dim, dim]
        offWv(l) = p: p = p + kv * d '      [kv_dim, dim]
        offWo(l) = p: p = p + d * d '       [dim, dim]
        offRmsFfn(l) = p: p = p + d
        offW1(l) = p: p = p + hd * d '      gate [hidden, dim]
        offW2(l) = p: p = p + d * hd '      down [dim, hidden]
        offW3(l) = p: p = p + hd * d '      up   [hidden, dim]
    NEXT
    offRmsFinal = p: p = p + d
    IF cfg.tied THEN
        offWcls = offTokenEmbedding ' tied: the output head IS the embedding table
    ELSE
        offWcls = p: p = p + cfg.vocabSize * d
    END IF
    nParams = p
END SUB

' The 4 bytes of a SINGLE as stored in memory (little-endian), for byte-exact
' comparison with the export script's printout.
FUNCTION FloatHex$ (x AS SINGLE)
    DIM s AS STRING, i AS LONG, h AS STRING
    s = MKS$(x)
    h = ""
    FOR i = 1 TO 4
        h = h + RIGHT$("0" + LCASE$(HEX$(ASC(MID$(s, i, 1)))), 2)
    NEXT
    FloatHex$ = h
END FUNCTION

SUB Fail (msg AS STRING)
    PRINT "error: "; msg
    SYSTEM
END SUB

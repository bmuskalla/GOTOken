' GOTOken - an LLM inference engine in BASIC, running SmolLM2-135M.
' Follows karpathy's llama2.c run.c as closely as the language allows.
'
' Usage:  gotoken                    model info + step 1 weight check
'         gotoken logits <token_id>  step 2: logits for one token, no transformer
'         gotoken kernels <token_id> step 3: RMSNorm and MatMul on layer 0, in isolation
'         gotoken layer <id> [<id> ...]  step 4: layer 0 over a token sequence, output at the last
'         gotoken forward <id> [<id> ...]      step 5: full forward, logits at the last position
'         gotoken generate <steps> <id> [...]  step 6: greedy decode with the KV cache
'         gotoken generate-nocache <steps> <id> [...]  step 5: same, re-forwarding the prefix
'         gotoken encode "<text>"              step 7: text -> token ids (and the chunks)
'         gotoken decode <id> [<id> ...]       step 7: token ids -> text
'         gotoken encode-batch <file>          step 7: many strings (INT32 count; INT32 len + bytes each)
'         gotoken complete <steps> "<text>"    step 7: greedy completion of a text prompt
' Set GOTOKEN_WEIGHTS / GOTOKEN_TOKENIZER to use files other than ./model/*.bin.
'
' Layout: QB64 has no modules, only textual includes. Declarations (.bi) go
' at the top, subs and functions (.bm) must follow all main-program code, so
' they are included at the bottom.
'
' Build (see build.sh):  ~/qb64/qb64 -x -c gotoken.bas -o build/gotoken

$CONSOLE:ONLY
OPTION _EXPLICIT

'$INCLUDE: 'src/model.bi'
'$INCLUDE: 'src/state.bi'
'$INCLUDE: 'src/tokenizer.bi'

DIM path AS STRING, tokPath AS STRING, cmd AS STRING, token AS LONG, t0 AS DOUBLE, n AS LONG, i AS LONG
DIM text AS STRING, f AS LONG, count AS LONG, k AS LONG
REDIM tokens(0 TO 0) AS LONG

' Like run.c: the checkpoint path comes from outside. QB64 chdirs into the
' executable's folder at startup, so the default is resolved from the
' directory the program was launched from.
path = ENVIRON$("GOTOKEN_WEIGHTS")
IF path = "" THEN path = _STARTDIR$ + "/model/weights.bin"
tokPath = ENVIRON$("GOTOKEN_TOKENIZER")
IF tokPath = "" THEN tokPath = _STARTDIR$ + "/model/tokenizer.bin"
cmd = COMMAND$(1)

' the tokenizer-only commands do not need 538 MB of weights
IF cmd = "encode" OR cmd = "decode" OR cmd = "encode-batch" OR cmd = "complete" THEN
    t0 = TIMER(0.001)
    LoadTokenizer tokPath
    PRINT "loaded"; tokVocabSize; "tokens in"; TIMER(0.001) - t0; "s"
END IF
IF cmd <> "encode" AND cmd <> "decode" AND cmd <> "encode-batch" THEN
    t0 = TIMER(0.001)
    LoadModel path
    AllocRunState
    PRINT "loaded"; nParams; "floats in"; TIMER(0.001) - t0; "s"
END IF

token = VAL(COMMAND$(2))
SELECT CASE cmd
    CASE ""
        PrintModelInfo
        PrintWeightCheck
    CASE "logits"
        ' The whole "model" of step 2: look the token up, score every vocab
        ' entry against it. No norm, no attention, no FFN. With tied weights
        ' this is just each embedding row dotted with this token's row.
        PRINT "token"; token
        t0 = TIMER(0.001)
        Embed token
        Classify
        PRINT "classified in"; TIMER(0.001) - t0; "s"
        PrintLogits 5
    CASE "kernels"
        PRINT "token"; token
        KernelCheck token
    CASE "layer"
        n = _COMMANDCOUNT - 1
        IF n < 1 THEN Fail "layer needs at least one token id"
        REDIM tokens(0 TO n - 1) AS LONG
        FOR i = 0 TO n - 1
            tokens(i) = VAL(COMMAND$(i + 2))
            PRINT "token"; tokens(i); " at position"; i
        NEXT
        LayerCheck tokens(), n
    CASE "forward"
        n = _COMMANDCOUNT - 1
        IF n < 1 THEN Fail "forward needs at least one token id"
        REDIM tokens(0 TO n - 1) AS LONG
        FOR i = 0 TO n - 1
            tokens(i) = VAL(COMMAND$(i + 2))
        NEXT
        t0 = TIMER(0.001)
        FOR i = 0 TO n - 1
            Forward tokens(i), i
        NEXT
        PRINT "forwarded"; n; "positions in"; TIMER(0.001) - t0; "s"
        PrintLogits 5
    CASE "generate", "generate-nocache"
        n = _COMMANDCOUNT - 2
        IF n < 1 THEN Fail "generate needs <steps> and at least one token id"
        REDIM tokens(0 TO n - 1) AS LONG
        FOR i = 0 TO n - 1
            tokens(i) = VAL(COMMAND$(i + 3))
        NEXT
        Generate tokens(), n, VAL(COMMAND$(2)), -(cmd = "generate")
    CASE "encode"
        text = COMMAND$(2)
        EncodeVerbose text
    CASE "decode"
        n = _COMMANDCOUNT - 1
        REDIM tokens(0 TO n - 1) AS LONG
        FOR i = 0 TO n - 1
            tokens(i) = VAL(COMMAND$(i + 2))
        NEXT
        PRINT "text: "; Decode$(tokens(), n)
    CASE "encode-batch"
        f = FREEFILE
        OPEN COMMAND$(2) FOR BINARY AS #f
        GET #f, , count
        t0 = TIMER(0.001)
        FOR k = 1 TO count
            GET #f, , n
            text = SPACE$(n)
            IF n > 0 THEN GET #f, , text
            Encode text, tokens(), n
            PRINT "ids:";
            FOR i = 0 TO n - 1
                PRINT tokens(i);
            NEXT
            PRINT
        NEXT
        CLOSE #f
        PRINT "encoded"; count; "strings in"; TIMER(0.001) - t0; "s"
    CASE "complete"
        text = COMMAND$(3)
        Encode text, tokens(), n
        IF n < 1 THEN Fail "empty prompt"
        PRINT "prompt ids:";
        FOR i = 0 TO n - 1
            PRINT tokens(i);
        NEXT
        PRINT
        Generate tokens(), n, VAL(COMMAND$(2)), 1
        PRINT "text: "; Decode$(genSeq(), genLen)
    CASE ELSE
        Fail "unknown command: " + cmd
END SELECT
SYSTEM

'$INCLUDE: 'src/util.bm'
'$INCLUDE: 'src/model.bm'
'$INCLUDE: 'src/kernels.bm'
'$INCLUDE: 'src/forward.bm'
'$INCLUDE: 'src/tokenizer.bm'
'$INCLUDE: 'src/generate.bm'
'$INCLUDE: 'src/checks.bm'

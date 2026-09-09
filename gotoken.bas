' GOTOken - an LLM inference engine in BASIC, running SmolLM2-135M.
' Follows karpathy's llama2.c run.c as closely as the language allows.
'
' Usage:  gotoken                    model info + step 1 weight check
'         gotoken logits <token_id>  step 2: logits for one token, no transformer
'         gotoken kernels <token_id> step 3: RMSNorm and MatMul on layer 0, in isolation
'         gotoken layer <id> [<id> ...]  step 4: layer 0 over a token sequence, output at the last
'         gotoken forward <id> [<id> ...]      step 5: full forward, logits at the last position
'         gotoken generate <steps> <id> [...]  step 5: greedy decode from the prompt ids
' Set GOTOKEN_WEIGHTS to use a weights.bin other than ./model/weights.bin.
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

DIM path AS STRING, cmd AS STRING, token AS LONG, t0 AS DOUBLE, n AS LONG, i AS LONG
REDIM tokens(0 TO 0) AS LONG

' Like run.c: the checkpoint path comes from outside. QB64 chdirs into the
' executable's folder at startup, so the default is resolved from the
' directory the program was launched from.
path = ENVIRON$("GOTOKEN_WEIGHTS")
IF path = "" THEN path = _STARTDIR$ + "/model/weights.bin"

t0 = TIMER(0.001)
LoadModel path
AllocRunState
PRINT "loaded"; nParams; "floats in"; TIMER(0.001) - t0; "s"

cmd = COMMAND$(1)
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
    CASE "generate"
        n = _COMMANDCOUNT - 2
        IF n < 1 THEN Fail "generate needs <steps> and at least one token id"
        REDIM tokens(0 TO n - 1) AS LONG
        FOR i = 0 TO n - 1
            tokens(i) = VAL(COMMAND$(i + 3))
        NEXT
        Generate tokens(), n, VAL(COMMAND$(2))
    CASE ELSE
        Fail "unknown command: " + cmd
END SELECT
SYSTEM

'$INCLUDE: 'src/util.bm'
'$INCLUDE: 'src/model.bm'
'$INCLUDE: 'src/kernels.bm'
'$INCLUDE: 'src/forward.bm'
'$INCLUDE: 'src/generate.bm'
'$INCLUDE: 'src/checks.bm'

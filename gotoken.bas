' GOTOken - an LLM inference engine in BASIC, running SmolLM2-135M.
' Follows karpathy's llama2.c run.c as closely as the language allows.
'
' Usage:  gotoken                        model info + step 1 weight check
'         gotoken <token_id>             step 2: logits for one token, no transformer
'         gotoken <token_id> <weights>   same, with an explicit weights.bin
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

DIM path AS STRING, token AS LONG, t0 AS DOUBLE

' Like run.c: the checkpoint path is an argument. QB64 chdirs into the
' executable's folder at startup, so the default is resolved from the
' directory the program was launched from.
path = COMMAND$(2)
IF path = "" THEN path = _STARTDIR$ + "/model/weights.bin"

t0 = TIMER(0.001)
LoadModel path
AllocRunState
PRINT "loaded"; nParams; "floats in"; TIMER(0.001) - t0; "s"

IF _COMMANDCOUNT = 0 THEN
    PrintModelInfo
    PrintWeightCheck
ELSE
    token = VAL(COMMAND$(1))
    PRINT "token"; token
    ' The whole "model" of step 2: look the token up, score every vocab entry
    ' against it. No norm, no attention, no FFN. With tied weights this is
    ' just each embedding row dotted with this token's row.
    t0 = TIMER(0.001)
    Embed token
    Classify
    PRINT "classified in"; TIMER(0.001) - t0; "s"
    PrintLogits 5
END IF
SYSTEM

'$INCLUDE: 'src/util.bm'
'$INCLUDE: 'src/model.bm'
'$INCLUDE: 'src/kernels.bm'
'$INCLUDE: 'src/forward.bm'
'$INCLUDE: 'src/checks.bm'

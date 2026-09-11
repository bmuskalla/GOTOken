' json.bi - a small JSON parser, enough for config.json, the safetensors
' header and tokenizer.json. The document becomes a tree of nodes held in
' parallel arrays; children of a node form a linked list.

CONST JSON_NULL = 0, JSON_BOOL = 1, JSON_NUMBER = 2, JSON_STRING = 3, JSON_ARRAY = 4, JSON_OBJECT = 5

DIM SHARED jsonText AS STRING '  the document being parsed
DIM SHARED jsonPos AS LONG '     1-based read position in jsonText
DIM SHARED jsonCount AS LONG '   nodes in use

REDIM SHARED jType(0 TO 0) AS LONG '   JSON_* kind
REDIM SHARED jValue(0 TO 0) AS STRING ' strings: decoded UTF-8 bytes; numbers: their text; bools: "1"/"0"
REDIM SHARED jKey(0 TO 0) AS STRING '   member name, when the node is an object member
REDIM SHARED jChild(0 TO 0) AS LONG '   first child (arrays, objects), -1 if none
REDIM SHARED jNext(0 TO 0) AS LONG '    next sibling, -1 if last
REDIM SHARED jLen(0 TO 0) AS LONG '     number of children

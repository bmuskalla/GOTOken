#!/bin/sh
# Download the SmolLM2-135M checkpoint from HuggingFace into a model directory
# (default ./model): config.json, model.safetensors (270 MB, bf16), tokenizer.json.
# The engine reads these three files directly.
set -e
DIR="${1:-$(dirname "$0")/model}"
BASE="https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main"
mkdir -p "$DIR"
for f in config.json model.safetensors tokenizer.json; do
    if [ -s "$DIR/$f" ]; then
        echo "$DIR/$f exists, skipping"
    else
        echo "fetching $f"
        curl -sSL -o "$DIR/$f" "$BASE/$f"
    fi
done
ls -l "$DIR"

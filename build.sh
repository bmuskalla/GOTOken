#!/bin/sh
# Compile gotoken.bas with QB64 (classic, https://qb64.com). Set QB64 to your
# qb64 binary if it is not at ~/qb64/qb64.
set -e
cd "$(dirname "$0")"
QB64="${QB64:-$HOME/qb64/qb64}"
mkdir -p build
"$QB64" -x -c "$PWD/gotoken.bas" -o "$PWD/build/gotoken"
echo "built build/gotoken"

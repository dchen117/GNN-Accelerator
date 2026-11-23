#!/usr/bin/env bash
# Build the LRU cache kernel used by sgcnsim/cache.py (a pure-Python fallback exists, but it is
# ~100x slower: PubMed/NELL need the compiled one).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
g++ -O2 -shared -fPIC -o "$HERE/_cache.so" "$HERE/cache.cpp"
echo "built $HERE/_cache.so"

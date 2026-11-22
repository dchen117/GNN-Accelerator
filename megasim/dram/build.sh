#!/usr/bin/env bash
# Build the Ramulator replay driver and export the HBM config (S-13).
#   RAMULATOR=/path/to/ramulator2 megasim/dram/build.sh
# Ramulator 2.1 must be built first (its README: mkdir build && cd build && cmake .. && make).
# Tested with CMU-SAFARI/ramulator2 commit 7bead6cb (2026-11-04).
set -euo pipefail
RAMULATOR=${RAMULATOR:-/home/user/ext/ramulator2}
HERE=$(cd "$(dirname "$0")" && pwd)
g++ -O2 -std=c++20 -o "$HERE/replay" "$HERE/replay.cpp" \
    -I"$RAMULATOR/src" -I"$RAMULATOR/ext/yaml-cpp/include" -I"$RAMULATOR/ext/fmt/include" \
    -L"$RAMULATOR" -lramulator -Wl,-rpath,"$RAMULATOR"
PYTHONPATH="$RAMULATOR/python" python -m ramulator export "$HERE/hbm_config.py" -o "$HERE/hbm1_256GBps.yaml"
echo "built $HERE/replay and $HERE/hbm1_256GBps.yaml"

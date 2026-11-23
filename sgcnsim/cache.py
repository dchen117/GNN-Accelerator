"""SGCN's global cache: set-associative, LRU, 64 B lines (G-05 in docs/sgcn/00-design.md).

    hit = lru(line_addresses, sets, ways)      # uint8 per access

Uses the compiled kernel (sgcnsim/build.sh) when present, else a pure-Python version (same
results, slow; fine for tests)."""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from pathlib import Path

import numpy as np

_SO = Path(__file__).resolve().parent / "_cache.so"
_lib = None
if _SO.exists():
    _lib = ctypes.CDLL(str(_SO))
    _lib.lru.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         ctypes.c_void_p]


def lru_python(lines: np.ndarray, sets: int, ways: int) -> np.ndarray:
    store = [OrderedDict() for _ in range(sets)]
    hit = np.zeros(lines.size, dtype=np.uint8)
    for i, l in enumerate(lines.tolist()):
        s = store[l % sets]
        if l in s:
            s.move_to_end(l)
            hit[i] = 1
        else:
            if len(s) >= ways:
                s.popitem(last=False)
            s[l] = None
    return hit


def lru(lines: np.ndarray, sets: int, ways: int) -> np.ndarray:
    lines = np.ascontiguousarray(lines, dtype=np.uint64)
    if _lib is None:
        return lru_python(lines, sets, ways)
    hit = np.zeros(lines.size, dtype=np.uint8)
    _lib.lru(lines.ctypes.data, lines.size, sets, ways, hit.ctypes.data)
    return hit

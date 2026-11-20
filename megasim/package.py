"""Adaptive-Package storage format (paper §V-B, Fig.9; decisions Q-17, Q-18, Q-19).

Layout of one layer's features in memory:

  package stream   packages back to back, each  Mode(2) | Bitwidth(3) | Val Array | zero pad
                   Mode 00/01/10 = 64/128/192 bits TOTAL, header included (Q-17); 11 unused.
                   Bitwidth field = b − 1 (b = 1..8 fits 3 bits; we use 2..8, Q-07).
                   Values are b-bit two's complement (Q-06/Q-22), MSB first, in node order
                   and, within a node, in ascending feature order.
  bitmap array     per node, F bits (1 = non-zero), padded to a multiple of 32 bits (Q-19).

Packing rule (Q-18, encoder p.8): the package register takes successive non-zero values until
the next value would not fit in 192 − 5 = 187 bits or the next value's node has a different
bitwidth; Mode is then the smallest length that holds 5 + k·b bits.
Nodes with no non-zeros contribute nothing to the stream (their bitmap is all zero), so they
neither close nor split a package. (Ours: the paper's "bitwidth changes" refers to values.)

Locating nodes (Q-19): no per-node pointers. The Decoder reads nodes in ID order, takes
popcount(bitmap_i) values for node i and keeps a counter of values left in the current
package, so nodes span packages and packages hold several nodes.

Bit order: the stream is one big-endian bit string (bit 0 = MSB of byte 0); packages are
multiples of 64 bits, so every package starts on an 8-byte boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADER_BITS = 5
MODE_LENGTHS = (64, 128, 192)              # Mode 00, 01, 10 (Q-17: total, header included)
MAX_VAL_BITS = MODE_LENGTHS[-1] - HEADER_BITS   # 187
BITMAP_WORD = 32                           # Q-19: bitmaps padded to whole 32-bit words


def capacity(b: int) -> int:
    """Values a full (192-bit) package holds at bitwidth b."""
    return MAX_VAL_BITS // b


@dataclass
class PackageStream:
    data: np.ndarray        # uint8 bytes of the whole stream
    mode: np.ndarray        # (P,) 0/1/2 per package
    bits: np.ndarray        # (P,) b per package
    count: np.ndarray       # (P,) values per package
    start_bit: np.ndarray   # (P,) offset of each package in the stream

    @property
    def num_packages(self) -> int:
        return self.mode.size

    @property
    def total_bits(self) -> int:
        return int(self.data.size * 8)

    def mode_histogram(self) -> dict:
        return {MODE_LENGTHS[m]: int((self.mode == m).sum()) for m in range(3)}


def bitmap_bits(n: int, F: int) -> int:
    """Stored bitmap size of a layer, in bits (F padded to 32 per node)."""
    return n * (-(-F // BITMAP_WORD) * BITMAP_WORD)


# --------------------------------------------------------------------------- encode
def encode(values: np.ndarray, node_bits_per_value: np.ndarray) -> PackageStream:
    """Greedy packing of a value sequence (already in stream order).

    values               int, the non-zeros in node order (b-bit two's complement range)
    node_bits_per_value  int, the bitwidth of each value's node
    """
    v = np.asarray(values, dtype=np.int64)
    b = np.asarray(node_bits_per_value, dtype=np.int64)
    nv = v.size
    if nv == 0:
        z = np.zeros(0, dtype=np.int64)
        return PackageStream(np.zeros(0, np.uint8), z, z, z, z)
    lim = 2 ** (b - 1)
    assert np.all((v >= -lim) & (v < lim)), "value outside its b-bit two's complement range"

    # runs of equal bitwidth; inside a run, packages of capacity(b) values (greedy)
    run_start = np.flatnonzero(np.concatenate([[True], b[1:] != b[:-1]]))
    run_len = np.diff(np.concatenate([run_start, [nv]]))
    run_b = b[run_start]
    cap = MAX_VAL_BITS // run_b
    n_pk = -(-run_len // cap)
    pk_run = np.repeat(np.arange(run_start.size), n_pk)
    pk_in_run = np.arange(pk_run.size) - np.repeat(np.cumsum(n_pk) - n_pk, n_pk)
    pk_bits = run_b[pk_run]
    pk_first = run_start[pk_run] + pk_in_run * cap[pk_run]
    pk_count = np.minimum(cap[pk_run], run_start[pk_run] + run_len[pk_run] - pk_first)

    need = HEADER_BITS + pk_count * pk_bits
    mode = np.searchsorted(np.array(MODE_LENGTHS), need)      # smallest length that fits
    length = np.array(MODE_LENGTHS)[mode]
    start = np.concatenate([[0], np.cumsum(length)[:-1]])
    total = int(length.sum())

    stream = np.zeros(total, dtype=np.uint8)                  # one entry per bit
    # header: mode (2 bits), then b − 1 (3 bits), MSB first
    hdr = (mode << 3) | (pk_bits - 1)
    for j in range(HEADER_BITS):
        stream[start + j] = (hdr >> (HEADER_BITS - 1 - j)) & 1
    # values: position = package start + 5 + index_in_package · b
    pk_of_val = np.repeat(np.arange(pk_count.size), pk_count)
    idx_in_pk = np.arange(nv) - pk_first[pk_of_val]
    pos = start[pk_of_val] + HEADER_BITS + idx_in_pk * b
    code = v & ((1 << b) - 1)                                 # two's complement, b bits
    for j in range(8):
        sel = j < b
        stream[pos[sel] + j] = (code[sel] >> (b[sel] - 1 - j)) & 1
    return PackageStream(np.packbits(stream), mode, pk_bits, pk_count, start)


# --------------------------------------------------------------------------- decode
def decode(data: np.ndarray, expected_bits: np.ndarray) -> np.ndarray:
    """Decoder model: walk the package stream and return the values in stream order.

    expected_bits: the bitwidth of each value's node, which the Decoder knows from the node's
    degree-table index (Q-26) and the popcount of its bitmap (Q-19). The header does not hold
    a value count, so this is how the Decoder knows where a padded package ends: it takes
    values while the package's mode has room AND the next value's node has the package's
    bitwidth (the encoder closed the package exactly when one of these failed, Q-18).
    The header's Bitwidth field is checked against the expected bitwidth.
    Header parsing is sequential, as in hardware; value extraction is vectorized.
    """
    b_exp = np.asarray(expected_bits, dtype=np.int64)
    nv = b_exp.size
    bits = np.unpackbits(data)
    # end (exclusive) of the equal-bitwidth run each value belongs to
    change = np.flatnonzero(np.concatenate([b_exp[1:] != b_exp[:-1], [True]])) + 1
    run_end = np.repeat(change, np.diff(np.concatenate([[0], change])))
    lengths = np.array(MODE_LENGTHS)
    starts, counts = [], []
    pos = got = 0
    while got < nv:
        hdr = 0
        for j in range(HEADER_BITS):
            hdr = (hdr << 1) | int(bits[pos + j])
        mode, b = hdr >> 3, (hdr & 7) + 1
        assert mode < 3, f"reserved Mode 11 at bit {pos}"
        assert b == b_exp[got], f"package at bit {pos}: Bitwidth {b}, node expects {b_exp[got]}"
        L = int(lengths[mode])
        k = min((L - HEADER_BITS) // b, int(run_end[got]) - got)
        starts.append(pos)
        counts.append(k)
        got += k
        pos += L
    assert pos <= bits.size
    counts = np.array(counts, dtype=np.int64)
    pk = np.repeat(np.arange(counts.size), counts)
    first = np.cumsum(counts) - counts
    vpos = np.array(starts, dtype=np.int64)[pk] + HEADER_BITS + (np.arange(nv) - first[pk]) * b_exp
    code = np.zeros(nv, dtype=np.int64)
    for j in range(8):
        sel = j < b_exp
        code[sel] = (code[sel] << 1) | bits[vpos[sel] + j]
    return np.where(code >= (1 << (b_exp - 1)), code - (1 << b_exp), code)

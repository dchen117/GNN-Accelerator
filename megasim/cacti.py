"""CACTI 7.0 runs for MEGA's on-chip buffers (S-14).

MEGA models its SRAM with CACTI-7.0 at TSMC 28 nm (§VI-A-3). This module writes a CACTI
config per buffer, runs CACTI and parses the per-access energies, leakage and area. The
results are cached in megasim/cacti_28nm.json (committed), so the energy model runs without
CACTI installed; `python -m megasim.cacti` regenerates the cache.

    CACTI=/path/to/cacti/cacti python -m megasim.cacti

Buffer organization (ours, S-14): each Tab.IV buffer is a scratch RAM ("ram", no tags) with
one read/write port, 64 B access width (the DRAM line and the widest regular access of the
datapath) and `banks` banks (ping-pong halves are banks). CACTI picks the internal subarray
organization for minimum ED² ("Optimize ED^2", the CACTI default). Cell/peripheral type and
bank count were chosen to match Tab.IV's measured areas (see BANKS and `python -m megasim.cacti`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CACTI = os.environ.get("CACTI", "/home/user/ext/cacti/cacti")
CACHE = Path(__file__).resolve().parent / "cacti_28nm.json"

# Tab.IV: size (KB), area (mm²), power (mW)
TAB_IV = {
    "agg_buf": (128, 0.540, 46.56),
    "comb_buf": (96, 0.452, 35.19),
    "input_buf": (64, 0.220, 22.88),
    "edge_buf": (24, 0.119, 9.44),
    "sparse_buf": (32, 0.154, 12.86),
    "weight_buf": (48, 0.190, 14.32),
}
BLOCK = 64          # bytes per access

# CACTI's own example config (cache.cfg in the CACTI tree) with these lines replaced
_SUBST = {
    "-size (bytes)": "-size (bytes) {size}",
    "-block size (bytes)": "-block size (bytes) {block}",
    "-associativity": "-associativity 1",
    "-UCA bank count": "-UCA bank count {banks}",
    "-technology (u)": "-technology (u) 0.028",
    "-Data array cell type": '-Data array cell type - "{cell}"',
    "-Data array peripheral type": '-Data array peripheral type - "{periph}"',
    "-Tag array cell type": '-Tag array cell type - "{cell}"',
    "-Tag array peripheral type": '-Tag array peripheral type - "{periph}"',
    "-output/input bus width": "-output/input bus width {bus}",
    "-cache type": '-cache type "ram"',
    "-Add ECC": '-Add ECC - "false"',
    "-Print input parameters": '-Print input parameters - "false"',
}


def _config(**kw) -> str:
    lines = (Path(CACTI).parent / "cache.cfg").read_text().splitlines()
    out = []
    for ln in lines:
        key = next((k for k in _SUBST if ln.startswith(k)), None)
        out.append(_SUBST[key].format(**kw) if key else ln)
    return "\n".join(out) + "\n"


_PAT = {
    "read_nJ": r"Total dynamic read energy per access \(nJ\):\s*([\d.eE+-]+)",
    "write_nJ": r"Total dynamic write energy per access \(nJ\):\s*([\d.eE+-]+)",
    "leak_bank_mW": r"Total leakage power of a bank \(mW\):\s*([\d.eE+-]+)",
    "gate_leak_bank_mW": r"Total gate leakage power of a bank \(mW\):\s*([\d.eE+-]+)",
    "area_mm2": r"Cache height x width \(mm\):\s*([\d.eE+-]+)\s*x\s*([\d.eE+-]+)",
    "cycle_ns": r"Cycle time \(ns\):\s*([\d.eE+-]+)",
}


def run(size: int, banks: int, cell: str = "itrs-lop", periph: str = "itrs-lop",
        block: int = BLOCK) -> dict:
    cfg = _config(size=size, block=block, banks=banks, cell=cell, periph=periph, bus=block * 8)
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", delete=False) as f:
        f.write(cfg)
        path = f.name
    try:
        out = subprocess.run([CACTI, "-infile", path], capture_output=True, text=True,
                             cwd=str(Path(CACTI).parent), timeout=600).stdout
    finally:
        os.unlink(path)
    r = {}
    for k, p in _PAT.items():
        m = re.search(p, out)
        if not m:
            raise RuntimeError(f"CACTI output has no {k}:\n{out[-2000:]}")
        r[k] = float(m.group(1)) * float(m.group(2)) if k == "area_mm2" else float(m.group(1))
    r["leak_mW"] = banks * (r["leak_bank_mW"] + r["gate_leak_bank_mW"])
    r.update(size=size, banks=banks, cell=cell, periph=periph, block=block)
    return r


# Chosen organization per buffer (see the area match printed by main()).
# Sweep (cells hp/lop/lstp × 1/2/4 banks, plus 8/16/32 for Agg/Comb), 2026-11-22:
#   * itrs-lop with these banks gives areas within 0–24 % of Tab.IV's measured areas (sum
#     1.48 vs 1.67 mm²) and CACTI leakage within 2–23 % of Tab.IV's buffer power (sum 151 vs
#     141 mW); hp leaks ~2× more, lstp ~1000× less with 2× the access energy.
#   * Agg/Comb need 8 banks to come near their Tab.IV area (16 overshoots the leakage by
#     1.7–1.9×); their 2–4 bank versions are half the measured area.
BANKS = {"agg_buf": 8, "comb_buf": 8, "input_buf": 4, "edge_buf": 4, "sparse_buf": 4,
         "weight_buf": 4}
CELL = "itrs-lop"


def load() -> dict:
    """Per-buffer CACTI results from the committed cache."""
    return json.loads(CACHE.read_text())["buffers"]


def main(argv=None):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(TAB_IV)) as ex:
        res = dict(zip(TAB_IV, ex.map(lambda nm: run(TAB_IV[nm][0] * 1024, BANKS[nm], CELL, CELL),
                                      TAB_IV)))
    rows = {}
    print(f"{'buffer':11s} {'KB':>4s} {'banks':>5s} {'area':>7s} {'TabIV':>6s} "
          f"{'rd pJ':>7s} {'wr pJ':>7s} {'leak mW':>8s} {'TabIV mW':>8s}")
    for name, (kb, area, mw) in TAB_IV.items():
        r = res[name]
        rows[name] = r
        print(f"{name:11s} {kb:4d} {BANKS[name]:5d} {r['area_mm2']:7.3f} {area:6.3f} "
              f"{r['read_nJ'] * 1e3:7.2f} {r['write_nJ'] * 1e3:7.2f} {r['leak_mW']:8.2f} {mw:8.2f}")
    CACHE.write_text(json.dumps({"cacti": "CACTI 7.0 (HewlettPackard/cacti 1ffd8df)",
                                 "technology_um": 0.028, "cell": CELL, "block_bytes": BLOCK,
                                 "buffers": rows}, indent=1))
    print(f"wrote {CACHE}")


if __name__ == "__main__":
    main(sys.argv[1:])

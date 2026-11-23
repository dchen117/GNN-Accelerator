"""CACTI 7.0 numbers for SGCN's global cache (G-07), same tool, node and cell type as MEGA's
buffers (megasim/cacti.py, S-14). Cached in sgcnsim/cacti_28nm.json (committed).

    CACTI=/path/to/cacti python -m sgcnsim.cacti        # regenerate

Organization: 392 KB, 16-way, 64 B lines, 1 read/write port, itrs-lop cells, CACTI's default
"normal" access mode (tags and data in parallel). Banks: 2, the count whose area (1.44 mm²)
is closest to MEGA's measured buffer area for the same 392 KB (Tab.IV 1.67 mm²); 1 bank gives
0.99 mm², 4 banks 2.93 mm², 16 banks has no valid organization.
"""

from __future__ import annotations

import json
from pathlib import Path

from megasim import cacti

CACHE = Path(__file__).resolve().parent / "cacti_28nm.json"
SIZE, WAYS, BANKS = 392 * 1024, 16, 2


def load() -> dict:
    return json.loads(CACHE.read_text())


def main():
    r = cacti.run(SIZE, BANKS, cacti.CELL, cacti.CELL, assoc=WAYS, ctype="cache")
    r["cacti"] = "CACTI 7.0 (HewlettPackard/cacti 1ffd8df)"
    r["technology_um"] = 0.028
    CACHE.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()

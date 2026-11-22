"""Fig.14/16/17/18-style tables from a `python -m megasim.sim --json` result file.

    python -m megasim.report runs/hw/megasim_m3.json [--baseline other.json]

MEGA's figures are normalized to HyGCN (Fig.14 speedup, Fig.16 DRAM-access reduction, Fig.17
energy saving, Fig.18 per-category energy). We print the absolute quantities behind them, per
run: time (Fig.14), DRAM bytes (Fig.16), energy (Fig.17) and its DRAM/SRAM/PU/leakage split
(Fig.18). With --baseline B, each quantity is also given as B / this, i.e. in the direction of
MEGA's figures with B in HyGCN's place (B = the design being compared against).
"""

from __future__ import annotations

import argparse
import json
import math


def _load(path):
    return {r["run"]: r for r in json.load(open(path))}


def table(res: dict, base: dict | None = None) -> str:
    cats = ["dram", "sram", "pu", "leakage"]
    hdr = ["run", "K", "time µs", "DRAM MB", "energy µJ"] + [f"{c} %" for c in cats] + \
          ["row hit", "DRAM stall"]
    if base:
        hdr += ["speedup", "DRAM red.", "energy sav."] + [f"{c} ×" for c in cats]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    geo = {"speedup": [], "dram": [], "energy": []}
    for name, r in res.items():
        e = r["energy"]
        ram = r.get("ramulator", {})
        # DRAM-induced stall share (Fig.20(a)): 1 − time with an ideal DRAM / time
        stall = f"{r['dram_stall_share']:.1%}" if "dram_stall_share" in r else "—"
        row = [name, r["subgraphs"], f"{r['time_us']:.1f}", f"{r['dram_MB_total']:.2f}",
               f"{e['total_uJ']:.1f}"] + [f"{100 * e['share'][c]:.1f}" for c in cats] + \
              [f"{ram['row_hit_rate']:.2f}" if ram else "—", stall]
        if base and name in base:
            b = base[name]
            sp = b["cycles"] / r["cycles"]
            dr = b["dram_MB_total"] / r["dram_MB_total"]
            en = b["energy"]["total_uJ"] / e["total_uJ"]
            geo["speedup"].append(sp), geo["dram"].append(dr), geo["energy"].append(en)
            row += [f"{sp:.2f}", f"{dr:.2f}", f"{en:.2f}"] + \
                   [f"{b['energy']['uJ'][c] / e['uJ'][c]:.2f}" for c in cats]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    if base and geo["speedup"]:
        g = {k: math.exp(sum(map(math.log, v)) / len(v)) for k, v in geo.items()}
        lines.append(f"\ngeomean: speedup {g['speedup']:.2f}×, DRAM reduction {g['dram']:.2f}×, "
                     f"energy saving {g['energy']:.2f}×")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("result")
    p.add_argument("--baseline", default=None)
    a = p.parse_args(argv)
    print(table(_load(a.result), _load(a.baseline) if a.baseline else None))


if __name__ == "__main__":
    main()

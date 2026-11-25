"""Energy model of MEGA (S-14), with the four categories of Fig.18: DRAM, SRAM, PU, leakage.

    e = energy(rep)          # rep = timing.simulate(...) or dram.ramulator.simulate_ramulator(...)
    e["total_uJ"], e["uJ"]   # {"dram", "sram", "pu", "leakage"}

DRAM     7 pJ/bit for every byte moved (HBM1.0, as in HyGCN [56], whose DRAM-energy
         methodology MEGA follows, §VI-A-3; HyGCN cites [32, 41] for the figure). Fixed
         pJ/bit is also what Fig.18 implies: its DRAM column equals Fig.16's access ratio.
SRAM     bytes read/written per buffer (timing.sram_accesses) / 64 B × CACTI 7.0 energy per
         64 B access at 28 nm (megasim/cacti_28nm.json, organization in megasim/cacti.py).
PU       Tab.IV power of each processing unit × the cycles it is busy (from the timing
         model) at 1 GHz: Tab.IV's figures are taken as the unit's power while working, and
         an idle unit as clock-gated (ours: the paper does not say how it turns power into
         energy). "Others" (0.80 mW) is charged for the whole run.
Leakage  CACTI leakage of all buffers × run time. PU leakage is inside the PU powers
         (we cannot separate it without the netlist).

Units MEGA does not have (ADDED_UNITS) are counted, never assumed free (user, 2026-11-25):
  expander  index → bitmap expander in front of the Input Buffer (S-15), used when X̄⁽⁰⁾'s
            positions come from DRAM as indices. Our estimate (docs/simulator/00-design.md
            S-15): ≈ 8,500 NAND2-eq (8 one-hot 8→256 decoders, 256-bit 8-input OR, 256-bit
            output register, 2 × 8 × 16-bit index FIFO, control) → 0.006 mm² at 0.5 µm² per
            NAND2 and 70 % utilization. Power = MEGA's own Decoder power density
            (Tab.IV 0.75 mW / 0.003 mm²) × that area = 1.5 mW while busy.
"""

from __future__ import annotations

from . import cacti

DRAM_PJ_PER_BIT = 7.0          # HBM1.0 (HyGCN [56])

# Tab.IV processing-unit powers (mW), and which timing-model unit keeps each busy
PU_MW = {"bse": 14.70, "aggregation": 28.92, "crossbar": 5.56, "condense": 1.19,
         "encoder": 1.81, "decoder": 0.75, "others": 0.80}
PU_BUSY = {"bse": "comb", "crossbar": "comb", "decoder": "comb", "aggregation": "agg",
           "encoder": "enc", "condense": "condense"}
TAB_IV_AREA_MM2 = 1.869        # measured total (Tab.IV)
# units we add to MEGA: area (mm²), power while busy (mW), timing-model busy key
# `present`: whether the configured hardware includes the unit (area is counted then, even on a
# dataset where it never runs; energy only for the cycles it works)
ADDED_UNITS = {"expander": {"area_mm2": 0.006, "mW": 1.5, "busy": "expander",
                            "present": lambda hw: hw.x0_format != "bitmap"}}


def energy(rep, block: int = cacti.BLOCK) -> dict:
    hw = rep.hw
    ns = 1.0 / hw.clock_ghz
    bufs = cacti.load()
    # DRAM
    dram_bytes = sum(rep.dram_bytes.values())
    dram_pJ = dram_bytes * 8 * DRAM_PJ_PER_BIT
    # SRAM dynamic
    sram_pJ_by = {}
    for b, a in rep.info["sram"].items():
        c = bufs[b]
        sram_pJ_by[b] = (a["read"] / block * c["read_nJ"] + a["write"] / block * c["write_nJ"]) * 1e3
    # PU
    busy = {}
    for s in rep.sweeps:
        for k, v in s.busy.items():
            busy[k] = busy.get(k, 0.0) + v
    pu_pJ_by = {u: PU_MW[u] * busy.get(k, 0.0) * ns for u, k in PU_BUSY.items()}   # mW·ns = pJ
    pu_pJ_by["others"] = PU_MW["others"] * rep.total_cycles * ns
    # added units: area if the hardware has them, energy for the cycles they work
    added = {u: c for u, c in ADDED_UNITS.items() if c["present"](hw)}
    for u, c in added.items():
        pu_pJ_by[u] = c["mW"] * busy.get(c["busy"], 0.0) * ns
    # leakage
    leak_mW = sum(c["leak_mW"] for c in bufs.values())
    leak_pJ = leak_mW * rep.total_cycles * ns
    uJ = {"dram": dram_pJ * 1e-6, "sram": sum(sram_pJ_by.values()) * 1e-6,
          "pu": sum(pu_pJ_by.values()) * 1e-6, "leakage": leak_pJ * 1e-6}
    total = sum(uJ.values())
    return {"total_uJ": total, "uJ": uJ, "share": {k: v / total for k, v in uJ.items()},
            "sram_uJ": {k: v * 1e-6 for k, v in sram_pJ_by.items()},
            "pu_uJ": {k: v * 1e-6 for k, v in pu_pJ_by.items()},
            "avg_power_mW": total * 1e-6 / (rep.total_cycles * ns * 1e-9) * 1e3,
            "dram_pJ_per_bit": DRAM_PJ_PER_BIT, "sram_leak_mW": leak_mW,
            "added_units": sorted(added),
            "area_mm2": TAB_IV_AREA_MM2 + sum(c["area_mm2"] for c in added.values())}

"""SGCN energy in MEGA's four Fig.18 categories (G-07), using MEGA's methodology (S-14) where
it applies, so the two designs are charged the same way:

DRAM     7 pJ/bit for every byte moved (as megasim.energy).
SRAM     global cache: every line an engine consumes is one read; every miss also one fill
         (write). CACTI 7.0 per-access energies (sgcnsim/cacti_28nm.json). The engines' small
         staging buffers and the systolic array's registers are not modelled (neither are
         MEGA's pipeline registers).
PU       energy per 32-bit MAC × MACs issued. Aggregation engines issue 16 MACs per line they
         consume (padding lanes included: a SIMD16 op runs on a whole line); the combination
         engine and the phase-A combination count the useful MACs. MAC energy: 32-bit integer
         multiply 3.1 pJ + add 0.1 pJ at 45 nm (Horowitz, ISSCC 2014), scaled to 28 nm by the
         feature-size ratio 28/45 (dynamic energy ∝ C ∝ feature size at equal voltage) =
         1.99 pJ. SGCN's own power numbers are for a different engine count and node (45 nm
         scaled to 32 nm, 8+8 engines), so they cannot be reused for Tab.V's configuration.
Leakage  CACTI leakage of the cache × run time (PU leakage inside PU energy, as for MEGA).
"""

from __future__ import annotations

from megasim.energy import DRAM_PJ_PER_BIT

from . import cacti

MAC_PJ = (3.1 + 0.1) * 28 / 45


def energy(rep) -> dict:
    hw = rep.hw
    ns = 1.0 / hw.clock_ghz
    c = cacti.load()
    dram_pJ = sum(rep.dram_bytes.values()) * 8 * DRAM_PJ_PER_BIT
    acc = sum(p["accesses"] for p in rep.phases)
    miss = sum(p["accesses"] - p["hits"] for p in rep.phases)
    sram_pJ = (acc * c["read_nJ"] + miss * c["write_nJ"]) * 1e3
    agg_macs = sum(p["busy"]["agg_engines"] * hw.agg_engines for p in rep.phases) * hw.simd
    comb_macs = rep.info["comb_macs"]
    pu_pJ = (agg_macs + comb_macs) * MAC_PJ
    leak_pJ = c["leak_mW"] * rep.total_cycles * ns
    uJ = {"dram": dram_pJ * 1e-6, "sram": sram_pJ * 1e-6, "pu": pu_pJ * 1e-6,
          "leakage": leak_pJ * 1e-6}
    total = sum(uJ.values())
    return {"total_uJ": total, "uJ": uJ, "share": {k: v / total for k, v in uJ.items()},
            "sram_accesses": {"read": acc, "fill": miss}, "macs": {"agg": agg_macs, "comb": comb_macs},
            "avg_power_mW": total * 1e-6 / (rep.total_cycles * ns * 1e-9) * 1e3,
            "dram_pJ_per_bit": DRAM_PJ_PER_BIT, "mac_pJ": MAC_PJ, "sram_leak_mW": c["leak_mW"]}

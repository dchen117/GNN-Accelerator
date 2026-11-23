# sgcnsim — SGCN simulator (MEGA Tab.V configuration)

SGCN (HPCA 2023) as MEGA's comparison point: 4 × SIMD16 aggregation engines, a 16-MAC
combination engine, a 392 KB 16-way LRU global cache, 32-bit data, HBM1 at 256 GB/s through
the same Ramulator model as megasim. The design and every decision are in
`docs/sgcn/00-design.md` (G-01 … G-07). The MEGA-vs-SGCN targets from MEGA's Fig.14/16/17
are also listed there.

## Run

```bash
sgcnsim/build.sh                       # LRU cache kernel (once; a slow Python fallback exists)
```

```bash
# needs megasim/dram/build.sh (Ramulator driver) as for megasim
python -m sgcnsim.sim runs/hw/gcn_cora runs/hw/gcn_citeseer runs/hw/gcn_pubmed runs/hw/gcn_nell \
    --json runs/hw/sgcn.json
```

```bash
# MEGA ÷ SGCN next to the paper's ratios
python -m sgcnsim.compare runs/hw/megasim_m3.json runs/hw/sgcn.json
```

```bash
python -m pytest tests/test_sgcnsim.py -q
```

`--dram uncontended` skips Ramulator: each transfer gets latency + its own transfer time and
no queueing. It is useful for quick debugging only.

## Code map

| File | What | Decisions |
|---|---|---|
| `workload.py` | FP32 GCN of the run (its pretraining checkpoint): graph CSR, X⁽⁰⁾ CSR, hidden ReLU outputs | G-02 |
| `timing.py` | Layouts (dense / in-place sliced BEICSR); phases A (X⁽⁰⁾W⁽⁰⁾ on the sparse engines), B (Ã·T + compress), C (Ã·X then combination); cache simulation; event schedule | G-03 … G-06 |
| `cache.py`, `cache.cpp` | Set-associative LRU over line addresses | G-05 |
| `energy.py`, `cacti.py`, `cacti_28nm.json` | DRAM 7 pJ/bit, CACTI cache read/fill/leakage, 1.99 pJ per 32-bit MAC | G-07 |
| `sim.py` | CLI. Its JSON has megasim's keys, so `megasim.report --baseline` also works | — |
| `compare.py` | Ratios MEGA ÷ SGCN next to the paper's per-dataset ratios and geomeans | Targets |

## Debugging

- `timing.build_plan(w)` holds every phase's access addresses, hits, misses and cycles before
  any timing.
- `timing.schedule(plan, dram=megasim.timing.IdealDram(hw))` gives the compute-only time.
- Phase reports (`phases` in the JSON) give start/end, engine and combination busy cycles,
  and cache accesses/hits.

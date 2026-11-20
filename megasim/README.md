# megasim — MEGA simulator

The simulator of the MEGA baseline. Design and the reasons for each choice:
`docs/simulator/00-design.md` (S-xx). The hardware it implements: Q-16 … Q-30, Q-35 in
`docs/mega-kb/04-open-questions.md`.

State (2026-11-20): **M1, the functional model, is done and bit-exact.** The timing model (M2)
and DRAM/energy (M3) are next.

## Run

```bash
# bit-exact check of the functional model against mega_pt.hw_stats.int_forward (Q-30)
python -m megasim.check runs/hw/* --json runs/hw/megasim_check.json
```

```bash
python -m pytest tests/test_megasim.py -q
```

A run directory needs `model_seed<s>.pt`, written by `python -m mega_pt.train ... --save-model`.

## Code map

| File | What | Decisions |
|---|---|---|
| `program.py` | Exports a trained run into a hardware program: W̄, per-node bitwidth and compacted degree-table index, 16-bit fixed-point tables (α′, c, e, γ), X̄⁽⁰⁾, binary A+I in CSR and CSC | S-05, S-07, Q-10, Q-11, Q-16, Q-26 |
| `package.py` | Adaptive-Package encoder (greedy, smallest Mode) and Decoder (walks headers, uses the node bitwidths to find where a padded package ends); bitmap sizes | S-06, Q-17, Q-18, Q-19 |
| `functional.py` | The datapath per layer: store/reload, bit-serial combination with the pass/tile split, Tile Accumulator, rescale, AUs, Encoder; range checks on every accumulator | S-04, Q-16, Q-22, Q-23, Q-25, Q-26 |
| `check.py` | CLI: runs `functional` and `int_forward(fmt="fx16")` on the same model and compares every intermediate exactly | Q-30 |

## Debugging

- `functional.run_layer` returns a `LayerResult` with every intermediate (`acc`, `B`, `psum`,
  `x_out`, `logits_code`), the width checks (`checks`) and the memory numbers (`memory`).
- `check.py` prints which intermediate first differs (`L<l>.equal`). The order inside a layer
  is x_in → acc → psum → (next x_in | logits).
- Any `*_overflow` / `au_may_saturate` that is true also fails the check: the model does not
  wrap or saturate, it reports.
- `package.decode` asserts on a header whose Bitwidth disagrees with the node's degree table,
  which is the first symptom of an encoder/decoder mismatch.

## Results: M1 functional check (2026-11-20)

All 8 saved models in `runs/hw` are **bit-exact** (X̄⁽⁰⁾, acc, psum, X̄⁽¹⁾, logits), and the
simulated test accuracy equals `hw_stats`' fx16 integer accuracy. Raw output:
`runs/hw/megasim_check.json` (written by the command above; `runs/` is not committed). The NELL model is the collapsed run (1.2 %, see `mega_pt/README.md`);
it is kept as a size and functional test only.

Per layer (L0 = X̄⁽⁰⁾, L1 = X̄⁽¹⁾). *values/el* is the Tab.VI-comparable number (Q-15), *stored/el*
includes package headers and padding and the bitmaps (Q-19). SA, TA = largest Shifter-Acc and
Tile Accumulator value; AU bound = largest Σ|B̄| into one partial sum (< 32,768 means no
saturation in any order).

| Run | L | density | values/el | stored/el | CR stored | pad % | SA max | TA max | AU bound |
|---|---|---|---|---|---|---|---|---|---|
| gcn_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 3.3 | 45 | 126 | 394 |
| | 1 | 0.690 | 2.532 | 3.744 | 8.5 | 7.7 | 545 | 1874 | 356 |
| gcn_citeseer | 0 | 0.009 | 0.017 | 1.020 | 31.4 | 3.2 | 150 | 330 | 170 |
| | 1 | 0.696 | 2.251 | 3.428 | 9.3 | 7.3 | 254 | 911 | 123 |
| gcn_pubmed | 0 | 0.066 | 0.132 | 1.160 | 27.6 | 3.2 | 75 | 128 | 316 |
| | 1 | 0.608 | 1.730 | 2.885 | 11.1 | 8.2 | 360 | 1256 | 314 |
| gin_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 3.3 | 75 | 177 | 450 |
| | 1 | 0.715 | 2.625 | 3.836 | 8.3 | 7.4 | 416 | 1292 | 361 |
| gin_citeseer | 0 | 0.009 | 0.017 | 1.020 | 31.4 | 3.1 | 45 | 139 | 190 |
| | 1 | 0.573 | 1.605 | 2.739 | 11.7 | 7.7 | 136 | 476 | 197 |
| gin_amazon-ratings | 0 | 0.722 | 1.451 | 2.566 | 12.5 | 3.2 | 680 | 737 | 247 |
| | 1 | 0.281 | 0.614 | 1.708 | 18.7 | 13.2 | 270 | 875 | 236 |
| sage_cora | 0 | 0.013 | 0.025 | 1.031 | 31.0 | 4.3 | 48 | 104 | 90 |
| | 1 | 0.659 | 2.074 | 3.175 | 10.1 | 4.6 | 708 | 2735 | 132 |

What this shows:
- **Widths hold with margin.** Shifter-Acc ≤ 708 (16-bit limit 32,767), Tile Accumulator
  ≤ 2,735 (32-bit), AU bound ≤ 2,124 incl. NELL. No overflow or saturation anywhere.
- **The bitmap costs about 1 bit per element in the stored format.** On hidden layers the
  stored size is values + ~1.0 bit (bitmap) + 5–13 % package padding, so CR_stored is roughly
  8–19× where the values-only number alone would suggest 12–20×. On X⁽⁰⁾ of bag-of-words
  sets the bitmap is nearly everything (CR_stored ≈ 31). This is MEGA's stated format (Q-19);
  it matters for the DRAM traffic M2 will report.
- **Most packages are full long ones** (192 bits); short/medium ones appear where the bitwidth
  changes between consecutive nodes, which is frequent in hidden layers (nodes are in dataset
  order; METIS relabelling in M2 will change this).

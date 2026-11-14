"""Train and evaluate the Degree-Aware quantized GCN.

Pipeline per seed (Q-14):
  1. FP32 GCN pretraining (Adam, lr 0.01, wd 5e-4, dropout 0.5). Keep the best-val-acc epoch.
  2. Calibrate α/β/γ from the FP32 model (Q-04, Q-08, Q-09).
  3. Quantization-aware fine-tuning with  L_total = L_task + λ·(M − M_target)²,  Eq.(4)–(5).
     Reported numbers come from the LAST QAT epoch, so bits and accuracy describe the same
     converged model (best-val selection could pick an epoch before the bits settle).

Examples
  python -m mega_pt.train --dataset cora --norm fp32 --seeds 1 --lam 0          # smoke test
  python -m mega_pt.train --dataset cora --norm fold --lam 1e-4 --target-dense-bits 2.0
  python -m mega_pt.train --dataset cora --norm per_source --frac-bits 6 --lam 1e-4 --m-target-kb 1000

Results go to runs/<run-name>/results.json (per-seed metrics, mean/std summary, args) and
runs/<run-name>/bits_seed<k>.json (learned per-degree bit/scale tables).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F

from .data import load_planetoid
from .metrics import accuracy, quant_report
from .model import ETA, NORM_MODES, GraphOps, QGCN


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cora", choices=["cora", "citeseer", "pubmed"])
    p.add_argument("--feat-norm", default="none", choices=["row", "none"],
                   help="input preprocessing. 'none' (raw binary/TF-IDF) is the default: row "
                        "normalization makes layer-0 values tiny, many round to 0, QAT lost ~3 pts on Cora")
    p.add_argument("--norm", default="fp32", choices=NORM_MODES, help="aggregation normalization mode (Q-11)")
    p.add_argument("--frac-bits", type=int, default=8, help="f in Q1.f for per_source mode")
    p.add_argument("--hidden", type=int, default=128, help="Tab.III: 128 for GCN")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01, help="weights, FP32 pretraining")
    # QAT learning rates. Tuned on Cora seed 0-1 (no memory penalty), see mega_pt/README.md:
    #   weights 1e-2 overfits during the extra QAT epochs (-0.7 pts vs 1e-3);
    #   α at 1e-2 drifts upward and the training loss RISES (STE bias on values that round
    #   to 0), costing up to 25 pts; 1e-3 is stable.
    p.add_argument("--lr-qat-weights", type=float, default=1e-3, help="weights during QAT")
    p.add_argument("--lr-alpha", type=float, default=1e-3, help="log α degree tables")
    p.add_argument("--lr-bits", type=float, default=1e-2, help="b degree tables (bits per step ≈ lr)")
    p.add_argument("--lr-colscale", type=float, default=1e-2, help="log β / log γ column scales")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay on weights only")
    p.add_argument("--epochs-fp32", type=int, default=200)
    p.add_argument("--epochs-qat", type=int, default=200)
    p.add_argument("--lam", type=float, default=0.0, help="λ in Eq.(5); 0 = no memory penalty")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--m-target-kb", type=float, help="M_target in KB (Eq.(4), dense)")
    g.add_argument("--target-dense-bits", type=float,
                   help="set M_target = N·Σ_l dim_l·bits/η, i.e. the dense memory at this mean bitwidth")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds 0..k-1 (Q-14: 10)")
    p.add_argument("--fp32-only", action="store_true", help="stop after FP32 pretraining")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    return p.parse_args(argv)


def train_fp32(model, g, ops, args):
    """Plain GCN training; returns test acc at the best-validation epoch."""
    opt = torch.optim.Adam(model.weight_params(), lr=args.lr, weight_decay=args.wd)
    best_val, best_state = -1.0, None
    for ep in range(args.epochs_fp32):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(g.x, ops)[g.train_idx], g.y[g.train_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = accuracy(model(g.x, ops), g.y, g.val_idx)
        if val > best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return accuracy(model(g.x, ops), g.y, g.test_idx), best_val


def train_qat(model, g, ops, args, m_target_kb):
    """Quantization-aware fine-tuning with the Eq.(5) loss. Returns the final-epoch model."""
    opt = torch.optim.Adam([
        {"params": model.weight_params(), "lr": args.lr_qat_weights, "weight_decay": args.wd},
        {"params": model.alpha_params(), "lr": args.lr_alpha, "weight_decay": 0.0},
        {"params": model.bit_params(), "lr": args.lr_bits, "weight_decay": 0.0},
        {"params": model.column_scale_params(), "lr": args.lr_colscale, "weight_decay": 0.0},
    ])
    for ep in range(args.epochs_qat):
        model.train()
        opt.zero_grad()
        logits = model(g.x, ops)
        task = F.cross_entropy(logits[g.train_idx], g.y[g.train_idx])
        mem = model.memory_kb(ops.deg)                        # Eq.(4) inner term, KB
        penalty = (mem - m_target_kb) ** 2 if args.lam > 0 else torch.zeros(())
        loss = task + args.lam * penalty                      # Eq.(5)
        loss.backward()
        opt.step()
        model.clamp_quant_params_()                           # keep b in [2, 8], scales > 0
        if args.log_every and (ep % args.log_every == 0 or ep == args.epochs_qat - 1):
            model.eval()
            with torch.no_grad():
                val = accuracy(model(g.x, ops), g.y, g.val_idx)
            print(f"  qat ep {ep:4d} task {task.item():.4f} mem {mem.item():9.1f} KB "
                  f"(target {m_target_kb:.1f}) val {val:.4f}")


def bit_tables(model) -> dict:
    return {f"layer{l}": {"bits": q.bits.detach().tolist(), "scale": q.scale.detach().tolist()}
            for l, q in enumerate(model.x_quant)}


def main(argv=None):
    args = parse_args(argv)
    g = load_planetoid(args.dataset, feat_norm=args.feat_norm)
    ops = GraphOps.from_graph(g)
    print(g.summary())
    dims = [g.x.shape[1], args.hidden, g.num_classes]

    # M_target (KB). Eq.(4) counts dense storage over each layer input (Q-10, Q-13).
    if args.target_dense_bits is not None:
        m_target = g.num_nodes * sum(dims[:-1]) * args.target_dense_bits / ETA
    else:
        m_target = args.m_target_kb if args.m_target_kb is not None else 0.0
    if args.lam > 0:
        print(f"M_target = {m_target:.1f} KB, λ = {args.lam}")

    name = args.run_name or (f"{args.dataset}_{args.norm}"
                             + (f"_f{args.frac_bits}" if args.norm == "per_source" else "")
                             + f"_lam{args.lam:g}" + (f"_mt{m_target:.0f}" if args.lam > 0 else ""))
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    per_seed = []
    for seed in range(args.seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        model = QGCN(dims, int(g.deg.max()), norm=args.norm, frac_bits=args.frac_bits, dropout=args.dropout)
        fp32_test, fp32_val = train_fp32(model, g, ops, args)
        row = {"seed": seed, "fp32_test": fp32_test, "fp32_val": fp32_val}
        if not args.fp32_only:
            model.calibrate(g.x, ops)
            model.quantize = True
            model.eval()
            with torch.no_grad():
                row["ptq_test"] = accuracy(model(g.x, ops), g.y, g.test_idx)  # before QAT, for debugging
            train_qat(model, g, ops, args, m_target)
            model.eval()
            with torch.no_grad():
                logits = model(g.x, ops)
            row["q_test"] = accuracy(logits, g.y, g.test_idx)
            row["q_val"] = accuracy(logits, g.y, g.val_idx)
            row.update(quant_report(model, g.x, ops))
            with open(os.path.join(out_dir, f"bits_seed{seed}.json"), "w") as f:
                json.dump(bit_tables(model), f)
        row["seconds"] = round(time.time() - t0, 1)
        per_seed.append(row)
        print(f"seed {seed}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            if k in ("fp32_test", "ptq_test", "q_test", "avg_bits_sparse", "avg_bits_dense_all",
                     "cr_sparse", "seconds") or "psum_max" in k or "lost_topdeg" in k))

    summary = {}
    for k, v in per_seed[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed":
            vals = [r[k] for r in per_seed]
            summary[k] = {"mean": statistics.fmean(vals),
                          "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "m_target_kb": m_target, "summary": summary, "per_seed": per_seed}, f, indent=1)

    print(f"\n== {name} ({args.seeds} seeds) -> {out_dir}/results.json")
    for k in ("fp32_test", "ptq_test", "q_test", "avg_bits_sparse", "cr_sparse",
              "avg_bits_dense_all", "memory_kb_eq4"):
        if k in summary:
            s = summary[k]
            scale = 100 if k.endswith("test") else 1
            print(f"  {k:20s} {s['mean'] * scale:8.2f} ± {s['std'] * scale:.2f}")


if __name__ == "__main__":
    main()

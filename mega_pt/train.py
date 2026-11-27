"""Train and evaluate the Degree-Aware quantized GCN / GIN / GraphSage (--model, Q-12).

GraphSage (--model sage): every training epoch (FP32 and QAT) uses a fresh neighbor sample
(--samples per node, plus self); evaluation, calibration and all reported metrics use ONE
fixed sample drawn from a generator seeded by the run seed (Q-12: sampling also at inference).

Pipeline per seed (Q-14):
  1. FP32 GCN pretraining (Adam, lr 0.01, wd 5e-4, dropout 0.5). Keep the best-val-acc epoch.
  2. Calibrate α/β/γ from the FP32 model (Q-04, Q-08, Q-09). Degree tables: MSE clip init.
  3. Quantization-aware fine-tuning with  L_total = L_task + λ·(M − M_target)²,  Eq.(4)–(5).
     b and α are both learned by gradient (default --x-param alpha --clip-refit 0, Q-04: the
     literal Eq.(2) form, chosen 2026-11-16 for fidelity to MEGA). Option: --x-param range
     --clip-refit 10 re-fits the clip ranges by MSE instead (matches Tab.VI, but uniform 2 bits).
     Reported numbers come from the LAST QAT epoch, so bits and accuracy describe the same
     converged model (best-val selection could pick an epoch before the bits settle).

Examples
  python -m mega_pt.train --dataset cora --seeds 1 --lam 0                       # smoke test
  python -m mega_pt.train --dataset pubmed --lam 1e-6 --target-dense-bits 2.0    # Tab.VI setting
  python -m mega_pt.train --dataset cora --lam 1e-5 --target-dense-bits 2.0      # (Cora needs 1e-5)
  python -m mega_pt.train --dataset cora --norm per_source --frac-bits 6 --lam 1e-7 --target-dense-bits 2.0
  python -m mega_pt.train --model gin --dataset citeseer --lam 1e-6 --target-dense-bits 2.0
  python -m mega_pt.train --model sage --dataset cora --lam 1e-6 --target-dense-bits 2.0

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

from .data import DATASETS, DEFAULT_FEAT_NORM, load_dataset
from .metrics import quant_report, score
from .model import ETA, MODELS, NORM_MODES, GraphOps, QGCN, residual_layers


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gcn", choices=MODELS,
                   help="Q-12: gcn; gin = GIN-0 (ε=0, one linear layer, sum over A+I); "
                        "sage = GraphSage-mean over self + sampled neighbors, single W")
    p.add_argument("--samples", type=int, default=25,
                   help="GraphSage neighbors sampled per node (Tab.III: 25); ignored otherwise")
    p.add_argument("--dataset", default="cora", choices=DATASETS,
                   help="MEGA's sets (cora/citeseer/pubmed/nell/reddit) or our extra evaluation "
                        "sets (docs/beyond-mega/01-extra-datasets.md). Datasets with several "
                        "published splits (roman-empire, amazon-ratings: 10) run seed s on split s%%10")
    p.add_argument("--feat-norm", default=None, choices=["row", "std", "none"],
                   help="input preprocessing; default per dataset (data.DEFAULT_FEAT_NORM): row = "
                        "row-normalize (Kipf; MEGA's bag-of-words sets, Q-14), std = per-column "
                        "z-score (embedding / mixed-unit features), none = raw (Reddit ships "
                        "already standardized). See README implementation choices")
    p.add_argument("--inductive", default=None, choices=["on", "off"],
                   help="train on the graph induced by the training nodes only (GraphSAGE's Reddit "
                        "protocol, Q-14); evaluate on the full graph. Default: on for reddit only")
    p.add_argument("--norm", default="fold", choices=NORM_MODES,
                   help="aggregation normalization mode (Q-11: decided fold)")
    p.add_argument("--x-param", default="alpha", choices=["range", "alpha"],
                   help="how the degree tables learn the scale: α directly (default, literal "
                        "Eq.(2)), or clip range c with α = c/q_max(b) (use with --clip-refit). "
                        "Same forward pass/hardware; see quant.py")
    p.add_argument("--x-init", default="mse", choices=["max", "mse"],
                   help="clip-range init of the degree tables: max|x| per degree, or the grid "
                        "fraction of it that minimizes quantization MSE at 4 bit; see quant.py")
    p.add_argument("--wb-init", default="mse", choices=["max", "mse"],
                   help="init of the 4-bit W/B per-column scales: MSE fit (default, Q-09 amended) "
                        "or max/7 (original Q-08/Q-09)")
    p.add_argument("--clip-refit", type=int, default=0,
                   help="0 (default) = scales learned by gradient, as the paper states. K > 0: "
                        "re-fit the degree tables' clip ranges by MSE at the current rounded b "
                        "every K QAT epochs instead (lr-alpha then ignored; pair with "
                        "--x-param range). See quant.py refit_clip_")
    p.add_argument("--lsq", action="store_true",
                   help="LSQ gradient scaling g=1/sqrt(N*Q_P) on the degree tables' learned "
                        "scale (only matters with --clip-refit 0, i.e. scales learned by gradient)")
    p.add_argument("--alpha-opt", default="adam", choices=["adam", "sgd"],
                   help="optimizer for the degree tables' scales. Adam normalizes each entry's "
                        "step, which cancels LSQ's per-entry constant g; LSQ itself used SGD")
    p.add_argument("--alpha-momentum", type=float, default=0.9, help="SGD momentum (--alpha-opt sgd)")
    p.add_argument("--x-init-bits", default=None,
                   help="starting bitwidth per layer, e.g. '4,2' (default: 4 everywhere, Q-04)")
    p.add_argument("--quant-parts", default="xwb",
                   help="DEBUG ablation: which quantizers are active (x = degree-aware features, "
                        "w = 4-bit weights, b = 4-bit B). Default 'xwb' is the real model")
    p.add_argument("--frac-bits", type=int, default=8, help="f in Q1.f for per_source mode")
    p.add_argument("--hidden", type=int, default=None,
                   help="default from Tab.III: 128 for GCN and GIN, 256 for GraphSage")
    p.add_argument("--layers", type=int, default=2,
                   help="GNN layers (Tab.III: 2). dims = [F] + [hidden]*(layers-1) + [classes]")
    p.add_argument("--residual", action="store_true",
                   help="identity residual X^(l+1) = ReLU(ÃX̄W̄ + X̄) on hidden→hidden layers "
                        "(ours, Q-36; needs --layers ≥ 3)")
    p.add_argument("--res-form", default="pre", choices=["pre", "post"],
                   help="residual added before (ReLU(AXW + X)) or after (ReLU(AXW) + X) the ReLU")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01, help="weights, FP32 pretraining")
    # QAT learning rates. Tuned on Cora seed 0-1 (no memory penalty), see mega_pt/README.md:
    #   weights 1e-2 overfits during the extra QAT epochs (-0.7 pts vs 1e-3);
    #   α at 1e-2 drifts upward and the training loss RISES (STE bias on values that round
    #   to 0), costing up to 25 pts; 1e-3 is stable.
    p.add_argument("--lr-qat-weights", type=float, default=1e-3, help="weights during QAT")
    p.add_argument("--lr-alpha", type=float, default=1e-3, help="log α / log c degree tables")
    p.add_argument("--lr-bits", type=float, default=2e-2, help="b degree tables (bits per step ≈ lr)")
    p.add_argument("--lr-colscale", type=float, default=1e-2, help="log β / log γ column scales")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay on weights only")
    p.add_argument("--epochs-fp32", type=int, default=200)
    p.add_argument("--epochs-qat", type=int, default=200)
    p.add_argument("--lam", type=float, default=0.0, help="λ in Eq.(5); 0 = no memory penalty")
    p.add_argument("--lam-rel", type=float, default=None,
                   help="set λ = lam_rel / M_target², i.e. penalty lam_rel·((M − M_target)/M_target)². "
                        "Scale-free: the same value pins M to a reachable M_target on any dataset "
                        "(300 ≈ λ 1e-5 on Roman-Empire 4-layer, which held M within 0.06%%). "
                        "Residual work only (2026-11-27); MEGA-baseline runs use --lam")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--m-target-kb", type=float, help="M_target in KB (Eq.(4), dense)")
    g.add_argument("--target-dense-bits", type=float,
                   help="set M_target = N·Σ_l dim_l·bits/η, i.e. the dense memory at this mean bitwidth")
    g.add_argument("--target-layer-bits", type=str,
                   help="comma list, one mean bitwidth per quantized layer input, e.g. '2,2.6': "
                        "M_target = N·Σ_l dim_l·bits_l/η. Preferred: it makes the target reachable "
                        "(b >= 2), so with a large λ the penalty acts as a memory constraint")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds 0..k-1 (Q-14: 10)")
    p.add_argument("--seed-start", type=int, default=0, help="first seed (DEBUG: rerun one seed)")
    p.add_argument("--fp32-only", action="store_true", help="stop after FP32 pretraining")
    p.add_argument("--fp32-cache", default=None,
                   help="directory to save / reuse the FP32-pretrained model per seed (keyed by "
                        "every argument that affects pretraining). Saves the ~20 min/seed "
                        "pretraining on Reddit when sweeping λ. The RNG states are saved too, so a "
                        "cached run continues exactly like an uncached one")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-model", action="store_true",
                   help="save the trained quantized model per seed (model_seed<s>.pt in the run "
                        "dir): input for mega_pt.hw_stats and the simulator")
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    return p.parse_args(argv)


def train_fp32(model, g, ops, args, train_ops):
    """Plain (unquantized) training; returns test acc at the best-validation epoch.
    ops: the evaluation graph; train_ops(): the graph for one training step (a fresh
    neighbor sample for GraphSage, else the same ops)."""
    opt = torch.optim.Adam(model.weight_params(), lr=args.lr, weight_decay=args.wd)
    best_val, best_state, best_ep = -1.0, None, -1
    curve = []                                 # val metric per epoch (convergence analysis)
    for ep in range(args.epochs_fp32):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(g.x, train_ops())[g.train_idx], g.y[g.train_idx])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = score(model(g.x, ops), g.y, g.val_idx, g.metric)
        curve.append(round(val, 5))
        if val > best_val:
            best_val, best_ep = val, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    train_fp32.best_epoch = best_ep            # read by main() for the results row
    train_fp32.val_curve = curve               # ditto (2026-11-26): epochs-to-threshold analysis
    with torch.no_grad():
        return score(model(g.x, ops), g.y, g.test_idx, g.metric), best_val


def train_qat(model, g, ops, args, m_target_kb, train_ops):
    """Quantization-aware fine-tuning with the Eq.(5) loss. Returns the final-epoch model."""
    groups = [
        {"params": model.weight_params(), "lr": args.lr_qat_weights, "weight_decay": args.wd},
        {"params": model.bit_params(), "lr": args.lr_bits, "weight_decay": 0.0},
        {"params": model.column_scale_params(), "lr": args.lr_colscale, "weight_decay": 0.0},
    ]
    # with --clip-refit the clip ranges are set by MSE re-fits, not by the optimizer
    alpha_lr = 0.0 if args.clip_refit else args.lr_alpha
    for q in model.x_quant:
        q.lsq = args.lsq
    opts = []
    if args.alpha_opt == "adam":
        groups.append({"params": model.alpha_params(), "lr": alpha_lr, "weight_decay": 0.0})
    else:
        # LSQ setup: plain SGD with momentum, so the per-degree factor g changes step sizes
        # (Adam divides it back out). Still on log c; see README "LSQ check".
        opts.append(torch.optim.SGD(model.alpha_params(), lr=alpha_lr,
                                    momentum=args.alpha_momentum))
    opts.append(torch.optim.Adam(groups))
    for ep in range(args.epochs_qat):
        if args.clip_refit and ep % args.clip_refit == 0:
            model.refit_clips_(g.x, ops)
        model.train()
        for o in opts:
            o.zero_grad()
        step_ops = train_ops()
        logits = model(g.x, step_ops)
        task = F.cross_entropy(logits[g.train_idx], g.y[g.train_idx])
        # Eq.(4) inner term, KB. Degrees of the graph this step runs on, except in inductive
        # mode: there val/test nodes are isolated in the training graph (degree 1), which would
        # misstate the memory of the graph the hardware actually stores, so count the
        # evaluation graph instead (for GraphSage: the fixed evaluation sample).
        mem = model.memory_kb(ops.deg if args.inductive == "on" else step_ops.deg)
        penalty = (mem - m_target_kb) ** 2 if args.lam > 0 else torch.zeros(())
        loss = task + args.lam * penalty                      # Eq.(5)
        loss.backward()
        for o in opts:
            o.step()
        model.clamp_quant_params_()                           # keep b in [2, 8], scales > 0
        if args.log_every and (ep % args.log_every == 0 or ep == args.epochs_qat - 1):
            model.eval()
            with torch.no_grad():
                val = score(model(g.x, ops), g.y, g.val_idx, g.metric)
            print(f"  qat ep {ep:4d} task {task.item():.4f} mem {mem.item():9.1f} KB "
                  f"(target {m_target_kb:.1f}) val {val:.4f}")
    if args.clip_refit:
        model.refit_clips_(g.x, ops)   # final ranges match the final b and weights


# Arguments that change FP32 pretraining; the cache key must include all of them.
FP32_KEYS = ("model", "dataset", "feat_norm", "inductive", "samples", "hidden", "dropout", "lr",
             "wd", "epochs_fp32", "norm", "frac_bits", "x_param", "layers", "residual", "res_form")


def _fp32_cache_path(args, seed: int):
    if not args.fp32_cache:
        return None
    # layers/residual are left out of the key at their defaults, so 2-layer caches made
    # before those options existed keep their names
    # (getattr defaults: sgcnsim calls this with the args of runs made before those options)
    layers, residual = getattr(args, "layers", 2), getattr(args, "residual", False)
    res_form = getattr(args, "res_form", "pre")
    key = "_".join(f"{k}={getattr(args, k, None)}" for k in FP32_KEYS
                   if not (k == "layers" and layers == 2) and not (k == "residual" and not residual)
                   and not (k == "res_form" and (res_form == "pre" or not residual)))
    import hashlib
    return os.path.join(args.fp32_cache, f"{args.model}_{args.dataset}_s{seed}_"
                        f"{hashlib.sha1(key.encode()).hexdigest()[:10]}.pt")


def bit_tables(model) -> dict:
    return {f"layer{l}": {"bits": q.bits.detach().tolist(), "scale": q.scale.detach().tolist()}
            for l, q in enumerate(model.x_quant)}


def main(argv=None):
    args = parse_args(argv)
    args.feat_norm = args.feat_norm or DEFAULT_FEAT_NORM[args.dataset]
    args.inductive = args.inductive or ("on" if args.dataset == "reddit" else "off")
    g_all = load_dataset(args.dataset, feat_norm=args.feat_norm)
    print(g_all.summary(), f"| feat_norm={args.feat_norm} inductive={args.inductive}"
          + (f" | {len(g_all.splits)} splits, seed s uses split s % {len(g_all.splits)}" if g_all.splits else ""))
    g = g_all
    sampled = args.model == "sage"
    if args.hidden is None:
        args.hidden = 256 if args.model == "sage" else 128          # Tab.III
    dims = [g.num_features] + [args.hidden] * (args.layers - 1) + [g.num_classes]
    if args.residual:
        assert residual_layers(dims), f"--residual needs a hidden→hidden layer, dims={dims}"

    # M_target (KB). Eq.(4) counts dense storage over each layer input (Q-10, Q-13).
    if args.target_layer_bits is not None:
        lb = [float(v) for v in args.target_layer_bits.split(",")]
        assert len(lb) == len(dims) - 1, f"need {len(dims) - 1} values, got {lb}"
        m_target = g.num_nodes * sum(d * b for d, b in zip(dims[:-1], lb)) / ETA
    elif args.target_dense_bits is not None:
        m_target = g.num_nodes * sum(dims[:-1]) * args.target_dense_bits / ETA
    else:
        m_target = args.m_target_kb if args.m_target_kb is not None else 0.0
    if args.lam_rel is not None:
        assert m_target > 0, "--lam-rel needs an M_target"
        args.lam = args.lam_rel / m_target ** 2
    if args.lam > 0:
        print(f"M_target = {m_target:.1f} KB, λ = {args.lam}")

    name = args.run_name or (("" if args.model == "gcn" else f"{args.model}_") + f"{args.dataset}_{args.norm}"
                             + (f"_f{args.frac_bits}" if args.norm == "per_source" else "")
                             + (f"_L{args.layers}" if args.layers != 2 else "") + (("_res" + ("post" if args.res_form == "post" else "")) if args.residual else "")
                             + f"_lam{args.lam:g}" + (f"_mt{m_target:.0f}" if args.lam > 0 else ""))
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    per_seed = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        g = g_all.use_split(seed)                  # no-op for single-split datasets
        # Training graph: the full graph (transductive, default) or, for Reddit, the graph
        # induced by the training nodes (inductive; val/test nodes and their edges unseen).
        g_train = g.train_subgraph() if args.inductive == "on" else g
        if sampled:
            # one fixed evaluation sample per run, and a separate stream for training samples
            ops = GraphOps.from_graph(g, "sage", args.samples, torch.Generator().manual_seed(seed))
            train_gen = torch.Generator().manual_seed(10_000 + seed)
            train_ops = lambda: GraphOps.from_graph(g_train, "sage", args.samples, train_gen)
            max_degree = args.samples + 1          # degree tables cover every sampled degree
        else:
            ops = GraphOps.from_graph(g, args.model)
            t_ops = ops if g_train is g else GraphOps.from_graph(g_train, args.model)
            train_ops = lambda: t_ops
            max_degree = int(g.deg.max())          # ≥ every training-graph degree too
        model = QGCN(dims, max_degree, norm=args.norm, frac_bits=args.frac_bits, dropout=args.dropout,
                     x_param=args.x_param, residual=args.residual, res_form=args.res_form)
        cache = _fp32_cache_path(args, seed)
        if cache and os.path.exists(cache):
            ck = torch.load(cache, weights_only=False)        # our own file (see _fp32_cache_path)
            model.load_state_dict(ck["model"])
            torch.set_rng_state(ck["rng"])
            if sampled:
                train_gen.set_state(ck["train_gen"])
            fp32_test, fp32_val = ck["fp32_test"], ck["fp32_val"]
            # older caches lack these; NaN marks "not measured" (the FP32 phase was skipped)
            fp32_seconds, fp32_best_epoch = ck.get("fp32_seconds", float("nan")), ck.get("fp32_best_epoch", -1)
            fp32_val_curve = ck.get("fp32_val_curve")
            print(f"  FP32 model loaded from {cache}")
        else:
            t_fp32 = time.time()
            fp32_test, fp32_val = train_fp32(model, g, ops, args, train_ops)
            fp32_seconds, fp32_best_epoch = round(time.time() - t_fp32, 1), train_fp32.best_epoch
            fp32_val_curve = train_fp32.val_curve
            if cache:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                torch.save({"model": model.state_dict(), "rng": torch.get_rng_state(),
                            "train_gen": train_gen.get_state() if sampled else None,
                            "fp32_test": fp32_test, "fp32_val": fp32_val,
                            "fp32_seconds": fp32_seconds, "fp32_best_epoch": fp32_best_epoch,
                            "fp32_val_curve": fp32_val_curve}, cache)
        # Phase timings (2026-11-25): wall seconds of the FP32 and QAT phases and the epoch the
        # FP32 best-val model came from (a late best epoch = still improving at --epochs-fp32).
        row = {"seed": seed, "fp32_test": fp32_test, "fp32_val": fp32_val,
               "fp32_seconds": fp32_seconds, "fp32_best_epoch": fp32_best_epoch}
        if fp32_val_curve is not None:        # list; the summary below skips non-scalars
            row["fp32_val_curve"] = fp32_val_curve
        if not args.fp32_only:
            model.calibrate(g.x, ops, args.x_init,
                            [float(v) for v in args.x_init_bits.split(",")] if args.x_init_bits else None,
                            args.wb_init)
            model.quantize = True
            model.quant_parts = args.quant_parts
            model.eval()
            with torch.no_grad():
                row["ptq_test"] = score(model(g.x, ops), g.y, g.test_idx, g.metric)  # before QAT, for debugging
            t_qat = time.time()
            train_qat(model, g, ops, args, m_target, train_ops)
            row["qat_seconds"] = round(time.time() - t_qat, 1)
            model.eval()
            with torch.no_grad():
                logits = model(g.x, ops)
            row["q_test"] = score(logits, g.y, g.test_idx, g.metric)
            row["q_val"] = score(logits, g.y, g.val_idx, g.metric)
            row.update(quant_report(model, g.x, ops))
            with open(os.path.join(out_dir, f"bits_seed{seed}.json"), "w") as f:
                json.dump(bit_tables(model), f)
            if args.save_model:
                # Everything needed to rebuild the exact model and graph sample:
                # hw_stats.load_run() re-creates dims / max_degree / ops from args + seed.
                torch.save({"state": model.state_dict(), "dims": dims, "max_degree": max_degree,
                            "args": vars(args), "seed": seed},
                           os.path.join(out_dir, f"model_seed{seed}.pt"))
        row["seconds"] = round(time.time() - t0, 1)
        per_seed.append(row)
        print(f"seed {seed}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            if k in ("fp32_test", "ptq_test", "q_test", "avg_bits_nz", "avg_bits_sparse", "avg_bits_dense_all",
                     "cr_sparse", "seconds", "fp32_seconds", "qat_seconds", "fp32_best_epoch") or "psum_max" in k or "lost_topdeg" in k))

    summary = {}
    for k, v in per_seed[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed":
            # NaN = not measured (fp32_seconds of a cached FP32 model); summarize the rest
            vals = [r[k] for r in per_seed if r[k] == r[k]]
            if not vals:
                continue
            summary[k] = {"mean": statistics.fmean(vals),
                          "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "m_target_kb": m_target, "summary": summary, "per_seed": per_seed}, f, indent=1)

    print(f"\n== {name} ({args.seeds} seeds) -> {out_dir}/results.json")
    for k in ("fp32_test", "ptq_test", "q_test", "avg_bits_nz", "avg_bits_sparse", "cr_sparse",
              "avg_bits_dense_all", "memory_kb_eq4"):
        if k in summary:
            s = summary[k]
            scale = 100 if k.endswith("test") else 1
            print(f"  {k:20s} {s['mean'] * scale:8.2f} ± {s['std'] * scale:.2f}")


if __name__ == "__main__":
    main()

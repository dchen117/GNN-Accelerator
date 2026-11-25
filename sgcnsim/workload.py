"""The workload SGCN runs: the FP32 GCN of a trained run (G-02 in docs/sgcn/00-design.md).

SGCN computes in 32-bit (MEGA Tab.V), so it runs the FP32 model, not MEGA's quantized one.
Every hardware run in runs/hw was QAT-fine-tuned from an FP32 pretraining whose weights are
cached in `args.fp32_cache` (mega_pt/train.py); we load that model, run its forward pass in
eval mode and keep what the hardware needs:

    graph       CSR by destination (A+I, rows = destinations, edge values Ã)
    x0          CSR of the input features X⁽⁰⁾ (values only matter for sparsity)
    hidden      per hidden layer l ≥ 1: the ReLU output X⁽ˡ⁾ (dense float32) → its zero pattern
    dims        [F0, hidden, ..., classes]   (args.layers deep; residual runs add + X⁽ˡ⁾)

    w = load_workload("runs/hw/gcn_cora")
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch

from mega_pt.data import load_dataset
from mega_pt.model import GraphOps, QGCN


@dataclass
class Workload:
    name: str
    n: int
    dims: list                      # [F0, F1, ..., classes]
    ptr: np.ndarray                 # (n+1,) CSR row pointer, rows = destinations (A+I)
    src: np.ndarray                 # (E,) source of each edge, sorted by destination
    x0: sp.csr_matrix               # (n, F0) input features
    hidden: list = field(default_factory=list)   # X⁽ˡ⁾ for l = 1 .. L-1, (n, F_l) float32
    fp32_test_acc: float = float("nan")

    @property
    def layers(self) -> int:
        return len(self.dims) - 1

    @property
    def edges(self) -> int:
        return int(self.src.size)


def _fp32_checkpoint(args: dict, seed: int) -> str:
    """The FP32 pretraining cache file of the run: the exact name train.py wrote, recomputed
    from the run's args (a glob on model/dataset/seed is ambiguous once 2-layer, 4-layer and
    residual caches share a directory)."""
    from types import SimpleNamespace
    from mega_pt.train import _fp32_cache_path
    path = _fp32_cache_path(SimpleNamespace(**args), seed)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"FP32 checkpoint {path} not found (train with --fp32-cache)")
    return path


def _to_csr(x: torch.Tensor) -> sp.csr_matrix:
    if x.is_sparse:
        x = x.coalesce()
        i = x.indices().numpy()
        return sp.csr_matrix((x.values().numpy(), (i[0], i[1])), shape=tuple(x.shape))
    return sp.csr_matrix(x.numpy())


@torch.no_grad()
def load_workload(run_dir: str, seed: int = 0) -> Workload:
    args = json.load(open(os.path.join(run_dir, "results.json")))["args"]
    if args["model"] != "gcn":
        # MEGA compares with SGCN on GCN only (Fig.14/16/17); GIN/Sage would need their own
        # FP32 checkpoints and aggregation rules.
        raise ValueError(f"sgcnsim supports GCN runs, got {args['model']}")
    g = load_dataset(args["dataset"], feat_norm=args["feat_norm"]).use_split(seed)
    ops = GraphOps.from_graph(g, "gcn")
    dims = [g.num_features] + [args["hidden"]] * (args.get("layers", 2) - 1) + [g.num_classes]
    model = QGCN(dims, int(g.deg.max()), norm=args["norm"], frac_bits=args["frac_bits"],
                 dropout=args["dropout"], x_param=args["x_param"],
                 residual=args.get("residual", False))
    ck = torch.load(_fp32_checkpoint(args, seed), weights_only=False)   # our own file
    model.load_state_dict(ck["model"])
    model.eval()                                   # quantize=False: plain FP32 forward
    # forward pass, keeping every hidden layer's input
    x, hidden = g.x, []
    for l, W in enumerate(model.weights):
        h = torch.sparse.mm(ops.a_norm, x @ W if not x.is_sparse else torch.sparse.mm(x, W))
        if l in model.residual_layers:     # identity residual (Q-36 / G-08), FP32
            h = h + x
        if l < len(model.weights) - 1:
            h = torch.relu(h)
            hidden.append(h.numpy().astype(np.float32))
        x = h
    acc = float((x.argmax(1)[g.test_idx] == g.y[g.test_idx]).float().mean())
    a = ops.a_norm.coalesce()                      # rows = destinations, sorted row-major
    dst, src = a.indices().numpy()
    ptr = np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=g.num_nodes))])
    return Workload(name=os.path.basename(os.path.normpath(run_dir)), n=g.num_nodes, dims=dims,
                    ptr=ptr.astype(np.int64), src=src.astype(np.int64), x0=_to_csr(g.x),
                    hidden=hidden, fp32_test_acc=acc)

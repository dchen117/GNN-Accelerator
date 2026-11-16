"""DEBUG tool: post-training-quantization (PTQ) accuracy vs. bitwidth, no QAT.

For each seed: train FP32, then for every per-layer bitwidth pair (b0, b1) set ALL degree
entries of layer l to b_l, fit each degree's clip range by MSE at that b on the FP32 layer
inputs, and report test accuracy and avg_bits_sparse. W and B stay at 4 bit.

Used on 2026-11-15 to show that low bits were not what broke QAT on CiteSeer/PubMed
(mega_pt/README.md, "Quantizer fix"). Takes the same options as train.py, e.g.

    python -m mega_pt.ptq_frontier --dataset pubmed --seeds 2
"""
import itertools
import sys

import torch
import torch.nn.functional as F

from .data import load_dataset
from .metrics import accuracy, quant_report
from .model import GraphOps, QGCN, _matmul
from .train import parse_args, train_fp32

BITS = [2, 3, 4, 5]


def main(argv=None):
    args = parse_args(argv)
    g = load_dataset(args.dataset, feat_norm=args.feat_norm)
    ops = GraphOps.from_graph(g)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        torch.manual_seed(seed)
        m = QGCN([g.num_features, args.hidden, g.num_classes], int(g.deg.max()), norm=args.norm,
                 frac_bits=args.frac_bits, dropout=args.dropout)
        fp, _ = train_fp32(m, g, ops, args)
        m.calibrate(g.x, ops, "mse", wb_init=args.wb_init)
        with torch.no_grad():                       # FP32 input of every layer
            xs, x = [g.x], g.x
            for l in range(len(m.weights) - 1):
                x = F.relu(torch.sparse.mm(ops.a_norm, _matmul(x, m.weights[l])))
                xs.append(x)
        m.quantize = True
        m.eval()
        for bits in itertools.product(BITS, repeat=len(m.weights)):
            with torch.no_grad():
                for l, b in enumerate(bits):
                    m.x_quant[l].init_from_activations(xs[l], ops.deg, "mse", bits=b)
                acc = accuracy(m(g.x, ops), g.y, g.test_idx)
                rep = quant_report(m, g.x, ops)
            print(f"seed {seed} fp32 {fp:.3f} b={bits} ptq {acc:.3f} "
                  f"sparse {rep['avg_bits_sparse']:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

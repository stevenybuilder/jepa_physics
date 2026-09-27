"""CPU-vs-GPU parity check for extracted meanpool (spec section 2, revised rule).

  python scripts/check_parity.py --ref A --test B [--ref2 C --test2 D] [--json out.json]

Each argument is an extraction output dir holding chunk_0000_meanpool.npy + chunk_0000.json.
Rule 1: per-layer relative error max|ref - test| / max|ref| < 1e-3 (ref = local CPU, test = box GPU).
Rule 2 (optional pair): the batch gap max|ref2 - test2| (GPU bs8 vs bs16, first rows matched by id)
must not exceed 2x the CPU-vs-CPU gap max|ref - cpu2| passed as --cpu2.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REL_TOL, GAP_RATIO = 1e-3, 2.0


def load(d):
    d = Path(d)
    ids = json.loads((d / "chunk_0000.json").read_text())["ids"]
    return ids, np.load(d / "chunk_0000_meanpool.npy")


def aligned(a, b):
    (ia, xa), (ib, xb) = load(a), load(b)
    common = [i for i in ia if i in ib]
    return xa[[ia.index(i) for i in common]], xb[[ib.index(i) for i in common]], common


p = argparse.ArgumentParser()
p.add_argument("--ref", required=True, help="local CPU")
p.add_argument("--test", required=True, help="box GPU, batch 8")
p.add_argument("--cpu2", help="box CPU")
p.add_argument("--test2", help="box GPU, batch 16")
p.add_argument("--json", type=Path)
args = p.parse_args()

a, b, ids = aligned(args.ref, args.test)
d = np.abs(a - b)
per_layer_abs = d.max(axis=(0, 2))
per_layer_rel = per_layer_abs / np.abs(a).max(axis=(0, 2))
out = {"ids": ids, "rel_tol": REL_TOL, "per_layer_max_abs": per_layer_abs.tolist(),
       "per_layer_rel": per_layer_rel.tolist(), "max_abs": float(d.max()),
       "max_rel": float(per_layer_rel.max()), "rule1_pass": bool(per_layer_rel.max() < REL_TOL)}
print(f"rule 1: max per-layer rel {per_layer_rel.max():.2e} (layer {per_layer_rel.argmax()}), "
      f"max|d| {d.max():.2e} -> {'PASS' if out['rule1_pass'] else 'FAIL'}")
if args.cpu2 and args.test2:
    x, y, _ = aligned(args.ref, args.cpu2)
    cpu_gap = float(np.abs(x - y).max())
    x, y, _ = aligned(args.test, args.test2)
    batch_gap = float(np.abs(x - y).max())
    out.update({"cpu_vs_cpu_gap": cpu_gap, "gpu_batch_gap": batch_gap, "gap_ratio_limit": GAP_RATIO,
                "rule2_pass": batch_gap <= GAP_RATIO * cpu_gap})
    print(f"rule 2: batch gap {batch_gap:.2e} vs cpu gap {cpu_gap:.2e} "
          f"(ratio {batch_gap / cpu_gap:.2f}) -> {'PASS' if out['rule2_pass'] else 'FAIL'}")
if args.json:
    args.json.write_text(json.dumps(out, indent=1))

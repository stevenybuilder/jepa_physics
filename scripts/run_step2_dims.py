"""Step 2 extra (spec §6 item 8): "how many dimensions is direction?" as four estimands, per layer,
direction vs speed, on train clips only (test is not read). Writes results/p1b_dims_four_ways.json.

  python scripts/run_step2_dims.py [--points 8 12] [--no-mlp] [--mlp-folds 5] [--eps 1e-3 1e-2 1e-1]

(a) literal K (the paper's count; read from p1b_*_dims.json when step 2 --all-layers has run, else
    recomputed) with the random-removal R² at the same rank; (b) whitened count, rank(I - T), at each
    shrinkage eps (the first eps is primary); (c) ridge and MLP R² before / after a LEACE erasure of the
    target (rank 2 for direction, 1 for speed); (d) split-half DFT spectrum of the 64 direction centroids.
A planted-ring control (clean, sheared, with a k = 3 harmonic, three copies) goes through the same code.
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.dims import leace_scores, planted_ring, split_half_spectrum, whitened_count  # noqa: E402
from wm.inlp import inlp, random_removal_curve  # noqa: E402
from wm.probes import (ALPHAS, N_POINTS, RESULTS, cv_select_alpha, layer_fraction, load_activations,  # noqa: E402
                       load_sweep, load_table, result_name, score, split_rows, standardized_layer, targets,
                       write_json)

parser = argparse.ArgumentParser()
parser.add_argument("--variables", nargs="+", default=["direction", "speed"])
parser.add_argument("--points", nargs="+", type=int, default=None, help="default: all 26 layer points")
parser.add_argument("--eps", nargs="+", type=float, default=[1e-2, 1e-3, 1e-1])
parser.add_argument("--no-mlp", action="store_true")
parser.add_argument("--mlp-folds", type=int, default=5)
parser.add_argument("--seeds", type=int, default=10, help="random-removal seeds")
parser.add_argument("--no-control", action="store_true")
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
args = parser.parse_args()
results = args.results or RESULTS


def four_ways(Xtr, Ytr, folds, kind, theta=None, alpha=None, literal=None):
    """All four estimands on one train matrix. literal = precomputed (K, dims) or None."""
    sf = partial(score, kind=kind)
    dummy = np.linspace(0, len(Xtr) - 1, 20).astype(int)     # inlp's test-score slot, unused here
    if literal is None:
        alpha = alpha or cv_select_alpha(Xtr, Ytr, folds, ALPHAS, sf)["alpha"]
        s, *_ = inlp(Xtr, Ytr, Xtr[dummy], Ytr[dummy], folds, alpha, sf, kind)
        literal = (s["K"], s["dims"])
    row = {"literal": {"K": literal[0], "dims": literal[1]}}
    if literal[1] > 0:
        alpha = alpha or cv_select_alpha(Xtr, Ytr, folds, ALPHAS, sf)["alpha"]
        rnd = random_removal_curve(Xtr, Ytr, Xtr[dummy], Ytr[dummy], folds, alpha, sf, [literal[1]], args.seeds)["rows"][0]
        row["literal"]["random_removal_cv_r2"] = {"mean": rnd["cv_r2"], "seed_sd": rnd["cv_r2_seed_sd"],
                                                  "seeds": args.seeds}
    row["whitened"] = [whitened_count(Xtr, Ytr, folds, kind, eps) for eps in args.eps]
    row["leace"] = leace_scores(Xtr, Ytr, folds, kind, args.eps[0], not args.no_mlp, args.mlp_folds)
    if theta is not None:
        row["dft"] = split_half_spectrum(Xtr, theta)
    return row


out = {"label": "extra (spec §6 item 8); does not replace the paper's count", "eps_primary": args.eps[0],
       "clips": "train only; 5 train folds", "variables": {}}
if not args.no_control:
    control = {}
    folds_c = np.arange(1200) % 5
    for name, kw in (("clean_ring", {}), ("sheared_ring_x30", {"shear": 30}),
                     ("ring_with_k3_harmonic", {"harmonics": {1: 1.0, 3: 0.7}}), ("three_copies", {"copies": 3})):
        X, th = planted_ring(**kw)
        r = np.radians(th)
        control[name] = {"planted": {k: (str(v) if isinstance(v, dict) else v) for k, v in kw.items()},
                         **four_ways(X, np.stack([np.sin(r), np.cos(r)], 1), folds_c, "circular", th)}
        c = control[name]
        print(f"control {name}: literal K {c['literal']['K']}, whitened rank {c['whitened'][0]['rank_I_minus_T']}, "
              f"LEACE ridge R2 {c['leace']['ridge_r2_after']:.3f}, DFT PR {c['dft']['participation_ratio']:.2f}")
    out["control"] = control

for variable in args.variables:
    df = load_table(variable)
    Y, kind, _ = targets(df, variable)
    tr, te, folds = split_rows(variable, df)
    acts = load_activations(variable, act_root=args.act_root)
    dims_path = results / (result_name("p1b", variable, variable, "meanpool") + "_dims.json")
    stored = {r["point"]: (r["K"], r["dims"]) for r in json.loads(dims_path.read_text())["layers"]} \
        if dims_path.exists() else {}
    try:
        sweep = load_sweep(variable, variable, "meanpool", results)
    except FileNotFoundError:
        sweep = None
    theta = df["theta_degrees"].to_numpy(float)[tr] if kind == "circular" else None
    rows = []
    for point in args.points if args.points is not None else range(N_POINTS):
        Xtr, _ = standardized_layer(acts, point, tr, te)
        alpha = sweep["layers"][point]["alpha"] if sweep else None
        row = {"point": point, "frac": layer_fraction(point), "post_ln": point == N_POINTS - 1,
               "literal_source": "p1b dims file" if point in stored else "recomputed",
               **four_ways(Xtr, Y[tr], folds, kind, theta, alpha, stored.get(point))}
        rows.append(row)
        print(f"{variable} point {point:2d}: literal dims {row['literal']['dims']}, whitened rank "
              f"{row['whitened'][0]['rank_I_minus_T']}, LEACE ridge R2 {row['leace']['ridge_r2_after']:.3f}"
              + (f", MLP R2 {row['leace']['mlp_r2_after']:.3f}" if "mlp_r2_after" in row["leace"] else "")
              + (f", DFT PR {row['dft']['participation_ratio']}" if "dft" in row else ""))
    out["variables"][variable] = {"kind": kind, "layers": rows}
write_json(results / "p1b_dims_four_ways.json", out)

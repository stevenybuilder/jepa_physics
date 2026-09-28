"""Paper-scale rows on constant-velocity clips only (the paper's velocity set is constant-speed, App. A l.669-675).

  python scripts/run_paperscale_velocity.py [--n-large 392] [--seeds 10] [--n-boot 200]

Same recipe as run_step1_paperscale.py (dual ridge, α on 5 folds, 90%-of-max onset, clip-bootstrap CI), direction
set, mean-pool, V-JEPA 2 and random-init, train rows of splits/split_v1.json only, restricted to motion == velocity:
  dir8_velocity          the paper's 8 directions, 5 random folds grouped by frame_hash, seed 0
  dir8_grouped_velocity  same clips, 5 folds holding out whole directions, seed 0 (pooled OOF R²)
  n{N}_velocity          --n-large random velocity train clips over all 64 directions (capped at the number
                         available), seeds 1000+s for the draw and s for the folds, s < --seeds (as `matched`)
The stored mixed-motion rows (results/p1a_paperscale_direction_direction.json) are copied beside them.
Writes results/p1a_paperscale_velocity_only.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import load_table  # noqa: E402
from wm.paperscale import (ci_from_draws, direction_grouped_folds, eight_direction_rows, motion_rows,  # noqa: E402
                           onset_draws, probe_rows, random_folds, summarise_curve)
from wm.probes import LAST_BLOCK, N_POINTS, RESULTS, availability, layer_matrix, load_activations, split_rows, \
    targets, write_json  # noqa: E402
from wm.splits import load_split  # noqa: E402

MODELS = ("vjepa2", "random")
SHOW = (1, 8)

parser = argparse.ArgumentParser()
parser.add_argument("--n-large", type=int, default=392)
parser.add_argument("--seeds", type=int, default=10)
parser.add_argument("--n-boot", type=int, default=200)
parser.add_argument("--results", type=Path, default=RESULTS)
args = parser.parse_args()

dataset, variable = "direction", "direction"
df = load_table(dataset)
Y, kind, score_fn = targets(df, variable)
theta = df["theta_degrees"].to_numpy(float)
motion = df["motion"].to_numpy()
train, _, _ = split_rows(dataset, df)
unit = np.array([load_split(dataset)["frame_hash"][str(i)] for i in df["id"]])
vel = motion_rows(train, motion, "velocity")
rows8 = vel[eight_direction_rows(theta[vel])]
n_large = min(args.n_large, len(vel))
large = f"n{n_large}_velocity"
specs = {"dir8_velocity": (rows8, random_folds(len(rows8), 5, 0, unit[rows8]), "fold_mean"),
         "dir8_grouped_velocity": (rows8, direction_grouped_folds(theta[rows8], 5, 0), "pooled")}
for s in range(args.seeds):
    r = np.sort(np.random.default_rng(1000 + s).choice(vel, n_large, replace=False))
    specs[f"{large}/{s}"] = (r, random_folds(n_large, 5, s, unit[r]), "fold_mean")
stored = json.loads((args.results / "p1a_paperscale_direction_direction.json").read_text())

out = {"dataset": dataset, "variable": variable, "pool": "meanpool", "label": "PART 1 EXTRA (paper scale, velocity only)",
       "why": "the paper's velocity set (App. A l.669-675) is constant-speed; the stored paper-scale rows mix "
              f"{stored['motion_counts_dir8']} clips",
       "subsample_pool": f"train rows of splits/split_v1.json with motion == velocity ({len(vel)} of {len(train)})",
       "n_dir8_velocity": int(len(rows8)), "dir8_theta": sorted(set(np.round(theta[rows8], 3))),
       "dir8_grouped_fold_directions": {int(k): sorted(set(np.round(theta[rows8][specs['dir8_grouped_velocity'][1] == k], 3)))
                                        for k in range(5)},
       "n_large_requested": args.n_large, "n_large_used": int(n_large),
       "n_large_note": ("enough velocity train clips" if n_large == args.n_large
                        else f"only {len(vel)} velocity train clips; used all"),
       "seeds": {"dir8": "folds seed 0 (as the stored rows)", "large": f"draw 1000+s, folds s, s < {args.seeds}"},
       "rule": stored["rule"], "stored_source": "results/p1a_paperscale_direction_direction.json", "models": {}}

for model in MODELS:
    acts = load_activations(dataset, "meanpool", model)
    fits = {k: [] for k in specs}
    for p in range(N_POINTS):
        X = layer_matrix(acts, p)
        for k, (rows, folds, select) in specs.items():
            fits[k].append(probe_rows(X, rows, Y, folds, score_fn, select))
        print(f"{model} point {p:2d}: dir8 {fits['dir8_velocity'][-1]['cv_mean']:.3f} "
              f"grouped {fits['dir8_grouped_velocity'][-1]['pooled']:.3f} {large}/0 {fits[large + '/0'][-1]['cv_mean']:.3f}",
              flush=True)
    res = {}
    for k in ("dir8_velocity", "dir8_grouped_velocity"):
        rows, folds, select = specs[k]
        curve = [f["cv_mean" if select == "fold_mean" else "pooled"] for f in fits[k]]
        s = summarise_curve(curve, [f["oof"] for f in fits[k]], Y[rows], score_fn, args.n_boot)
        res[k] = {"n_clips": int(len(rows)), "curve": curve, "curve_sd": [f["cv_sd"] for f in fits[k]],
                  "alpha": [f["alpha"] for f in fits[k]], **s}
    curves = np.array([[f["cv_mean"] for f in fits[f"{large}/{s}"]] for s in range(args.seeds)])
    draws = []
    for s in range(args.seeds):
        draws += onset_draws(Y[specs[f"{large}/{s}"][0]], [f["oof"] for f in fits[f"{large}/{s}"]], score_fn,
                             max(1, args.n_boot // args.seeds), seed=s)
    ci, n_none = ci_from_draws(draws)
    av = availability(curves.mean(axis=0)[:LAST_BLOCK + 1])
    res[large] = {"n_clips": int(n_large), "curve": curves.mean(axis=0).tolist(),
                  "curve_sd": curves.std(axis=0, ddof=1).tolist(), "curve_per_seed": curves.tolist(),
                  "onset": av["onset"], "peak": av["peak"], "peak_score": av["peak_score"],
                  "onset_per_seed": [availability(c[:LAST_BLOCK + 1])["onset"] for c in curves],
                  "onset_ci": ci, "onset_boot_draws_without_onset": n_none}
    for c in res.values():
        c["at_point"] = {str(p): c["curve"][p] for p in SHOW}
    out["models"][model] = res

out["summary"] = {m: {**{f"{c}_mixed_stored": {"block1": r["block1"], "block8": r["block8"], "onset": r["onset"],
                                               "onset_ci": r["onset_ci"], "n_clips": r["n_clips"]}
                         for c, r in stored["summary"][m].items() if c in ("dir8", "dir8_grouped")},
                      **{c: {"block1": r["at_point"]["1"], "block8": r["at_point"]["8"], "onset": r["onset"],
                             "onset_ci": r["onset_ci"], "n_clips": r["n_clips"]} for c, r in out["models"][m].items()}}
                  for m in MODELS}
write_json(args.results / "p1a_paperscale_velocity_only.json", out)
for m, cs in out["summary"].items():
    for c, s in cs.items():
        print(f"{m:7s} {c:28s} n={s['n_clips']:4d} block1={s['block1']:.3f} block8={s['block8']:.3f} "
              f"onset={s['onset']} CI={s['onset_ci']}")

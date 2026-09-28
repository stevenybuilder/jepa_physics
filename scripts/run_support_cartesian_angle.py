"""Is the (vx, vy)-before-(sin θ, cos θ) onset gap the normalisation v/‖v‖? (REPORT §5 "Cartesian vs polar")

  python scripts/run_support_cartesian_angle.py [--points 1 2 3 9] [--pool meanpool]

Refits the two probes of run_step1_support.py (a) the same way (wm.support.layer_cv: direction-set train clips with
motion == velocity, all-NaN clips excluded, standardised on those rows, α on the split's folds) at the given points,
then scores the angle atan2(v̂y, v̂x) of the (vx, vy) probe's out-of-fold predictions against the truth and against
the direct direction probe's out-of-fold predictions (wm.support.cartesian_angle_agreement). The refit CV R² is
checked against results/p1a_support_onset_direction_{pool}.json. Writes results/p1a_support_cartesian_angle.json.
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import load_table  # noqa: E402
from wm.probes import RESULTS, all_nan_clips, layer_matrix, load_activations, split_rows, targets, write_json  # noqa: E402
from wm.support import cartesian_angle_agreement, layer_cv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--points", type=int, nargs="+", default=[1, 2, 3, 9])
parser.add_argument("--pool", default="meanpool")
parser.add_argument("--results", type=Path, default=RESULTS)
args = parser.parse_args()

dataset = "direction"
acts = load_activations(dataset, args.pool)
get_X = lru_cache(maxsize=2)(lambda p: layer_matrix(acts, p))
df = load_table(dataset)
tr, _, folds = split_rows(dataset, df)
keep = (df["motion"].to_numpy()[tr] == "velocity") & ~all_nan_clips(acts)[tr]
rows, folds = tr[keep], folds[keep]
Yd, _, sd = targets(df, "direction")
Yv, _, sv = targets(df, "vxvy")
ref = json.loads((args.results / f"p1a_support_onset_{dataset}_{args.pool}.json").read_text())
ref_r2 = {t: {r["point"]: r["cv_mean"] for r in ref["targets"][t]["layers"]} for t in ("vxvy", "sincos")}

cart_rows, cart_oof = layer_cv(get_X, Yv, rows, folds, sv, args.points)
dir_rows, dir_oof = layer_cv(get_X, Yd, rows, folds, sd, args.points)
per_point = []
for i, p in enumerate(args.points):
    a = cartesian_angle_agreement(cart_oof[i], dir_oof[i], Yd[rows])
    per_point.append({"point": p, **a,
                      "refit_cv_r2": {"vxvy": cart_rows[i]["cv_mean"], "sincos": dir_rows[i]["cv_mean"]},
                      "stored_cv_r2": {"vxvy": ref_r2["vxvy"][p], "sincos": ref_r2["sincos"][p]},
                      "alpha": {"vxvy": cart_rows[i]["alpha"], "sincos": dir_rows[i]["alpha"]}})
    print(f"point {p}: cart-angle vs truth R2 {a['cartesian_angle_vs_truth']['r2']:.3f} "
          f"MAE {a['cartesian_angle_vs_truth']['mae_deg']:.2f} | direct R2 {a['direct_probe_vs_truth']['r2']:.3f} "
          f"(unit {a['direct_probe_unit_vs_truth']['r2']:.3f}) MAE {a['direct_probe_vs_truth']['mae_deg']:.2f} | "
          f"cart vs direct R2 {a['cartesian_angle_vs_direct_probe']['r2']:.3f} "
          f"MAE {a['cartesian_angle_vs_direct_probe']['mae_deg']:.2f}", flush=True)

write_json(args.results / "p1a_support_cartesian_angle.json", {
    "dataset": dataset, "pool": args.pool, "clips": "direction-set train clips with motion == velocity",
    "n_clips": int(len(rows)), "label": "extra: angle of the (vx, vy) probe vs the direct (sin, cos) probe",
    "method": ("out-of-fold predictions of both probes (wm.support.layer_cv, as run_step1_support.py (a)); "
               "Cartesian angle = atan2(v_y_hat, v_x_hat) -> (sin, cos); scored against the truth (circular MAE, "
               "sin/cos R2) and against the direct probe's OOF (sin, cos) predictions"),
    "reference": f"results/p1a_support_onset_{dataset}_{args.pool}.json", "points": per_point})

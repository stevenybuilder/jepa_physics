"""Same-clip direction vs speed probe counts (paper App. A reads both off one velocity set).

The stored step-2 counts compare direction K (supplied direction set) with speed K (supplied speed set): two clip
sets. Here both variables run on the SAME clips: the 750 constant-velocity clips (motion == 'velocity') of the
supplied direction set, with their splits/split_v1.json train/test/fold assignment restricted to those clips. Same
machinery as scripts/run_step2.py (wm.inlp.inlp: ridge, nested + paper protocols, C.11 and Fig. 22 stop rules;
random-removal band), same standardisation rule (train-standardised meanpool, stats from the restricted train rows).
α is re-chosen on the restricted train rows by the step-1 rule (fold-mean R² over the 5 folds; nested also re-picks
per fold as stored). Writes results/p1b_sameclip_direction_vs_speed.json.

  python scripts/run_sameclip.py [--points 8 9 19 22] [--seeds 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.inlp import inlp, random_removal_curve, rank_schedule, write_result  # noqa: E402
from wm.probes import (RESULTS, cv_select_alpha, load_activations, load_sweep, load_table, split_rows,  # noqa: E402
                       standardized_layer, targets)

VARS = ("direction", "speed")


def restrict(tr, te, folds, keep):
    """Split rows (table positions) restricted to rows where keep is True; folds follow their train rows."""
    k_tr, k_te = keep[tr], keep[te]
    return tr[k_tr], te[k_te], folds[k_tr]


def band_at(rows, dims, key):
    """Random-removal band value (seed mean, seed SD) at the largest scored rank <= dims."""
    r = max((r for r in rows if r["dims_removed"] <= dims), key=lambda r: r["dims_removed"], default=None)
    return None if r is None or key not in r else {"dims_removed": r["dims_removed"], key: r[key],
                                                   key + "_seed_sd": r[key + "_seed_sd"]}


def one_variable(Xtr, Ytr, Xte, Yte, folds, kind, score_fn, seeds=10):
    """Probe sequence under both protocols + random band; compact K / dims per protocol and stop rule."""
    alpha = cv_select_alpha(Xtr, Ytr, folds, score_fn=score_fn)["alpha"]
    summary, _, _, _ = inlp(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, kind, protocols=("nested", "paper"))
    rand = random_removal_curve(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, rank_schedule(summary), seeds,
                                summary["alpha_folds"])
    m = summary["m"]
    out = {"m": m, "alpha": float(alpha), "alpha_folds": summary["alpha_folds"]}
    for proto, block, prefix in (("nested", summary, "cv"), ("paper", summary["paper"], "test")):
        K, Kl = block["K"], block["K_loose"]
        out[proto] = {"K_C11": K, "dims_C11": K * m, "K_fig22": Kl, "dims_fig22": Kl * m,
                      "K_fig22_censored": block["K_loose_censored"], "hit_round_cap": block["hit_round_cap"],
                      "r2_round1": block["rounds"][0][f"{prefix}_r2"],
                      "r2_curve": [r[f"{prefix}_r2"] for r in block["rounds"]],
                      "random_band_at_dims_C11": band_at(rand["rows"], K * m, f"{prefix}_r2")}
        if proto == "nested":
            out[proto].update({"K_folds": block["K_folds"], "K_fold_min": block["K_fold_min"],
                               "K_fold_max": block["K_fold_max"]})
    out["random"] = rand
    return out


def stored_cross_set(var, point, results_dir):
    """Stored cross-set K (direction from the direction set, speed from the speed set), with the source keys."""
    out = {}
    f = Path(results_dir) / f"p1b_{var}_{var}_meanpool_L{point}.json"
    if f.exists():
        d = json.loads(f.read_text())
        out["step2_file"] = {"file": f"results/{f.name}", "nested_K_C11 [K]": d["K"],
                             "nested_K_fig22 [K_loose]": d["K_loose"], "paper_K_C11 [paper.K]": d["paper"]["K"],
                             "paper_K_fig22 [paper.K_loose]": d["paper"]["K_loose"]}
    s = json.loads((Path(results_dir) / "p1b_stop_rules.json").read_text())["cells"].get(f"{var}_L{point}")
    if s:
        out["stop_rules_file"] = {"file": "results/p1b_stop_rules.json",
                                  **{f"{p}_K_{r} [cells.{var}_L{point}.{p}.K.{r}]": s[p]["K"][r]
                                     for p in ("nested", "paper") for r in ("stored_C11", "fig22")}}
    return out or None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, nargs="+", default=None,
                    help="default: 8, 9 and each variable's stored step-1 CV peak (direction 22, speed 19)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    peaks = {v: load_sweep(v, v, "meanpool", args.results)["availability"]["peak"] for v in VARS}
    points = args.points or sorted({8, 9, *peaks.values()})

    df = load_table("direction")
    keep = (df["motion"] == "velocity").to_numpy()
    if not np.isfinite(df.loc[keep, "speed_mps"].to_numpy(float)).all():
        sys.exit("speed_mps missing for constant-velocity direction-set clips; stop")
    tr, te, folds = restrict(*split_rows("direction", df), keep)
    acts = load_activations("direction", "meanpool")
    Ys = {v: targets(df, v) for v in VARS}
    cells = {}
    for p in points:
        Xtr, Xte = standardized_layer(acts, p, tr, te)
        cell = {"point": p, "is_peak_of": [v for v in VARS if peaks[v] == p]}
        for v in VARS:
            Y, kind, score_fn = Ys[v]
            cell[v] = one_variable(Xtr, Y[tr], Xte, Y[te], folds, kind, score_fn, args.seeds)
            cell[v]["stored_cross_set"] = stored_cross_set(v, p, args.results)
        cmp = {}
        for proto in ("nested", "paper"):
            for rule in ("C11", "fig22"):
                kd, ks = cell["direction"][proto][f"K_{rule}"], cell["speed"][proto][f"K_{rule}"]
                cmp[f"{proto}_{rule}"] = {"K_direction": kd, "dims_direction_2K": 2 * kd, "K_speed": ks,
                                          "dims_speed_K": ks, "speed_fewer_probes": ks < kd,
                                          "speed_fewer_dims": ks < 2 * kd}
        cell["comparison"] = cmp
        cells[f"L{p}"] = cell
        print(p, {k: (c["K_direction"], c["K_speed"]) for k, c in cmp.items()}, flush=True)

    out = {"item": "Same-clip direction vs speed probe counts (ledger #171; REPORT §3.2)",
           "clips": {"dataset": "direction (supplied)", "filter": "motion == 'velocity' (constant velocity)",
                     "n": int(keep.sum()), "n_train": int(len(tr)), "n_test": int(len(te)),
                     "folds": {int(k): int((folds == k).sum()) for k in np.unique(folds)},
                     "split": "splits/split_v1.json direction split restricted to these clips",
                     "speed_labels": "speed_mps from the direction-set metadata (1-7 m/s, all 750 present)"},
           "method": {"sequence": "wm.inlp.inlp (scripts/run_step2.py machinery): ridge, QR, project out span(W_k)",
                      "standardisation": "per-feature z-score, stats from the restricted train rows (step-1 rule)",
                      "alpha": "re-chosen on the restricted train rows by the step-1 CV rule (paper protocol); "
                               "nested re-picks per fold as stored",
                      "stop_rules": {"C11": "direction R2 < 0.1 or circular MAE > 80 deg; speed R2 < 0.05 or MAE > "
                                            "0.9 x mean-predictor MAE (C.11 l.1211-1212, stored rule)",
                                     "fig22": "direction R2 < 0.3, speed R2 < 0.1 (Fig. 22 caption l.1172)"},
                      "K": "probes kept before the first at-chance round; dims = 2K for direction (sin, cos), K "
                           "for speed",
                      "protocols": {"nested": "fold-mean curve over 5 held-out folds (pooled K)",
                                    "paper": "all-train sequence scored on test"},
                      "random_band": f"random orthonormal removal, {args.seeds} seeds, same α rule; value at the "
                                     "treatment's C.11 dims"},
           "peaks": peaks, "cells": cells}
    write_result(Path(args.results) / "p1b_sameclip_direction_vs_speed.json", out,
                 test_read="paper protocol scores the restricted test split at every round (as stored step 2)")
    return out


if __name__ == "__main__":
    main()

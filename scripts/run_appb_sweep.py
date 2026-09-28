"""The paper's App. B probe recipe (refs/physics_paper.txt l.689-693: linear probe, 20-config sweep over lr
{1e-4, 3e-4, 1e-3, 3e-3, 5e-3} x weight decay {0.01, 0.1, 0.4, 0.8}, best by validation, 5-fold CV, mean ± SD across
folds) at the EARLY layer points, where the onset verdict is decided, plus the stored CV-peak point for reference.
Reuses wm.adam_probe.cv_adam (as scripts/check_probe_recipe.py does) on the stored ridge sweeps' exact inputs
(wm.probes.load_activations meanpool, train-standardised features, split_v1 train folds). The onset uses the stored
files' rule and bootstrap (wm.probes.availability, wm.probes.bootstrap_onset). Writes results/p1a_appB_sweep.json.

  python scripts/run_appb_sweep.py [--variables direction speed] [--points 0 1 ... 10] [--folds3-points ...] [--jobs 4]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.adam_probe import LRS, WDS, cv_adam  # noqa: E402
from wm.probes import (RESULTS, availability, bootstrap_onset, load_activations, load_sweep, load_table,  # noqa: E402
                       split_rows, standardized_layer, targets, write_json)


def merge_folds(folds, k):
    """Coarsen the stored 5 folds to k by merging whole folds (fold f -> f mod k), so no clip changes side."""
    ids = np.unique(folds)
    return np.array([int(np.where(ids == f)[0][0]) % k for f in folds])


def fit_point(job):
    variable, point, n_folds, epochs, act_root = job
    df = load_table(variable)
    Y, _, score_fn = targets(df, variable)
    tr, te, folds = split_rows(variable, df)
    if n_folds != len(np.unique(folds)):
        folds = merge_folds(folds, n_folds)
    Xtr, _ = standardized_layer(load_activations(variable, act_root=act_root), point, tr, te)
    t0 = time.time()
    fit = cv_adam(Xtr, Y[tr], folds, score_fn, epochs)
    print(f"{variable} point {point:2d} ({n_folds} folds): R2 {fit['cv_mean']:.3f}±{fit['cv_sd']:.3f} "
          f"lr={fit['lr']} wd={fit['wd']} [{time.time() - t0:.0f}s]", flush=True)
    return variable, point, n_folds, fit, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variables", nargs="+", default=["direction", "speed"])
    parser.add_argument("--points", nargs="+", type=int, default=list(range(0, 11)))
    parser.add_argument("--folds3-points", nargs="*", type=int, default=[],
                        help="points fitted with 3 folds (merged from the stored 5) to save time")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--act-root", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    results = Path(args.results or RESULTS)

    stored = {v: load_sweep(v, v, "meanpool", results) for v in args.variables}
    jobs = []
    for v in args.variables:
        epochs = 100 if v == "direction" else 50
        pts = sorted(set(args.points) | {stored[v]["availability"]["peak"]})
        jobs += [(v, p, 3 if p in args.folds3_points else 5, epochs, args.act_root) for p in pts]
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        fits = list(ex.map(fit_point, jobs))
    wall = time.time() - t0

    out = {"recipe": {"source": "physics paper App. B (refs/physics_paper.txt l.689-693), via wm.adam_probe.cv_adam",
                      "grid": {"lr": list(LRS), "weight_decay": list(WDS)},
                      "selection": "best (lr, wd) by fold-mean validation R2, chosen independently at each point",
                      "reported": "fold mean ± SD (ddof=1) R2 of the chosen config on the held-out fold",
                      "epochs": "100 direction, 50 speed (paper C.11; App. B gives none); minibatch 64",
                      "weight_decay": "coupled L2 on W only (torch.optim.Adam weight_decay); targets standardised per fit"},
           "inputs": "meanpool V-JEPA 2 activations (artifacts/activations/<var>/vjepa2/meanpool.npy), train-standardised "
                     "per point, split_v1 train clips and folds: identical to the stored ridge sweeps (wm.probes.layer_sweep)",
           "onset_rule": ("wm.probes.availability (first point >= 90% of the curve max) and wm.probes.bootstrap_onset "
                          "(200-draw clip bootstrap of the out-of-fold predictions, seed 0), imported unchanged, applied "
                          "to the curve over the fitted points only (0..10 plus the stored CV peak): the peak supplies "
                          "the max. The same restriction is applied to the stored ridge curve as a check "
                          "(ridge_onset_same_points must equal the stored onset)."),
           "point_0_note": "point 0 (embedding) is fitted too, so 'first point' is decided on the same points as the stored rule",
           "wall_seconds": round(wall, 1), "variables": {}}
    for v in args.variables:
        rows = sorted([f for f in fits if f[0] == v], key=lambda f: f[1])
        st = stored[v]
        sfile = f"results/p1a_{v}_{v}_meanpool.json"
        df = load_table(v)
        Y, _, score_fn = targets(df, v)
        tr, _, _ = split_rows(v, df)
        points = [r[1] for r in rows]
        per_point = []
        for _, p, k, fit, secs in rows:
            sl = st["layers"][p]
            per_point.append({"point": p, "n_folds": k, "sweep_cv_mean": fit["cv_mean"], "sweep_cv_sd": fit["cv_sd"],
                              "lr": fit["lr"], "wd": fit["wd"], "grid": fit["grid"],
                              "ridge_cv_mean": sl["cv_mean"], "ridge_cv_sd": sl["cv_sd"], "ridge_alpha": sl["alpha"],
                              "ridge_source": f"{sfile} layers[{p}].cv_mean", "seconds": round(secs, 1)})
        curve = [r[3]["cv_mean"] for r in rows]
        av = availability(curve)
        ci, n_none = bootstrap_onset(Y[tr], [r[3]["oof"] for r in rows], score_fn, 200, 0)
        ridge_av = availability([st["layers"][p]["cv_mean"] for p in points])
        out["variables"][v] = {
            "points": points, "per_point": per_point,
            "sweep_onset": points[av["onset"]] if av["onset"] is not None else None,
            "sweep_onset_ci": None if ci is None else [points[ci[0]], points[ci[1]]],
            "sweep_onset_boot_draws_without_onset": n_none,
            "sweep_max_point": points[av["peak"]], "sweep_max": av["peak_score"],
            "ridge_onset_stored": st["availability"]["onset"], "ridge_onset_ci_stored": st["availability"]["onset_ci"],
            "ridge_peak_stored": st["availability"]["peak"],
            "ridge_stored_source": f"{sfile} availability",
            "ridge_onset_same_points": points[ridge_av["onset"]] if ridge_av["onset"] is not None else None}
        o = out["variables"][v]
        print(f"{v}: sweep onset {o['sweep_onset']} CI {o['sweep_onset_ci']} | ridge stored {o['ridge_onset_stored']} "
              f"{o['ridge_onset_ci_stored']} (same points: {o['ridge_onset_same_points']})")
    if args.folds3_points:
        out["fold_note"] = (f"points {sorted(args.folds3_points)} use 3 folds (stored folds merged f -> f mod 3) because "
                            "the full 5-fold grid exceeded the ~30 min budget; all other points use the stored 5 folds")
    write_json(results / "p1a_appB_sweep.json", out)


if __name__ == "__main__":
    main()

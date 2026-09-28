"""Step 1 support runs on stored activations (spec 5.1 (a)-(c)). All Part 1 extras, not the paper's protocol.

(a) Cartesian vs polar onset: onset layer of (vx, vy) vs (sin θ, cos θ) on constant-speed clips, same
    availability rule as step 1, with a paired clip bootstrap for the onset difference.
(b) Direction transfer: direction probe fit on direction-set train clips, read on every speed-set and
    acceleration-set clip (disjoint clips, different speed and start ranges).
(c) Spatial generalisation: direction probe fit on train clips with start x < 0, read on train clips with
    start x > 0, and the reverse.

Only `train` rows of the direction set are used for fitting and for (a) and (c); the direction set's `test`
clips are never read here. Start position and motion type come from load_table (metadata.json fields
start_position_xy_m and motion), so no extra metadata read is needed.
"""
from functools import partial

import numpy as np

from wm.metrics import radius_summary
from wm.probes import (ALPHAS, LAST_BLOCK, N_POINTS, Standardizer, availability, cv_select_alpha, fit_ridge,
                       layer_fraction, predict, score)

ONSET_RULE = ("onset = first point with CV R2 >= 90% of max over points 0..24 (step-1 rule); CI: clip bootstrap "
              "of fixed out-of-fold predictions; the difference CI resamples the same clips for both targets")


def layer_cv(get_X, Y, rows, folds, score_fn, points=range(LAST_BLOCK + 1)):
    """CV curve over layer points on a subset of clips: standardise on `rows`, pick α on `folds`.
    Returns per-point rows and the out-of-fold predictions (for the bootstrap)."""
    out, oofs = [], []
    for p in points:
        Xr = get_X(p)[rows]
        X = Standardizer().fit(Xr).transform(Xr)
        cv = cv_select_alpha(X, Y[rows], folds, ALPHAS, score_fn)
        out.append({"point": p, "frac": layer_fraction(p), "alpha": cv["alpha"], "cv_mean": cv["cv_mean"],
                    "cv_sd": cv["cv_sd"], "cv_mae_mean": cv["cv_mae_mean"]})
        oofs.append(cv["oof"])
    return out, oofs


def onset_compare(get_X, targets, rows, folds, n_boot=200, seed=0):
    """Onset layer of each target in `targets` = {name: (Y [N, m], score_fn)} on the same clips.

    Returns per target the CV curve, availability and a 95% bootstrap CI of the onset; for the first
    two targets also the paired bootstrap of onset[first] − onset[second] (negative: first is earlier).
    """
    names = list(targets)
    res, oofs = {}, {}
    for name, (Y, score_fn) in targets.items():
        curve_rows, oofs[name] = layer_cv(get_X, Y, rows, folds, score_fn)
        av = availability([r["cv_mean"] for r in curve_rows])
        av["onset_frac"] = None if av["onset"] is None else layer_fraction(av["onset"])
        res[name] = {"layers": curve_rows, "availability": av}
    rng = np.random.default_rng(seed)
    n = len(rows)
    boots = {name: [] for name in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for name, (Y, score_fn) in targets.items():
            Yr = Y[rows][idx]
            boots[name].append(availability([score_fn(Yr, oof[idx])["r2"] for oof in oofs[name]])["onset"])
    for name in names:
        ok = [o for o in boots[name] if o is not None]
        res[name]["availability"]["onset_ci"] = ([int(np.percentile(ok, 2.5)), int(np.percentile(ok, 97.5))]
                                                 if ok else None)
        res[name]["availability"]["onset_boot_draws_without_onset"] = len(boots[name]) - len(ok)
    out = {"targets": res, "n_clips": int(n), "n_boot": n_boot, "rule": ONSET_RULE}
    if len(names) >= 2:
        a, b = names[:2]
        diff = [x - y for x, y in zip(boots[a], boots[b]) if x is not None and y is not None]
        oa, ob = res[a]["availability"]["onset"], res[b]["availability"]["onset"]
        out["onset_difference"] = {
            "definition": f"onset[{a}] - onset[{b}] (layer points; negative = {a} earlier)",
            "point_estimate": None if oa is None or ob is None else int(oa - ob),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))] if diff else None,
            "frac_draws_first_earlier": float(np.mean(np.array(diff) < 0)) if diff else None}
    return out


def direction_score(Y, P):
    """R², circular MAE and readout radius of direction predictions P against Y = (sin θ, cos θ)."""
    s = score(Y, P, "circular")
    return {"r2": s["r2"], "mae": s["mae"], "radius": radius_summary(P)}


def transfer_curve(get_X_src, Y_src, rows, folds, sets, points=range(N_POINTS)):
    """Direction probe fit on `rows` of the source (α by CV on `folds`, standardised on `rows`), read on
    each target set: sets = {name: (get_X [N_t, d] by point, Y_t, {subset_name: bool mask} or None)}.
    The source CV score (fold mean) is the in-distribution reference; no source test clip is read."""
    circ = partial(score, kind="circular")
    out = []
    for p in points:
        X = get_X_src(p)
        st = Standardizer().fit(X[rows])
        Xs = st.transform(X[rows])
        cv = cv_select_alpha(Xs, Y_src[rows], folds, ALPHAS, circ)
        W, b = fit_ridge(Xs, Y_src[rows], cv["alpha"])
        row = {"point": p, "frac": layer_fraction(p), "post_ln": p == N_POINTS - 1, "alpha": cv["alpha"],
               "source_cv_r2": cv["cv_mean"], "source_cv_mae": cv["cv_mae_mean"], "targets": {}}
        for name, (get_X_t, Y_t, subsets) in sets.items():
            P = predict(st.transform(get_X_t(p)), W, b)
            t = direction_score(Y_t, P)
            t["n"] = int(len(Y_t))
            for sub, mask in (subsets or {}).items():
                if mask.sum() > 1:
                    t[sub] = {**direction_score(Y_t[mask], P[mask]), "n": int(mask.sum())}
            row["targets"][name] = t
        out.append(row)
    return out


def spatial_curve(get_X, Y, side_a, side_b, folds_a, folds_b, points=range(N_POINTS)):
    """Direction probe fit on side A (α by CV on A's folds, standardised on A), read on side B, and the
    reverse. side_a / side_b are row-index arrays of disjoint clips (start x < 0 / x > 0)."""
    circ = partial(score, kind="circular")
    out = []
    for p in points:
        X = get_X(p)
        row = {"point": p, "frac": layer_fraction(p), "post_ln": p == N_POINTS - 1}
        for name, fit_rows, fit_folds, eval_rows in (("neg_to_pos", side_a, folds_a, side_b),
                                                     ("pos_to_neg", side_b, folds_b, side_a)):
            st = Standardizer().fit(X[fit_rows])
            Xf = st.transform(X[fit_rows])
            cv = cv_select_alpha(Xf, Y[fit_rows], fit_folds, ALPHAS, circ)
            W, b = fit_ridge(Xf, Y[fit_rows], cv["alpha"])
            cross = direction_score(Y[eval_rows], predict(st.transform(X[eval_rows]), W, b))
            row[name] = {"alpha": cv["alpha"], "within_cv_r2": cv["cv_mean"], "within_cv_mae": cv["cv_mae_mean"],
                         "cross_r2": cross["r2"], "cross_mae": cross["mae"], "cross_radius": cross["radius"],
                         "n_fit": int(len(fit_rows)), "n_eval": int(len(eval_rows))}
        out.append(row)
    return out


def cartesian_angle_agreement(P_cart, P_dir, Y_dir):
    """Angle of a Cartesian (vx, vy) probe's predictions vs a direct (sin θ, cos θ) probe's, on the same clips.

    P_cart [n, 2] predicts magnitude · (cos θ, sin θ) (wm.probes.targets 'vxvy'); its angle atan2(v̂y, v̂x) is turned
    into the unit vector (sin, cos). P_dir [n, 2] predicts (sin θ, cos θ); Y_dir [n, 2] is the truth. Returns each
    readout scored against the truth (circular MAE, sin/cos R²; the direct probe both raw and normalised to unit length, the
    like-for-like comparison with the unit-length Cartesian angle),
    and the Cartesian angle scored against the direct probe's predictions (circular MAE between the two angles, and
    the R² of the Cartesian unit vector against the direct probe's (sin, cos) predictions)."""
    ang = np.arctan2(P_cart[:, 1], P_cart[:, 0])
    U = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    D = P_dir / np.linalg.norm(P_dir, axis=1, keepdims=True)
    cart, direct, direct_u = (score(Y_dir, M, "circular") for M in (U, P_dir, D))
    agree = score(P_dir, U, "circular")
    return {"cartesian_angle_vs_truth": {"r2": cart["r2"], "mae_deg": cart["mae"]},
            "direct_probe_vs_truth": {"r2": direct["r2"], "mae_deg": direct["mae"]},
            "direct_probe_unit_vs_truth": {"r2": direct_u["r2"], "mae_deg": direct_u["mae"]},
            "cartesian_angle_vs_direct_probe": {"r2": agree["r2"], "mae_deg": agree["mae"]}}

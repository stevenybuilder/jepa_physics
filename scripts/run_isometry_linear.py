"""Goodfire A.5 isometry test with its linear baseline, and with behaviour spaces that are not built from the steered
activations (PART2_SECOND_LOOK.md section B). CPU only, stored activations.

Goodfire §2.3 / A.5: Pearson r between pairwise geodesic distances on the activation manifold M_h and on the behaviour
manifold M_y (r = 0.89-0.999 across their tasks), against r between straight-line (chord) distances in activation
space and the same M_y geodesics (r = 0.36-0.89; 0.06 in the mountain-car world model). run_part2 stores the geodesic
r (isometry_probe, isometry_behaviour) but never the chord baseline, and its M_y (Eq. 9 at the steered layer) is a
function of the steered activations.

Here, per direction layer, on the 64 direction values (W = 64 vertices, K = 0 interior points as Goodfire uses for
W >= 81; A.5's shared-geodesic exclusion is then empty):
  activation side (knot folds 0-2, PCA-64 fit on them): the count-weighted periodic smoothing spline that run_part2
      steers with, and Goodfire's interpolating periodic spline; intrinsic angle chosen as in run_part2
      (choose_angle_source), the true direction with --angle labels (causalab's alphabet / age 8B and all 70B runners,
      intrinsic_mode: parameter), or Goodfire's own label-free coordinate with --angle goodfire (--goodfire-method
      only; mf.goodfire_periodic_angle on all 64 knot centroids, what the weekdays / months 8B runners get by
      inheriting intrinsic_mode: pca, kept even when the periodicity test fails), or causalab's full pca-mode rule
      with --angle goodfire_full (that angle when the test passes, else PC1 with a natural cubic and a non-periodic
      path, causalab spline/train.py; the same rule on the behaviour side, spline/belief_fit.py). d_geo = arc length along the spline (short way round); d_lin = Euclidean distance
      between the same vertex points in PCA-64.
  behaviour side (probe folds 3-4, disjoint clips), each a periodic smoothing spline through per-value centroids
      with the true angle as coordinate, distances = arc length:
      eq9_same_layer   Goodfire Eq. 9 M_y at the steered layer (Hellinger geodesics; circular, for reference)
      encoder_out      the encoder's last point (25, post-LN) activations, PCA-64
      predictor        the V-JEPA 2 predictor's own unedited forecast of tubelets 4-7 from frames 1-8
                       (artifacts/session2/native/pred_pooled_all.npy, mean over the 4 steps), PCA-64
      concept          the true angular distance |d theta| (the conceptual metric d_Z)
  CIs: 200 bootstrap draws resampling clips within each value on both sides (PCA bases fixed), raw percentile
      intervals (biased low: centroid noise pulls the draws below the estimate); --angle goodfire_full uses the same
      draws for four activation coordinates at once and reports bias and bias-corrected intervals (BOOT_SCHEME).

  python scripts/run_isometry_linear.py --layers 8 12 22
  python scripts/run_isometry_linear.py --goodfire-method --angle goodfire   # -> p2_isometry_goodfire_coord.json
  python scripts/run_isometry_linear.py --goodfire-method --angle goodfire_full   # -> p2_isometry_goodfire_full.json
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

PRED = PROJECT_ROOT / "artifacts" / "session2" / "native" / "pred_pooled_all.npy"
N_INTERIOR_NOTE = ("0 here (vertices only); causalab's published runners use n_interior_per_pair = 1, adding one "
                   "interior point between each adjacent vertex pair")


def geo_matrix(curve, n=4000):
    """Pairwise arc length between the curve's vertices (curve.coords), short way round, from one dense table."""
    tt, s = mf.arc_length_table(curve, n)
    total = s[-1]
    lo = curve.t_range()[0]
    t = (curve.coords - lo) % mf.TWO_PI + lo if curve.periodic else curve.coords
    sv = np.interp(t, tt, s)
    D = np.abs(sv[:, None] - sv[None])
    return np.minimum(D, total - D) if curve.periodic else D


def path_length_matrix(curve, values, n_steps=150, chunk=256):
    """Goodfire's geodesic (causalab scores/isometry.py _decoded_path_length_batched): for each vertex pair, the
    short-way coordinate path cut into n_steps equal sub-intervals, decoded by the spline, lengths of the n_steps
    straight pieces summed. Rows / columns in the order of `values`."""
    t = curve.coord_of_value(np.asarray(values, dtype=float))
    ia, ib = np.triu_indices(len(t), 1)
    s = np.linspace(0.0, 1.0, n_steps + 1)
    L = np.empty(len(ia))
    for c in range(0, len(ia), chunk):
        ta, tb = t[ia[c:c + chunk]], t[ib[c:c + chunk]]
        P = curve(ta[:, None] + s[None] * curve.step(ta, tb)[:, None])          # [b, n_steps + 1, dim]
        L[c:c + chunk] = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
    G = np.zeros((len(t), len(t)))
    G[ia, ib] = G[ib, ia] = L
    return G


def angle_choice(cent, angle="auto"):
    """Coordinate for the activation spline: "auto" = choose_angle_source (unsupervised angle, labels fallback);
    "labels" = the true direction (causalab's alphabet / age 8B and 70B runners, intrinsic_mode: parameter; weekdays /
    months 8B inherit intrinsic_mode: pca); "goodfire" = mf.goodfire_periodic_angle on these centroids (causalab
    spline/builders.py remap_periodic_to_angle: atan2 of PC1/PC2 each divided by sqrt(variance)), kept whether or not
    their periodicity test passes; "goodfire_full" = causalab's pca-mode rule (pca_mode_branch): that angle when the
    test passes, else PC1 with a natural cubic. Returns the choose_angle_source dict, {"angle": "labels", "plane":
    "activation"}, or {"angle": "goodfire" | "goodfire_full", "plane": "activation", "goodfire":
    goodfire_periodic_angle dict, "branch": "goodfire_pca_angle" | "pc1_fallback"}."""
    if angle == "labels":
        return {"angle": "labels", "plane": "activation"}
    if angle in ("goodfire", "goodfire_full"):
        g = mf.goodfire_periodic_angle(cent["C"])
        branch = "goodfire_pca_angle" if angle == "goodfire" else pca_mode_branch(g)
        return {"angle": angle, "plane": "activation", "goodfire": g, "branch": branch}
    if angle != "auto":
        raise ValueError(f"angle must be 'auto', 'labels', 'goodfire' or 'goodfire_full', not {angle!r}")
    return mf.choose_angle_source(cent["C"], cent["values"])


def pca_mode_branch(g):
    """causalab pca-mode branch for a goodfire_periodic_angle result (spline/train.py and belief_fit.py: periodic
    pair detected -> remap to angle, else the top intrinsic_dim = 1 PCA component, no periodic dims)."""
    return "goodfire_pca_angle" if g["passes"] else "pc1_fallback"


def pca_mode_curve(C, Cpca, values, branch, smooth=None):
    """causalab pca-mode spline through the points C [m, d], coordinate from the same centroids in that side's PCA
    basis Cpca [m, >=2]: branch "goodfire_pca_angle" = periodic cubic at goodfire_periodic_angle(Cpca) (recomputed on
    these centroids), "pc1_fallback" = natural cubic at Cpca[:, 0], non-periodic (so the geodesic path is the direct
    coordinate path, not the short way round). coord_source is set to the branch."""
    if branch == "goodfire_pca_angle":
        c = mf.periodic_cubic(C, values, angle=mf.goodfire_periodic_angle(Cpca)["angle"], smooth=smooth)
    elif branch == "pc1_fallback":
        c = mf.natural_cubic(C, Cpca[:, 0], values=np.asarray(values, dtype=float), smooth=smooth)
    else:
        raise ValueError(branch)
    return replace(c, coord_source=branch)


def act_curve(cent, choice, spline):
    """Activation spline through cent with choice's coordinate; "goodfire" / "goodfire_full" go through
    pca_mode_curve with choice["branch"] (angle recomputed on these same centroids; "goodfire" is always the periodic
    angle branch, coord_source "goodfire_pca_angle")."""
    if choice["angle"] not in ("goodfire", "goodfire_full"):
        return mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=spline)
    smooth = {"count": cent["count"], "sd": cent["sd_coord"]} if spline == "smooth" else None
    return pca_mode_curve(cent["C"], cent["C"], cent["values"], choice["branch"], smooth=smooth)


def span_coords(C):
    """C [m, d] in an orthonormal basis of its own affine span (m - 1 columns): an exact isometry of the centroids,
    and of any spline through them (a cubic spline is linear in its knot values), so path lengths are unchanged and
    1024-d splines cost what 63-d ones do."""
    Cc = C - C.mean(axis=0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    return Cc @ Vt[:len(C) - 1].T


def beh_pca_mode(pred, yp, bp=None, branch=None, span=False):
    """causalab pca-mode behaviour manifold (belief_fit.fit_belief_tps_pca with the PCA of output_manifold/main.py,
    fit on per-example sqrt p): here PCA-64 fit on the per-clip predictor forecasts of the probe clips (continuous
    1024-d vectors, not distributions, so no square root), per-value centroids projected into it, periodicity test on
    their PC1/PC2, then the interpolating pca_mode_curve through the full 1024-d centroids. bp / branch: fixed from the
    point estimate in bootstrap draws; span: fit the spline through span_coords of the centroids (same distances).
    Returns (curve, bp, goodfire_periodic_angle dict, PCA coords of centroids)."""
    bc = mf.centroids(pred, yp)
    bp = mf.fit_pca(pred, 64) if bp is None else bp
    bC = bp.project(bc["C"])
    g = mf.goodfire_periodic_angle(bC)
    branch = pca_mode_branch(g) if branch is None else branch
    return pca_mode_curve(span_coords(bc["C"]) if span else bc["C"], bC, bc["values"], branch), bp, g, bC


def circ_lin_corr(x, theta_deg):
    """Circular-linear correlation (Mardia) of a linear coordinate x with the angle theta, in [0, 1]."""
    th = np.radians(np.asarray(theta_deg, dtype=float))
    rc, rs, cs = (np.corrcoef(x, np.cos(th))[0, 1], np.corrcoef(x, np.sin(th))[0, 1],
                  np.corrcoef(np.cos(th), np.sin(th))[0, 1])
    return float(np.sqrt((rc ** 2 + rs ** 2 - 2 * rc * rs * cs) / (1 - cs ** 2)))


def side_record(g, Cpca, values):
    """Periodicity test and coordinate-vs-labels summary for one side (activation or behaviour)."""
    rec = {"branch": pca_mode_branch(g), "passes": g["passes"], "rel_diff": g["rel_diff"],
           "eigenvalues_pc1_pc2": g["eigenvalues"],
           "atan2_angle_circular_corr_with_labels": mf.compare_angles(g["angle"], values)["circular_corr"],
           "pc1_circular_linear_corr_with_labels": circ_lin_corr(Cpca[:, 0], values)}
    rec["coordinate_used"] = ("sqrtvar_atan2_angle" if g["passes"] else "pc1")
    return rec


VARIANTS = ("labels", "label_free_knot_order", "goodfire", "goodfire_full")


def variant_matrices(Zk, yk, pred, yp, fixed, n_steps=150):
    """Geodesic / chord vs behaviour-geodesic Pearson r for the four activation coordinates on one (re)sample.
    fixed: point-estimate choices held fixed in bootstrap draws (label-free plane, goodfire_full branches, behaviour
    PCA basis); per-centroid angles are recomputed on the sample. Behaviour: labels-angle interpolating spline in the
    full 1024-d forecast space for labels / label_free_knot_order / goodfire (goodfire_method), the pca-mode
    behaviour manifold (beh_pca_mode) for goodfire_full; both behaviour splines through span_coords (exact)."""
    cent = mf.centroids(Zk, yk)
    values = cent["values"]
    cb = mf.centroids(pred, yp)
    B_lab = path_length_matrix(mf.fit_curve({**cb, "C": span_coords(cb["C"])}, True, angle="labels", spline="interp"),
                               values, n_steps)
    B_full = path_length_matrix(beh_pca_mode(pred, yp, fixed["bp"], fixed["beh_branch"], span=True)[0], values,
                                n_steps)
    choices = {"labels": {"angle": "labels", "plane": "activation"}, "label_free_knot_order": fixed["auto"],
               "goodfire": {"angle": "goodfire", "branch": "goodfire_pca_angle"},
               "goodfire_full": {"angle": "goodfire_full", "branch": fixed["act_branch"]}}
    out = {}
    for name, ch in choices.items():
        B = B_full if name == "goodfire_full" else B_lab
        out[name] = {}
        for sp in ("interp", "smooth"):
            c = act_curve(cent, ch, sp)
            G = path_length_matrix(c, values, n_steps)
            P = by_value(np.linalg.norm(c.points[:, None] - c.points[None], axis=-1), c.values, values)
            out[name][sp] = {"geo_pearson": corr(G, B)[0], "lin_pearson": corr(P, B)[0]}
    return out


def boot_inputs(L):
    """Knot-fold PCA-64 activations and probe-fold predictor forecasts with their labels (goodfire_method's inputs)."""
    d = load_inputs("direction", L)
    knot, probe = d["role"] == "knot", d["role"] == "probe"
    pca = mf.fit_pca(d["X"][knot], 64)
    return pca.project(d["X"][knot]), d["y"][knot], np.load(PRED).astype(np.float64).mean(1)[probe], d["y"][probe]


def resample_idx(y, rng):
    """Row indices of resample(Z, y, rng) (same rng calls, same order)."""
    return np.concatenate([rng.choice(np.flatnonzero(y == v), (y == v).sum()) for v in np.unique(y)])


def boot_chunk(L, idx, fixed, n_steps):
    Zk, yk, pred, yp = boot_inputs(L)
    return [variant_matrices(Zk[ik], yk[ik], pred[ip], yp[ip], fixed, n_steps) for ik, ip in idx]


BOOT_SCHEME = ("paired within-value clip resampling (run_layer's draws: knot clips then probe clips per draw, PCA "
               "bases and coordinate choices fixed, all four variants and both arms on the same draw); geo - lin "
               "bootstrapped per draw; per statistic: point estimate, bootstrap mean, bias = mean - point, raw 95% "
               "percentile CI, and the bias-corrected CI = the percentile CI shifted by -bias (recentred on the "
               "full-sample estimate). Resampling clips within a value adds noise to every centroid and biases the "
               "raw percentile interval downward (p2_isometry_linear.json: 20 of 48 intervals exclude their own "
               "estimate). BCa not used: its jackknife acceleration would need one refit per clip under this "
               "stratified scheme. Verdicts use the corrected geo - lin interval.")


def boot_summary(point, draws):
    """Point estimate, bootstrap mean, bias, raw percentile 95% CI and bias-corrected (recentred) 95% CI."""
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    bias = float(np.mean(draws) - point)
    return {"point": float(point), "boot_mean": float(np.mean(draws)), "bias": bias, "ci95_percentile": [lo, hi],
            "ci95_corrected": [lo - bias, hi - bias]}


def bootstrap_variants(L, n_boot=200, seed=0, n_steps=150, workers=1):
    """95% CIs on geo_pearson, lin_pearson and the paired geo - lin for the four VARIANTS (boot_summary per statistic;
    scheme in BOOT_SCHEME): n_boot draws of run_layer's resampling, clips within each value, knot clips (activation
    side) then probe clips (behaviour side), from default_rng(seed); PCA bases and the point estimate's coordinate
    choices fixed; the same draw is shared by all four variants and both arms (paired). The raw percentile interval
    is biased low by centroid noise, so each statistic also carries the bias and the interval recentred on the
    full-sample estimate. Draw indices are generated in order in
    this process, then evaluated in `workers` processes (results do not depend on workers)."""
    Zk, yk, pred, yp = boot_inputs(L)
    cent = mf.centroids(Zk, yk)
    auto = mf.choose_angle_source(cent["C"], cent["values"])
    _, bp, gb, _ = beh_pca_mode(pred, yp)
    fixed = {"auto": {k: auto[k] for k in ("angle", "plane")}, "bp": bp, "beh_branch": pca_mode_branch(gb),
             "act_branch": pca_mode_branch(mf.goodfire_periodic_angle(cent["C"]))}
    point = variant_matrices(Zk, yk, pred, yp, fixed, n_steps)
    rng = np.random.default_rng(seed)
    idx = []
    for _ in range(n_boot):
        ik = resample_idx(yk, rng)
        idx.append((ik, resample_idx(yp, rng)))
    chunks = [idx[i::workers] for i in range(workers)]
    if workers == 1:
        parts = [boot_chunk(L, idx, fixed, n_steps)]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(boot_chunk, [L] * workers, chunks, [fixed] * workers, [n_steps] * workers))
    boots = [None] * n_boot
    for i, part in enumerate(parts):
        boots[i::workers] = part
    res = {}
    for name in VARIANTS:
        res[name] = {}
        for sp in ("interp", "smooth"):
            g = np.array([b[name][sp]["geo_pearson"] for b in boots])
            li = np.array([b[name][sp]["lin_pearson"] for b in boots])
            pt = point[name][sp]
            res[name][sp] = {"geo_pearson": boot_summary(pt["geo_pearson"], g),
                             "lin_pearson": boot_summary(pt["lin_pearson"], li),
                             "geo_minus_lin": boot_summary(pt["geo_pearson"] - pt["lin_pearson"], g - li)}
    return {"n_boot": n_boot, "seed": seed, "scheme": BOOT_SCHEME, "fixed_choices": {"label_free_knot_order": fixed["auto"],
            "goodfire_full_activation_branch": fixed["act_branch"], "goodfire_full_behaviour_branch":
            fixed["beh_branch"]}, "variants": res}


def goodfire_method(L, n_steps=150, angle="auto"):
    """Isometry modelled on causalab compute_isometry_from_manifolds (path_mode geometric), with n_interior_per_pair 0
    (causalab's runners use 1): behaviour manifold = INTERPOLATING periodic spline (labels angle) through per-value
    centroids of the predictor's forecast in its full 1024-d space (probe folds), activation manifold = the knot-fold
    PCA-64 spline (interpolating as theirs, and the smoothing spline run_part2 steers with; coordinate from
    angle_choice: "auto" = choose_angle_source, "labels" = true direction as causalab's parameter-mode runners,
    "goodfire" = goodfire_periodic_angle on all 64 knot centroids as their pca-mode runners, "goodfire_full" = their
    full pca-mode rule on both sides: that angle if the periodicity test passes else PC1 natural cubic, and the
    behaviour manifold from beh_pca_mode instead of the labels-angle spline); both distances =
    path_length_matrix with 150 sub-intervals; Pearson r over the upper triangle. Chord baseline = PCA-64 chord between
    the same spline vertices, recomputed here."""
    d = load_inputs("direction", L)
    knot, probe = d["role"] == "knot", d["role"] == "probe"
    y, values = d["y"], np.unique(d["y"])
    pca = mf.fit_pca(d["X"][knot], 64)
    cent = mf.centroids(pca.project(d["X"][knot]), y[knot])
    choice = angle_choice(cent, angle)
    pred = np.load(PRED).astype(np.float64).mean(1)[probe]
    if angle == "goodfire_full":
        beh, _, gb, bC = beh_pca_mode(pred, y[probe])
    else:
        beh = mf.fit_curve(mf.centroids(pred, y[probe]), True, angle="labels", spline="interp")
    B = path_length_matrix(beh, values, n_steps)
    out = {"angle_choice": {k: choice[k] for k in ("angle", "plane")}, "behaviour_dim": int(pred.shape[1])}
    if angle == "goodfire_full":
        out["activation_side"] = side_record(choice["goodfire"], cent["C"], cent["values"])
        out["behaviour_side"] = side_record(gb, bC, values)
        out["coord_source"] = {"activation": choice["branch"], "behaviour": beh.coord_source}
    if angle in ("goodfire", "goodfire_full"):
        g = choice["goodfire"]
        out["goodfire_periodicity_test"] = {"passes": g["passes"], "rel_diff": g["rel_diff"],
                                            "eigenvalues_pc1_pc2": g["eigenvalues"], "n_centroids": len(values)}
        out["angle_vs_labels"] = {q: mf.compare_angles(g["angle"], cent["values"])[q]
                                  for q in ("circular_corr", "max_dev_deg", "mean_dev_deg")}
    for sp in ("interp", "smooth"):
        c = act_curve(cent, choice, sp)
        G = path_length_matrix(c, values, n_steps)
        P = by_value(np.linalg.norm(c.points[:, None] - c.points[None], axis=-1), c.values, values)
        r, rho = corr(G, B)
        rl, rhol = corr(P, B)
        out[sp] = {"geo_pearson": r, "geo_spearman": rho, "lin_pearson": rl, "lin_spearman": rhol}
    return out


def by_value(G, curve_values, values):
    """Reorder a vertex-ordered matrix into sorted-value order."""
    idx = {float(v): i for i, v in enumerate(curve_values)}
    o = [idx[float(v)] for v in values]
    return G[np.ix_(o, o)]


def act_curves(Z, y, rng=None, choice=None, angle="auto"):
    """Smoothing (run_part2) and interpolating (Goodfire A.3) periodic splines through per-value centroids of Z.
    choice: the point estimate's angle source and plane, held fixed in bootstrap draws (otherwise a draw can switch
    plane and reorder the knots, which is a different pipeline, not sampling noise)."""
    if rng is not None:
        Z, y = resample(Z, y, rng)
    cent = mf.centroids(Z, y)
    if choice is None:
        choice = angle_choice(cent, angle)
    out = {}
    for sp in ("smooth", "interp"):
        c = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=sp)
        G = by_value(geo_matrix(c), c.values, cent["values"])
        P = by_value(np.linalg.norm(c.points[:, None] - c.points[None], axis=-1), c.values, cent["values"])
        out[sp] = {"geo": G, "lin": P}
    return out, cent["values"], choice


def beh_curve(Z, y, rng=None):
    if rng is not None:
        Z, y = resample(Z, y, rng)
    cent = mf.centroids(Z, y)
    c = mf.fit_curve(cent, True, angle="labels", spline="smooth")
    return by_value(geo_matrix(c), c.values, cent["values"])


def resample(Z, y, rng):
    idx = np.concatenate([rng.choice(np.flatnonzero(y == v), (y == v).sum()) for v in np.unique(y)])
    return Z[idx], y[idx]


def corr(A, B):
    iu = np.triu_indices(len(A), 1)
    return float(np.corrcoef(A[iu], B[iu])[0, 1]), float(spearmanr(A[iu], B[iu])[0])


def run_layer(L, n_boot, seed=0, angle="auto"):
    d = load_inputs("direction", L)
    knot, probe = d["role"] == "knot", d["role"] == "probe"
    y = d["y"]
    pca = mf.fit_pca(d["X"][knot], 64)
    Zk = pca.project(d["X"][knot])
    d25 = load_inputs("direction", 25)
    pca25 = mf.fit_pca(d25["X"][probe], 64)
    Z25 = pca25.project(d25["X"][probe])
    pred = np.load(PRED).astype(np.float64).mean(1)
    pcap = mf.fit_pca(pred[probe], 64)
    Zp = pcap.project(pred[probe])
    yk, yp = y[knot], y[probe]
    values = np.unique(y)

    # Eq. 9 M_y at the steered layer: probe-fold reference smoothing spline in the knot PCA (as run_part2 builds it)
    ref_curve = mf.fit_curve(mf.centroids(pca.project(d["X"][probe]), yp), True, angle="labels", spline="smooth")
    bm = mf.BehaviourManifold(d["X"][probe], yp, True, curve=ref_curve, pca=pca)
    eq9 = bm.geodesic_matrix(values)
    dth = np.abs(((values[:, None] - values[None]) + 180) % 360 - 180)

    def all_corrs(act, beh):
        out = {}
        for sp, M in act.items():
            for kind in ("geo", "lin"):
                for name, B in beh.items():
                    r, rho = corr(M[kind], B)
                    out[f"{sp}.{kind}.{name}"] = {"pearson": r, "spearman": rho}
        return out

    act, vals, choice = act_curves(Zk, yk, angle=angle)
    assert np.allclose(vals, values)
    beh = {"eq9_same_layer": eq9, "encoder_out": beh_curve(Z25, yp), "predictor": beh_curve(Zp, yp), "concept": dth}
    point = all_corrs(act, beh)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        a, _, _ = act_curves(Zk, yk, rng, choice)
        b = {"eq9_same_layer": eq9, "encoder_out": beh_curve(Z25, yp, rng), "predictor": beh_curve(Zp, yp, rng),
             "concept": dth}
        boots.append(all_corrs(a, b))
    res = {}
    for key, v in point.items():
        pb = np.array([b[key]["pearson"] for b in boots])
        res[key] = {**v, "pearson_ci95": [float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))]}
    diffs = {}
    for sp in act:
        for name in beh:
            dd = np.array([b[f"{sp}.geo.{name}"]["pearson"] - b[f"{sp}.lin.{name}"]["pearson"] for b in boots])
            diffs[f"{sp}.{name}"] = {"geo_minus_lin": point[f"{sp}.geo.{name}"]["pearson"]
                                     - point[f"{sp}.lin.{name}"]["pearson"],
                                     "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    return {**layer_role("direction", L, "direction"), "angle_choice": {k: choice[k] for k in ("angle", "plane")},
            "n_knot_clips": int(knot.sum()), "n_probe_clips": int(probe.sum()), "correlations": res,
            "geo_minus_lin": diffs}


def run_goodfire_full(layers, n_boot, workers=6):
    """--angle goodfire_full -> results/p2_isometry_goodfire_full.json: goodfire_method with the pca-mode rule on both
    sides, the three stored coordinates (goodfire_coord.json) for reference, and bootstrap_variants per point."""
    prev = json.loads((PROJECT_ROOT / "results" / "p2_isometry_goodfire_coord.json").read_text())["layers"]
    res = {"provenance": provenance(None, seeds={"bootstrap": 0}, layers=layers, pool="meanpool", k=64,
                                    angle="goodfire_full", n_steps=150, n_boot=n_boot, bootstrap_scheme=BOOT_SCHEME,
                                    n_interior_per_pair=N_INTERIOR_NOTE),
           "method": goodfire_method.__doc__, "bootstrap_method": bootstrap_variants.__doc__,
           "behaviour_method": beh_pca_mode.__doc__,
           "angle_source": "causalab_pca_mode_goodfire_angle_or_pc1_fallback_both_sides", "layers": {}}
    boots = {L: bootstrap_variants(L, n_boot, workers=workers) for L in layers}
    for L in layers:
        new = goodfire_method(L, angle="goodfire_full")
        b = boots[L]["variants"]
        ref = {"goodfire_full": new, "goodfire": prev[str(L)]["goodfire_angle"], "labels": prev[str(L)]["labels_angle"],
               "label_free_knot_order": prev[str(L)]["unsupervised_angle"]}
        for k, r in ref.items():                             # bootstrap point estimates = the stored / direct values
            for sp in ("interp", "smooth"):
                for q in ("geo_pearson", "lin_pearson"):
                    assert abs(b[k][sp][q]["point"] - r[sp][q]) < 1e-8, (L, k, sp, q, b[k][sp][q], r[sp][q])
        res["layers"][str(L)] = {"goodfire_full_angle": new, **{k: prev[str(L)][k] for k in (
            "goodfire_angle", "labels_angle", "unsupervised_angle")}, "bootstrap": boots[L]}
        print(L, json.dumps({k: {sp: [round(v[sp][q]["point"], 3) for q in ("geo_pearson", "lin_pearson")]
                                 for sp in ("interp", "smooth")} for k, v in b.items()}), flush=True)
    g = {L: res["layers"][str(L)]["goodfire_full_angle"] for L in layers}
    res["branches"] = {str(L): v["coord_source"] for L, v in g.items()}
    res["periodicity_test_passes"] = {str(L): {s: v[f"{s}_side"]["passes"] for s in ("activation", "behaviour")}
                                      for L, v in g.items()}
    res["circular_corr_with_labels"] = {str(L): {s: v[f"{s}_side"]["atan2_angle_circular_corr_with_labels"]
                                                 for s in ("activation", "behaviour")} for L, v in g.items()}
    res["geo_minus_lin_verdict"] = {str(L): {k: {sp: ("spline" if lo > 0 else "chord" if hi < 0 else "tie")
                                                 for sp, (lo, hi) in ((sp, vv[sp]["geo_minus_lin"]["ci95_corrected"])
                                                                      for sp in ("interp", "smooth"))}
                                             for k, vv in res["layers"][str(L)]["bootstrap"]["variants"].items()}
                                    for L in layers}
    path = PROJECT_ROOT / "results" / "p2_isometry_goodfire_full.json"
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[8, 12, 22])
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p2_isometry_linear.json"))
    p.add_argument("--goodfire-method", action="store_true",
                   help="only the causalab-faithful variant (goodfire_method) -> results/p2_isometry_goodfire_method.json")
    p.add_argument("--angle", default="auto", choices=("auto", "labels", "goodfire", "goodfire_full"),
                   help="activation-spline coordinate: auto = choose_angle_source (unsupervised angle), labels = true "
                        "direction (intrinsic_mode: parameter, alphabet / age 8B and 70B runners), goodfire = "
                        "goodfire_periodic_angle on all 64 knot centroids (intrinsic_mode: pca, weekdays / months 8B; "
                        "--goodfire-method only), goodfire_full = causalab's pca-mode rule on both sides (PC1 "
                        "fallback when the periodicity test fails) + bootstrap CIs for four coordinates "
                        "(--goodfire-method only)")
    args = p.parse_args(argv)
    if args.angle in ("goodfire", "goodfire_full") and not args.goodfire_method:
        p.error(f"--angle {args.angle} needs --goodfire-method")
    if args.angle == "goodfire_full":
        run_goodfire_full(args.layers, args.n_boot)
        return
    if args.angle == "goodfire":
        lab = json.loads((PROJECT_ROOT / "results" / "p2_isometry_goodfire_labels.json").read_text())["layers"]
        uns = json.loads((PROJECT_ROOT / "results" / "p2_isometry_goodfire_method.json").read_text())["layers"]
        keep = ("geo_pearson", "lin_pearson")
        res = {"provenance": provenance(None, layers=args.layers, pool="meanpool", k=64, angle="goodfire",
                                        n_interior_per_pair=N_INTERIOR_NOTE),
               "method": goodfire_method.__doc__, "angle_source": "goodfire_pca_sqrtvar_atan2", "layers": {}}
        for L in args.layers:
            new = goodfire_method(L, angle="goodfire")
            res["layers"][str(L)] = {
                "goodfire_angle": new,
                "labels_angle": {sp: {q: lab[str(L)]["labels_angle"][sp][q] for q in keep} for sp in ("interp", "smooth")},
                "unsupervised_angle": {"angle_choice": uns[str(L)]["new"]["angle_choice"],
                                       **{sp: {q: uns[str(L)]["new"][sp][q] for q in keep} for sp in ("interp", "smooth")}}}
            print(L, new, flush=True)
        g = {L: res["layers"][str(L)]["goodfire_angle"] for L in args.layers}
        res["periodicity_test_passes"] = {str(L): v["goodfire_periodicity_test"]["passes"] for L, v in g.items()}
        res["circular_corr_with_labels"] = {str(L): v["angle_vs_labels"]["circular_corr"] for L, v in g.items()}
        res["geo_below_chord_goodfire_angle"] = {str(L): {sp: v[sp]["geo_pearson"] < v[sp]["lin_pearson"]
                                                          for sp in ("interp", "smooth")} for L, v in g.items()}
        path = PROJECT_ROOT / "results" / "p2_isometry_goodfire_coord.json"
        path.write_text(json.dumps(res, indent=1))
        print("wrote", path)
        return
    if args.goodfire_method and args.angle == "labels":
        prev = json.loads((PROJECT_ROOT / "results" / "p2_isometry_goodfire_method.json").read_text())["layers"]
        res = {"provenance": provenance(None, layers=args.layers, pool="meanpool", k=64, angle="labels",
                                        n_interior_per_pair=N_INTERIOR_NOTE),
               "method": goodfire_method.__doc__, "layers": {}}
        for L in args.layers:
            new = goodfire_method(L, angle="labels")
            old = prev[str(L)]["new"]
            res["layers"][str(L)] = {"labels_angle": new, "unsupervised_angle": {
                "angle_choice": old["angle_choice"],
                **{sp: {q: old[sp][q] for q in ("geo_pearson", "lin_pearson")} for sp in ("interp", "smooth")}}}
            print(L, new, flush=True)
        rev = {L: {sp: v[sp]["geo_pearson"] < v[sp]["lin_pearson"] for sp in ("interp", "smooth")}
               for L, v in ((L, res["layers"][str(L)]["labels_angle"]) for L in args.layers)}
        res["geo_below_chord_labels_angle"] = {str(L): r for L, r in rev.items()}
        path = PROJECT_ROOT / "results" / "p2_isometry_goodfire_labels.json"
        path.write_text(json.dumps(res, indent=1))
        print("wrote", path)
        return
    if args.goodfire_method:
        stored = json.loads((PROJECT_ROOT / "results" / "p2_isometry_linear.json").read_text())["layers"]
        res = {"provenance": provenance(None, layers=args.layers, pool="meanpool", k=64,
                                        n_interior_per_pair=N_INTERIOR_NOTE),
               "method": goodfire_method.__doc__, "layers": {}}
        for L in args.layers:
            c = stored[str(L)]["correlations"]
            res["layers"][str(L)] = {"new": goodfire_method(L), "stored": {
                q: c[q]["pearson"] for q in ("smooth.geo.predictor", "smooth.lin.predictor", "interp.geo.predictor",
                                             "interp.lin.predictor")}}
            print(L, res["layers"][str(L)]["new"], flush=True)
        path = PROJECT_ROOT / "results" / "p2_isometry_goodfire_method.json"
        path.write_text(json.dumps(res, indent=1))
        print("wrote", path)
        return
    out = {"provenance": provenance(None, seeds={"bootstrap": 0}, layers=args.layers, pool="meanpool", k=64),
           "method": ("Goodfire A.5: Pearson r over the upper triangle of 64 x 64 pairwise distance matrices; "
                      "key = activation_spline.distance.behaviour_space; activation_spline smooth = run_part2's "
                      "count-weighted smoothing spline, interp = Goodfire's interpolating spline; distance geo = arc "
                      "length along it, lin = chord in PCA-64 between the same vertices"),
           "caveat": ("eq9_same_layer is built from the steered layer's own activations (circular, as in Goodfire's "
                      "mountain-car B.1); encoder_out, predictor and concept are not. For a planar circle the chord "
                      "is a monotone function of the arc (2R sin(d/2)), so Pearson geo-vs-lin gaps on a ring are "
                      "small by construction; Spearman is reported beside Pearson."),
           "layers": {}}
    for L in args.layers:
        out["layers"][str(L)] = run_layer(L, args.n_boot, angle=args.angle)
        r = out["layers"][str(L)]["correlations"]
        print(f"L{L}: " + "  ".join(f"{b}: geo {r[f'smooth.geo.{b}']['pearson']:.3f} lin {r[f'smooth.lin.{b}']['pearson']:.3f}"
                                     for b in ("eq9_same_layer", "encoder_out", "predictor", "concept")), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

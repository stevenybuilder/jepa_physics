"""Cheap geometry checks to run before trusting any spline (nonobvious_components.md §6-7, lessons file D.2).

Each check is one function returning a small dict of plain numbers. They all work on stored activations.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import subspace_angles

from wm.manifold import TWO_PI, Curve, centroids, fit_curve, fit_pca

# ---------------------------------------------------------------- helpers -----------------------------------------


def align(per_value, values, curve_values):
    """Reorder a per-value array (indexed like `values`) into the curve's knot order."""
    idx = {float(v): i for i, v in enumerate(values)}
    return np.asarray(per_value)[[idx[float(v)] for v in curve_values]]


def _spline(coords, points, periodic):
    if periodic:
        return CubicSpline(np.append(coords, coords[0] + TWO_PI), np.vstack([points, points[:1]]), bc_type="periodic")
    return CubicSpline(coords, points, bc_type="natural")


def _polyline(coords, points, periodic, t):
    if periodic:
        points = np.vstack([points, points[:1]])
        coords = np.append(coords, coords[0] + TWO_PI)
        t = (t - coords[0]) % TWO_PI + coords[0]
    return np.array([np.interp(t, coords, points[:, j]) for j in range(points.shape[1])])


# ---------------------------------------------------------------- (i) leave one value out -------------------------

def loo_reconstruction(curve, noise=None, stride=1):
    """Rebuild knots the fit did not see, with a cubic spline and with a straight polyline through the others.

    stride=1: leave each knot out in turn and fit on all the rest (leave-one-value-out).
    stride=s: fit on every s-th knot (each of the s offsets in turn) and rebuild the knots in between, so each
              rebuilt knot sits inside a gap of s knot spacings.
    Why stride: with 64 dense knots the bend over one spacing is tiny (a ring's chord misses the arc by
    R(1 - cos(2pi/64)) = 0.5% of R), so centroid noise decides stride=1. Larger strides show the curvature that a
    steer of that size meets. (A hole cut into the dense knots is not used: a spline pinned by noisy, closely
    spaced knots overshoots across a hole.) Sequential curves: knots beyond the outermost fit knots are skipped.
    noise: optional per-knot centroid noise floor (e.g. spread / sqrt(count)) in the curve's knot order.
    Returns per-knot errors and a summary: mean errors, mean ln(cubic/line) (negative = cubic better, the
    jepa_steering convention), fraction of knots where the cubic wins.
    """
    c, P, per = curve.coords, curve.points, curve.periodic
    m = len(c)
    if stride == 1:
        fits = [np.arange(m) != i for i in range(m)]
    else:
        fits = [np.arange(m) % stride == o for o in range(stride)]
    err_cubic, err_line, which = [], [], []
    for keep in fits:
        fit_idx = np.flatnonzero(keep)
        if len(fit_idx) < 4:
            continue
        spline = _spline(c[keep], P[keep], per)
        targets = np.flatnonzero(~keep)
        if not per:
            targets = targets[(targets > fit_idx.min()) & (targets < fit_idx.max())]
        for i in targets:
            err_cubic.append(np.linalg.norm(spline(c[i]) - P[i]))
            err_line.append(np.linalg.norm(_polyline(c[keep], P[keep], per, c[i]) - P[i]))
            which.append(i)
    ec, el = np.array(err_cubic), np.array(err_line)
    out = {"stride": stride, "values": curve.values[which].tolist(), "err_cubic": ec.tolist(), "err_line": el.tolist(),
           "mean_err_cubic": float(ec.mean()), "mean_err_line": float(el.mean()),
           "mean_log_ratio": float(np.mean(np.log(ec / el))), "frac_cubic_wins": float(np.mean(ec < el)),
           "winner": "cubic" if ec.mean() < el.mean() else "line"}
    if noise is not None:
        nz = np.asarray(noise)[which]
        out["mean_noise_floor"] = float(nz.mean())
        out["line_err_over_noise"] = float(el.mean() / nz.mean())
    return out


def loo_from_activations(X, labels, periodic, k=64, angle="unsupervised", plane="activation", stride=1):
    """PCA -> centroids -> curve -> loo_reconstruction, in one call (used for the precision check)."""
    pca = fit_pca(X, k)
    cent = centroids(pca.project(X), labels)
    curve = fit_curve(cent, periodic, angle=angle, plane=plane)
    noise = align(cent["spread"] / np.sqrt(cent["count"]), cent["values"], curve.values)
    return loo_reconstruction(curve, noise, stride)


# ---------------------------------------------------------------- (ii) precision -----------------------------------

def bf16_round(X):
    """Round float32 values to the nearest bfloat16 (round half to even) and return them as float32.
    bfloat16 keeps the top 16 bits of a float32; the rounding adds half of the dropped part first."""
    u = np.ascontiguousarray(X, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def precision_check(X, labels, periodic, k=64, angle="unsupervised", plane="activation", stride=1):
    """Leave-one-out spline-vs-line in float32 and after rounding the activations to bfloat16."""
    fp32 = loo_from_activations(np.asarray(X, np.float32), labels, periodic, k, angle, plane, stride)
    bf16 = loo_from_activations(bf16_round(X), labels, periodic, k, angle, plane, stride)
    keys = ("mean_err_cubic", "mean_err_line", "mean_log_ratio", "frac_cubic_wins", "winner")
    return {"fp32": {q: fp32[q] for q in keys}, "bf16": {q: bf16[q] for q in keys},
            "winner_same": fp32["winner"] == bf16["winner"]}


# ---------------------------------------------------------------- (iii) spread vs curvature -----------------------

def spread_vs_curvature(curve, spread, count):
    """Local bend of the centroid curve vs the noise around it.

    bend_i = distance from knot i to the chord between its neighbours (at knot i's coordinate), i.e. a second
    difference. If the bend is smaller than the centroid noise (spread / sqrt(count)), a spline is fitting noise.
    spread, count: in the curve's knot order.
    """
    c, P = curve.coords, curve.points
    m = len(c)
    idx = range(m) if curve.periodic else range(1, m - 1)
    bend = []
    for i in idx:
        a, b = (i - 1) % m, (i + 1) % m
        ta, tb, ti = c[a], c[b], c[i]
        if curve.periodic:
            ta, tb = ti - ((ti - ta) % TWO_PI), ti + ((tb - ti) % TWO_PI)
        w = (ti - ta) / (tb - ta)
        bend.append(np.linalg.norm(P[i] - ((1 - w) * P[a] + w * P[b])))
    bend = np.array(bend)
    spread, count = np.asarray(spread, float)[list(idx)], np.asarray(count, float)[list(idx)]
    noise = spread / np.sqrt(count)
    return {"median_bend": float(np.median(bend)), "median_spread": float(np.median(spread)),
            "median_centroid_noise": float(np.median(noise)),
            "bend_over_noise": float(np.median(bend / noise)), "bend_over_spread": float(np.median(bend / spread))}


# ---------------------------------------------------------------- (iv) participation ratio ------------------------

def participation_ratio(M):
    """(sum of covariance eigenvalues)^2 / sum of squared eigenvalues: the effective number of dimensions."""
    M = np.asarray(M, dtype=float)
    M = M - M.mean(axis=0)
    lam = np.linalg.svd(M, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def participation_check(Z, labels):
    """Participation ratio of the centroids vs of the within-value residuals (clip minus its value's centroid)."""
    cent = centroids(Z, labels)
    lookup = {float(v): c for v, c in zip(cent["values"], cent["C"])}
    resid = np.asarray(Z) - np.array([lookup[float(v)] for v in labels])
    return {"pr_centroids": participation_ratio(cent["C"]), "pr_residuals": participation_ratio(resid)}


# ---------------------------------------------------------------- (v) knot spacing --------------------------------

def _r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def knot_spacing(curve):
    """Cumulative chord length along the knots (sorted by value) against the value: linear or log-like?
    Fits s = a + b*v and s = a + b*log(v) by least squares and reports both R^2. Needs values > 0."""
    order = np.argsort(curve.values)
    v, P = curve.values[order].astype(float), curve.points[order]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    out = {}
    for name, x in (("linear", v), ("log", np.log(v))):
        b, a = np.polyfit(x, s, 1)
        out[f"r2_{name}"] = _r2(s, a + b * x)
    out["better"] = "log" if out["r2_log"] > out["r2_linear"] else "linear"
    return out


# ---------------------------------------------------------------- (vi) shared vs clip-specific delta --------------

def shared_delta_fraction(deltas):
    """Fraction of edit energy in the edit shared by all clips: n ||mean delta||^2 / sum_i ||delta_i||^2.
    1 = every clip got the same constant shift; ~1/n for independent random edits."""
    D = np.asarray(deltas, dtype=float).reshape(len(deltas), -1)
    total = (D ** 2).sum()
    return float(len(D) * (D.mean(axis=0) ** 2).sum() / total) if total > 0 else float("nan")


# ---------------------------------------------------------------- (vii) principal angles --------------------------

def principal_angles_to_centroid_plane(V, C, n_plane=2):
    """Principal angles (degrees) between span(V) and the plane of the top n_plane principal axes of the centroids.
    V [D, r] and C [m, D] must be in the same coordinate system (e.g. both train-standardised, as INLP bases are)."""
    Cc = np.asarray(C, float) - np.asarray(C, float).mean(axis=0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    ang = np.degrees(subspace_angles(np.asarray(V, float), Vt[:n_plane].T))
    return {"angles_deg": np.sort(ang).tolist(), "max_deg": float(ang.max()), "min_deg": float(ang.min())}


# ---------------------------------------------------------------- direction: circle or cone -----------------------

def ring_radius_by_group(Z, theta, group):
    """For direction: per group (e.g. motion type or speed bin), the mean distance of that group's per-angle
    centroids from the group's own centre. If the radius changes with speed, the manifold is a cone, not a ring."""
    Z, theta, group = np.asarray(Z), np.asarray(theta), np.asarray(group)
    out = {}
    for g in np.unique(group):
        C = centroids(Z[group == g], theta[group == g])["C"]
        out[str(g)] = float(np.linalg.norm(C - C.mean(axis=0), axis=1).mean())
    return out


# ---------------------------------------------------------------- controls ----------------------------------------

def _curve_like(curve, points):
    """A curve with the same knots (values, coordinates, type) as `curve` but different points."""
    spline = _spline(curve.coords, points, curve.periodic)
    return Curve(spline=spline, values=curve.values, coords=curve.coords, points=points,
                 periodic=curve.periodic, coord_source=curve.coord_source + "+control")


def shuffled_curve(curve, rng):
    """Control ("shuffled_unmatched"): the same centroids assigned to the knots in a random order, same spline type.
    Unmatched: a path along it does not share the spline path's endpoints, so endpoint metrics favour the spline
    trivially; the endpoint-matched control is manifold.endpoint_matched_random."""
    return _curve_like(curve, curve.points[rng.permutation(len(curve.points))])


def random_smooth_curve(curve, rng, n_freq=3):
    """Control ("random_unmatched"; endpoints not matched, see shuffled_curve): a random smooth curve through the
    same PCA space, with the same knots, the same total arc length
    and the same centre as `curve`. Points = a few random low-frequency sines/cosines of the knot coordinate
    (Fourier modes for a periodic curve), then rescaled."""
    t = curve.coords
    lo, hi = curve.t_range()
    u = (t - lo) / (hi - lo)                             # 0..1 along the curve
    k = curve.points.shape[1]
    P = np.zeros((len(t), k))
    for j in range(1, n_freq + 1):
        a, b = rng.standard_normal(k), rng.standard_normal(k)
        freq = 2 * np.pi * j if curve.periodic else np.pi * j
        P += (np.outer(np.cos(freq * u), a) + np.outer(np.sin(freq * u), b)) / j
    ref = _curve_like(curve, curve.points)
    rnd = _curve_like(curve, P)

    def total_length(c):
        tt = np.linspace(lo, hi, 2001)
        return np.linalg.norm(np.diff(c(tt), axis=0), axis=1).sum()

    P = (P - P.mean(axis=0)) * (total_length(ref) / total_length(rnd)) + curve.points.mean(axis=0)
    return _curve_like(curve, P)


# ---------------------------------------------------------------- circular chart (supervised check) ---------------

def fit_circular_chart(X, theta_deg):
    """Least-squares chart X ~ mu + A [cos theta, sin theta] (jepa_steering's 2 Sep chart). Returns dict(mu [D],
    A [D, 2]). Fit on training rows only."""
    r = np.radians(np.asarray(theta_deg, dtype=float))
    M = np.stack([np.ones_like(r), np.cos(r), np.sin(r)], axis=1)       # [n, 3]
    coef = np.linalg.lstsq(M, np.asarray(X, dtype=float), rcond=None)[0]  # [3, D]
    return {"mu": coef[0], "A": coef[1:].T}


def chart_decode(chart, X):
    """Angle (degrees, [0, 360)) of each row on the fitted chart plane: least-squares coordinates c of (x - mu) in
    span(A), then atan2(c_sin, c_cos)."""
    c = (np.asarray(X, dtype=float) - chart["mu"]) @ np.linalg.pinv(chart["A"]).T
    return np.degrees(np.arctan2(c[:, 1], c[:, 0])) % 360.0


def chart_check(X, theta_deg, pca, plane_used="activation"):
    """Supervised circular chart vs Goodfire's unsupervised atan2(PC2, PC1) on the same training rows.

    Reports, for the per-value centroids and for the individual clips: circular correlation and aligned mean
    absolute deviation between the chart angle and the unsupervised angle (orientation and zero of atan2 are free),
    the chart's error against the true angle, and the principal angles between span(A) and the top-2 PC plane.
    plane_used: the plane the pipeline chose for its own unsupervised angle, reported alongside.
    """
    from wm.manifold import compare_angles, unsupervised_angle, value_error
    X, theta = np.asarray(X, dtype=float), np.asarray(theta_deg, dtype=float)
    chart = fit_circular_chart(X, theta)
    values = np.unique(theta)
    C_full = np.array([X[theta == v].mean(0) for v in values])
    chart_c = chart_decode(chart, C_full)
    unsup_c = unsupervised_angle(pca.project(C_full), "activation")
    chart_x = chart_decode(chart, X)
    Zx = pca.project(X)[:, :2]
    unsup_x = np.arctan2(Zx[:, 1] - Zx[:, 1].mean(), Zx[:, 0] - Zx[:, 0].mean()) % TWO_PI

    def agree(unsup, chart_deg):
        c = compare_angles(unsup, chart_deg)
        return {"circular_corr": abs(c["circular_corr"]), "mae_deg": c["mean_dev_deg"], "max_dev_deg": c["max_dev_deg"],
                "orientation": c["orientation"]}

    ang = np.degrees(subspace_angles(chart["A"], pca.components[:2].T))
    out = {"centroids_chart_vs_pc12": agree(unsup_c, chart_c),
           "clips_chart_vs_pc12": agree(unsup_x, chart_x),
           "chart_vs_true_mae_deg": {"centroids": float(np.mean(value_error(chart_c, values, True))),
                                     "clips_in_sample": float(np.mean(value_error(chart_x, theta, True)))},
           "principal_angles_chart_vs_pc12_deg": np.sort(ang).tolist(),
           "chart_radius": float(np.linalg.svd(chart["A"], compute_uv=False).mean()),
           "pipeline_plane": plane_used}
    if plane_used == "centroid":
        out["centroids_chart_vs_centroid_plane"] = agree(unsupervised_angle(pca.project(C_full), "centroid"), chart_c)
    return out


# ---------------------------------------------------------------- planted-ring positive control -------------------

def within_value_sd(X, labels, U):
    """Pooled within-value SD of X projected on each column of U [D, q], averaged over the columns (RMS)."""
    P = np.asarray(X, dtype=float) @ U
    labels = np.asarray(labels)
    resid = P.copy()
    for v in np.unique(labels):
        rows = labels == v
        resid[rows] -= P[rows].mean(0)
    return float(np.sqrt((resid ** 2).mean()))


def planted_ring_control(X, labels, radii=(0.5, 1, 2, 4), k=64, n_values=64, seed=0, stride=16, min_corr=0.95):
    """Plant a ring of known radius into real activations and check that the Part 2 pipeline recovers it.

    Ring: two random orthonormal directions Q [D, 2]; synthetic labels theta on a 64-value grid; X' = X + r * sd *
    [cos theta, sin theta] Q^T, with sd the pooled within-value SD of X along Q (grouped by the real `labels`), so r
    is in units of within-value SD. theta is assigned stratified by the real label: rows sorted by the real label
    are cut into blocks of 64 and each block gets a random permutation of the 64 theta values. Each planted value
    then averages clips spread evenly over the real label's range, so the real variable (whose sampling noise in a
    randomly assigned centroid is larger than a small ring) cancels in the planted centroids. Other real variance
    is left as it is; that is the noise the control is meant to face.
    Pipeline: PCA-k -> centroids by theta -> choose_angle_source (atan2 in the activation or centroid plane) ->
    periodic spline -> knot-subsampling LOO at `stride` (cubic gain = line error / cubic error).
    Recovered = |circular corr| (Fisher-Lee, unsupervised atan2 angle vs planted theta, best of the activation and
    centroid planes) > min_corr, and the cubic beats the line at the stride. `pipeline_angle_source` says whether
    choose_angle_source would also have accepted the unsupervised angle (it additionally demands strict knot order,
    which noisy dense knots can fail while the angle is still recovered).
    Returns dict(sd, rows=[one per r]).
    """
    from wm.manifold import centroids as _centroids, choose_angle_source, fit_curve
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((X.shape[1], 2)))[0]
    grid = np.arange(n_values) * 360.0 / n_values
    order = np.argsort(np.asarray(labels, dtype=float), kind="stable")
    theta = np.empty(len(X))
    for i in range(0, len(X), n_values):
        blk = order[i:i + n_values]
        theta[blk] = rng.permutation(grid)[:len(blk)]
    sd = within_value_sd(X, labels, Q)
    ring = np.stack([np.cos(np.radians(theta)), np.sin(np.radians(theta))], 1) @ Q.T
    rows = []
    for r in radii:
        Xp = X + r * sd * ring
        pca = fit_pca(Xp, k)
        cent = _centroids(pca.project(Xp), theta)
        choice = choose_angle_source(cent["C"], cent["values"])
        best_plane = max(choice["checks"], key=lambda pl: abs(choice["checks"][pl]["circular_corr"]))
        used = choice["checks"][best_plane]
        curve = fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"])
        noise = align(cent["spread"] / np.sqrt(cent["count"]), cent["values"], curve.values)
        loo = loo_reconstruction(curve, noise, stride=stride)
        corr = abs(used["circular_corr"])
        rows.append({"r_within_sd": float(r), "ring_radius": float(r * sd),
                     "ring_var_in_pca_frac": float(((pca.components @ Q) ** 2).sum() / 2),
                     "pipeline_angle_source": choice["angle"], "pipeline_plane": choice["plane"],
                     "best_plane": best_plane, "circular_corr": corr,
                     "max_dev_deg": used["max_dev_deg"], "loo_stride": stride,
                     "loo_err_line": loo["mean_err_line"], "loo_err_cubic": loo["mean_err_cubic"],
                     "cubic_gain": loo["mean_err_line"] / loo["mean_err_cubic"], "loo_winner": loo["winner"],
                     "recovered": bool(corr > min_corr and loo["winner"] == "cubic")})
    return {"sd": sd, "n_values": n_values, "seed": seed, "min_corr": min_corr, "rows": rows}


# ---------------------------------------------------------------- held-out block reconstruction ------------------

def heldout_reconstruction(X, labels, periodic, k=64, angle="unsupervised", plane="activation", seeds=(0, 1, 2, 3)):
    """Which curve should steering use? For each held-out design (train rows only): fit PCA + centroids on the kept
    values, then rebuild the held-out centroids (computed from the same rows, never used by the fit) with the
    interpolating spline, the count-weighted smoothing spline, and the chord between kept neighbours.
    Contiguous blocks are averaged over `seeds`. Errors in PCA units; the lowest wins. A development decision:
    make it here, on train, before any test read."""
    from wm.manifold import centroids as _centroids, heldout_design, piecewise_linear_point
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    values = np.unique(labels)
    designs = [("scattered", 0), *[("contiguous", s) for s in seeds]] + ([] if periodic else [("extrapolation", 0)])
    acc = {}
    for design, seed in designs:
        mask, _ = heldout_design(values, design, periodic, seed=seed)
        rows = ~np.isin(labels, values[mask])
        pca = fit_pca(X[rows], k)
        cent = _centroids(pca.project(X[rows]), labels[rows])
        truth = np.array([pca.project(X[labels == v]).mean(0) for v in values[mask]])
        errs = {}
        for sp in ("interp", "smooth"):
            curve = fit_curve(cent, periodic, angle=angle, plane=plane, spline=sp)
            errs[sp] = float(np.linalg.norm(curve(curve.coord_of_value(values[mask])) - truth, axis=1).mean())
            if sp == "interp":
                errs["chord"] = float(np.linalg.norm(piecewise_linear_point(curve, values[mask]) - truth, axis=1).mean())
        acc.setdefault(design, []).append(errs)
    out = {}
    for design, runs in acc.items():
        mean = {q: float(np.mean([r[q] for r in runs])) for q in ("interp", "smooth", "chord")}
        out[design] = {**mean, "n_seeds": len(runs), "best": min(mean, key=mean.get)}
    return out


# ---------------------------------------------------------------- ring or velocity plane --------------------------

def procrustes_r2(Y, T):
    """Scaled orthogonal Procrustes of rows Y [m, D] onto 2-D targets T [m, 2] (both centred): Y ~ s T R^T with
    R [D, 2] orthonormal. Returns dict(r2 = fraction of Y's variance explained, scale s, R, means)."""
    Y, T = np.asarray(Y, dtype=float), np.asarray(T, dtype=float)
    my, mt = Y.mean(0), T.mean(0)
    Yc, Tc = Y - my, T - mt
    U, S, Vt = np.linalg.svd(Yc.T @ Tc, full_matrices=False)
    R = U @ Vt
    s = S.sum() / (Tc ** 2).sum()
    resid = Yc - s * Tc @ R.T
    return {"r2": float(1 - (resid ** 2).sum() / (Yc ** 2).sum()), "scale": float(s), "R": R, "mu_y": my, "mu_t": mt}


def velocity_plane_check(X, theta_deg, speed, n_speed_bins=8, n_dir_bins=16):
    """Is direction coded as a ring (cos, sin) independent of speed, or as a velocity plane (v cos, v sin)?
    On a dataset that crosses speed and direction (the speed set): cell centroids over speed bin x direction bin,
    Procrustes R^2 onto each target, and the ring radius (direction-bin centroids about their centre) per speed bin.
    A ring predicts a flat radius; a velocity plane predicts radius proportional to speed."""
    X, th, v = np.asarray(X, dtype=float), np.asarray(theta_deg, dtype=float), np.asarray(speed, dtype=float)
    sb = np.clip(np.searchsorted(np.quantile(v, np.linspace(0, 1, n_speed_bins + 1)[1:-1]), v, side="right"),
                 0, n_speed_bins - 1)
    db = (np.floor(th / (360.0 / n_dir_bins)).astype(int)) % n_dir_bins
    cells, Tr, Tv = [], [], []
    for i in range(n_speed_bins):
        for j in range(n_dir_bins):
            rows = (sb == i) & (db == j)
            if rows.sum() == 0:
                continue
            a = np.radians(np.angle(np.exp(1j * np.radians(th[rows])).mean(), deg=True))
            cells.append(X[rows].mean(0))
            Tr.append([np.cos(a), np.sin(a)])
            Tv.append([v[rows].mean() * np.cos(a), v[rows].mean() * np.sin(a)])
    cells, Tr, Tv = np.array(cells), np.array(Tr), np.array(Tv)
    ring, vel = procrustes_r2(cells, Tr), procrustes_r2(cells, Tv)
    radius = []
    for i in range(n_speed_bins):
        C = np.array([X[(sb == i) & (db == j)].mean(0) for j in range(n_dir_bins) if ((sb == i) & (db == j)).any()])
        radius.append({"speed_mean": float(v[sb == i].mean()),
                       "radius": float(np.linalg.norm(C - C.mean(0), axis=1).mean()), "n_dir_bins": len(C)})
    r, s = np.array([x["radius"] for x in radius]), np.array([x["speed_mean"] for x in radius])
    return {"procrustes_r2_ring": ring["r2"], "procrustes_r2_velocity": vel["r2"],
            "better": "velocity_plane" if vel["r2"] > ring["r2"] else "ring", "n_cells": len(cells),
            "radius_by_speed": radius, "radius_ratio_top_bottom": float(r[-1] / r[0]),
            "speed_ratio_top_bottom": float(s[-1] / s[0]), "radius_speed_corr": float(np.corrcoef(r, s)[0, 1]),
            "_velocity_fit": vel}


# ---------------------------------------------------------------- nuisance regression and alternative subspaces -----

def nuisance_matrix(df, variable):
    """Nuisance covariates for `variable` (Engels et al. 2405.14860 App. K style): every physical covariate except the
    variable itself: speed, acceleration, motion type (one-hot), start_x, start_y, and (cos, sin) of direction when
    direction is not the variable. Constant columns dropped. Returns (N [n, c], column names). The QA json holds
    only a dataset summary of disk visibility, so a per-clip visible fraction is not included."""
    cols, names = [], []
    th = np.radians(df["theta_degrees"].to_numpy(float))
    cand = {"speed_mps": df["speed_mps"], "acceleration_mps2": df["acceleration_mps2"],
            "start_x": df["start_x"] if "start_x" in df else None, "start_y": df["start_y"] if "start_y" in df else None}
    skip = {"speed": "speed_mps", "acceleration": "acceleration_mps2"}.get(variable)
    for name, col in cand.items():
        if col is not None and name != skip:
            cols.append(np.asarray(col, float))
            names.append(name)
    if variable != "direction":
        cols += [np.cos(th), np.sin(th)]
        names += ["cos_theta", "sin_theta"]
    if "motion" in df:
        for mtype in sorted(df["motion"].unique())[1:]:
            cols.append((df["motion"] == mtype).to_numpy(float))
            names.append(f"motion={mtype}")
    N = np.stack(cols, 1)
    keep = N.std(0) > 1e-12
    return N[:, keep], [n for n, k in zip(names, keep) if k]


def regress_out(X, N, train):
    """Standardise X on train rows, fit X_std ~ [1, N] by least squares on train rows only, and return the residual
    for every row (the same projection applied to all clips)."""
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-8
    Xs = (X - mu) / sd
    M = np.column_stack([np.ones(len(N)), N])
    B = np.linalg.lstsq(M[train], Xs[train], rcond=None)[0]
    R = Xs - M @ B
    r2 = 1 - (R[train] ** 2).sum() / (((Xs[train] - Xs[train].mean(0)) ** 2).sum())
    return R, float(r2)


def fit_subspace(X, labels, kind="pca", k=64, basis=None):
    """The subspace in which centroids, splines and steering paths live, as a PCA-like object (project / lift /
    complement). kind: "pca" (top-k PCA, Goodfire A.3), "chart" (the 2-D supervised circular-chart plane, for when
    the unsupervised angle check fails) or "inlp" (span of a Part 1 INLP basis, basis [D, r], orthonormalised)."""
    from wm.manifold import PCA
    X = np.asarray(X, dtype=np.float64)
    if kind == "pca":
        return fit_pca(X, k)
    if kind == "chart":
        B = fit_circular_chart(X, labels)["A"]
    elif kind == "inlp":
        B = np.asarray(basis, dtype=float)
    else:
        raise ValueError(kind)
    Qb = np.linalg.qr(B)[0].T                                           # [r, D]
    mean = X.mean(0)
    Zc = (X - mean) @ Qb.T
    var = Zc.var(0, ddof=1)
    order = np.argsort(var)[::-1]
    return PCA(mean=mean, components=Qb[order], explained=var[order])


def load_basis_matrix(path, X_train=None):
    """[D, r] basis from a .npy, or the Q of an INLP .npz (train-standardised coordinates, mapped to raw-space
    directions by dividing by the train SD when X_train is given)."""
    if str(path).endswith(".npz"):
        Q = np.load(path)["Q"]
        if X_train is not None:
            Q = Q / (np.asarray(X_train).std(0) + 1e-8)[:, None]
        return Q
    return np.load(path)


def expected_sagitta(radius, n_values, strides, noise):
    """Per stride: the chord-vs-arc gap at the middle of the gap the LOO test leaves (2 spacings for stride 1, else
    `stride` spacings), r (1 - cos(gap / 2)), and its ratio to the centroid noise. Curvature is detectable only where
    the ratio is well above 1."""
    out = {}
    for s in strides:
        gap = np.radians((2 if s == 1 else s) * 360.0 / n_values)
        sag = radius * (1 - np.cos(gap / 2))
        out[str(s)] = {"gap_deg": float(np.degrees(gap)), "expected_sagitta": float(sag),
                       "sagitta_over_centroid_noise": float(sag / noise)}
    return out

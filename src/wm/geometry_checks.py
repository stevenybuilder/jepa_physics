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
    """Control: the same centroids assigned to the knots in a random order (shuffled labels), same spline type."""
    return _curve_like(curve, curve.points[rng.permutation(len(curve.points))])


def random_smooth_curve(curve, rng, n_freq=3):
    """Control: a random smooth curve through the same PCA space, with the same knots, the same total arc length
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

"""Part 2 core: Goodfire-style manifold steering (arXiv steering paper, App. A.3-A.7) adapted to V-JEPA.

Pipeline, all on one layer of `meanpool` activations X [n, D]:
  1. PCA-64 fit on training rows only. The manifold lives in the PCA space; the orthogonal complement of every
     activation is kept untouched by manifold steering (A.3, A.6).
  2. One centroid per label value in PCA space.
  3. A cubic spline through the centroids: natural for sequential variables (speed, acceleration; intrinsic
     coordinate = the physical value), periodic for direction (intrinsic coordinate = atan2(PC2, PC1) of the
     centroids, found without labels, then checked against the true angle).
  4. Steering paths of K waypoints: a straight line between centroids (linear) or a walk along the spline
     (manifold). Both are written as additive edits x + delta by default, so they are compared like for like.
  5. Metrics: off-manifold energy (distance to the curve, distance to the nearest real activation), path lengths,
     isometry, and two pluggable readouts.

What plays Goodfire's behaviour manifold M_y. V-JEPA has no output distribution, so there is no M_y in the
paper's sense. Two stand-ins are provided, and neither is called "behaviour":
  - ProbeReadout: a ridge probe fit on clips disjoint from the knots and from the steered clips. M_y is the
    probe's output space (the predicted value, or (sin, cos) for direction). Weakest rung: it is a probe again.
  - NearestRealReadout: agreement with V-JEPA's own activations on real clips that have the target value.
    M_y is the set of real-clip activation centroids at each value, at the layer being read. Applied at later
    layers after propagating the edit, this is the strongest offline readout (the implementation-pitfalls note §5.3, §8).
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import CubicSpline, splev, splrep
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TWO_PI = 2 * np.pi


def wrap_pi(a):
    """Wrap angles (radians) to (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(a, dtype=float)))


# ---------------------------------------------------------------- 1. PCA ------------------------------------------

@dataclass
class PCA:
    mean: np.ndarray        # [D]
    components: np.ndarray  # [k, D], orthonormal rows, sorted by variance
    explained: np.ndarray   # [k] variance of each component

    def project(self, X):
        """Full space [.., D] -> PCA coordinates [.., k]."""
        return (np.asarray(X) - self.mean) @ self.components.T

    def lift(self, Z):
        """PCA coordinates [.., k] -> full space [.., D] (the point inside the PCA subspace)."""
        return self.mean + np.asarray(Z) @ self.components

    def lift_delta(self, dZ):
        """A displacement in PCA coordinates -> the same displacement in full space (no mean)."""
        return np.asarray(dZ) @ self.components

    def complement(self, X):
        """The part of X outside the PCA subspace. X == lift(project(X)) + complement(X)."""
        return np.asarray(X) - self.lift(self.project(X))


def fit_pca(X_train, k=64):
    """PCA by SVD of the centred training rows. k is capped at min(n - 1, D)."""
    X = np.asarray(X_train, dtype=np.float64)
    mean = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, X.shape[0] - 1, X.shape[1])
    return PCA(mean=mean, components=Vt[:k], explained=s[:k] ** 2 / (X.shape[0] - 1))


# ---------------------------------------------------------------- 2. centroids ------------------------------------

def centroids(Z, labels):
    """Per label value: centroid, clip count, and spread (mean distance of the clips to their centroid).

    Returns dict(values [m], C [m, k], count [m], spread [m], sd_coord [k]), values sorted. sd_coord is the pooled
    within-value SD of each coordinate (what the smoothing spline weights by).
    """
    Z, labels = np.asarray(Z), np.asarray(labels)
    values = np.unique(labels)
    C, count, spread, ss = [], [], [], 0.0
    for v in values:
        rows = Z[labels == v]
        c = rows.mean(axis=0)
        C.append(c)
        count.append(len(rows))
        spread.append(np.linalg.norm(rows - c, axis=1).mean())
        ss = ss + ((rows - c) ** 2).sum(axis=0)
    dof = max(len(Z) - len(values), 1)
    return {"values": values, "C": np.array(C), "count": np.array(count), "spread": np.array(spread),
            "sd_coord": np.sqrt(ss / dof)}


# ---------------------------------------------------------------- 3. curves ---------------------------------------

@dataclass
class Curve:
    """A cubic spline through centroids in PCA space, with its intrinsic coordinate t.

    values: label value of each knot; coords: intrinsic coordinate of each knot (same order).
    For a periodic curve t is an angle in radians with period 2*pi, and values are degrees.
    """
    spline: CubicSpline
    values: np.ndarray
    coords: np.ndarray
    points: np.ndarray
    periodic: bool
    coord_source: str       # "value", "unsupervised_angle" or "labels_angle" (callers may retag, e.g. "goodfire_pca_angle")
    kind: str = "interpolating"   # or "smoothing" (count-weighted, Goodfire B.1); points are then the smoothed knots
    aim: str = "coord"            # periodic held-out values: "coord" (legacy) or "arc" (value_neighbours; issue #213)

    def __call__(self, t):
        return self.spline(np.asarray(t, dtype=float))

    def t_range(self):
        if self.periodic:
            return self.coords.min(), self.coords.min() + TWO_PI
        return self.coords.min(), self.coords.max()

    def dense(self, n=2000):
        """n points along the curve (for the periodic curve the endpoint is not repeated)."""
        lo, hi = self.t_range()
        t = np.linspace(lo, hi, n, endpoint=not self.periodic)
        return t, self(t)

    def project(self, Z, n=2000):
        """Intrinsic coordinate of the closest curve point to each row of Z (grid search over n points)."""
        t, pts = self.dense(n)
        _, idx = cKDTree(pts).query(np.atleast_2d(Z))
        return t[idx] if np.ndim(Z) > 1 else t[idx[0]]

    def distance(self, Z, n=2000):
        """Distance from each row of Z [.., k] to the curve (brute force against n dense points; a KD-tree is
        slower than BLAS in 64 dimensions)."""
        Z = np.asarray(Z)
        _, pts = self.dense(n)
        return min_distance(Z, pts)

    def step(self, ta, tb):
        """Signed coordinate change from ta to tb; the short way round for a periodic curve."""
        d = np.asarray(tb, dtype=float) - np.asarray(ta, dtype=float)
        return wrap_pi(d) if self.periodic else d

    def coord_of_value(self, v):
        """Intrinsic coordinate of a label value, by linear interpolation between the knots.

        For the value coordinate this is the identity. For an angle coordinate it maps true theta (degrees) to
        the intrinsic angle, which is how held-out target values are reached when t was found without labels.
        """
        order = np.argsort(self.values)
        xv, tc = self.values[order].astype(float), self.coords[order].astype(float)
        if not self.periodic:
            return interp_extrap(v, xv, tc)
        if self.aim == "arc":
            return self._coord_by_arc(v)
        tc = np.unwrap(np.append(tc, tc[0]))            # continuous around the loop, either orientation
        xv = np.append(xv, xv[0] + 360.0)
        x = (np.asarray(v, dtype=float) - xv[0]) % 360.0 + xv[0]
        return np.interp(x, xv, tc) % TWO_PI

    def value_neighbours(self, v):
        """Periodic curves: for each value (degrees) the knot indices a, b of its VALUE neighbours (the kept values
        just below and just above it round the circle) and its fraction f in [0, 1) between them; a knot value
        gives a == b, f == 0. The knots' positions in angle order play no part, so a knot order that is not
        monotone in the value (label-free angle, issue #213) cannot put foreign knots between a and b."""
        v = np.atleast_1d(np.asarray(v, dtype=float))
        kv = self.values.astype(float)
        below = (v[:, None] - kv[None, :]) % 360.0                     # [n, m] distance down to each knot value
        above = (kv[None, :] - v[:, None]) % 360.0
        a = np.argmin(below, axis=1)
        exact = np.isclose(below[np.arange(len(v)), a], 0.0, atol=1e-9) | np.isclose(below[np.arange(len(v)), a],
                                                                                        360.0, atol=1e-9)
        b = np.argmin(np.where(above > 1e-9, above, np.inf), axis=1)
        db, da = below[np.arange(len(v)), a], above[np.arange(len(v)), b]
        f = np.where(exact, 0.0, db / np.where(exact, 1.0, db + da))
        return a, np.where(exact, a, b), f

    def _coord_by_arc(self, v, n=2000):
        """Issue #213 aim: the coordinate at arc-length fraction f (the value's fraction between its value
        neighbours a, b) along the fitted curve from knot a to knot b, walked the short way in angle order.
        Equals the knot's coordinate at a knot value."""
        v_arr = np.asarray(v, dtype=float)
        a, b, f = self.value_neighbours(v_arr.ravel())
        out = np.empty(len(f))
        cache = {}
        for i, (ia, ib, fi) in enumerate(zip(a, b, f)):
            if ia == ib or fi == 0.0:
                out[i] = self.coords[ia]
                continue
            if (ia, ib) not in cache:
                ta = float(self.coords[ia])
                t = ta + np.linspace(0.0, 1.0, n + 1) * float(wrap_pi(self.coords[ib] - ta))
                s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self(t), axis=0), axis=1))])
                cache[(ia, ib)] = (t, s)
            t, s = cache[(ia, ib)]
            out[i] = np.interp(fi * s[-1], s, t)
        out = out % TWO_PI
        return out.reshape(v_arr.shape) if v_arr.ndim else float(out[0])


def interp_extrap(x, xp, fp):
    """np.interp inside [xp[0], xp[-1]], and the end segments extended linearly outside it (np.interp would clamp).
    Used for extrapolation targets: the line continues its last chord, never stops at the last knot."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    return np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)


class SmoothSpline:
    """Count-weighted cubic smoothing spline, one univariate spline per coordinate (Goodfire B.1: "weighted by the
    square root of bin counts"). Weight of knot j in coordinate c = sqrt(count_j) / sd_c, i.e. 1 / (the centroid's
    standard error), and smoothing s = number of knots (the expected chi-square), so no free parameter.
    Callable like a CubicSpline: spline(t, nu=0) -> [..., k]. Periodic: period 2*pi from the first coordinate."""

    def __init__(self, coord, P, count, sd, periodic):
        coord, P = np.asarray(coord, float), np.asarray(P, float)
        w = np.sqrt(np.asarray(count, float))
        self.periodic, self.t0 = periodic, float(coord[0])
        if periodic:
            coord, P, w = np.append(coord, coord[0] + TWO_PI), np.vstack([P, P[:1]]), np.append(w, w[0])
        m = len(coord) - (1 if periodic else 0)
        self.tck = [splrep(coord, P[:, c], w=w / max(sd[c], 1e-12), k=3, s=float(m), per=int(periodic))
                    for c in range(P.shape[1])]

    def __call__(self, t, nu=0):
        t = np.asarray(t, dtype=float)
        tt = (t - self.t0) % TWO_PI + self.t0 if self.periodic else t
        out = np.stack([splev(tt.ravel(), tck, der=nu) for tck in self.tck], axis=-1)
        return out.reshape(t.shape + (len(self.tck),))


class LinearExtension:
    """A scalar-coordinate spline continued past its end knots along the end tangent: f(t) inside [lo, hi], and
    f(end) + (t - end) f'(end) outside (explicit linear continuation; splev ext=3 would clamp to the end value, and
    ext=0 / CubicSpline extrapolate=True extend the end cubic piece). nu=1 is the end tangent outside, nu>=2 zero."""

    def __init__(self, spline, lo, hi):
        self.spline, self.lo, self.hi = spline, float(lo), float(hi)

    def __call__(self, t, nu=0):
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, self.lo, self.hi)
        if nu == 0:
            return self.spline(tc) + (t - tc)[..., None] * self.spline(tc, 1)
        if nu == 1:
            return self.spline(tc, 1)
        return np.where((t != tc)[..., None], 0.0, self.spline(t, nu))


def natural_cubic(C, coord, values=None, smooth=None, extend="cubic"):
    """Natural cubic spline (zero second derivative at the ends) through centroids C [m, k] at coord [m].
    smooth=dict(count [m], sd [k]): count-weighted smoothing spline instead (SmoothSpline).
    extend: "cubic" (default: the end cubic piece is extended) or "linear" (LinearExtension along the end tangent)."""
    coord = np.asarray(coord, dtype=float)
    order = np.argsort(coord)
    values = coord if values is None else np.asarray(values)
    wrap = (lambda sp: LinearExtension(sp, coord.min(), coord.max())) if extend == "linear" else (lambda sp: sp)
    if smooth is not None:
        sp = wrap(SmoothSpline(coord[order], C[order], np.asarray(smooth["count"])[order], smooth["sd"], False))
        return Curve(spline=sp, values=values[order], coords=coord[order], points=sp(coord[order]), periodic=False,
                     coord_source="value", kind="smoothing")
    return Curve(spline=wrap(CubicSpline(coord[order], C[order], bc_type="natural")), values=values[order],
                 coords=coord[order], points=C[order], periodic=False, coord_source="value")


def unsupervised_angle(C, plane="activation"):
    """Intrinsic angle of each centroid without labels: atan2(PC2, PC1) (Goodfire A.3), in [0, 2*pi).

    plane="activation": the first two axes of the activation PCA (C must be PCA coordinates, as from fit_pca).
    plane="centroid":  the first two principal axes of the centred centroids themselves (finds the ring even when
                       PC1/PC2 of the activations are taken by another variable, e.g. speed).
    """
    C = np.asarray(C, dtype=float)
    if plane == "activation":
        P = C[:, :2] - C[:, :2].mean(axis=0)
    else:
        Cc = C - C.mean(axis=0)
        _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
        P = Cc @ Vt[:2].T
    return np.arctan2(P[:, 1], P[:, 0]) % TWO_PI


def goodfire_periodic_angle(C, eigenvalue_tol=0.45, min_variance_fraction=0.1):
    """Label-free angle as Goodfire's code computes it (causalab spline/builders.py detect_periodic_dims +
    remap_periodic_to_angle, called from spline/train.py in PCA mode with 2 * intrinsic_dim = 2 columns): the first two
    activation-PCA columns of the centroids C, eigenvalues = their variance across centroids (unbiased); the pair is
    periodic iff each holds >= min_variance_fraction of the two-column total and |l1 - l2| / max < eigenvalue_tol.
    Angle = atan2(centred PC2 / sqrt(l2), centred PC1 / sqrt(l1)) in [0, 2*pi); no centroid-plane fallback (their
    code then uses the top PCA components with no periodic coordinate). The angle is returned whether or not the test
    passes. Returns dict(angle, passes, eigenvalues, rel_diff)."""
    P = np.asarray(C, dtype=float)[:, :2]
    lam = P.var(axis=0, ddof=1)
    rel = float(abs(lam[0] - lam[1]) / lam.max())
    passes = bool(lam.min() >= min_variance_fraction * lam.sum() and rel < eigenvalue_tol)
    Pc = (P - P.mean(axis=0)) / np.sqrt(lam)
    return {"angle": np.arctan2(Pc[:, 1], Pc[:, 0]) % TWO_PI, "passes": passes, "eigenvalues": lam.tolist(),
            "rel_diff": rel}


def compare_angles(t_intrinsic, theta_deg):
    """How well an unsupervised angle matches the true angle.

    Allows either orientation and any offset (the sign and zero of atan2 are arbitrary). Returns circular
    correlation (Fisher & Lee 1983; signed by orientation), the aligned max and mean absolute deviation in degrees,
    the orientation, and whether the knots keep their order around the loop. Fisher-Lee, not Jammalamadaka: the
    latter centres on each sample's circular mean, which is undefined for angles spread evenly round the circle
    (exactly the 64-value grid), and it then scores near-perfect agreement as ~0.6.
    """
    a = np.asarray(t_intrinsic, dtype=float)
    b = np.radians(np.asarray(theta_deg, dtype=float))

    def circ_mean(x):
        return np.angle(np.exp(1j * x).mean())

    rho = fisher_lee(a, b)

    best = None
    for sign in (1, -1):
        offset = circ_mean(a - sign * b)
        dev = np.abs(np.degrees(wrap_pi(a - sign * b - offset)))
        if best is None or dev.mean() < best["mean_dev_deg"]:
            best = {"orientation": sign, "offset_deg": float(np.degrees(offset)),
                    "max_dev_deg": float(dev.max()), "mean_dev_deg": float(dev.mean())}
    ordered = a[np.argsort(b)]
    steps = np.diff(np.unwrap(ordered))
    order_frac = float(np.mean(steps * best["orientation"] > 0))
    # coarse order: circular means of consecutive blocks of ~45 degrees must run round the loop in order
    nb = max(3, min(8, len(a) // 4))
    blocks = [np.angle(np.exp(1j * g).mean()) for g in np.array_split(ordered, nb)]
    coarse = np.diff(np.unwrap(np.append(blocks, blocks[0])))
    return {"circular_corr": rho, **best, "order_preserved": order_frac == 1.0, "order_frac": order_frac,
            "order_coarse_preserved": bool(np.all(coarse * best["orientation"] > 0))}


def fisher_lee(a, b):
    """Fisher-Lee circular-circular correlation of angles a, b (radians): sum_{i<j} sin(a_i-a_j) sin(b_i-b_j),
    normalised; rotation invariant, +1 same orientation, -1 reflected. Closed form, O(n)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    A, B = np.sum(np.cos(a) * np.cos(b)), np.sum(np.sin(a) * np.sin(b))
    C, D = np.sum(np.cos(a) * np.sin(b)), np.sum(np.sin(a) * np.cos(b))
    E, F = np.sum(np.cos(2 * a)), np.sum(np.sin(2 * a))
    G, H = np.sum(np.cos(2 * b)), np.sum(np.sin(2 * b))
    den = np.sqrt((n ** 2 - E ** 2 - F ** 2) * (n ** 2 - G ** 2 - H ** 2))
    return float(4 * (A * B - C * D) / den) if den > 0 else float("nan")


def choose_angle_source(C, values_deg, min_corr=0.9, max_dev_deg=None):
    """Pick the intrinsic angle for a ring: atan2(PC2, PC1) in the activation-PCA plane, else in the centroid plane,
    else the labels (flagged). A plane is accepted when |circular corr| (Fisher-Lee) >= min_corr in EITHER
    orientation (a ring has no handedness; the sign is recorded) and the knots run round the loop in order up to
    reversal at the 45-degree block scale (order_coarse_preserved). The aligned max deviation is always reported in
    the checks; it is a criterion only when max_dev_deg is given (a noisy ring can have one knot 30+ degrees off and
    still be the loop). Swaps of neighbouring knots 5.6 degrees apart are noise with 10-20 clips per value; their
    fraction is reported (order_frac), not required. The spline is fit through the knots in the order of their
    unsupervised angle, i.e. their geometric order round the loop; the value -> coordinate map is interpolated in
    value order.
    Returns dict(angle="unsupervised"|"labels", plane, orientation, max_dev_deg, circular_corr, checks)."""
    checks = {pl: compare_angles(unsupervised_angle(C, pl), values_deg) for pl in ("activation", "centroid")}
    for pl in ("activation", "centroid"):
        c = checks[pl]
        if (abs(c["circular_corr"]) >= min_corr and c["order_coarse_preserved"]
                and (max_dev_deg is None or c["max_dev_deg"] < max_dev_deg)):
            return {"angle": "unsupervised", "plane": pl, "orientation": c["orientation"],
                    "max_dev_deg": c["max_dev_deg"], "circular_corr": c["circular_corr"], "checks": checks}
    return {"angle": "labels", "plane": "activation", "orientation": None, "max_dev_deg": None, "circular_corr": None,
            "checks": checks}


def periodic_cubic(C, values_deg, angle=None, plane="activation", smooth=None):
    """Closed (periodic) cubic spline through centroids C [m, k] of a circular variable.

    angle=None: intrinsic angle derived without labels (unsupervised_angle). Otherwise pass angles in radians;
    labels_angle(values_deg) is the flagged fallback when the ring is not in the chosen plane.
    """
    values_deg = np.asarray(values_deg, dtype=float)
    source = "unsupervised_angle" if angle is None else "labels_angle"
    t = unsupervised_angle(C, plane) if angle is None else np.asarray(angle, dtype=float) % TWO_PI
    order = np.argsort(t)
    t, P, v = t[order], np.asarray(C)[order], values_deg[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("two centroids share an intrinsic angle; the loop cannot be parameterised")
    if smooth is not None:
        sp = SmoothSpline(t, P, np.asarray(smooth["count"])[order], smooth["sd"], True)
        return Curve(spline=sp, values=v, coords=t, points=sp(t), periodic=True, coord_source=source,
                     kind="smoothing")
    spline = CubicSpline(np.append(t, t[0] + TWO_PI), np.vstack([P, P[:1]]), bc_type="periodic")
    return Curve(spline=spline, values=v, coords=t, points=P, periodic=True, coord_source=source)


def labels_angle(values_deg):
    """Fallback intrinsic angle: the true direction, in radians. Mark results that use it."""
    return np.radians(np.asarray(values_deg, dtype=float)) % TWO_PI


def arc_length_table(curve, n=2000):
    """Cumulative arc length s(t) on a dense grid: (t [n+1], s [n+1]). Arc length is in PCA units."""
    lo, hi = curve.t_range()
    t = np.linspace(lo, hi, n + 1)
    seg = np.linalg.norm(np.diff(curve(t), axis=0), axis=1)
    return t, np.concatenate([[0.0], np.cumsum(seg)])


def arc_length(curve, ta, tb, n=200):
    """Length along the curve from ta to tb (the short way round if periodic), summing n straight segments."""
    t = np.asarray(ta, dtype=float) + np.linspace(0, 1, n + 1) * curve.step(ta, tb)
    return float(np.linalg.norm(np.diff(curve(t), axis=0), axis=1).sum())


def arc_length_coords(curve, t, n=2000):
    """Re-express intrinsic coordinates t as arc length from the curve's start (for equal-speed paths)."""
    tt, s = arc_length_table(curve, n)
    lo = curve.t_range()[0]
    t = np.asarray(t, dtype=float)
    if curve.periodic:
        t = (t - lo) % TWO_PI + lo
    return np.interp(t, tt, s)


def heldout_mask(values, every=4, periodic=False):
    """Boolean mask of held-out label values: every 4th value, starting at index 2 so that, for a sequential
    variable, both endpoints stay knots (held-out values are always interior). 16 of 64 values."""
    m = len(values)
    mask = (np.arange(m) % every) == (every // 2)
    if not periodic:
        mask[0] = mask[-1] = False
    return mask


HOLDOUT_DESIGNS = ("scattered", "contiguous", "extrapolation")


def heldout_design(values, design, periodic, seed=0, block=8, every=4, extend="cubic"):
    """Held-out label values for one of three designs (project spec section 6 item 1). values must be sorted (np.unique).

    scattered:     every 4th value (heldout_mask; interior for scalars). A local interpolation check only.
    contiguous:    one block of `block` consecutive values. Direction: a 45 degree arc (8 of 64) starting at an index
                   drawn from `seed`, wrapping round 0/360. Scalars: an interior block (at least one kept knot on
                   each side), start drawn from `seed`. The spline-vs-line claim rests on this design.
    extrapolation: the top `block` values (scalars only; direction has no extrapolation).
    extend: the scalar spline's extension beyond the end knots ("cubic" or "linear", as fit_curve); only the
            extrapolation note depends on it.
    Returns (mask [m] bool, info dict naming the design and the held-out values).
    """
    values = np.asarray(values)
    m = len(values)
    info = {"design": design}
    if design == "scattered":
        mask = heldout_mask(values, every, periodic)
    elif design == "contiguous":
        rng = np.random.default_rng(seed)
        start = int(rng.integers(m)) if periodic else int(rng.integers(1, m - block))
        idx = (start + np.arange(block)) % m
        mask = np.zeros(m, bool)
        mask[idx] = True
        info.update({"seed": seed, "block_start_index": start, "block_first_value": float(values[idx[0]]),
                     "block_last_value": float(values[idx[-1]])})
        if periodic:
            info["arc_width_deg"] = float(block * 360.0 / m)
    elif design == "extrapolation":
        if periodic:
            raise ValueError("direction is periodic: there is no extrapolation design")
        mask = np.zeros(m, bool)
        mask[np.argsort(values)[-block:]] = True
        how = "extends its end cubic piece" if extend == "cubic" else "continues along its end tangent"
        info["note"] = f"targets lie beyond the last kept knot: the spline {how}, the line extends its last chord"
    else:
        raise ValueError(f"unknown held-out design {design!r}; choose from {HOLDOUT_DESIGNS}")
    info["held_out_values"] = values[mask].astype(float).tolist()
    return mask, info


def sagitta(curve, values, chord_curve=None):
    """Per target value: the distance (PCA units) between the curve point and the chord point between the kept
    neighbouring knots, i.e. how far apart the spline's and the line's targets are. Also names the neighbours.
    The chord runs between curve.points (the smoothed knots for a smoothing spline) unless chord_curve is given
    (e.g. raw_knot_curve: the chord between the raw kept centroids)."""
    values = np.asarray(values, dtype=float)
    chord = curve if chord_curve is None else chord_curve
    dist = np.linalg.norm(curve(curve.coord_of_value(values)) - piecewise_linear_point(chord, values), axis=-1)
    kv = np.sort(curve.values.astype(float))
    out = []
    for v, s in zip(values, np.atleast_1d(dist)):
        if curve.periodic:
            below = kv[kv < v].max() if (kv < v).any() else kv.max()
            above = kv[kv > v].min() if (kv > v).any() else kv.min()
        else:
            below = kv[kv < v].max() if (kv < v).any() else None
            above = kv[kv > v].min() if (kv > v).any() else None
            if above is None:                       # extrapolation: the last chord is extended
                below, above = kv[-2], kv[-1]
        out.append({"value": float(v), "sagitta": float(s), "chord_from": float(below), "chord_to": float(above)})
    return out


def fit_curve(cent, periodic, keep=None, angle="unsupervised", plane="activation", spline="interp", extend="cubic"):
    """Fit the right spline to a centroids() dict, optionally on a subset of values (keep mask).

    angle: "unsupervised" or "labels" (periodic only).
    spline: "interp" (Goodfire A.3: passes through every centroid) or "smooth" (Goodfire B.1: count-weighted
            smoothing spline; with ~10-20 clips per value an interpolating spline chases centroid noise and
            overshoots across a held-out block).
    extend: scalars only, beyond the end knots: "cubic" (end piece extended) or "linear" (end tangent).
    """
    keep = np.ones(len(cent["values"]), bool) if keep is None else np.asarray(keep)
    C, v = cent["C"][keep], cent["values"][keep]
    smooth = {"count": cent["count"][keep], "sd": cent["sd_coord"]} if spline == "smooth" else None
    if not periodic:
        return natural_cubic(C, v, smooth=smooth, extend=extend)
    return periodic_cubic(C, v, angle=None if angle == "unsupervised" else labels_angle(v), plane=plane,
                          smooth=smooth)


def raw_knot_curve(curve, cent):
    """The polyline through the RAW kept centroids cent["C"], in the curve's knot order, coordinates and periodicity
    (spline=None: only piecewise_linear_point uses it). This is the paper's linear baseline chord between raw
    centroids c_a, c_b. For an interpolating spline curve.points are already these centroids; for a smoothing spline
    curve.points are the smoothed knots, so the chord through them is not the raw-centroid chord."""
    idx = {float(v): i for i, v in enumerate(cent["values"])}
    C = np.asarray(cent["C"])[[idx[float(v)] for v in curve.values]]
    return Curve(spline=None, values=curve.values, coords=curve.coords, points=C, periodic=curve.periodic,
                 coord_source=curve.coord_source, kind="raw_knots", aim=curve.aim)


# ---------------------------------------------------------------- 4. steering paths -------------------------------

def linear_path(x, ca, cb, K):
    """Straight-line steer in full space: x + s (cb - ca), s = 0..1 in K steps. Returns [K, D] (or [n, K, D])."""
    s = np.linspace(0.0, 1.0, K)[:, None]                                  # [K, 1]
    diff = np.asarray(cb, dtype=float) - np.asarray(ca, dtype=float)       # [D] or [n, D]
    return np.asarray(x, dtype=float)[..., None, :] + s * diff[..., None, :]


def manifold_path(x, pca, curve, ta, tb, K, mode="shift"):
    """Walk along the spline from coordinate ta to tb in K equal steps and lift back to full space.

    Only the top-k PCA components change; the orthogonal complement of x is kept exactly.
      mode="shift"   (default): z_k = z + curve(t_k) - curve(ta). An additive edit, like linear_path; the clip keeps
                     its own offset from the curve. A zero-length path returns x unchanged.
      mode="replace" (Goodfire A.6): z_k = curve(t_k). The waypoint sits on the curve exactly.
    x [D] with scalar ta -> [K, D];  x [n, D] with ta [n] -> [n, K, D].
    """
    x = np.asarray(x, dtype=float)
    single = x.ndim == 1
    X = np.atleast_2d(x)
    out = pca.lift(manifold_coords(pca.project(X), curve, ta, tb, K, mode)) + pca.complement(X)[:, None, :]
    return out[0] if single else out


def manifold_coords(Z, curve, ta, tb, K, mode="shift"):
    """The PCA-coordinate waypoints [n, K, k] of manifold_path for clips at PCA coordinates Z [n, k]."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    ta = np.broadcast_to(np.asarray(ta, dtype=float), (len(Z),))
    tb = np.broadcast_to(np.asarray(tb, dtype=float), (len(Z),))
    s = np.linspace(0.0, 1.0, K)
    t = ta[:, None] + s[None, :] * curve.step(ta, tb)[:, None]          # [n, K]
    on_curve = curve(t)                                                  # [n, K, k]
    if mode == "shift":
        return Z[:, None, :] + on_curve - on_curve[:, :1, :]
    if mode == "replace":
        return on_curve
    if mode == "transport":
        return on_curve + transport_offset(Z - on_curve[:, 0], curve, t)
    raise ValueError(mode)


def curve_frame(curve, t):
    """Unit tangent T and unit in-plane normal N of a closed curve at t (N = the direction from the loop's centre,
    made orthogonal to T). Returns (T, N), each [..., k]."""
    P = curve(t)
    T = curve.spline(np.asarray(t, dtype=float), 1)
    T = T / np.linalg.norm(T, axis=-1, keepdims=True)
    N = P - curve.points.mean(axis=0)
    N = N - np.sum(N * T, axis=-1, keepdims=True) * T
    return T, N / np.linalg.norm(N, axis=-1, keepdims=True)


def transport_offset(o, curve, t):
    """Carry each clip's offset o [n, k] from the curve along the loop, rotating its tangent and normal parts with
    the loop's frame (o_k = o - (o.T0)T0 - (o.N0)N0 + (o.T0)T_k + (o.N0)N_k) instead of shifting it rigidly.
    A rigid shift puts an outward offset on the inner side after a 180 degree steer; transport keeps it outward.
    Exact rotation for a planar loop. Periodic curves only. t [n, K]; returns [n, K, k]."""
    if not curve.periodic:
        raise ValueError("transport is defined for a closed curve (direction); use mode='shift' for scalars")
    T, N = curve_frame(curve, t)                                         # [n, K, k]
    a, b = np.sum(o * T[:, 0], -1), np.sum(o * N[:, 0], -1)              # [n]
    rest = o - a[:, None] * T[:, 0] - b[:, None] * N[:, 0]
    return rest[:, None] + a[:, None, None] * T + b[:, None, None] * N


def endpoint_matched_random(Zk, coef, iters=40):
    """Endpoint-matched random control for paths Zk [n, K, k]: the path's own chord (traversed with its arc-length
    spacing f) plus a random smooth bend g(f) = sum_j coef_j sin(j pi f), which is zero at both ends, scaled per
    clip so the path length equals the original path's (bisection on the amplitude). Same endpoints, dose, waypoint
    count and length as the spline; only the shape of the bend is random. coef [n_freq, k], one draw."""
    Zk = np.asarray(Zk, dtype=float)
    chord, _ = chord_coords(Zk)
    seg = lambda P: np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
    target = seg(Zk)
    s = np.concatenate([np.zeros((len(Zk), 1)), np.cumsum(np.linalg.norm(np.diff(Zk, axis=1), axis=-1), 1)], 1)
    f = s / np.where(s[:, -1:] > 0, s[:, -1:], 1.0)
    j = np.arange(1, len(coef) + 1)
    bend = np.einsum("nkj,jd->nkd", np.sin(np.pi * f[..., None] * j), np.asarray(coef, float))   # [n, K, k]
    lo, hi = np.zeros(len(Zk)), np.full(len(Zk), 1.0)
    for _ in range(60):                                   # grow hi until it overshoots the target length
        short = seg(chord + hi[:, None, None] * bend) < target
        if not short.any():
            break
        hi = np.where(short, hi * 2, hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        longer = seg(chord + mid[:, None, None] * bend) > target
        hi, lo = np.where(longer, mid, hi), np.where(longer, lo, mid)
    return chord + ((lo + hi) / 2)[:, None, None] * bend


def linear_coords(Z, pa, pb, K):
    """Straight-line steer inside the PCA subspace: z + s (pb - pa), s = 0..1 in K steps. Z [n, k]; pa, pb [n, k]
    or [k] (the polyline points at the source and target values). Returns [n, K, k]. Lifted with the clip's own
    complement this is the matched-support linear arm: same subspace and same off-subspace residual as the spline."""
    s = np.linspace(0.0, 1.0, K)[None, :, None]
    diff = np.broadcast_to(np.asarray(pb, float) - np.asarray(pa, float), np.shape(Z))
    return np.asarray(Z, float)[:, None, :] + s * diff[:, None, :]


def chord_coords(Zk):
    """Curvature controls for a path of waypoints Zk [n, K, k] (jepa_steering's registered action-geometry arms).

    projected: the straight chord between the path's own endpoints, traversed with the path's arc-length spacing
               (waypoint k sits at fraction s_k / s_K of the chord, s = cumulative length along the path).
    reflected: 2 * projected - path, the bend flipped to the other side of the chord.
    Both have the path's endpoints, dose and waypoint count. Returns (projected, reflected), each [n, K, k].
    """
    Zk = np.asarray(Zk, dtype=float)
    seg = np.linalg.norm(np.diff(Zk, axis=1), axis=-1)                    # [n, K-1]
    s = np.concatenate([np.zeros((len(Zk), 1)), np.cumsum(seg, axis=1)], axis=1)
    total = s[:, -1:]
    uniform = np.broadcast_to(np.linspace(0.0, 1.0, Zk.shape[1]), s.shape)
    f = np.where(total > 0, s / np.where(total > 0, total, 1.0), uniform)  # zero-length path: uniform fractions
    a, b = Zk[:, :1], Zk[:, -1:]
    projected = a + f[..., None] * (b - a)
    return projected, 2 * projected - Zk


def steer_to_value(x, pca, curve, target_value, K=2, source_value=None, mode="shift"):
    """Steer x to a target label value along the curve (works for values that were never knots).

    Start coordinate: x projected onto the curve, or the coordinate of source_value if given. End coordinate:
    curve.coord_of_value(target_value). Returns the K waypoints; the last one is the steered activation.
    """
    ta = curve.project(pca.project(x)) if source_value is None else curve.coord_of_value(source_value)
    return manifold_path(x, pca, curve, ta, curve.coord_of_value(target_value), K, mode=mode)


def piecewise_linear_point(curve, v):
    """The point at value v on the polyline through the knots (the 'line' that the spline is compared with).
    Periodic with curve.aim == "arc" (issue #213): the chord between the value's two VALUE-neighbouring knots at its
    value fraction, (1 - f) P_a + f P_b (A.9's chord between c_a and c_b). Identical to the legacy coordinate-order
    polyline whenever the knot order is monotone in the value; when it is not (label-free angle) the legacy polyline
    runs through foreign knots. aim == "coord" keeps the legacy polyline byte for byte."""
    if curve.periodic and curve.aim == "arc":
        v_arr = np.asarray(v, dtype=float)
        a, b, f = curve.value_neighbours(v_arr.ravel())
        P = np.asarray(curve.points, dtype=float)
        out = (1.0 - f)[:, None] * P[a] + f[:, None] * P[b]
        return out.reshape(v_arr.shape + (P.shape[1],))
    t = curve.coord_of_value(v)
    if curve.periodic:
        P = np.vstack([curve.points, curve.points[:1]])
        tc = np.append(curve.coords, curve.coords[0] + TWO_PI)
        t = (np.asarray(t) - tc[0]) % TWO_PI + tc[0]
    else:
        P, tc = curve.points, curve.coords
        return np.stack([interp_extrap(t, tc, P[:, j]) for j in range(P.shape[1])], axis=-1)
    return np.stack([np.interp(t, tc, P[:, j]) for j in range(P.shape[1])], axis=-1)


# ---------------------------------------------------------------- 5. metrics --------------------------------------

def min_distance(A, B, chunk=4096):
    """Distance from each row of A [.., D] to its nearest row of B [m, D] (brute force, chunked)."""
    A = np.asarray(A, dtype=np.float64)
    flat = A.reshape(-1, A.shape[-1])
    B = np.asarray(B, dtype=np.float64)
    b2 = (B ** 2).sum(axis=1)
    out = np.empty(len(flat))
    for i in range(0, len(flat), chunk):
        a = flat[i:i + chunk]
        d2 = (a ** 2).sum(axis=1)[:, None] + b2[None, :] - 2 * a @ B.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out.reshape(A.shape[:-1])


def knn_distance(A, B, k=5, chunk=2048):
    """Mean distance from each row of A [.., D] to its k nearest rows of B [m, D] (a kNN density proxy)."""
    A = np.asarray(A, dtype=np.float64)
    flat = A.reshape(-1, A.shape[-1])
    B = np.asarray(B, dtype=np.float64)
    k = min(k, len(B))
    b2 = (B ** 2).sum(axis=1)
    out = np.empty(len(flat))
    for i in range(0, len(flat), chunk):
        a = flat[i:i + chunk]
        d2 = np.maximum((a ** 2).sum(axis=1)[:, None] + b2[None, :] - 2 * a @ B.T, 0.0)
        out[i:i + chunk] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, :k]).mean(axis=1)
    return out.reshape(A.shape[:-1])


def off_manifold_energy(W, pca, curve, X_real=None, knn=1):
    """Per-waypoint distances for W [.., D]: to `curve` (PCA space) and, if X_real is given, the mean distance to
    the knn nearest real activations (full space; a density proxy). Means are what Goodfire-style summaries use.

    Goodfire A.7 scores naturalness as the closest-point distance to a manifold fit to unintervened data. Our
    choice (not A.7's): pass a reference curve and real clips that NEITHER steering arm was built from, because
    scoring against the manifold arm's own spline makes the manifold arm win by construction."""
    out = {"to_curve": curve.distance(pca.project(W))}
    if X_real is not None:
        out["to_nearest_real"] = min_distance(W, X_real) if knn == 1 else knn_distance(W, X_real, knn)
    return out


def path_lengths(W, readout_fn):
    """Length of a path W [K, D] in activation space and in readout space (readout_fn: [K, D] -> [K, q])."""
    R = np.asarray(readout_fn(W)).reshape(len(W), -1)
    return {"activation": float(np.linalg.norm(np.diff(W, axis=0), axis=1).sum()),
            "readout": float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum())}


def geodesic_matrix(curve, n=200):
    """Pairwise arc length between the curve's knots, along the curve."""
    m = len(curve.coords)
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            G[i, j] = G[j, i] = arc_length(curve, curve.coords[i], curve.coords[j], n)
    return G


def isometry(curve, readout_points, metric="euclidean"):
    """Pearson correlation between knot-pair geodesic distances on the activation curve and distances between the
    readouts of the same knots (readout_points [m, q], in the curve's knot order, or an [m, m] distance matrix with
    metric="precomputed"). Goodfire A.5 compares geodesic with geodesic, so for a (sin, cos) probe readout use
    metric="angular" (the angle between readouts, wrapped to [0, 180]), not the Euclidean chord. A.5 without
    interior points: the knots here are already dense (48-64 per curve, like their ages task, K=0)."""
    G = geodesic_matrix(curve)
    R = np.asarray(readout_points, dtype=float)
    if metric == "precomputed":
        Dr = R
    elif metric == "angular":
        ang = np.arctan2(R[:, 0], R[:, 1])
        Dr = np.abs(wrap_pi(ang[:, None] - ang[None]))
    else:
        R = R.reshape(len(G), -1)
        Dr = np.linalg.norm(R[:, None] - R[None], axis=-1)
    iu = np.triu_indices(len(G), 1)
    return float(np.corrcoef(G[iu], Dr[iu])[0, 1])


# ---------------------------------------------------------------- behaviour manifold M_y (Goodfire section 5) -----

class BehaviourManifold:
    """Goodfire section 5 Eq. 9 / App. B.1 Eq. 10: for a model with no output distribution, a behaviour is built from
    the activations themselves. Default ("spline", literal): p(z) = softmax_b(-||z - mu_b||_2 / tau), tau = 0.5,
    UNsquared L2, mu_b = B = 128 points evenly spaced in the intrinsic coordinate along a fitted activation spline
    (`curve`, in the PCA-k coordinates of `pca`, where the spline lives). b_i = mean p over real clips at value i;
    M_y = a spline through sqrt(b_i) in the tangent plane of the unit sphere at the normalised mean (log map; decoded
    by the exp map, App. A.4), parameterised by the value (radians for direction). Energy E_BC = sum over waypoints
    of the Bhattacharyya distance -log sum_i sqrt(p_i q_i) to the nearest point q of M_y (App. A.7).

    Option "centroid_sq": the 64 raw per-value centroids in full space and squared distance divided by the mean
    within-value squared distance (tau in units of within-value spread).

    CAVEAT (circularity): at the steered layer F is itself a distance-to-centroid function of the edited
    activation, so behaviour_energy and isometry_behaviour there partly restate the activation geometry the arms
    were built on. They become informative only at later layers (after propagating the edit) or through the
    predictor. Build it from real clips (and a curve) that neither steering arm was built from.
    """
    source = ("Goodfire Manifold Steering, section 5 Eq. 9 and App. B.1 Eq. 10 (unsquared L2, tau = 0.5, B = 128 "
              "spline points); Hellinger/tangent spline App. A.4")
    caveat = ("at the steered layer the behaviour function F is itself a distance-to-centroid function of the edited "
              "activation, so behaviour_energy and isometry_behaviour are partly circular there; they become "
              "informative only at later layers (propagation) or through the predictor")

    def __init__(self, X_real, values_real, periodic, tau=0.5, mode="spline", curve=None, pca=None, n_bins=128,
                 n_grid=2000):
        X, v = np.asarray(X_real, dtype=float), np.asarray(values_real, dtype=float)
        self.values = np.unique(v)
        self.periodic, self.tau, self.mode, self.pca = periodic, tau, mode, pca
        if mode == "spline":
            lo, hi = curve.t_range()
            bin_t = np.linspace(lo, hi, n_bins, endpoint=not curve.periodic)
            self.mu = curve(bin_t)                                                          # [B, k]
            self.s2 = 1.0
            # value of each bin: the reference curve's coordinate is the value (radians for direction)
            bv = np.degrees(bin_t) % 360.0 if periodic else bin_t
            dist = (np.abs((bv[:, None] - self.values[None] + 180.0) % 360.0 - 180.0) if periodic
                    else np.abs(bv[:, None] - self.values[None]))
            self.bin_to_value = np.eye(len(self.values))[dist.argmin(1)]                    # [B, values]
        elif mode == "centroid_sq":
            self.mu = np.array([X[v == u].mean(0) for u in self.values])                   # [B, D]
            idx = np.searchsorted(self.values, v)
            self.s2 = float(((X - self.mu[idx]) ** 2).sum(1).mean())
            self.bin_to_value = np.eye(len(self.values))
        else:
            raise ValueError(mode)
        self.scale = "none (literal)" if mode == "spline" else "within-value mean squared distance"
        P = self.F(X)
        b = np.array([P[v == u].mean(0) for u in self.values])                             # [values, B]
        h = np.sqrt(b)
        base = h.mean(0)
        self.base = base / np.linalg.norm(base)
        tangent = self._log(h)
        coord = np.radians(self.values) if periodic else self.values
        if periodic:
            self.spline = CubicSpline(np.append(coord, coord[0] + TWO_PI), np.vstack([tangent, tangent[:1]]),
                                      bc_type="periodic")
            self.grid_t = np.linspace(coord[0], coord[0] + TWO_PI, n_grid, endpoint=False)
        else:
            self.spline = CubicSpline(coord, tangent, bc_type="natural")
            self.grid_t = np.linspace(coord[0], coord[-1], n_grid)
        self.grid = self.decode(self.grid_t)                                               # [G, B] unit vectors

    def F(self, Z):
        """Eq. 9 / 10: distribution over the B bins for activations Z [.., D]."""
        Z = np.asarray(Z, dtype=float)
        flat = Z.reshape(-1, Z.shape[-1])
        if self.mode == "spline":
            flat = self.pca.project(flat)
        d2 = np.maximum((flat ** 2).sum(1)[:, None] + (self.mu ** 2).sum(1)[None] - 2 * flat @ self.mu.T, 0.0)
        logits = -(np.sqrt(d2) if self.mode == "spline" else d2) / (self.tau * self.s2)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return (p / p.sum(1, keepdims=True)).reshape(Z.shape[:-1] + (len(self.mu),))

    def _log(self, h):
        c = np.clip(h @ self.base, -1.0, 1.0)
        th = np.arccos(c)[:, None]
        u = h - c[:, None] * self.base
        n = np.linalg.norm(u, axis=1, keepdims=True)
        return np.where(n > 1e-12, th * u / np.where(n > 1e-12, n, 1.0), 0.0)

    def decode(self, t):
        """Exp map of the tangent spline at coordinates t: points of M_y (unit-norm sqrt-probabilities)."""
        tv = self.spline(np.asarray(t, dtype=float))
        n = np.linalg.norm(tv, axis=-1, keepdims=True)
        return np.cos(n) * self.base + np.sin(n) * tv / np.where(n > 1e-12, n, 1.0)

    def value_distribution(self, Z, P=None):
        """Eq. 9 probability aggregated over label VALUES (each bin assigned to its nearest value): [.., n_values]."""
        return (self.F(Z) if P is None else P) @ self.bin_to_value

    def entropy(self, Z, P=None):
        """Entropy (nats) of the Eq. 9 distribution over bins, per row. Real clips far from the curve give broad
        distributions; points on the curve give sharp ones."""
        P = self.F(Z) if P is None else P
        return -(P * np.log(np.clip(P, 1e-300, 1.0))).sum(-1)

    def bhattacharyya(self, Z, P=None):
        """Per-row Bhattacharyya distance of p(z) to the nearest grid point of M_y: -log max_q sum sqrt(p q).
        P: precomputed F(Z) (saves recomputing it)."""
        sp = np.sqrt(self.F(Z) if P is None else P)
        flat = sp.reshape(-1, sp.shape[-1])
        bc = np.empty(len(flat))
        for i in range(0, len(flat), 4096):
            bc[i:i + 4096] = (flat[i:i + 4096] @ self.grid.T).max(1)
        return -np.log(np.clip(bc, 1e-300, 1.0)).reshape(sp.shape[:-1])

    def geodesic(self, v_a, v_b, n=150):
        """Hellinger length along M_y between two values (App. A.5: 150 sub-intervals, d_H = ||.||_2 / sqrt 2)."""
        ta = np.radians(v_a) if self.periodic else float(v_a)
        tb = np.radians(v_b) if self.periodic else float(v_b)
        step = wrap_pi(tb - ta) if self.periodic else tb - ta
        pts = self.decode(ta + np.linspace(0, 1, n + 1) * step)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() / np.sqrt(2))

    def geodesic_matrix(self, values):
        m = len(values)
        G = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                G[i, j] = G[j, i] = self.geodesic(values[i], values[j])
        return G


# ---------------------------------------------------------------- readouts ("what plays M_y") ---------------------

def value_error(pred_value, target_value, periodic):
    """|pred - target|, wrap-aware in degrees for a periodic variable."""
    d = np.asarray(pred_value, dtype=float) - np.asarray(target_value, dtype=float)
    return np.abs((d + 180.0) % 360.0 - 180.0) if periodic else np.abs(d)


class Readout(Protocol):
    """Scores steered activations against a target value. 1 = moved all the way to the target, 0 = no closer
    than before steering, negative = moved away."""
    name: str
    plays_M_y: str

    def score(self, X_steered, X_orig, target_value) -> np.ndarray: ...


class ProbeReadout:
    """Ridge probe (standardise + RidgeCV) fit on clips disjoint from the knots and from the steered clips.
    Predicts the value (scalars) or (sin, cos) decoded with atan2 (direction)."""
    name = "probe"
    plays_M_y = "the evaluation probe's output space (predicted value); a probe, not behaviour"

    def __init__(self, X, values, periodic, alphas=np.logspace(-2, 5, 15)):
        self.periodic = periodic
        y = np.asarray(values, dtype=float)
        if periodic:
            y = np.stack([np.sin(np.radians(y)), np.cos(np.radians(y))], axis=1)
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(X, y)

    def raw(self, X):
        """Probe output: [n] values, or [n, 2] (sin, cos)."""
        return self.model.predict(np.asarray(X).reshape(-1, np.shape(X)[-1]))

    def predict(self, X):
        out = self.raw(X)
        if self.periodic:
            return np.degrees(np.arctan2(out[:, 0], out[:, 1])) % 360.0
        return out.ravel()

    def score(self, X_steered, X_orig, target_value):
        e_after = value_error(self.predict(X_steered), target_value, self.periodic)
        e_before = value_error(self.predict(X_orig), target_value, self.periodic)
        return 1.0 - e_after / np.maximum(e_before, 1e-12)


class NearestRealReadout:
    """Agreement with V-JEPA's own activations on real clips at the target value (lessons file, section C):
    R = 1 - D(steered, real target) / D(unsteered, real target), D = distance to the mean of the real clips that
    have the target value, at the layer these activations come from. Use real clips disjoint from the knots."""
    name = "nearest_real"
    plays_M_y = "centroids of real-clip activations at each value, at the layer read"

    def __init__(self, X_real, values_real):
        self.X_real = np.asarray(X_real, dtype=float)
        self.values_real = np.asarray(values_real)

    def target_mean(self, target_value):
        rows = np.isclose(self.values_real, target_value)
        if not rows.any():
            raise ValueError(f"no real clip with value {target_value}")
        return self.X_real[rows].mean(axis=0)

    def score(self, X_steered, X_orig, target_value):
        return nearest_real_agreement(X_steered, X_orig, self.target_mean(target_value))


def nearest_real_agreement(X_steered, X_orig, real_target_mean):
    """1 - D(steered, real)/D(unsteered, real), per row."""
    d_after = np.linalg.norm(np.asarray(X_steered) - real_target_mean, axis=-1)
    d_before = np.linalg.norm(np.asarray(X_orig) - real_target_mean, axis=-1)
    return 1.0 - d_after / np.maximum(d_before, 1e-12)


def path_value_stats(Pv, src, tgt, values, periodic, arc_sign=None):
    """Where the Eq. 9 value-mass sits along a steering path (Goodfire Fig. 4 analogue: "smooth and ordered").

    Pv [n, K, V]: probability over the V label values at each waypoint. Values are re-expressed as offsets from each
    clip's source along a given arc (direction: arc_sign [n] = +1 for increasing label angle, -1 for decreasing;
    run_part2 passes the spline's traversed arc for every arm; default the short way round by label, + at exactly
    180), so the target sits at +span;
    scalars in value-index steps towards the target.
    Returns dict:
      intermediate [n, K]  mass on values strictly between source and target on the traversed arc
      ordering [n]         Spearman correlation between waypoint index and the argmax value's offset (0 if constant)
      argmax_on_arc [n]    fraction of waypoints whose argmax lies on the traversed arc, endpoints included
      aligned [n, K, n_off], offsets [n_off]   mass by offset in (-180, 180] (direction) / -(V-1)..V-1 (scalars)
      arc_sign [n]"""
    from scipy.stats import rankdata
    values = np.asarray(values, dtype=float)
    src, n = np.asarray(src, dtype=float), len(Pv)
    tgt = np.broadcast_to(np.asarray(tgt, dtype=float), (n,))
    V = len(values)
    if periodic:
        step = 360.0 / V
        if arc_sign is None:
            arc_sign = np.where(wrap_pi(np.radians(tgt - src)) < 0, -1.0, 1.0)
        sign = np.broadcast_to(np.asarray(arc_sign, dtype=float), (n,))
        fwd = ((values[None] - src[:, None]) * sign[:, None]) % 360.0                      # [n, V] in [0, 360)
        span = ((tgt - src) * sign) % 360.0
        # offsets for the argmax / aligned view: behind the source (the rest of the circle past the midpoint of the
        # untraversed arc) counts as negative
        off = np.where(fwd > span[:, None] + (360.0 - span[:, None]) / 2, fwd - 360.0, fwd)
        al = np.where(fwd > 180.0 + 1e-9, fwd - 360.0, fwd)
        offsets = np.arange(-V // 2 + 1, V // 2 + 1) * step
        idx = np.clip(np.rint(al / step).astype(int) + V // 2 - 1, 0, V - 1)
    else:
        si = np.abs(values[None] - src[:, None]).argmin(1)
        ti = np.abs(values[None] - tgt[:, None]).argmin(1)
        sign = np.where(ti < si, -1.0, 1.0)
        fwd = off = (np.arange(V)[None] - si[:, None]) * sign[:, None]
        span = np.abs(ti - si).astype(float)
        offsets = np.arange(-(V - 1), V).astype(float)
        idx = (off + V - 1).astype(int)
    between = (fwd > 1e-9) & (fwd < span[:, None] - 1e-9)                                  # [n, V]
    on_arc = (fwd > -1e-9) & (fwd < span[:, None] + 1e-9)
    intermediate = (Pv * between[:, None, :]).sum(-1)
    am = Pv.argmax(-1)                                                                     # [n, K]
    am_off = np.take_along_axis(off, am, axis=1)
    am_on = np.take_along_axis(on_arc, am, axis=1)
    k = rankdata(np.arange(Pv.shape[1]))
    ordering = np.zeros(n)
    for i in range(n):
        r = rankdata(am_off[i])
        if r.std() > 0:
            ordering[i] = np.corrcoef(k, r)[0, 1]
    aligned = np.zeros((n, Pv.shape[1], len(offsets)))
    for i in range(n):
        np.add.at(aligned[i], (slice(None), idx[i]), Pv[i])
    return {"intermediate": intermediate, "ordering": ordering, "argmax_on_arc": am_on.mean(1), "aligned": aligned,
            "offsets": offsets, "arc_sign": np.asarray(sign, dtype=float)}

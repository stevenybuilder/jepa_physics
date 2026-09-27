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
    layers after propagating the edit, this is the strongest offline readout (nonobvious_components.md §5.3, §8).
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.interpolate import CubicSpline
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

    Returns dict(values [m], C [m, k], count [m], spread [m]), values sorted.
    """
    Z, labels = np.asarray(Z), np.asarray(labels)
    values = np.unique(labels)
    C, count, spread = [], [], []
    for v in values:
        rows = Z[labels == v]
        c = rows.mean(axis=0)
        C.append(c)
        count.append(len(rows))
        spread.append(np.linalg.norm(rows - c, axis=1).mean())
    return {"values": values, "C": np.array(C), "count": np.array(count), "spread": np.array(spread)}


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
    coord_source: str       # "value", "unsupervised_angle" or "labels_angle"

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
        """Distance from each row of Z [.., k] to the curve."""
        Z = np.asarray(Z)
        _, pts = self.dense(n)
        d, _ = cKDTree(pts).query(Z.reshape(-1, Z.shape[-1]))
        return d.reshape(Z.shape[:-1])

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
        tc = np.unwrap(np.append(tc, tc[0]))            # continuous around the loop, either orientation
        xv = np.append(xv, xv[0] + 360.0)
        x = (np.asarray(v, dtype=float) - xv[0]) % 360.0 + xv[0]
        return np.interp(x, xv, tc) % TWO_PI


def interp_extrap(x, xp, fp):
    """np.interp inside [xp[0], xp[-1]], and the end segments extended linearly outside it (np.interp would clamp).
    Used for extrapolation targets: the line continues its last chord, never stops at the last knot."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    return np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)


def natural_cubic(C, coord, values=None):
    """Natural cubic spline (zero second derivative at the ends) through centroids C [m, k] at coord [m]."""
    coord = np.asarray(coord, dtype=float)
    order = np.argsort(coord)
    values = coord if values is None else np.asarray(values)
    return Curve(spline=CubicSpline(coord[order], C[order], bc_type="natural"), values=values[order],
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
    steps = np.diff(np.unwrap(a[np.argsort(b)]))
    order_preserved = bool(np.all(steps * best["orientation"] > 0))
    return {"circular_corr": rho, **best, "order_preserved": order_preserved}


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


def choose_angle_source(C, values_deg, max_dev_deg=30.0):
    """Pick the intrinsic angle for a ring: atan2(PC2, PC1) in the activation-PCA plane if it tracks the true angle
    (aligned max deviation < max_dev_deg and knot order kept), else in the centroid plane, else the labels (flag).
    Returns dict(angle="unsupervised"|"labels", plane, checks={plane: compare_angles(...)})."""
    checks = {pl: compare_angles(unsupervised_angle(C, pl), values_deg) for pl in ("activation", "centroid")}
    for pl in ("activation", "centroid"):
        if checks[pl]["max_dev_deg"] < max_dev_deg and checks[pl]["order_preserved"]:
            return {"angle": "unsupervised", "plane": pl, "checks": checks}
    return {"angle": "labels", "plane": "activation", "checks": checks}


def periodic_cubic(C, values_deg, angle=None, plane="activation"):
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


def heldout_design(values, design, periodic, seed=0, block=8, every=4):
    """Held-out label values for one of three designs (spec.md section 6 item 1). values must be sorted (np.unique).

    scattered:     every 4th value (heldout_mask; interior for scalars). A local interpolation check only.
    contiguous:    one block of `block` consecutive values. Direction: a 45 degree arc (8 of 64) starting at an index
                   drawn from `seed`, wrapping round 0/360. Scalars: an interior block (at least one kept knot on
                   each side), start drawn from `seed`. The spline-vs-line claim rests on this design.
    extrapolation: the top `block` values (scalars only; direction has no extrapolation).
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
        info["note"] = ("targets lie beyond the last kept knot: the spline extends its end cubic piece, the line "
                        "extends its last chord")
    else:
        raise ValueError(f"unknown held-out design {design!r}; choose from {HOLDOUT_DESIGNS}")
    info["held_out_values"] = values[mask].astype(float).tolist()
    return mask, info


def sagitta(curve, values):
    """Per target value: the distance (PCA units) between the curve point and the chord point between the kept
    neighbouring knots, i.e. how far apart the spline's and the line's targets are. Also names the neighbours."""
    values = np.asarray(values, dtype=float)
    dist = np.linalg.norm(curve(curve.coord_of_value(values)) - piecewise_linear_point(curve, values), axis=-1)
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


def fit_curve(cent, periodic, keep=None, angle="unsupervised", plane="activation"):
    """Fit the right spline to a centroids() dict, optionally on a subset of values (keep mask).

    angle: "unsupervised" or "labels" (periodic only).
    """
    keep = np.ones(len(cent["values"]), bool) if keep is None else np.asarray(keep)
    C, v = cent["C"][keep], cent["values"][keep]
    if not periodic:
        return natural_cubic(C, v)
    return periodic_cubic(C, v, angle=None if angle == "unsupervised" else labels_angle(v), plane=plane)


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
    raise ValueError(mode)


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
    """The point at value v on the polyline through the knots (the 'line' that the spline is compared with)."""
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


def off_manifold_energy(W, pca, curve, X_real=None):
    """Per-waypoint distances for W [.., D]: to the fitted curve (PCA space) and, if X_real is given, to the
    nearest real training activation (full space; a density proxy). Means are what Goodfire-style summaries use."""
    out = {"to_curve": curve.distance(pca.project(W))}
    if X_real is not None:
        out["to_nearest_real"] = min_distance(W, X_real)
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


def isometry(curve, readout_points):
    """Pearson correlation between knot-pair geodesic distances on the activation curve and Euclidean distances
    between the readouts of the same knots (readout_points [m, q], in the curve's knot order). A.5 without the
    interior points: the knots here are already dense (48-64 per curve)."""
    G = geodesic_matrix(curve)
    R = np.asarray(readout_points, dtype=float).reshape(len(G), -1)
    Dr = np.linalg.norm(R[:, None] - R[None], axis=-1)
    iu = np.triu_indices(len(G), 1)
    return float(np.corrcoef(G[iu], Dr[iu])[0, 1])


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

"""Pure helpers for the Part 2 new-variable experiments: within-clip time (clock vs odometer) and the joint
(direction, speed) velocity sheet. Used by scripts/run_time_manifold.py and scripts/run_velocity_sheet.py.

Time convention: timepool step t (0..7) pools tubelet t = frames 2t, 2t+1 at 24 fps; its time is the midpoint of the
two frames, tau_t = (2t + 0.5) / 24 s (frame k is the scene at k / 24 s, render_twin.py).
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FPS = 24.0
N_STEPS = 8
ALPHAS = np.logspace(-2, 5, 15)


def step_times(n_steps=N_STEPS, fps=FPS):
    """tau_t in seconds, the midpoint of tubelet t's two frames."""
    return (2 * np.arange(n_steps) + 0.5) / fps


def step_distance(speed, accel, n_steps=N_STEPS):
    """Distance travelled from the start at each step, s = v tau + a tau^2 / 2. speed, accel [n] -> [n, T]."""
    tau = step_times(n_steps)
    return np.asarray(speed, float)[:, None] * tau + 0.5 * np.asarray(accel, float)[:, None] * tau ** 2


def step_position(start_xy, theta_deg, speed, accel, n_steps=N_STEPS):
    """Disk position (m) at each step [n, T, 2]."""
    th = np.radians(np.asarray(theta_deg, float))
    u = np.stack([np.cos(th), np.sin(th)], -1)
    return np.asarray(start_xy, float)[:, None, :] + step_distance(speed, accel, n_steps)[..., None] * u[:, None, :]


def remove_clip_mean(F):
    """[n, T, D] -> the per-step residual after removing each clip's own mean over T."""
    F = np.asarray(F, dtype=np.float64)
    return F - F.mean(axis=1, keepdims=True)


def ridge(X, Y):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X, Y)


def grouped_cv_predict(X, Y, fold, fit=ridge):
    """Out-of-fold predictions, one fold held out at a time (fold: a group id per row, all rows of a clip share it)."""
    Y = np.asarray(Y, float)
    out = np.zeros_like(Y)
    for f in np.unique(fold):
        te = fold == f
        out[te] = fit(X[~te], Y[~te]).predict(X[te]).reshape(out[te].shape)
    return out


def r2(y, yhat):
    """R^2 pooled over all output columns."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def nearest_coord(Z, grid_t, grid_pts):
    """Intrinsic coordinate of the nearest of grid_pts [m, k] (at grid_t [m]) to each row of Z [.., k]."""
    Z = np.asarray(Z, float)
    flat = Z.reshape(-1, Z.shape[-1])
    d2 = (flat ** 2).sum(1)[:, None] + (grid_pts ** 2).sum(1)[None] - 2 * flat @ grid_pts.T
    return np.asarray(grid_t)[d2.argmin(1)].reshape(Z.shape[:-1])


def per_clip_slope(coord, t=None):
    """OLS slope of coord [n, T] against t (default 0..T-1), per clip."""
    coord = np.asarray(coord, float)
    t = np.arange(coord.shape[1], dtype=float) if t is None else np.asarray(t, float)
    tc = t - t.mean()
    return (coord - coord.mean(1, keepdims=True)) @ tc / (tc ** 2).sum()


def odometer_coord(dist, ref_speed=None, ref_accel=None, n_steps=N_STEPS, fps=FPS):
    """Odometer prediction for a time coordinate indexed by a reference clip: the (fractional) step at which a
    reference clip (constant ref_speed, or constant ref_accel from rest) has covered the distance `dist` [n, T].
    A clock predicts coordinate = t; an odometer predicts this."""
    dist = np.asarray(dist, float)
    if ref_speed is not None:
        tau = dist / ref_speed
    else:
        tau = np.sqrt(2 * np.maximum(dist, 0) / ref_accel)
    return (tau * fps - 0.5) / 2.0


def clock_odometer_fit(coord, t_grid, odo):
    """Least squares coord = a + b t + c odo over all rows (flattened); plus R^2 of the clock-only and odometer-only
    single-predictor fits. Clock: b ~ 1, c ~ 0; odometer: b ~ 0, c ~ 1."""
    y, t, o = (np.asarray(v, float).ravel() for v in (coord, t_grid, odo))
    A = np.stack([np.ones_like(t), t, o], 1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    def fit_r2(cols):
        B = A[:, cols]
        res = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
        return float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"a": float(coef[0]), "b_clock": float(coef[1]), "c_odometer": float(coef[2]),
            "r2_clock_only": fit_r2([0, 1]), "r2_odometer_only": fit_r2([0, 2]), "r2_both": fit_r2([0, 1, 2])}


def cluster_bootstrap(values, groups=None, stat=np.mean, n_boot=1000, seed=0):
    """Statistic of per-row values with a 95% percentile CI, resampling groups (clips) with replacement."""
    values = np.asarray(values, float)
    groups = np.arange(len(values)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = [stat(values[np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(stat(values)), "ci95": [float(lo), float(hi)], "n": int(len(values)), "n_clips": int(len(uniq))}


def step_variance_explained(R, t_idx, C):
    """1 - sum ||R - C[t]||^2 / sum ||R||^2 for rows R [m, k] at steps t_idx [m] and per-step points C [T, k]
    (R already has each clip's mean removed, so the null model is 0)."""
    R = np.asarray(R, float)
    return float(1 - ((R - C[np.asarray(t_idx)]) ** 2).sum() / (R ** 2).sum())


def ring_radius_corrected(C_theta, noise):
    """Ring radius of per-direction centroids C_theta [m, k] (RMS distance from their mean), with the centroid noise
    (RMS per-centroid standard error, same units) subtracted in quadrature. Returns (raw, corrected)."""
    C = np.asarray(C_theta, float)
    raw2 = float(((C - C.mean(0)) ** 2).sum(1).mean())
    return float(np.sqrt(raw2)), float(np.sqrt(max(raw2 - noise ** 2, 0.0)))


def cone_cylinder_fit(C, band, dirbin, n_dir):
    """Centroids C [m, k] indexed by (band, dirbin). Fits, on these centroids,
      cylinder: C = m_band + u_dir                 (same ring at every speed band)
      cone    : C = m_band + r_band * u_dir         (the ring's size scales per band; a true cone has r ~ speed)
    Returns dict(m [B, k], u [n_dir, k], r [B], bands) so predict_sheet can rebuild any (band, dir) cell."""
    C, band, dirbin = np.asarray(C, float), np.asarray(band), np.asarray(dirbin)
    bands = np.unique(band)
    m = np.array([C[band == b].mean(0) for b in bands])
    bi = np.searchsorted(bands, band)
    D = C - m[bi]
    u = np.array([D[dirbin == j].mean(0) if (dirbin == j).any() else np.zeros(C.shape[1]) for j in range(n_dir)])
    r = np.ones(len(bands))
    for _ in range(20):                                     # alternating least squares for r_band, u_dir
        r = np.array([(D[bi == i] * u[dirbin[bi == i]]).sum() / max((u[dirbin[bi == i]] ** 2).sum(), 1e-12)
                      for i in range(len(bands))])
        r = r / np.sqrt((r ** 2).mean())
        u = np.array([(r[bi[dirbin == j], None] * D[dirbin == j]).sum(0) / max((r[bi[dirbin == j]] ** 2).sum(), 1e-12)
                      if (dirbin == j).any() else np.zeros(C.shape[1]) for j in range(n_dir)])
    u_cyl = np.array([D[dirbin == j].mean(0) if (dirbin == j).any() else np.zeros(C.shape[1]) for j in range(n_dir)])
    return {"bands": bands, "m": m, "u_cyl": u_cyl, "u_cone": u, "r_cone": r}


def predict_sheet(fit, band, dirbin, kind):
    bi = np.searchsorted(fit["bands"], np.asarray(band))
    dirbin = np.asarray(dirbin)
    if kind == "cylinder":
        return fit["m"][bi] + fit["u_cyl"][dirbin]
    return fit["m"][bi] + fit["r_cone"][bi, None] * fit["u_cone"][dirbin]


def angular_bins(theta_deg, n_bins):
    """Bin index of each angle, bins of 360/n_bins degrees starting at 0."""
    return (np.floor(np.asarray(theta_deg, float) % 360.0 / (360.0 / n_bins))).astype(int) % n_bins

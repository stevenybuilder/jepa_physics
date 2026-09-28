"""Pure helpers for the velocity-sheet v2 audit (scripts/run_velocity_sheet_v2.py).

Hold-out designs over a (direction bin, speed bin) grid, a cone frame of a fitted sheet P(theta, v)
(axis c(v), radius rho(v), unit ring U(theta, v)), the radius x shape factorial ablation of the sheet edit,
per-row norm matching, a minimum-norm probe edit, an inverse-distance cell-centroid estimate (the non-oracle
"difference of means" chord target) and a vectorised clip bootstrap.
"""
import numpy as np

BLOCK2 = [(1, (2, 3)), (5, (3, 4)), (9, (4, 5)), (13, (3, 4))]          # v1 blocks (2 dir bins x 2 speed bins)
BLOCK3 = [(1, (2, 3, 4)), (5, (3, 4, 5)), (9, (1, 2, 3)), (13, (4, 5, 6))]  # 3 x 3


def holdout(dbin, sbin, design, seed, n_dir=16):
    """Held-out mask and target cells.
    block2 / block3: the cells (held dir bins) x (held speed bins) are unseen; their direction values are seen at
      other speeds and their speeds at other directions.
    cross: the held dir bins at EVERY speed and the held speed bins at EVERY direction are unseen, so a target cell
      (intersection) has neither its direction nor its speed seen anywhere in the fitting data.
    Returns (held [N] bool, targets [(d, s)], held_dirs, held_spds)."""
    if design == "block3":
        b0, sp = BLOCK3[seed]
        hd = [(b0 + i) % n_dir for i in range(3)]
    elif design in ("block2", "cross"):
        b0, sp = BLOCK2[seed]
        hd = [(b0 + i) % n_dir for i in range(2)]
    else:
        raise ValueError(design)
    hs = list(sp)
    ind, ins = np.isin(dbin, hd), np.isin(sbin, hs)
    held = (ind | ins) if design == "cross" else (ind & ins)
    return held, [(d, s) for d in hd for s in hs], hd, hs


def circ_mean_deg(theta_deg):
    return float(np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(theta_deg, float))).mean())) % 360)


class ConeFrame:
    """Cone frame of a sheet P(theta_deg [n], v [n]) -> [n, k]: axis c(v) = mean over theta of P(., v),
    radius rho(v) = RMS over theta of |P(., v) - c(v)|, unit ring U(theta, v) = (P(theta, v) - c(v)) / rho(v).
    Means over theta use n_theta equally spaced angles."""

    def __init__(self, P, n_theta=64):
        self.P = P
        self.grid = np.arange(n_theta) * 360.0 / n_theta

    def _slice(self, v):
        v = np.asarray(v, float)
        n, m = len(v), len(self.grid)
        S = self.P(np.tile(self.grid, n), np.repeat(v, m)).reshape(n, m, -1)
        c = S.mean(1)
        rho = np.sqrt(((S - c[:, None]) ** 2).sum(-1).mean(1))
        return c, rho

    def center(self, v):
        return self._slice(v)[0]

    def radius(self, v):
        return self._slice(v)[1]

    def unit(self, theta, v):
        c, rho = self._slice(v)
        return (self.P(np.asarray(theta, float), np.asarray(v, float)) - c) / rho[:, None]


def factorial_delta(frame, th_s, v_s, th_t, v_t, radius=True, shape=True):
    """Sheet edit with the radius and/or the ring shape frozen at the SOURCE speed:
        delta = c(v_t) - c(v_s) + rho(v_r) U(th_t, v_u) - rho(v_s) U(th_s, v_s),
    v_r = v_t if radius else v_s, v_u = v_t if shape else v_s. radius=shape=True is exactly P(t) - P(s);
    radius=shape=False is the global (direction-independent) speed axis plus the source-speed ring."""
    th_s, v_s, th_t, v_t = (np.asarray(a, float) for a in (th_s, v_s, th_t, v_t))
    c_s, r_s = frame._slice(v_s)
    c_t, r_t = frame._slice(v_t)
    u_s = frame.unit(th_s, v_s)
    u_t = frame.unit(th_t, v_t if shape else v_s)
    return c_t - c_s + (r_t if radius else r_s)[:, None] * u_t - r_s[:, None] * u_s


def rescale_rows(delta, ref_norm, eps=1e-9):
    """Scale each row of delta [n, ..., k] (last axis = feature; the norm is taken on the row's LAST waypoint if
    delta is [n, K, k]) to have norm ref_norm [n]."""
    delta = np.asarray(delta, float)
    end = delta[:, -1] if delta.ndim == 3 else delta
    s = np.asarray(ref_norm, float) / np.maximum(np.linalg.norm(end, axis=-1), eps)
    return delta * s.reshape((-1,) + (1,) * (delta.ndim - 1))


def min_norm_probe_edit(W, b, z0, y_target):
    """Smallest delta with W (z0 + delta) + b = y_target for a linear readout W [q, k], b [q] (q <= k):
    delta = W^T (W W^T)^-1 (y_target - W z0 - b). z0 [n, k], y_target [n, q]."""
    W = np.asarray(W, float)
    r = np.asarray(y_target, float) - (np.asarray(z0, float) @ W.T + b)
    return r @ np.linalg.solve(W @ W.T, W)


def idw_cell_estimate(cell_d, cell_s, C, target_d, target_s, n_dir=16, k=4, s_scale=1.0):
    """Non-oracle estimate of an unseen cell centroid: inverse-distance weighted mean of the k nearest seen cell
    centroids in (dir bin, speed bin) grid units (direction distance circular; speed distance x s_scale)."""
    dd = np.abs(np.asarray(cell_d) - target_d) % n_dir
    dd = np.minimum(dd, n_dir - dd)
    dist = np.hypot(dd, s_scale * (np.asarray(cell_s) - target_s))
    nn = np.argsort(dist, kind="stable")[:k]
    w = 1.0 / np.maximum(dist[nn], 1e-9)
    return (w[:, None] * np.asarray(C)[nn]).sum(0) / w.sum()


def clip_boot(values, clips, n_boot=1000, seed=0):
    """Mean of per-row values with a 95% percentile CI resampling clips (all rows of a clip together).
    Same estimator as timeman.cluster_bootstrap (row-weighted mean of the drawn clips), vectorised."""
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, clips = values[ok], np.asarray(clips)[ok]
    uniq, inv = np.unique(clips, return_inverse=True)
    S = np.bincount(inv, values, len(uniq))
    N = np.bincount(inv, None, len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), (n_boot, len(uniq)))
    boots = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(len(values)),
            "n_clips": int(len(uniq))}

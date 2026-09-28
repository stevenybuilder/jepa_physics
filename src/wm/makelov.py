"""Pure helpers for the Makelov-style ranking of INLP directions (Makelov et al., "Towards principled evaluations of
sparse autoencoders for interpretability and control") and the 16-arc conceptor aggregate.

Directions: INLP's Q is orthonormal in step 1's train-standardised coordinates (z = (x - mu) / sd). A unit step q in z
moves the raw activation by sd * q, so the raw-space edit direction is normalise(sd * q) (the convention of step-3
steering, which edits in z)."""
import numpy as np
from scipy.stats import spearmanr


def raw_directions(Q, sd, n=None):
    """Q [D, K] (orthonormal in standardised coordinates), sd [D] -> unit raw-space directions [n, D]."""
    Q = np.asarray(Q, float)[:, : (Q.shape[1] if n is None else n)]
    R = (np.asarray(sd, float)[:, None] * Q).T
    return R / np.linalg.norm(R, axis=1, keepdims=True)


def scaled_edits(dirs, norms):
    """dirs [n, D] unit, norms [C] -> edits [C, n, D] with ||edit[c, k]|| = norms[c]."""
    return np.asarray(norms, float)[:, None, None] * np.asarray(dirs, float)[None]


def wrap_deg(a):
    """Signed angle difference wrapped to (-180, 180]."""
    return -((180.0 - np.asarray(a, float)) % 360.0 - 180.0)


def order_spearman(moves):
    """Spearman between INLP order (0, 1, ..., n-1) and the per-direction move. Negative = earlier INLP directions move
    the readout more."""
    moves = np.asarray(moves, float)
    return float(spearmanr(np.arange(len(moves)), moves).correlation)


def carrier_bootstrap_spearman(M, n_boot=2000, seed=0):
    """M [C, n]: per-carrier move of each direction. Resample carriers with replacement, average, Spearman with the
    INLP order. Returns dict(rho, ci95, n_boot)."""
    M = np.asarray(M, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M.shape[0], (n_boot, M.shape[0]))
    boots = np.array([order_spearman(M[i].mean(0)) for i in idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"rho": order_spearman(M.mean(0)), "ci95": [float(lo), float(hi)], "n_boot": int(n_boot)}


def permutation_p(moves, n_perm=10000, seed=0):
    """Two-sided permutation p-value of the order Spearman (directions shuffled)."""
    moves = np.asarray(moves, float)
    r0 = abs(order_spearman(moves))
    rng = np.random.default_rng(seed)
    null = np.array([abs(order_spearman(rng.permutation(moves))) for _ in range(n_perm)])
    return float((1 + (null >= r0 - 1e-12).sum()) / (1 + n_perm))


def arc_clip_bootstrap(values_by_arc, ids_by_arc, n_boot=2000, seed=0):
    """Mean over arcs of each arc's row mean, with a 95% CI from resampling clips WITHIN each arc (a clip keeps all
    its rows/targets together; arcs are fixed strata). values_by_arc: list of [n_rows] arrays; ids_by_arc: matching
    clip ids. Returns dict(mean, ci95, n_arcs, n_boot)."""
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    means = []
    for v, g in zip(values_by_arc, ids_by_arc):
        v = np.asarray(v, float)
        uniq, inv = np.unique(np.asarray(g), return_inverse=True)
        sums, cnt = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
        idx = rng.integers(0, len(uniq), (n_boot, len(uniq)))
        boots += sums[idx].sum(1) / cnt[idx].sum(1)
        means.append(v.mean())
    boots /= len(values_by_arc)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(np.mean(means)), "ci95": [float(lo), float(hi)], "n_arcs": len(values_by_arc),
            "n_boot": int(n_boot)}

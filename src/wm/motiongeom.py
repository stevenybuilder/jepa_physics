"""Pure helpers for the motion-geometry audit (scripts/run_motion_geometry.py): variable subspaces from a joint
linear encoding model, principal angles, union-vs-sum effective dimension, projector strength, residualised edits,
coordinate-system feature maps for planar motion, and a Holm correction.

Conventions: activations are rows; a subspace is an orthonormal basis Q [D, k] (columns); angles in degrees.
"""
import numpy as np


def orth(B, tol=1e-10):
    """Orthonormal basis [D, r] of the column span of B [D, k] (r = numerical rank)."""
    B = np.atleast_2d(np.asarray(B, float))
    if B.shape[0] < B.shape[1]:
        B = B.T
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    r = int((s > tol * max(s.max(), 1e-300)).sum())
    return U[:, :r]


def principal_angles_deg(Q1, Q2):
    """Principal angles (ascending) between span(Q1) and span(Q2), both orthonormal [D, k]."""
    s = np.linalg.svd(np.asarray(Q1).T @ np.asarray(Q2), compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def overlap(Q1, Q2):
    """Mean squared cosine of the principal angles, ||Q1^T Q2||_F^2 / min(k1, k2): 0 orthogonal, 1 nested."""
    M = np.asarray(Q1).T @ np.asarray(Q2)
    return float((M ** 2).sum() / min(M.shape))


def union_dimension(bases):
    """Effective dimension of the union of subspaces. Stack the orthonormal bases [D, sum k]; the squared singular
    values s^2 of the stack sum to sum k. Orthogonal subspaces give all s^2 = 1 (participation ratio = sum k);
    shared directions pile up (one s^2 > 1, one ~0). Returns dict(sum_dims, participation_ratio, rank_s2_gt_0.5,
    s2 [sorted desc])."""
    S = np.hstack([np.asarray(Q) for Q in bases])
    s2 = np.linalg.svd(S, compute_uv=False) ** 2
    return {"sum_dims": int(S.shape[1]), "participation_ratio": float(s2.sum() ** 2 / (s2 ** 2).sum()),
            "rank_s2_gt_0.5": int((s2 > 0.5).sum()), "s2": [float(x) for x in np.sort(s2)[::-1]]}


def fourier_features(theta_deg, K=1):
    """[cos t, sin t, cos 2t, sin 2t, ..., cos Kt, sin Kt] for t = theta in radians. [n, 2K]."""
    t = np.radians(np.asarray(theta_deg, float))
    return np.column_stack([f(k * t) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def fit_linear_encoding(X, F):
    """OLS X ~ a + F B. X [n, D], F [n, p]. Returns (a [D], B [p, D])."""
    X, F = np.asarray(X, float), np.asarray(F, float)
    Fm, Xm = F.mean(0), X.mean(0)
    B, *_ = np.linalg.lstsq(F - Fm, X - Xm, rcond=None)
    return Xm - Fm @ B, B


def projector_strength(X, Q, mean=None):
    """||P_S (x - mean)|| per row."""
    X = np.asarray(X, float)
    if mean is not None:
        X = X - mean
    return np.linalg.norm(X @ np.asarray(Q), axis=1)


def residualise(delta, Q, renorm=True, eps=1e-12):
    """(I - Q Q^T) delta per row, optionally rescaled back to each row's original norm."""
    delta = np.asarray(delta, float)
    Q = np.asarray(Q, float)
    r = delta - (delta @ Q) @ Q.T
    if renorm:
        r = r * (np.linalg.norm(delta, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), eps))
    return r


def captured_variance(X, Q, mean=None):
    """Fraction of the total variance of X inside span(Q)."""
    X = np.asarray(X, float)
    X = X - (X.mean(0) if mean is None else mean)
    return float(((X @ Q) ** 2).sum() / (X ** 2).sum())


# ---- coordinate systems for planar motion (theta in degrees, v in m/s) ----------------------------------------

COORDS = {
    "polar":            lambda th, v: np.column_stack([np.asarray(th, float), v]),
    "log_polar":        lambda th, v: np.column_stack([np.asarray(th, float), np.log(v)]),
    "cartesian":        lambda th, v: np.asarray(v, float)[:, None] * fourier_features(th, 1),
    "fourier_polar":    lambda th, v: np.column_stack([fourier_features(th, 1), v]),
    "fourier_logpolar": lambda th, v: np.column_stack([fourier_features(th, 1), np.log(v)]),
    "fourier2_polar":   lambda th, v: np.column_stack([fourier_features(th, 2), v]),
}


def coords_to_state(name, C):
    """Invert a coordinate map's (predicted) coordinates C [n, p] to (theta_deg, v)."""
    C = np.asarray(C, float)
    if name == "polar":
        return C[:, 0] % 360.0, C[:, 1]
    if name == "log_polar":
        return C[:, 0] % 360.0, np.exp(C[:, 1])
    if name == "cartesian":
        return np.degrees(np.arctan2(C[:, 1], C[:, 0])) % 360.0, np.hypot(C[:, 0], C[:, 1])
    th = np.degrees(np.arctan2(C[:, 1], C[:, 0])) % 360.0
    if name in ("fourier_polar", "fourier2_polar"):
        return th, C[:, -1]
    if name == "fourier_logpolar":
        return th, np.exp(C[:, -1])
    raise ValueError(name)


def circ_err_deg(a, b):
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def clip_boot(values, clips, n_boot=1000, seed=0):
    """Mean with a 95% percentile CI resampling clips (all rows of a clip together)."""
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
            "n_clips": int(len(uniq)), "p_two_sided_vs_0": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


class SVDRidge:
    """Ridge on train-standardised features with the penalty chosen by GCV over a grid, from ONE SVD of the design,
    so many target sets share the decomposition. fit(Y) -> self; predict(X) -> [n, q] (or [n] for 1-D Y).
    The GCV score is summed over target columns (one alpha per call)."""

    def __init__(self, X, alphas=np.logspace(-1, 5, 13)):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        self.U, self.s, self.Vt = np.linalg.svd(Z, full_matrices=False)
        self.alphas = np.asarray(alphas, float)
        self.n = len(X)

    def fit(self, Y):
        Y = np.asarray(Y, float)
        self.vec = Y.ndim == 1
        Y2 = Y[:, None] if self.vec else Y
        self.ym = Y2.mean(0)
        UtY = self.U.T @ (Y2 - self.ym)
        s2 = self.s ** 2
        best = None
        for a in self.alphas:
            f = s2 / (s2 + a)
            res = (Y2 - self.ym) - self.U @ (f[:, None] * UtY)
            gcv = (res ** 2).sum() / (self.n - f.sum()) ** 2
            if best is None or gcv < best[0]:
                best = (gcv, a)
        self.alpha = best[1]
        self.coef = self.Vt.T @ ((self.s / (s2 + self.alpha))[:, None] * UtY)
        return self

    def predict(self, X):
        P = ((np.asarray(X, float) - self.mu) / self.sd) @ self.coef + self.ym
        return P[:, 0] if self.vec else P


def oblique_residualise(delta, B_enc, W_dec, renorm=True, eps=1e-12):
    """Remove from delta the part a linear decoder W_dec [q, D] would read, moving only along the nuisance's
    ENCODING directions B_enc [D, q]: delta - B (W B)^-1 W delta. Afterwards W_dec @ delta' = 0 exactly, and the
    change lies in span(B) (the oblique projector along span(B) onto ker(W)). Optionally rescale to the old norm."""
    delta = np.asarray(delta, float)
    B, W = np.asarray(B_enc, float), np.asarray(W_dec, float)
    r = delta - np.linalg.solve(W @ B, W @ delta.T).T @ B.T
    if renorm:
        r = r * (np.linalg.norm(delta, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), eps))
    return r

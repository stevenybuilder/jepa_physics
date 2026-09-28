"""Conceptor steering (COAST, arXiv 2605.17144) in the Part 2 PCA-k edit space.

Formulas (page numbers of the arXiv PDF):
  conceptor      C = R (R + alpha^-2 I)^-1,  R = X~^T X~ / N, X~ mean-centred      (Eq. 1, p. 4; A.9.1, p. 27)
  NOT            not C = I - C                                                        (p. 4)
  AND            A and B = (A^-1 + B^-1 - I)^-1, pseudoinverses in the code          (Eq. 3, p. 4; Eq. 7-8, p. 27-28)
  contrastive    C_steer = C_success and not C_failure                               (Eq. 4, p. 4)
  gate           h' = h M^T,  M = (1 - beta) I + beta C_steer, beta in [0, 1]         (Eq. 5, p. 5; Eq. 9, p. 28)
  overlap        tr(Cs Cf) / sqrt(tr(Cs^2) tr(Cf^2)); alpha kept if mean overlap in [0.85, 0.95], else closest
                 (Eq. 11 and Stage 2, p. 30)
  quota          q(C) = tr(C) / d                                                     (Eq. 10, p. 30)
Here "success" = the target condition and "failure" = the source condition.
"""
import dataclasses

import numpy as np

OVERLAP_BAND = (0.85, 0.95)                 # COAST A.10.2 Stage 2
ALPHA_GRID = (0.1, 0.5, 1.0, 2.0, 10.0)     # COAST Table 14 (pi0.5 rows)


def conceptor(X, alpha, w=None):
    """C = R (R + alpha^-2 I)^-1 with R the (optionally weighted) covariance of the mean-centred rows of X [N, d].
    Computed in float64 by eigendecomposition of R (same matrix as the literal product; symmetric by construction)."""
    X = np.asarray(X, dtype=np.float64)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    Xc = X - w @ X
    R = (Xc * w[:, None]).T @ Xc
    lam, U = np.linalg.eigh(R)
    lam = np.clip(lam, 0.0, None)
    return (U * (lam / (lam + alpha ** -2))) @ U.T


def NOT(C):
    return np.eye(len(C)) - C


def AND_pinv(A, B):
    """COAST's implementation (A.9.1 pseudocode, p. 27-28): pinv(pinv(A) + pinv(B) - I)."""
    I = np.eye(len(A))
    return np.linalg.pinv(np.linalg.pinv(A) + np.linalg.pinv(B) - I)


def AND_jaeger(A, B, tol=1e-10):
    """Jaeger (2014) AND for possibly singular conceptors (quoted from memory; COAST gives only the invertible form):
    A and B = U (U^T (A^+ + B^+ - I) U)^-1 U^T, U an orthonormal basis of range(A) intersect range(B)."""
    d = len(A)

    def rng(M):
        s, V = np.linalg.eigh((M + M.T) / 2)
        return V[:, s > tol]
    Ua, Ub = rng(A), rng(B)
    # intersection of ranges = null space of [I - Pa] stacked with [I - Pb]
    P = np.vstack([np.eye(d) - Ua @ Ua.T, np.eye(d) - Ub @ Ub.T])
    _, s, Vt = np.linalg.svd(P)
    U = Vt[np.concatenate([s, np.zeros(d - len(s))]) < 1e-8].T
    if U.shape[1] == 0:
        return np.zeros((d, d))
    core = U.T @ (np.linalg.pinv(A) + np.linalg.pinv(B) - np.eye(d)) @ U
    return U @ np.linalg.inv(core) @ U.T


def contrastive(C_success, C_failure, mode="pinv"):
    """C_steer = C_success AND NOT C_failure (Eq. 4)."""
    return (AND_pinv if mode == "pinv" else AND_jaeger)(C_success, NOT(C_failure))


def overlap(A, B):
    """COAST Eq. 11 similarity tr(AB) / sqrt(tr(A^2) tr(B^2))."""
    den = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
    return float(np.trace(A @ B) / den) if den > 0 else 0.0


def select_alpha(mean_overlap, band=OVERLAP_BAND):
    """COAST Stage 2: alphas whose mean overlap is in the band; if none, the closest one. Several in band: we take the
    one nearest the band centre (COAST keeps them all and picks by rollouts, which we do not have)."""
    a = np.array(sorted(mean_overlap), dtype=float)
    o = np.array([mean_overlap[x] for x in a])
    inb = (o >= band[0]) & (o <= band[1])
    dist = np.abs(o - np.mean(band)) if inb.any() else np.minimum(np.abs(o - band[0]), np.abs(o - band[1]))
    dist = np.where(inb, dist, np.inf) if inb.any() else dist
    return float(a[int(np.argmin(dist))]), bool(inb.any())


def chord_weights(curve, v):
    """Weights over the curve's knots with which the chord (mf.piecewise_linear_point) builds its point at value v:
    the point is exactly weights @ curve.points. Obtained by running the chord on one-hot knot points, so it matches
    whichever chord rule the installed wm.manifold uses. Returns [len(v), m]."""
    from wm import manifold as mf
    m = len(curve.values)
    eye = dataclasses.replace(curve, points=np.eye(m))
    return np.atleast_2d(mf.piecewise_linear_point(eye, np.atleast_1d(np.asarray(v, dtype=float))))


def condition_states(Z_knot, y_knot, knot_values, weights, tol=1e-9):
    """Knot-clip rows and per-row weights for a condition given as weights over knot values (each value's weight
    split equally over its clips). Returns (Z [n, k], w [n])."""
    rows, ws = [], []
    for v, wv in zip(knot_values, weights):
        if wv <= tol:
            continue
        idx = np.flatnonzero(np.isclose(y_knot, v))
        rows.append(Z_knot[idx])
        ws.append(np.full(len(idx), wv / len(idx)))
    return np.concatenate(rows), np.concatenate(ws)


def gate_path(U, C, betas):
    """COAST gate along a beta schedule: U [n, k] (or [k]), C [n, k, k] (or [k, k]); u' = u + beta (C - I) u, which
    is u M^T for symmetric C. Returns [n, K, k]."""
    U = np.atleast_2d(U)
    C = np.broadcast_to(C, (len(U),) + np.shape(C)[-2:])
    step = np.einsum("nij,nj->ni", C, U) - U
    return U[:, None] + np.asarray(betas)[None, :, None] * step[:, None]


def aimed_path(Z, C, mu, s):
    """Target-aimed variant: z' = z + s C (mu - z), s over the schedule. Z [n, k], C [n, k, k], mu [n, k] or [k]."""
    Z = np.atleast_2d(Z)
    C = np.broadcast_to(C, (len(Z),) + np.shape(C)[-2:])
    step = np.einsum("nij,nj->ni", C, np.broadcast_to(mu, Z.shape) - Z)
    return Z[:, None] + np.asarray(s)[None, :, None] * step[:, None]


def random_projector(k, r, rng):
    """Orthogonal projector onto a uniformly random r-dimensional subspace of R^k."""
    Q = np.linalg.qr(rng.standard_normal((k, max(1, int(r)))))[0]
    return Q @ Q.T

""""How many dimensions is direction?" as four estimands (spec §6 item 8). A labelled extra beside the
paper's count, not a replacement.

(a) literal: the paper's orthogonal probe sequence K (inlp.inlp) against the random-removal band.
(b) whitened: the same sequence in the whitened metric (Jin et al. 2608.10566, Alg. 1): whiten with the
    train covariance, iterate probe -> QR -> project, and report rank(I - T) of the composite erasure T
    mapped back to the original coordinates. Affine-invariant; a clean linear ring returns 2.
(c) LEACE: closed-form least-squares concept erasure (Belrose et al. 2306.03819) of the target, then a
    fresh ridge and a small MLP are fit on the erased features; R² near 0 means the rank-m erasure removed
    what a linear (ridge) or nonlinear (MLP) probe could find.
(d) split-half DFT spectrum of the 64 direction centroids for harmonics k = 0..8, and the participation
    ratio of the k >= 1 power. Mass only at k = 1 is a flat ring (2 dims); mass at k >= 2 is a curved ring.
"""
from functools import partial

import numpy as np

from wm.inlp import inlp
from wm.probes import ALPHAS, cv_select_alpha, score


def whitener(X, eps=1e-2):
    """(μ, S^{-1/2}, S^{1/2}) of the train covariance, shrunk as S = Σ + eps·mean(eig Σ)·I so that the
    near-null directions of an n ≈ d sample do not dominate (eps is reported with every result)."""
    mu = X.mean(axis=0)
    lam, U = np.linalg.eigh(np.cov(X - mu, rowvar=False, bias=True))
    lam = np.maximum(lam, 0) + eps * max(lam.mean(), 1e-12)
    return mu, (U / np.sqrt(lam)) @ U.T, (U * np.sqrt(lam)) @ U.T


def whitened_count(Xtr, Ytr, folds, kind, eps=1e-2, max_rounds=None):
    """Estimand (b). Probe sequence on Z = (X − μ) S^{-1/2} with the paper's stopping rule; α by CV at
    round 1 and then fixed, as in step 2. Returns K, the number of rounds, and rank(I − T), where
    T = S^{1/2} (I − QQᵀ) S^{-1/2} is the composite erasure in the original coordinates."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    mu, S_mhalf, S_half = whitener(Xtr, eps)
    Z = (Xtr - mu) @ S_mhalf
    score_fn = partial(score, kind=kind)
    alpha = cv_select_alpha(Z, Ytr, folds, ALPHAS, score_fn)["alpha"]
    dummy = np.linspace(0, len(Z) - 1, 20).astype(int)     # inlp's test-score slot, unused here
    summary, Q, _, _ = inlp(Z, Ytr, Z[dummy], Ytr[dummy], folds, alpha, score_fn, kind, max_rounds)
    I_minus_T = S_half @ Q @ Q.T @ S_mhalf if Q.shape[1] else np.zeros((Z.shape[1], Z.shape[1]))
    return {"K": summary["K"], "dims": summary["dims"], "rank_I_minus_T": int(np.linalg.matrix_rank(I_minus_T, tol=1e-8)),
            "alpha": float(alpha), "eps": eps, "cv_r2_by_round": [r["cv_r2"] for r in summary["rounds"]]}


def leace_fit(X, Y, eps=1e-2):
    """LEACE eraser for targets Y (closed form, 2306.03819 Thm. 4.3): with W = S^{-1/2},
    r(x) = x − W⁺ P_{WΣ_xz} W (x − μ), P the orthogonal projector onto the columns of WΣ_xz.
    Returns r as a function of X [n, d]."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    mu, W, W_pinv = whitener(X, eps)
    sxz = (X - mu).T @ (Y - Y.mean(axis=0)) / len(X)
    U, s, _ = np.linalg.svd(W @ sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else U[:, :0]
    E = W_pinv @ U @ U.T @ W
    return lambda Xn: Xn - (Xn - mu) @ E.T


def mlp_cv_r2(X, Y, folds, hidden=64, max_iter=300, seed=0, n_folds=None):
    """Fold-mean R² of a one-hidden-layer MLP (sklearn, adam, early stopping), standardised inputs."""
    from sklearn.neural_network import MLPRegressor
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    r2s = []
    for k in np.unique(folds)[:n_folds]:
        val = folds == k
        mlp = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=max_iter, early_stopping=True, random_state=seed)
        mlp.fit(X[~val], Y[~val] if Y.shape[1] > 1 else Y[~val, 0])
        P = mlp.predict(X[val]).reshape(val.sum(), -1)
        r2s.append(score(Y[val], P, "scalar")["r2"])
    return float(np.mean(r2s))


def leace_scores(Xtr, Ytr, folds, kind, eps=1e-2, mlp=True, mlp_folds=None):
    """Estimand (c). The eraser is fit on all of train, then fresh probes are scored on the 5 folds of the
    erased train set. Also reports the same probes before erasure (the ceiling). R² for direction is the
    mean over (sin, cos)."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    erase = leace_fit(Xtr, Ytr, eps)
    Xe = erase(Xtr)
    sf = partial(score, kind=kind)
    out = {"rank": Ytr.shape[1], "eps": eps,
           "ridge_r2_before": cv_select_alpha(Xtr, Ytr, folds, ALPHAS, sf)["cv_mean"],
           "ridge_r2_after": cv_select_alpha(Xe, Ytr, folds, ALPHAS, sf)["cv_mean"]}
    if mlp:
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        out["mlp_r2_before"] = mlp_cv_r2((Xtr - mu) / sd, Ytr, folds, n_folds=mlp_folds)
        mu, sd = Xe.mean(axis=0), Xe.std(axis=0) + 1e-8
        out["mlp_r2_after"] = mlp_cv_r2((Xe - mu) / sd, Ytr, folds, n_folds=mlp_folds)
    return out


def split_half_spectrum(X, theta_deg, kmax=8, seed=0):
    """Estimand (d). Split the clips of each direction value into two random halves, average each half
    into 64 centroids C_1, C_2 [64, d], take the DFT over the (equally spaced) direction index, and use the
    cross-power P_k = Σ_d Re(F_1[k] · conj F_2[k]) (noise is independent across halves, so it cancels in
    expectation). Returns P_0..P_kmax, the fraction of k >= 1 power per harmonic, and the participation
    ratio PR = (Σ P_k)² / Σ P_k² over k >= 1 with negative P_k clipped to 0 (1 = one harmonic)."""
    rng = np.random.default_rng(seed)
    values = np.unique(theta_deg)
    C1, C2 = [], []
    for v in values:
        rows = rng.permutation(np.flatnonzero(theta_deg == v))
        half = len(rows) // 2
        assert half >= 1, f"direction {v} needs >= 2 clips for a split-half spectrum"
        C1.append(X[rows[:half]].mean(axis=0))
        C2.append(X[rows[half:2 * half]].mean(axis=0))
    ang = np.radians(values)
    basis = np.exp(-1j * np.outer(np.arange(kmax + 1), ang)) / len(values)        # [k, 64]
    F1, F2 = basis @ np.array(C1), basis @ np.array(C2)
    P = np.real(np.sum(F1 * np.conj(F2), axis=1))
    Pk = np.clip(P[1:], 0, None)
    frac = (Pk / Pk.sum()).tolist() if Pk.sum() > 0 else [0.0] * kmax
    pr = float(Pk.sum() ** 2 / np.sum(Pk ** 2)) if Pk.sum() > 0 else None
    return {"k": list(range(kmax + 1)), "power": P.tolist(), "frac_k_ge_1": frac, "participation_ratio": pr,
            "n_values": int(len(values))}


def planted_ring(n=1200, d=64, harmonics=None, copies=1, noise=0.3, shear=None, seed=0):
    """Control data: θ on 64 values; each copy writes cos kθ, sin kθ (amplitude a_k) for each harmonic
    k in `harmonics` = {k: a_k} into its own slots, with independent noise; the rest is unit noise;
    one random rotation. shear = condition number of an extra anisotropic linear map applied after the
    rotation (a feature shear: changes the Euclidean metric, not the information)."""
    harmonics = harmonics or {1: 1.0}
    rng = np.random.default_rng(seed)
    theta = rng.choice(np.arange(64) * 360 / 64, n)
    t = np.radians(theta)
    cols = []
    for _ in range(copies):
        for k, a in harmonics.items():
            cols += [a * np.cos(k * t) + noise * rng.standard_normal(n), a * np.sin(k * t) + noise * rng.standard_normal(n)]
    Zc = np.stack(cols, axis=1)
    Z = np.hstack([Zc, rng.standard_normal((n, d - Zc.shape[1]))])
    X = Z @ np.linalg.qr(rng.standard_normal((d, d)))[0].T
    if shear:
        R = np.linalg.qr(rng.standard_normal((d, d)))[0]
        X = X @ (R * np.logspace(0, np.log10(shear), d)) @ R.T
    return X, theta

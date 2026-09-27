""""How many dimensions is direction?" as four estimands (spec §6 item 8). A labelled extra beside the
paper's count, not a replacement.

(a) literal: the paper's orthogonal probe sequence K (inlp.inlp) against the random-removal band.
(b) whitened: the probe sequence in the whitened metric with the target residualised on the directions
    already removed (Jin et al. 2608.10566, Alg. 1); reports the whitened K. Affine-invariant; a clean
    linear ring returns K = 1 (2 dims).
(c) LEACE: closed-form least-squares concept erasure (Belrose et al. 2306.03819) of the target, then a
    fresh ridge and a small MLP are fit on the erased features; R² near 0 means the rank-m erasure removed
    what a linear (ridge) or nonlinear (MLP) probe could find.
(d) split-half DFT spectrum of the 64 direction centroids for harmonics k = 0..8, and the participation
    ratio of the k >= 1 power. Mass only at k = 1 is a flat ring (2 dims); mass at k >= 2 is a curved ring.
"""
from functools import partial

import numpy as np

from wm.probes import ALPHAS, cv_select_alpha, fit_ridge, predict, ridge_path, score


def whitener(X, eps=1e-2):
    """(μ, S^{-1/2}, S^{1/2}) of the train covariance, shrunk as S = Σ + eps·mean(eig Σ)·I so that the
    near-null directions of an n ≈ d sample do not dominate (eps is reported with every result)."""
    mu = X.mean(axis=0)
    lam, U = np.linalg.eigh(np.cov(X - mu, rowvar=False, bias=True))
    lam = np.maximum(lam, 0) + eps * max(lam.mean(), 1e-12)
    return mu, (U / np.sqrt(lam)) @ U.T, (U * np.sqrt(lam)) @ U.T


def _whitened_sequence(Xfit, Yfit, Xeval, Yeval, eps, alpha=None, fit_folds=None):
    """Rounds of the whitened, residualised probe sequence fit on (Xfit, Yfit) only: whitener, α (CV over fit_folds
    at round 1 when alpha is None), each round's residualisation and probe. Yields (alpha, R² of the residualised
    target on (Xeval, Yeval)) per round, then extends Q with the round's probe."""
    mu, S_mhalf, _ = whitener(Xfit, eps)
    Z, Ze = (Xfit - mu) @ S_mhalf, (Xeval - mu) @ S_mhalf
    if alpha is None:
        alpha = cv_select_alpha(Z, Yfit, fit_folds, ALPHAS, partial(score, kind="scalar"))["alpha"]
    Q = np.zeros((Z.shape[1], 0))
    while True:
        Yk, Yek = _residualise(Z @ Q, Yfit, Ze @ Q, Yeval)
        W, b = fit_ridge(Z - (Z @ Q) @ Q.T, Yk, alpha)
        yield float(alpha), score(Yek, predict(Ze - (Ze @ Q) @ Q.T, W, b), "scalar")["r2"]
        W = W - Q @ (Q.T @ W)
        Q = np.hstack([Q, np.linalg.qr(W)[0]])


def whitened_count(Xtr, Ytr, folds, kind, eps=1e-2, max_rounds=None, Xte=None, Yte=None, Yfit=None):
    """Estimand (b), after Jin et al. Alg. 1: the probe sequence in the whitened metric with the target
    residualised on the directions already removed. Round k: Z_k = Z − Z Q Qᵀ; Y_k = Y minus its
    least-squares fit on the removed coordinates Z Q; a ridge probe Z_k → Y_k; its weights' QR extends Q.
    Nested protocol (as inlp.nested_curve): for each of the 5 folds the whitener, α (CV over the fold's training
    part, at round 1, then fixed), residualisations and the whole sequence are fit on the fold's training part and
    every round is scored on the held-out fold. Stops when the fold-mean R² of the residualised target falls below
    the paper's threshold (0.1 direction, 0.05 scalars) and every fold has; K = pooled (fold-mean curve), with the
    fold-wise K. If Xte is given, also the paper-style variant: fit on all train (α by CV over the 5 folds), each
    round scored on test ('paper'). Reports the whitened K (and K·m). Yfit (default Ytr): the labels the nested fits
    use (scoring always uses Ytr); differs only in the leakage test."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    Yfit = Ytr if Yfit is None else np.asarray(Yfit, float).reshape(len(Yfit), -1)
    m = Ytr.shape[1]
    max_rounds = max_rounds or Xtr.shape[1] // m
    thresh = 0.1 if kind == "circular" else 0.05
    ks = np.unique(folds)
    seqs = [_whitened_sequence(Xtr[folds != k], Yfit[folds != k], Xtr[folds == k], Ytr[folds == k], eps,
                               fit_folds=folds[folds != k]) for k in ks]
    r2_by_round, sd_by_round, folds_by_round, fold_K, K, alphas = [], [], [], [None] * len(ks), None, None
    for k in range(1, max_rounds + 1):
        steps = [next(g) for g in seqs]
        alphas = [a for a, _ in steps]
        r2s = [r for _, r in steps]
        for f, r in enumerate(r2s):
            if fold_K[f] is None and r < thresh:
                fold_K[f] = k - 1
        r2_by_round.append(float(np.mean(r2s)))
        sd_by_round.append(float(np.std(r2s, ddof=1)))
        folds_by_round.append([float(r) for r in r2s])
        if K is None and r2_by_round[-1] < thresh:
            K = k - 1
        if K is not None and None not in fold_K:
            break
    hit_cap = K is None
    K = len(r2_by_round) if hit_cap else K
    fk = [len(r2_by_round) if x is None else x for x in fold_K]
    out = {"K": K, "dims": K * m, "protocol": "nested", "alpha_folds": alphas, "eps": eps, "stop_r2": thresh,
           "cv_r2_by_round": r2_by_round, "cv_r2_sd_by_round": sd_by_round,
           "cv_r2_folds_by_round": folds_by_round, "hit_round_cap": hit_cap,
           "K_folds": fk, "K_fold_mean": float(np.mean(fk)), "K_fold_min": int(min(fk)), "K_fold_max": int(max(fk))}
    if Xte is not None:
        seq = _whitened_sequence(Xtr, Ytr, Xte, np.asarray(Yte, float).reshape(len(Yte), -1), eps, fit_folds=folds)
        test_r2, Kp = [], None
        while Kp is None and len(test_r2) < max_rounds:
            alpha, r = next(seq)
            test_r2.append(r)
            if r < thresh:
                Kp = len(test_r2) - 1
        Kp = len(test_r2) if Kp is None else Kp
        out["paper"] = {"K": Kp, "dims": Kp * m, "alpha": alpha, "test_r2_by_round": test_r2,
                        "hit_round_cap": test_r2[-1] >= thresh, "protocol": "fit on all train, scored on test"}
    return out


def _residualise(A_fit, Y_fit, A_apply, Y_apply):
    """Residuals of Y on the columns of A (with intercept), fit on (A_fit, Y_fit), applied to both."""
    if A_fit.shape[1] == 0:
        return Y_fit - Y_fit.mean(axis=0), Y_apply - Y_fit.mean(axis=0)
    A1 = np.hstack([A_fit, np.ones((len(A_fit), 1))])
    B = np.linalg.lstsq(A1, Y_fit, rcond=None)[0]
    return Y_fit - A1 @ B, Y_apply - np.hstack([A_apply, np.ones((len(A_apply), 1))]) @ B


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
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    return _mlp_folds(lambda k: (X[folds != k], X[folds == k]), Y, folds, n_folds, hidden, max_iter, seed)


def leace_scores(Xtr, Ytr, folds, kind, eps=1e-2, mlp=True, mlp_folds=None):
    """Estimand (c). Inside each of the 5 folds the eraser is fit on the fold's training part only and
    applied to both parts; a fresh ridge (α picked by the fold-mean score over the grid) and an MLP are
    then fit on the erased training part and scored on the erased held-out part. Also reports the same
    probes before erasure (the ceiling). R² for direction is the mean over (sin, cos)."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    sf = partial(score, kind=kind)
    fold_ids = np.unique(folds)
    erased = {}
    for k in fold_ids:
        val = folds == k
        erase = leace_fit(Xtr[~val], Ytr[~val], eps)
        erased[k] = (erase(Xtr[~val]), erase(Xtr[val]))
    curves = np.zeros((len(ALPHAS), len(fold_ids)))
    for j, k in enumerate(fold_ids):
        val = folds == k
        for i, (W, b) in enumerate(ridge_path(erased[k][0], Ytr[~val], ALPHAS)):
            curves[i, j] = sf(Ytr[val], predict(erased[k][1], W, b))["r2"]
    out = {"rank": Ytr.shape[1], "eps": eps, "eraser_fit": "per fold, on the fold's training part",
           "ridge_r2_before": cv_select_alpha(Xtr, Ytr, folds, ALPHAS, sf)["cv_mean"],
           "ridge_r2_after": float(curves.mean(axis=1).max())}
    if mlp:
        out["mlp_r2_before"] = _mlp_folds(lambda k: (Xtr[folds != k], Xtr[folds == k]), Ytr, folds, mlp_folds)
        out["mlp_r2_after"] = _mlp_folds(lambda k: erased[k], Ytr, folds, mlp_folds)
    return out


def _mlp_folds(parts, Y, folds, n_folds=None, hidden=64, max_iter=300, seed=0):
    """Fold-mean MLP R² where parts(k) gives (X_train_part, X_heldout_part), standardised on the train part."""
    from sklearn.neural_network import MLPRegressor
    r2s = []
    for k in np.unique(folds)[:n_folds]:
        val = folds == k
        A, B = parts(k)
        mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-8
        mlp = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=max_iter, early_stopping=True, random_state=seed)
        mlp.fit((A - mu) / sd, Y[~val] if Y.shape[1] > 1 else Y[~val, 0])
        r2s.append(score(Y[val], mlp.predict((B - mu) / sd).reshape(val.sum(), -1), "scalar")["r2"])
    return float(np.mean(r2s))


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

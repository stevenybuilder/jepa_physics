"""The paper's probe recipe (App. B), for the spec §5.1 parity check against closed-form ridge.

Linear probe f(h) = Wh + b trained with Adam + weight decay and MSE, swept over the App. B grid of 20
configurations (learning rate x weight decay), chosen on the 5-fold validation score. App. B does not
give epochs or batch size; we use C.11's 100 epochs (direction) / 50 (scalars) and minibatches of 64.
Weight decay is Adam's coupled L2 on W only (the bias is unpenalised, as in our ridge). Implemented in
numpy with all 20 configurations trained side by side (same minibatches), so the grid costs one run.
"""
import numpy as np

LRS = (1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
WDS = (0.01, 0.1, 0.4, 0.8)
BETAS, EPS = (0.9, 0.999), 1e-8   # torch.optim.Adam defaults


def fit_adam_grid(X, Y, configs, epochs, batch=64, seed=0):
    """Train one linear probe per (lr, wd) in `configs` at once, each with its own Adam state, on the same
    minibatch sequence. Adam as in torch.optim.Adam with weight_decay (coupled L2: g += wd·W), applied
    to W only; loss = MSE averaged over batch and outputs; init as torch.nn.Linear (U(±1/√d)).
    Returns W [G, d, m] and b [G, m]."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    (n, d), m, G = X.shape, Y.shape[1], len(configs)
    lr = np.array([c[0] for c in configs])[:, None, None]
    wd = np.array([c[1] for c in configs])[:, None, None]
    rng = np.random.default_rng(seed)
    bound = 1 / np.sqrt(d)
    W = np.repeat(rng.uniform(-bound, bound, (1, d, m)), G, axis=0)
    b = np.repeat(rng.uniform(-bound, bound, (1, m)), G, axis=0)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    (b1, b2), t = BETAS, 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), int(np.ceil(n / batch))):
            Xb, Yb = X[idx], Y[idx]
            R = np.einsum("bd,gdm->gbm", Xb, W) + b[:, None, :] - Yb          # residuals [G, B, m]
            scale = 2.0 / (len(idx) * m)
            gW = scale * np.einsum("bd,gbm->gdm", Xb, R) + wd * W
            gb = scale * R.sum(axis=1)
            t += 1
            mW, vW = b1 * mW + (1 - b1) * gW, b2 * vW + (1 - b2) * gW ** 2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            W = W - lr * (mW / c1) / (np.sqrt(vW / c2) + EPS)
            b = b - lr[:, :, 0] * (mb / c1) / (np.sqrt(vb / c2) + EPS)
    return W, b


def fit_adam(X, Y, lr, wd, epochs, batch=64, seed=0):
    """One configuration: W [d, m], b [m]."""
    W, b = fit_adam_grid(X, Y, [(lr, wd)], epochs, batch, seed)
    return W[0], b[0]


def cv_adam(X, Y, folds, score_fn, epochs, lrs=LRS, wds=WDS, batch=64, seed=0):
    """Grid over (lr, wd), each scored by the fold-mean R² (fit fold-out, score fold-in). Returns the best
    configuration with its fold mean ± SD (R² and MAE) and the whole grid."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    configs = [(lr, wd) for lr in lrs for wd in wds]
    fold_scores = [[] for _ in configs]
    for k in np.unique(folds):
        val = folds == k
        W, b = fit_adam_grid(X[~val], Y[~val], configs, epochs, batch, seed)
        for g in range(len(configs)):
            fold_scores[g].append(score_fn(Y[val], X[val] @ W[g] + b[g]))
    grid, best = [], None
    for (lr, wd), fs in zip(configs, fold_scores):
        r2s, maes = [f["r2"] for f in fs], [f["mae"] for f in fs]
        row = {"lr": lr, "wd": wd, "cv_mean": float(np.mean(r2s)), "cv_sd": float(np.std(r2s, ddof=1)),
               "cv_mae_mean": float(np.mean(maes)), "cv_mae_sd": float(np.std(maes, ddof=1))}
        grid.append(row)
        if best is None or row["cv_mean"] > best["cv_mean"]:
            best = row
    return {**best, "epochs": epochs, "batch": batch, "grid": grid}


def parity(ridge, adam):
    """Parity rule: |CV R² difference| within the larger of the two fold SDs."""
    diff = adam["cv_mean"] - ridge["cv_mean"]
    tol = max(ridge["cv_sd"], adam["cv_sd"])
    return {"adam_minus_ridge_r2": float(diff), "tolerance_fold_sd": float(tol), "within": bool(abs(diff) <= tol)}

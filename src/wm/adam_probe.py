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


def init_linear(d, m, seed=0):
    """torch.nn.Linear-style init U(±1/√d) from a fresh RNG with this seed: W [d, m], b [m]. fit_adam_grid starts
    from exactly this (then keeps drawing minibatch permutations from the same RNG)."""
    rng = np.random.default_rng(seed)
    bound = 1 / np.sqrt(d)
    return rng.uniform(-bound, bound, (d, m)), rng.uniform(-bound, bound, m), rng


def target_scaler(Y):
    """(mean, sd) of the fitting rows' targets, per column (sd floored at 1e-8)."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    return Y.mean(axis=0), np.maximum(Y.std(axis=0), 1e-8)


def fit_adam_grid(X, Y, configs, epochs, batch=64, seed=0, decoupled=False, standardize_y=True):
    """Train one linear probe per (lr, wd) in `configs` at once, each with its own Adam state, on the same
    minibatch sequence. Adam as in torch.optim.Adam with weight_decay (coupled L2: g += wd·W), applied
    to W only; loss = MSE averaged over batch and outputs; init as torch.nn.Linear (U(±1/√d)).
    decoupled=True is AdamW instead (W ← W(1 − lr·wd) before the Adam step, no L2 in the gradient).
    standardize_y (default): train on targets standardised with the fitting rows' mean and SD and return the probe
    in physical units (W·sd, b·sd + mean), so predictions and scores are physical. Without it a speed target far
    from 0 needs the bias to travel ~its mean at ≤ lr per step, which 50 epochs at lr 1e-3 cannot do (round-1 train
    R² < 0). Weight decay then acts on the standardised-unit weights.
    Returns W [G, d, m] and b [G, m]."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    y_mean, y_sd = target_scaler(Y) if standardize_y else (np.zeros(Y.shape[1]), np.ones(Y.shape[1]))
    Y = (Y - y_mean) / y_sd
    (n, d), m, G = X.shape, Y.shape[1], len(configs)
    lr = np.array([c[0] for c in configs])[:, None, None]
    wd = np.array([c[1] for c in configs])[:, None, None]
    W0, b0, rng = init_linear(d, m, seed)
    W = np.repeat(W0[None], G, axis=0)
    b = np.repeat(b0[None], G, axis=0)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    (b1, b2), t = BETAS, 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(n), int(np.ceil(n / batch))):
            Xb, Yb = X[idx], Y[idx]
            R = np.einsum("bd,gdm->gbm", Xb, W) + b[:, None, :] - Yb          # residuals [G, B, m]
            scale = 2.0 / (len(idx) * m)
            gW = scale * np.einsum("bd,gbm->gdm", Xb, R) + (0.0 if decoupled else wd * W)
            if decoupled:
                W = W * (1 - lr * wd)
            gb = scale * R.sum(axis=1)
            t += 1
            mW, vW = b1 * mW + (1 - b1) * gW, b2 * vW + (1 - b2) * gW ** 2
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb ** 2
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            W = W - lr * (mW / c1) / (np.sqrt(vW / c2) + EPS)
            b = b - lr[:, :, 0] * (mb / c1) / (np.sqrt(vb / c2) + EPS)
    return W * y_sd, b * y_sd + y_mean


def fit_adam(X, Y, lr, wd, epochs, batch=64, seed=0, decoupled=False, standardize_y=True):
    """One configuration: W [d, m], b [m] (physical units)."""
    W, b = fit_adam_grid(X, Y, [(lr, wd)], epochs, batch, seed, decoupled, standardize_y)
    return W[0], b[0]


def cv_adam(X, Y, folds, score_fn, epochs, lrs=LRS, wds=WDS, batch=64, seed=0, decoupled=False, standardize_y=True):
    """Grid over (lr, wd), each scored by the fold-mean R² (fit fold-out, score fold-in). Returns the best
    configuration with its fold mean ± SD (R² and MAE), its out-of-fold predictions ('oof', for the
    bootstrap) and the whole grid."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    configs = [(lr, wd) for lr in lrs for wd in wds]
    fold_scores = [[] for _ in configs]
    oof = np.zeros((len(configs),) + Y.shape)
    for k in np.unique(folds):
        val = folds == k
        W, b = fit_adam_grid(X[~val], Y[~val], configs, epochs, batch, seed, decoupled, standardize_y)
        for g in range(len(configs)):
            oof[g][val] = X[val] @ W[g] + b[g]
            fold_scores[g].append(score_fn(Y[val], oof[g][val]))
    grid, best = [], None
    for g, ((lr, wd), fs) in enumerate(zip(configs, fold_scores)):
        r2s, maes = [f["r2"] for f in fs], [f["mae"] for f in fs]
        row = {"lr": lr, "wd": wd, "cv_mean": float(np.mean(r2s)), "cv_sd": float(np.std(r2s, ddof=1)),
               "cv_mae_mean": float(np.mean(maes)), "cv_mae_sd": float(np.std(maes, ddof=1))}
        grid.append(row)
        if best is None or row["cv_mean"] > best["cv_mean"]:
            best, best_g = row, g
    return {**best, "epochs": epochs, "batch": batch,
            "weight_decay_type": "decoupled (AdamW)" if decoupled else "coupled L2 (Adam)",
            "targets": ("standardised on each fit's rows (mean, SD), scored in physical units" if standardize_y
                        else "raw"),
            "grid": grid, "oof": oof[best_g]}


def pooled_r2_ci(Y, oof, score_fn, n_boot=200, seed=0):
    """Pooled out-of-fold R² and its 95% clip-bootstrap CI (predictions fixed, clips resampled)."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    rng = np.random.default_rng(seed)
    draws = [score_fn(Y[idx], oof[idx])["r2"] for idx in (rng.integers(0, len(Y), len(Y)) for _ in range(n_boot))]
    return score_fn(Y, oof)["r2"], [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def parity(Y, ridge_oof, adam_oof, score_fn, n_boot=200, seed=0):
    """Parity rule (spec §5.1): the ridge probe's pooled out-of-fold R² lies inside the 95% bootstrap CI of
    the Adam probe's pooled out-of-fold R² (same folds, same clips)."""
    r_ridge = score_fn(np.asarray(Y, float).reshape(len(Y), -1), ridge_oof)["r2"]
    r_adam, ci = pooled_r2_ci(Y, adam_oof, score_fn, n_boot, seed)
    return {"ridge_pooled_oof_r2": float(r_ridge), "adam_pooled_oof_r2": float(r_adam), "adam_ci95": ci,
            "within": bool(ci[0] <= r_ridge <= ci[1]),
            "rule": "ridge pooled OOF R2 inside the 95% clip-bootstrap CI of the Adam probe's pooled OOF R2"}

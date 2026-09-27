"""Spec §5.1 probe-recipe parity: Adam + weight decay matches closed-form ridge on a planted code."""
from functools import partial

import numpy as np

from wm.adam_probe import cv_adam, fit_adam, fit_adam_grid, parity
from wm.probes import cv_select_alpha, score


def test_adam_matches_ridge_on_planted_direction_code():
    rng = np.random.default_rng(0)
    n, d = 500, 20
    th = rng.uniform(0, 2 * np.pi, n)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    X = np.hstack([Y + 0.2 * rng.standard_normal((n, 2)), rng.standard_normal((n, d - 2))])
    X = X @ np.linalg.qr(rng.standard_normal((d, d)))[0].T
    folds = np.arange(n) % 5
    sf = partial(score, kind="circular")
    ridge = cv_select_alpha(X, Y, folds, score_fn=sf)
    adam = cv_adam(X, Y, folds, sf, epochs=40, lrs=(1e-3, 5e-3), wds=(0.01, 0.8))
    assert len(adam["grid"]) == 4 and adam["cv_mean"] > 0.85 and adam["weight_decay_type"] == "coupled L2 (Adam)"
    par = parity(Y, ridge["oof"], adam["oof"], sf)
    assert par["within"] and par["adam_ci95"][0] < par["adam_pooled_oof_r2"] < par["adam_ci95"][1]
    adamw = cv_adam(X, Y, folds, sf, epochs=40, lrs=(5e-3,), wds=(0.01,), decoupled=True)
    assert adamw["weight_decay_type"] == "decoupled (AdamW)" and adamw["cv_mean"] > 0.85
    # a crippled probe (far too few epochs at the smallest lr) must fail parity
    weak = cv_adam(X, Y, folds, sf, epochs=1, lrs=(1e-4,), wds=(0.8,))
    assert not parity(Y, ridge["oof"], weak["oof"], sf)["within"]


def test_fit_adam_recovers_linear_map_and_leaves_bias_unpenalised():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5))
    W_true = rng.standard_normal((5, 1))
    y = X @ W_true + 3.0
    W, b = fit_adam(X, y, lr=5e-3, wd=0.0, epochs=200, batch=64)
    assert np.allclose(W, W_true, atol=0.05) and abs(b[0] - 3.0) < 0.05
    _, b_wd = fit_adam(X, y, lr=5e-3, wd=0.8, epochs=200, batch=64)
    assert abs(b_wd[0] - 3.0) < 0.2                          # weight decay does not pull the bias to 0


def test_adamw_matches_torch():
    import torch
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((200, 8)), rng.standard_normal((200, 2))
    W, b = fit_adam_grid(X, Y, [(3e-3, 0.4)], epochs=20, batch=200, decoupled=True)
    r = np.random.default_rng(0)
    bound = 1 / np.sqrt(8)
    W0, b0 = r.uniform(-bound, bound, (1, 8, 2))[0], r.uniform(-bound, bound, (1, 2))[0]
    lin = torch.nn.Linear(8, 2).double()
    lin.weight.data, lin.bias.data = torch.tensor(W0.T.copy()), torch.tensor(b0.copy())
    opt = torch.optim.AdamW([{"params": [lin.weight], "weight_decay": 0.4},
                             {"params": [lin.bias], "weight_decay": 0.0}], lr=3e-3)
    for _ in range(20):
        opt.zero_grad()
        torch.nn.functional.mse_loss(lin(torch.tensor(X)), torch.tensor(Y)).backward()
        opt.step()
    assert np.allclose(lin.weight.detach().numpy().T, W[0], atol=1e-12) and np.allclose(lin.bias.detach().numpy(), b[0])

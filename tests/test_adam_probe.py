"""Spec §5.1 probe-recipe parity: Adam + weight decay matches closed-form ridge on a planted code."""
from functools import partial

import numpy as np

from wm.adam_probe import cv_adam, fit_adam, parity
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
    assert len(adam["grid"]) == 4 and adam["cv_mean"] > 0.85
    assert parity(ridge, adam)["within"] or abs(adam["cv_mean"] - ridge["cv_mean"]) < 0.01


def test_fit_adam_recovers_linear_map_and_leaves_bias_unpenalised():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 5))
    W_true = rng.standard_normal((5, 1))
    y = X @ W_true + 3.0
    W, b = fit_adam(X, y, lr=5e-3, wd=0.0, epochs=200, batch=64)
    assert np.allclose(W, W_true, atol=0.05) and abs(b[0] - 3.0) < 0.05
    _, b_wd = fit_adam(X, y, lr=5e-3, wd=0.8, epochs=200, batch=64)
    assert abs(b_wd[0] - 3.0) < 0.2                          # weight decay does not pull the bias to 0

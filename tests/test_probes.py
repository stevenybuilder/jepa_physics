import numpy as np
import pandas as pd

from wm.probes import (Standardizer, availability, cv_select_alpha, fit_ridge, predict, ridge_path, score,
                       targets)


def test_ridge_recovers_planted_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 20))
    W_true, b_true = rng.standard_normal((20, 2)), np.array([3.0, -1.0])
    Y = X @ W_true + b_true
    W, b = fit_ridge(X, Y, 1e-8)
    assert np.allclose(W, W_true, atol=1e-6) and np.allclose(b, b_true, atol=1e-6)
    W1, b1 = fit_ridge(X, Y[:, 0], 1e-8)          # 1-D target works too
    assert W1.shape == (20, 1) and np.allclose(predict(X, W1, b1)[:, 0], Y[:, 0], atol=1e-6)


def test_standardizer_uses_train_stats_and_floors_std():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [np.nan, 5.0]])
    st = Standardizer().fit(X)
    assert st.std[1] == 1e-6 and np.allclose(st.mean, [2.0, 5.0])
    Z = st.transform(X)
    assert np.allclose(Z[:2, 0], [-1, 1]) and Z[2, 0] == 0.0


def test_circular_targets_and_wraparound():
    df = pd.DataFrame({"theta_degrees": [0.0, 90.0, 180.0, 359.0], "speed_mps": 1.0, "acceleration_mps2": 0.0})
    Y, kind, score_fn = targets(df, "direction")
    assert kind == "circular" and np.allclose(Y[1], [1, 0]) and np.allclose(Y[2], [0, -1], atol=1e-12)
    assert score_fn(Y, Y)["mae"] < 1e-9
    near = np.radians(np.array([1.0, 91.0, 181.0, 1.0]))          # 359 -> 1 is a 2 degree error
    s = score_fn(Y, np.stack([np.sin(near), np.cos(near)], 1))
    assert abs(s["mae"] - 1.25) < 1e-9
    Yv, kind_v, _ = targets(df, "vxvy")
    assert kind_v == "vector" and np.allclose(Yv[1], [0, 1], atol=1e-12)


def test_planted_sin_cos_code_decodes_with_zero_circular_mae():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 360, 600)
    r = np.radians(theta)
    X = np.hstack([np.stack([np.sin(r), np.cos(r)], 1) @ rng.standard_normal((2, 10)),
                   rng.standard_normal((600, 5))])
    Y = np.stack([np.sin(r), np.cos(r)], 1)
    folds = np.arange(600) % 5
    cv = cv_select_alpha(X, Y, folds, score_fn=lambda a, b: score(a, b, "circular"))
    assert cv["cv_mean"] > 0.999 and cv["fold_scores"][0]["mae"] < 0.5


def test_cv_picks_strong_alpha_for_pure_noise_and_availability_rule():
    rng = np.random.default_rng(2)
    X, y = rng.standard_normal((300, 50)), rng.standard_normal(300)
    cv = cv_select_alpha(X, y, np.arange(300) % 5)
    assert cv["alpha"] >= 1e3 and len(cv["curve"]) == 13
    av = availability([0.0, 0.2, 0.5, 0.92, 1.0, 0.95, 0.8])
    assert av["onset"] == 3 and av["peak"] == 4 and abs(av["decline"] - 0.2) < 1e-12


def test_ridge_path_matches_fit_ridge():
    rng = np.random.default_rng(5)
    X, Y = rng.standard_normal((80, 30)), rng.standard_normal((80, 2))
    for alpha, (W, b) in zip([0.1, 10.0], ridge_path(X, Y, [0.1, 10.0])):
        W2, b2 = fit_ridge(X, Y, alpha)
        assert np.allclose(W, W2) and np.allclose(b, b2)

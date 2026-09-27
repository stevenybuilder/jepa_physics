import numpy as np
import pandas as pd

from wm.probes import (Standardizer, availability, bootstrap_onset, cv_select_alpha, precision_onset,
                       selectivity_onset, fit_ridge, predict, ridge_path, score,
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


def test_selectivity_and_precision_onsets():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((400, 1))
    noise = lambda sd: y + sd * rng.standard_normal((400, 1))
    model = [noise(3.0) if p < 6 else noise(0.2) for p in range(26)]      # informative from point 6
    random = [noise(0.8) for _ in range(26)]                              # moderately informative everywhere
    onset, cis = selectivity_onset(y, model, random, lambda a, b: score(a, b, "scalar"), n_boot=100)
    assert onset == 6 and cis[5][1] < 0 < cis[6][0] and len(cis) == 25
    rows = [{"cv_mae_mean": m} for m in [5, 4, 2, 1.15, 1.05, 1.0, 1.2] + [1.3] * 19]
    assert precision_onset(rows, "scalar") == 4                           # 1.05 <= 1.1 * min
    rows = [{"cv_acc15_mean": a} for a in [0.1, 0.5, 0.85, 0.95, 0.9] + [0.9] * 21]
    assert precision_onset(rows, "circular") == 3                         # 0.95 is max; 0.855 needed


def test_bootstrap_onset_ci_and_no_onset_count():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((300, 1))
    oofs = [y + (3.0 if p < 10 else 0.3) * rng.standard_normal((300, 1)) for p in range(25)]
    ci, n_none = bootstrap_onset(y, oofs, lambda a, b: score(a, b, "scalar"), n_boot=100)
    assert ci[0] <= 10 <= ci[1] and ci[1] - ci[0] <= 2 and n_none == 0
    flat = [rng.standard_normal((300, 1)) for _ in range(25)]                # no information anywhere
    ci, n_none = bootstrap_onset(y, flat, lambda a, b: score(a, b, "scalar"), n_boot=50)
    assert ci is None and n_none == 50

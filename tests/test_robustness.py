import numpy as np

from wm.inlp import ridge_fit
from wm.probes import fit_ridge, score
from wm.robustness import (binned_score, decay_stats, direction_bin, equal_count_edges, in_sample_and_oof,
                           run_rounds, start_cells, value_grouped_folds)


def test_start_cells_grid():
    assert start_cells([-1.99, -1.01, 0.5, 1.99], [-1.5, 1.5, 0.0, 1.99]).tolist() == [0, 3, 2002, 3003]


def test_value_grouped_folds_hold_out_whole_values():
    v = np.repeat(np.arange(12.0), 7)
    f = value_grouped_folds(v)
    assert set(f) == set(range(5))
    assert all(len(set(f[v == x])) == 1 for x in np.unique(v))


def test_direction_bin_and_edges():
    assert direction_bin([0, 44.9, 45, 359, -1]).tolist() == [0, 0, 1, 7, 7]
    e = equal_count_edges(np.arange(80.0))
    assert len(e) == 7 and np.all(np.bincount(np.digitize(np.arange(80.0), e)) == 10)


def test_binned_score_perfect_and_scalar():
    th = np.radians(np.arange(0, 360, 5.625))
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    assert binned_score("circular")(Y, 0.5 * Y)["bacc8"] == 1.0
    y = np.arange(80.0)[:, None]
    fn = binned_score("scalar", equal_count_edges(y[:, 0]))
    assert fn(y, y)["bacc8"] == 1.0 and fn(y, y[::-1])["bacc8"] < 0.2


def test_decay_stats_detects_teeth():
    v = [1.0, 0.8, 0.95, 0.6, 0.75, 0.4]
    s = decay_stats(v)
    assert s["isolated_dips"] == [2, 4] and s["drop_lag1_autocorr"] < 0


def test_run_rounds_matches_single_ridge_and_oof_helper():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    Y = X[:, :1] + 0.1 * rng.standard_normal((200, 1))
    rows = run_rounds(X[:150], Y[:150], ridge_fit(1.0), lambda a, b: score(a, b, "scalar"), X[150:], Y[150:], 3)
    W, b = fit_ridge(X[:150], Y[:150], 1.0)
    assert np.isclose(rows[0]["r2"], score(Y[150:], X[150:] @ W + b, "scalar")["r2"]) and rows[1]["r2"] < rows[0]["r2"]
    _, _, rep = in_sample_and_oof(lambda A, B: fit_ridge(A, B, 1.0), X, Y, np.arange(200) % 5, "scalar")
    assert rep["in_sample"]["r2"] >= rep["out_of_fold"]["r2_fold_mean"] > 0.9


def test_k_below_first_crossing_or_censored():
    from wm.robustness import k_below
    assert k_below([0.5, 0.2, 0.08, 0.04], 0.1) == 2 and k_below([0.5, 0.2, 0.08, 0.04], 0.05) == 3
    assert k_below([0.5, 0.2], 0.05) is None and k_below([0.01], 0.05) == 0


def test_lockstep_until_stops_on_mean_r2_and_matches_run_rounds():
    from wm.inlp import probe_sequence
    from wm.robustness import lockstep_until
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    Y = X[:, :3] @ np.ones((3, 1)) + 0.1 * rng.standard_normal((200, 1))
    fn = lambda a, b: score(a, b, "scalar")  # noqa: E731
    rows = lockstep_until([probe_sequence(X[:150], Y[:150], ridge_fit(1.0), fn, X[150:], Y[150:])], 0.05, 10)
    ref = run_rounds(X[:150], Y[:150], ridge_fit(1.0), fn, X[150:], Y[150:], len(rows))
    assert np.allclose([r["r2"] for r in rows], [r["r2"] for r in ref])
    assert rows[-1]["r2"] < 0.05 and all(r["r2"] >= 0.05 for r in rows[:-1])

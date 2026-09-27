"""Paper-scale step 1 (wm.paperscale) and the random-feature floor (wm.randfeat) on tiny synthetic inputs."""
from functools import partial

import numpy as np

from wm.paperscale import (direction_grouped_folds, dual_cv, eight_direction_rows, onset_draws, random_folds,
                           summarise_curve)
from wm.pixels import probe_row
from wm.probes import ALPHAS, cv_select_alpha, score
from wm.randfeat import centroid_inputs, random_relu_features


def test_eight_direction_rows_picks_multiples_of_45():
    theta = np.arange(64) * 5.625
    rows = eight_direction_rows(np.concatenate([theta, theta]))
    assert len(rows) == 16 and set(np.round(np.concatenate([theta, theta])[rows], 3)) == set(range(0, 360, 45))


def test_grouped_folds_hold_out_whole_directions():
    rng = np.random.default_rng(0)
    theta = rng.choice(np.arange(0, 360, 45), 190).astype(float)
    folds = direction_grouped_folds(theta, 5, seed=3)
    per_fold = [set(theta[folds == k]) for k in range(5)]
    assert all(per_fold)                                        # every fold used
    for i in range(5):                                          # a direction never appears in two folds
        for j in range(i + 1, 5):
            assert not per_fold[i] & per_fold[j]
    assert sorted(len(s) for s in per_fold) == [1, 1, 2, 2, 2]
    assert set().union(*per_fold) == set(range(0, 360, 45))
    rf = random_folds(190, 5, 0)                                # random folds, by contrast, mix directions
    assert np.bincount(rf).tolist() == [38] * 5 and len(set(theta[rf == 0])) == 8


def test_dual_cv_equals_wm_probes_primal():
    rng = np.random.default_rng(1)
    for n, d in ((60, 150), (200, 30)):                         # n < d and n > d
        X = rng.standard_normal((n, d))
        th = rng.uniform(0, 2 * np.pi, n)
        Y = np.stack([np.sin(th), np.cos(th)], 1) + 0.1 * (X[:, :2] @ rng.standard_normal((2, 2)))
        folds = np.arange(n) % 5
        fn = partial(score, kind="circular")
        a, b = dual_cv(X, Y, folds, fn), cv_select_alpha(X, Y, folds, ALPHAS, fn)
        assert a["alpha"] == b["alpha"] and np.isclose(a["cv_mean"], b["cv_mean"], atol=1e-8)
        assert np.allclose(a["curve_fold_mean"], b["curve"], atol=1e-8) and np.allclose(a["oof"], b["oof"], atol=1e-7)


def test_grouped_cv_uses_pooled_score_and_generalises_across_directions():
    rng = np.random.default_rng(2)
    theta = rng.choice(np.arange(0, 360, 45), 200).astype(float)
    t = np.radians(theta)
    X = np.stack([np.sin(t), np.cos(t)], 1) @ rng.standard_normal((2, 40)) + 0.05 * rng.standard_normal((200, 40))
    Y = np.stack([np.sin(t), np.cos(t)], 1)
    folds = direction_grouped_folds(theta, 5, 0)
    res = dual_cv(X, Y, folds, partial(score, kind="circular"), select="pooled")
    assert res["cv_mean"] is None                               # single-direction folds: fold R² undefined
    assert res["pooled"] > 0.95                                 # a linear code interpolates to unseen directions


def test_onset_summary_on_planted_rising_curve():
    rng = np.random.default_rng(3)
    n, Y = 150, rng.standard_normal((150, 1))
    oofs = [Y * min(1.0, p / 8) + 0.2 * rng.standard_normal((n, 1)) for p in range(25)]
    fn = partial(score, kind="scalar")
    curve = [fn(Y, o)["r2"] for o in oofs]
    s = summarise_curve(curve, oofs, Y, fn, n_boot=50)
    assert 5 <= s["onset"] <= 8 and s["onset_ci"][0] <= s["onset"] <= s["onset_ci"][1]
    assert len(onset_draws(Y, oofs, fn, 10)) == 10


def test_random_features_reproduce_planted_nonlinear_target():
    rng = np.random.default_rng(4)
    n = 1000
    Z = rng.uniform(-1, 1, (n, 4))
    y = (np.sqrt((Z[:, :2] ** 2).sum(1)))[:, None]              # a norm: not linear in Z, like speed from positions
    tr, te = np.arange(800), np.arange(800, n)
    folds = tr % 5
    fn = partial(score, kind="scalar")
    Ftr, Fte = random_relu_features(Z[tr], Z[te], 1024, seed=0)
    rf, _ = probe_row(Ftr, Fte, y[tr], y[te], folds, "scalar", fn)
    lin, _ = probe_row(Z[tr], Z[te], y[tr], y[te], folds, "scalar", fn)
    assert rf["cv_mean"] > 0.9 and rf["test_r2"] > 0.9 and lin["cv_mean"] < 0.1
    F2, _ = random_relu_features(Z[tr], Z[te], 1024, seed=0)
    assert np.array_equal(Ftr, F2)                              # fixed seed -> identical features


def test_centroid_inputs_shapes_and_nan_fill():
    cen = np.random.default_rng(5).uniform(0, 256, (20, 16, 2))
    cen[3, 5] = np.nan
    tr, te = np.arange(15), np.arange(15, 20)
    Ztr, Zte = centroid_inputs(cen, tr, te, "centroid")
    Dtr, _ = centroid_inputs(cen, tr, te, "diff")
    assert Ztr.shape == (15, 32) and Zte.shape == (5, 32) and Dtr.shape == (15, 30)
    assert np.isfinite(Ztr).all() and Ztr[3, 10] == 0.0          # NaN -> train mean (0 after z-scoring)

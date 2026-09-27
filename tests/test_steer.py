from functools import partial

import numpy as np
import pytest

from wm.inlp import inlp
from wm.probes import fit_ridge, predict, score
from wm.steer import build_basis, encode, evaluate, readout_weights, steer

from synthetic import copies_code, split


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    theta = rng.choice(np.arange(64) * 360 / 64, 1500)
    Y = encode(theta, "circular")
    X = copies_code(Y, 3, d=40, seed=0)
    tr, te, folds = split(len(Y))
    summary, Q, W, b = inlp(X[tr], Y[tr], X[te], Y[te], folds, 1.0, partial(score, kind="circular"), "circular")
    probes = {"Q": Q, "W": W, "b": b}
    eval_W, eval_b = fit_ridge(X[te], Y[te], 1.0)          # evaluation probe: test activations only
    return X[te], theta[te], probes, eval_W, eval_b


def test_steering_probes_read_target_exactly(world):
    X, theta, probes, _, _ = world
    V = build_basis(probes["W"])
    K = len(probes["W"])
    Xs = steer(X, V, probes, encode(90.0, "circular"), K)
    Wt, bt = readout_weights(probes, K)
    assert np.allclose(Xs @ Wt + bt, np.tile(encode(90.0, "circular"), K), atol=1e-8)
    x_perp = X - (X @ V) @ V.T
    assert np.allclose(Xs - (Xs @ V) @ V.T, x_perp)       # only the subspace coordinates move


def test_zero_dose_identity(world):
    X, theta, probes, _, _ = world
    V = build_basis(probes["W"])
    assert np.array_equal(steer(X, V, probes, encode(90.0, "circular"), 0), X)
    Wt, bt = readout_weights(probes, 1)
    current = X[:1] @ Wt + bt                              # target = the probes' own readout
    assert np.allclose(steer(X[:1], V, probes, current, 1), X[:1], atol=1e-12)


def test_true_value_target_is_identity_for_exact_probes():
    rng = np.random.default_rng(4)
    theta = rng.uniform(0, 360, 400)
    Y = encode(theta, "circular")
    X = np.hstack([Y, rng.standard_normal((400, 6))])      # noise-free code: probes read θ exactly
    W, b = fit_ridge(X, Y, 1e-10)
    probes = {"Q": np.linalg.qr(W)[0], "W": W[None], "b": b[None]}
    Xs = steer(X, build_basis(probes["W"]), probes, Y, 1)
    assert np.allclose(Xs, X, atol=1e-6)


def test_held_out_probe_reads_target_and_true_error_rises(world):
    X, theta, probes, eval_W, eval_b = world
    res = evaluate(X, theta, probes, eval_W, eval_b, "circular", 90.0, np.arange(64) * 360 / 64)
    s = res["single"]
    assert s[0]["mae_to_target"] > 60 and s[0]["mae_to_true"] < 15
    assert s[-1]["mae_to_target"] < 10 and s[-1]["mae_to_true"] > 60
    assert res["all_targets"][-1]["mae_to_target"] < 10
    shift = np.array(res["per_clip_at_K"]["shift"])
    ratio = np.array(res["per_clip_at_K"]["norm_ratio"])
    assert np.all(np.abs(ratio[shift < 15] - 1) < 0.1)
    assert len(res["shift_bins"]["mae_to_target"]) == res["K"] + 1

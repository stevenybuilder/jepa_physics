from functools import partial

import numpy as np
import pytest

from wm.inlp import inlp
from wm.probes import fit_ridge, predict, score
from wm.steer import (build_basis, empirical_p, encode, eval_probe_cv, evaluate, matched_targets, random_nulls,
                      readout_weights, steer)

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


ALL_T = np.arange(64) * 360 / 64


@pytest.fixture(scope="module")
def shrunk_world():
    """Noisy planted code: ridge readouts shrink well below radius 1."""
    rng = np.random.default_rng(0)
    theta = rng.choice(ALL_T, 1500)
    Y = encode(theta, "circular")
    X = copies_code(Y, 3, d=40, seed=0, noise0=0.6, ratio=1.3)
    tr, te, folds = split(len(Y))
    _, Q, W, b = inlp(X[tr], Y[tr], X[te], Y[te], folds, 1.0, partial(score, kind="circular"), "circular")
    eval_W, eval_b, report = eval_probe_cv(X[te], Y[te], "circular")
    return X[tr], theta[tr], X[te], theta[te], {"Q": Q, "W": W, "b": b}, eval_W, eval_b, report


def test_matched_target_recovers_planted_radius():
    rng = np.random.default_rng(1)
    theta = rng.choice(ALL_T, 800)
    rho = 0.4                                              # every clip's code has radius 0.4
    X = np.hstack([rho * encode(theta, "circular"), rng.standard_normal((800, 5))])
    W = np.zeros((7, 2)); W[0, 0] = W[1, 1] = 1.0          # a probe that reads the code exactly
    probes = {"Q": W[:, :2].copy(), "W": W[None], "b": np.zeros((1, 2))}
    t = matched_targets(X, theta, probes, [90.0, 180.0])
    assert np.allclose(t, [[rho, 0.0], [0.0, -rho]], atol=1e-12)


def test_matched_arm_stays_at_data_radius_unit_arm_overshoots(shrunk_world):
    Xtr, th_tr, X, theta, probes, eval_W, eval_b, _ = shrunk_world
    m_all = matched_targets(Xtr, th_tr, probes, ALL_T)
    unit = evaluate(X, theta, probes, eval_W, eval_b, "circular", 90.0, ALL_T)
    matched = evaluate(X, theta, probes, eval_W, eval_b, "circular", 90.0, ALL_T,
                       single_full=m_all[16], all_full=m_all)
    r0 = unit["single"][0]["radius"]["median"]
    assert r0 < 0.85                                       # shrinkage is present
    assert unit["single"][-1]["radius"]["median"] > 0.95   # unit target pushes beyond the data
    assert abs(matched["single"][-1]["radius"]["median"] - r0) < 0.15
    assert abs(matched["single"][-1]["norm_ratio_median"] - 1) < abs(unit["single"][-1]["norm_ratio_median"] - 1)
    assert matched["single"][-1]["mae_to_target"] < 20 and len(matched["per_clip_at_K"]["radius"]) == len(X)


def test_eval_probe_reports_in_sample_and_out_of_fold(shrunk_world):
    report = shrunk_world[-1]
    assert report["out_of_fold"]["r2_fold_mean"] < report["in_sample"]["r2"]
    assert report["alpha"] > 0


def test_empirical_p_resolution():
    assert empirical_p(0.0, np.ones(20)) == 1 / 21 and empirical_p(2.0, np.ones(20)) == 1.0


def test_random_nulls_exclude_planted_and_contain_random(world):
    X, theta, probes, eval_W, eval_b = world
    nulls = random_nulls(X, theta, probes, eval_W, eval_b, "circular", 90.0, n_draws=20)
    last = nulls["rows"][-1]
    assert last["random_orientation"]["empirical_p_to_target"] == 1 / 21      # planted beats every draw
    assert last["random_basis"]["empirical_p_to_target"] == 1 / 21
    assert last["learned"]["mae_to_target"] < last["random_orientation"]["mae_to_target"]["p05"]
    assert len(last["random_basis"]["mae_to_target"]["draws"]) == 20
    # A random "learned" basis is exchangeable with the orientation null: strictly inside the draws.
    rng = np.random.default_rng(7)
    d = X.shape[1]
    W = np.linalg.qr(rng.standard_normal((d, 6)))[0].reshape(d, 3, 2).transpose(1, 0, 2)
    rand = {"Q": np.linalg.qr(np.hstack(list(W)))[0], "W": W, "b": np.zeros((3, 2))}
    for row in random_nulls(X, theta, rand, eval_W, eval_b, "circular", 90.0, n_draws=20)["rows"]:
        assert 1 / 21 < row["random_orientation"]["empirical_p_to_target"] < 1.0


def test_orientation_null_keeps_per_clip_dose(world):
    from wm.steer import steer_delta
    X, theta, probes, _, _ = world
    V = build_basis(probes["W"])
    dc = steer_delta(X, V, probes, encode(90.0, "circular"), len(probes["W"]))
    Q = np.linalg.qr(np.random.default_rng(3).standard_normal(V.shape))[0]
    assert np.allclose(np.linalg.norm(dc @ Q.T, axis=1), np.linalg.norm(dc @ V.T, axis=1))


def test_residualised_basis_equals_paper_raw_stacked_basis(world):
    """C.12 Eq. 8: V_N = QR([W_1ᵀ … W_Nᵀ]) from the raw probe weights. Our probes are stored residualised
    (W_k − Q_<k Q_<kᵀ W_k); the spans must agree up to float error for every N, and steering with the
    full-K basis at n = N must equal steering with the paper's V_N."""
    from wm.inlp import project_out
    rng = np.random.default_rng(0)
    theta_all = rng.choice(np.arange(64) * 360 / 64, 1500)
    Y = encode(theta_all, "circular")
    X = copies_code(Y, 3, d=40, seed=0)
    tr, te, folds = split(len(Y))
    X_te, theta, probes, _, _ = world
    Q, K = probes["Q"], len(probes["W"])
    raw = [fit_ridge(project_out(X[tr], Q[:, :2 * k]), Y[tr], 1.0)[0] for k in range(K)]   # W_k as trained
    for N in range(1, K + 1):
        V_raw = np.linalg.qr(np.hstack(raw[:N]))[0]
        V_res = build_basis(probes["W"][:N])
        assert np.allclose(V_raw @ V_raw.T, V_res @ V_res.T, atol=1e-8)
        assert np.allclose(V_res @ V_res.T, Q[:, :2 * N] @ Q[:, :2 * N].T, atol=1e-8)
        full = steer(X_te, build_basis(probes["W"]), probes, encode(90.0, "circular"), N)
        paper = steer(X_te, V_raw, probes, encode(90.0, "circular"), N)
        assert np.allclose(full, paper, atol=1e-8)


def test_off_target_speed_readout_unmoved_when_codes_orthogonal_moved_when_shared():
    rng = np.random.default_rng(5)
    n = 1500
    theta = rng.choice(ALL_T, n)
    speed = rng.uniform(1, 7, n)
    Yd = encode(theta, "circular")
    tr, te, folds = split(n)
    s_code = ((speed - 4) / 2)[:, None]
    for shared, bound in ((False, 0.1), (True, None)):
        # shared: speed is written on the same axis as the first direction copy's sin
        Z = np.hstack([Yd + 0.1 * rng.standard_normal((n, 2)), Yd + 0.2 * rng.standard_normal((n, 2)),
                       s_code + 0.05 * rng.standard_normal((n, 1)), rng.standard_normal((n, 11))])
        if shared:
            Z[:, 0] += s_code[:, 0]
        X = Z @ np.linalg.qr(np.random.default_rng(0).standard_normal((16, 16)))[0].T
        _, Q, W, b = inlp(X[tr], Yd[tr], X[te], Yd[te], folds, 1.0, partial(score, kind="circular"), "circular")
        probes = {"Q": Q, "W": W, "b": b}
        eW, eb = fit_ridge(X[te], Yd[te], 1.0)
        sW, sb = fit_ridge(X[tr], speed[tr], 1.0)
        off = {"W": sW, "b": sb, "kind": "scalar", "mask": np.ones(te.sum(), bool), "labels": speed[te]}
        res = evaluate(X[te], theta[te], probes, eW, eb, "circular", 90.0, ALL_T[::8], off=off)
        ch = [row["off_target"]["mean_abs_change"] for row in res["single"]]
        assert ch[0] == 0.0 and res["single"][0]["off_target"]["mae_to_true"] < 0.5
        if shared:
            assert max(ch) > 0.5                     # steering direction drags the speed readout
        else:
            assert max(ch) < bound                   # speed readout untouched (m/s)


def test_grouped_folds_and_stratified_halves_keep_identical_clips_together():
    from wm.steer import grouped_folds, stratified_halves
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "e"] * 5)
    labels = np.repeat(np.arange(8), 5)
    groups = np.char.add(groups, labels.astype(str))                     # units nested in labels
    folds = grouped_folds(groups)
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    in_a = stratified_halves(groups, labels)
    for g in np.unique(groups):
        assert len(set(in_a[groups == g])) == 1
    assert 0.3 < in_a.mean() < 0.7 and all(in_a[labels == v].any() or (~in_a[labels == v]).any() for v in range(8))

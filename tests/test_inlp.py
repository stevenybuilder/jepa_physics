from functools import partial

import numpy as np

from wm.inlp import inlp, project_out, random_removal_curve, sawtooth
from wm.probes import score

from synthetic import copies_code, split


def run(Y, n_copies, kind):
    X = copies_code(Y, n_copies, d=40, seed=0)
    tr, te, folds = split(len(Y))
    score_fn = partial(score, kind=kind)
    summary, Q, W, b = inlp(X[tr], Y[tr], X[te], Y[te], folds, 1.0, score_fn, kind, protocols=("nested", "paper"))
    rand = random_removal_curve(X[tr], Y[tr], X[te], Y[te], folds, 1.0, score_fn,
                                [r["dims_removed"] for r in summary["rounds"]], seeds=3)
    return summary, Q, W, rand


def test_scalar_rank_recovered_and_random_removal_decays_slower():
    y = np.random.default_rng(0).uniform(-1.7, 1.7, 1200)[:, None]
    summary, Q, W, rand = run(y, 4, "scalar")
    assert summary["K"] == 4 and summary["dims"] == 4
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)
    last = summary["rounds"][-1]
    assert last["cv_r2"] < 0.05
    assert rand["rows"][-1]["cv_r2"] > 0.8          # random rank-4 removal barely hurts


def test_pair_code_gives_2K_dims():
    theta = np.radians(np.random.default_rng(1).uniform(0, 360, 1200))
    Y = np.stack([np.sin(theta), np.cos(theta)], 1)
    summary, Q, W, rand = run(Y, 3, "circular")
    assert summary["K"] == 3 and summary["dims"] == 6 and W.shape == (3, 40, 2)
    assert summary["K_r2_03"] is not None and summary["K_r2_03"] <= summary["K"]
    assert rand["rows"][-1]["cv_r2"] > 0.8
    st = sawtooth(summary, W)
    assert len(st["consecutive_angle_deg"]) == 2 and all(0 <= a <= 90 for a in st["consecutive_angle_deg"])


def test_project_out_removes_span():
    rng = np.random.default_rng(3)
    Q = np.linalg.qr(rng.standard_normal((10, 3)))[0]
    X = project_out(rng.standard_normal((50, 10)), Q)
    assert np.abs(X @ Q).max() < 1e-12


def test_principal_angles_planted_orthogonal_identical_and_random():
    from wm.inlp import covectors_into, subspace_overlap
    rng = np.random.default_rng(0)
    d = 60
    R = np.linalg.qr(rng.standard_normal((d, d)))[0]
    A, B = R[:, :4], R[:, 4:6]                       # orthogonal codes
    o = subspace_overlap(A, B)
    assert np.allclose(o["angles_deg"], 90, atol=1e-6) and o["overlap"] < 1e-12
    assert abs(o["mean_angle_deg"] - 90) < 1e-6 and abs(o["grassmann_distance_rad"] - np.sqrt(2) * np.pi / 2) < 1e-6
    assert o["overlap_B_from_A"] < 1e-12 and o["random_expectation_B_from_A"] == 2 / d
    half = subspace_overlap(A, np.hstack([A[:, :1], B[:, :1]]))     # B shares one of its two directions with A
    assert abs(half["overlap_A_from_B"] - 0.5) < 1e-10 and abs(half["overlap_B_from_A"] - 0.25) < 1e-10
    assert o["random_expectation"] == 4 / d
    s = subspace_overlap(A, A[:, :2] @ np.array([[2.0, 1.0], [0.5, -1.0]]))   # same span, other basis
    assert np.allclose(s["angles_deg"], 0, atol=1e-5) and abs(s["overlap"] - 1) < 1e-10
    assert s["grassmann_distance_rad"] < 1e-6 and s["mean_angle_deg"] < 1e-5
    lo, hi = o["random_overlap_p05_p95"]
    assert lo < 4 / d < hi
    # covector map: a readout that is identical in raw space maps onto itself
    sA, sB = rng.uniform(0.5, 2, d), rng.uniform(0.5, 2, d)
    w_raw = rng.standard_normal((d, 1))
    mapped = covectors_into(w_raw * sB[:, None], sB, sA)   # w in B coords -> A coords
    assert subspace_overlap(mapped, w_raw * sA[:, None])["min_angle_deg"] < 1e-5


def test_acc15_per_round_sawtooth_on_all_metrics_and_loose_K():
    theta = np.radians(np.random.default_rng(1).uniform(0, 360, 1200))
    Y = np.stack([np.sin(theta), np.cos(theta)], 1)
    summary, Q, W, rand = run(Y, 3, "circular")
    rounds = summary["rounds"]
    assert all(0 <= r["cv_acc15"] <= 1 for r in rounds)
    assert all("test_acc15" in r for r in summary["paper"]["rounds"])
    assert rounds[0]["cv_acc15"] > 0.9 and rounds[-1]["cv_acc15"] < 0.3
    assert "cv_acc15" in rand["rows"][0]
    st = sawtooth(summary, W)
    assert set(st["by_metric"]) == {"cv_r2", "cv_mae", "cv_acc15"}
    assert all(len(v["drops"]) == len(rounds) - 1 for v in st["by_metric"].values())
    assert st["by_metric"]["cv_mae"]["drops"][-1] > 0          # MAE rises as information is removed
    y = np.random.default_rng(0).uniform(-1.7, 1.7, 1200)[:, None]
    s2, *_ = run(y, 4, "scalar")
    assert s2["K_r2_01"] == s2["K_loose"] and s2["K_loose"] <= s2["K"] and "R2 < 0.1" in s2["loose_threshold"]


def test_planted_rank_nested_and_paper_K_and_bug2_fields():
    """Planted rank-r code: nested pooled K = r exactly, every fold's K and the paper-protocol K within ±1."""
    y = np.random.default_rng(7).uniform(-1.7, 1.7, 1200)[:, None]   # seed 0 would equal split()'s test draw
    s, Q, W, _ = run(y, 4, "scalar")
    assert s["K"] == 4 and s["protocol"] == "nested" and s["K_probes"] == 4 and "dims_2K" not in s
    assert all(abs(k - 4) <= 1 for k in s["K_folds"]) and s["K_fold_min"] <= 4 <= s["K_fold_max"]
    assert len(s["alpha_folds"]) == 5 and "fold's training rows" in s["alpha_folds_source"] and s["alpha"] == 1.0
    assert abs(s["paper"]["K"] - 4) <= 1 and s["paper"]["protocol"] == "paper" and "test" in s["paper"]["test_read"]
    assert all("test_r2" in r and "cv_r2" not in r for r in s["paper"]["rounds"])
    theta = np.radians(np.random.default_rng(1).uniform(0, 360, 1200))
    s, Q, W, rand = run(np.stack([np.sin(theta), np.cos(theta)], 1), 3, "circular")
    assert s["K"] == 3 and abs(s["paper"]["K"] - 3) <= 1 and all(abs(k - 3) <= 1 for k in s["K_folds"])
    assert s["K_probes"] == 3 and s["dims_2K"] == 6 == s["dims"] and s["paper"]["dims_2K"] == 2 * s["paper"]["K"]
    assert Q.shape == (40, 6) and np.allclose(Q.T @ Q, np.eye(6), atol=1e-8)
    row = rand["rows"][-1]
    assert {"cv_r2", "test_r2", "cv_acc15", "test_acc15"} <= set(row) and rand["protocols"]["cv_*"] == "nested"


def test_nested_curve_never_sees_the_scored_folds_labels():
    """Permuting the held-out fold's labels in the labels the removals are fit on leaves that fold's nested curve
    unchanged; the old (leaky) path, removals fit on all train rows, does change."""
    from wm.inlp import nested_curve, probe_sequence, ridge_fit
    from wm.probes import fit_ridge, predict
    theta = np.radians(np.random.default_rng(1).uniform(0, 360, 1200))
    Y = np.stack([np.sin(theta), np.cos(theta)], 1)
    X = copies_code(Y, 3, d=40, seed=0)
    tr, te, folds = split(len(Y))
    X, Y = X[tr], Y[tr]
    sf = partial(score, kind="circular")
    val = folds == folds.min()
    Yperm = Y.copy()
    Yperm[val] = Y[val][np.random.default_rng(5).permutation(val.sum())]
    base = nested_curve(X, Y, folds, 1.0, sf, "circular")
    perm = nested_curve(X, Y, folds, 1.0, sf, "circular", Yfit=Yperm)
    n = min(len(base["rounds"]), len(perm["rounds"]))
    assert n >= 3
    assert np.allclose([r["fold_r2"][0] for r in base["rounds"][:n]], [r["fold_r2"][0] for r in perm["rounds"][:n]],
                       atol=1e-10)

    def leaky(Yfit, rounds=3):   # the removed code: Q_k fit on all train rows, the scored fold projected through it
        seq, out = probe_sequence(X, Yfit, ridge_fit(1.0)), []
        Q = np.zeros((40, 0))
        for _ in range(rounds):
            Xk = X - X @ Q @ Q.T
            W, b = fit_ridge(Xk[~val], Y[~val], 1.0)
            out.append(sf(Y[val], predict(Xk[val], W, b))["r2"])
            _, Wk, *_ = next(seq)
            Q = np.hstack([Q, np.linalg.qr(Wk)[0]])
        return np.array(out)
    assert np.abs(leaky(Y) - leaky(Yperm)).max() > 1e-3


def test_isolated_dips():
    from wm.inlp import isolated_dips
    assert isolated_dips([0.9, 0.8, 0.3, 0.8, 0.7, 0.2, 0.1]) == [3]


def test_adam_sequence_logs_and_flags_untrained_rounds(monkeypatch):
    from wm.inlp import adam_sequence
    theta = np.radians(np.random.default_rng(1).uniform(0, 360, 1200))
    Y = np.stack([np.sin(theta), np.cos(theta)], 1)
    X = copies_code(Y, 3, d=40, seed=0)
    tr, te, _ = split(len(Y))
    s, W = adam_sequence(X[tr], Y[tr], X[te], Y[te], partial(score, kind="circular"), "circular", max_rounds=12,
                         batch=64)
    r = s["rounds"]
    assert r[0]["test_acc15"] > 0.5 and not r[0]["failed_to_train"] and W.shape == (len(r), 40, 2)
    assert {"W_norm", "W_move_from_init", "train_mse", "train_r2", "train_acc15", "test_r2"} <= set(r[0])
    assert s["recipe"]["lr"] == 1e-3 and s["recipe"]["weight_decay"] == 1e-4 and s["recipe"]["epochs_used"] == 100
    assert s["K_first"] <= s["K_patience"] and "dips_vs_failed" in s["sawtooth"]
    # a round whose probe never trains (round 2 returns its init): flagged, scores low, removes ~nothing, and the
    # next round recovers -> an isolated dip at round 2 that the dips-vs-failed table attributes to the failure
    import wm.inlp as inlp_mod
    from wm.adam_probe import init_linear
    real, calls = inlp_mod.fit_adam, {"n": 0}

    def flaky(X, Yf, lr, wd, epochs, batch, seed, decoupled):
        calls["n"] += 1
        return init_linear(X.shape[1], Yf.shape[1], seed)[:2] if calls["n"] == 2 else real(X, Yf, lr, wd, epochs, batch, seed, decoupled)
    monkeypatch.setattr(inlp_mod, "fit_adam", flaky)
    s2, _ = adam_sequence(X[tr], Y[tr], X[te], Y[te], partial(score, kind="circular"), "circular", max_rounds=8,
                          batch=64, patience=2)
    r2 = s2["rounds"]
    assert r2[1]["failed_to_train"] and not r2[0]["failed_to_train"] and not r2[2]["failed_to_train"]
    assert r2[2]["test_acc15"] > r2[1]["test_acc15"] + 0.3
    dv = s2["sawtooth"]["dips_vs_failed"]
    assert 2 in dv["isolated_dips"] and dv["dips_that_failed"] == [2] and dv["p_dip_given_failed"] == 1.0
    assert r2[1]["at_chance_on_train"] and not r2[0]["at_chance_on_train"] and "movement only" in s2["failed_rule"]
    assert r2[-1]["at_chance_on_train"] and not r2[-1]["failed_to_train"]   # info exhausted, but the probe moved
    assert s2["K_first"] == 1 and s2["K_patience"] == 3     # the paper's first-at-chance rule stops at the tooth


def test_step_layers_paper_layer_is_point_9_with_alt_8():
    from wm.provenance import PAPER_LAYER, PAPER_LAYER_ALT, step_layers
    sweep = {"availability": {"peak": 20, "onset": 9}}
    assert PAPER_LAYER == 9 and PAPER_LAYER_ALT == 8
    assert step_layers(sweep, "all") == {20: "peak", 9: "onset+paper_layer", 8: "paper_layer_alt"}
    assert step_layers(sweep, "all", alt=False) == {20: "peak", 9: "onset+paper_layer"}
    assert step_layers(sweep, "paper", alt=False) == {9: "paper_layer"} and step_layers(sweep, "peak") == {20: "peak"}

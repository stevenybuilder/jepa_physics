from functools import partial

import numpy as np

from wm.inlp import inlp, project_out, random_removal_curve, sawtooth
from wm.probes import score

from synthetic import copies_code, split


def run(Y, n_copies, kind):
    X = copies_code(Y, n_copies, d=40, seed=0)
    tr, te, folds = split(len(Y))
    score_fn = partial(score, kind=kind)
    summary, Q, W, b = inlp(X[tr], Y[tr], X[te], Y[te], folds, 1.0, score_fn, kind)
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
    assert o["random_expectation"] == 4 / d
    s = subspace_overlap(A, A[:, :2] @ np.array([[2.0, 1.0], [0.5, -1.0]]))   # same span, other basis
    assert np.allclose(s["angles_deg"], 0, atol=1e-5) and abs(s["overlap"] - 1) < 1e-10
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
    assert all(0 <= r["cv_acc15"] <= 1 and "test_acc15" in r for r in rounds)
    assert rounds[0]["cv_acc15"] > 0.9 and rounds[-1]["cv_acc15"] < 0.3
    assert "cv_acc15" in rand["rows"][0]
    st = sawtooth(summary, W)
    assert set(st["by_metric"]) == {"cv_r2", "cv_mae", "cv_acc15"}
    assert all(len(v["drops"]) == len(rounds) - 1 for v in st["by_metric"].values())
    assert st["by_metric"]["cv_mae"]["drops"][-1] > 0          # MAE rises as information is removed
    y = np.random.default_rng(0).uniform(-1.7, 1.7, 1200)[:, None]
    s2, *_ = run(y, 4, "scalar")
    assert s2["K_r2_01"] == s2["K_loose"] and s2["K_loose"] <= s2["K"] and "R2 < 0.1" in s2["loose_threshold"]

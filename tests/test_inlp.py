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

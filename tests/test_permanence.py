"""wm.permanence on a tiny synthetic time pool: a persistent code (direction still written into absent-step tokens)
vs a vanishing one (absent-step tokens carry only noise)."""
import numpy as np

from wm.permanence import absence, absence_counts, run

N, T, D = 400, 8, 24


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    Y = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    trailing = np.zeros(N, int)
    lose = rng.random(N) < 0.3
    trailing[lose] = rng.integers(1, 4, lose.sum())
    mask = np.ones((N, T, 4, 4), bool)
    for i in range(N):
        if trailing[i]:
            mask[i, T - trailing[i]:] = False
    basis = np.linalg.qr(rng.standard_normal((D, 2)))[0]
    code = (3 * Y) @ basis.T                                              # [N, D]
    vis = mask.any(axis=(2, 3))
    tp = np.zeros((N, 2, T, D), np.float32)
    for p, persistent in ((0, True), (1, False)):
        x = np.repeat(code[:, None, :], T, axis=1) + 0.3 * rng.standard_normal((N, T, D))
        if not persistent:
            x[~vis] = rng.standard_normal(((~vis).sum(), D))
        tp[:, p] = x
    order = rng.permutation(N)
    te, tr = np.sort(order[:100]), np.sort(order[100:])
    folds = rng.integers(0, 5, len(tr))
    return tp, mask, Y, tr, te, folds, trailing


def test_absence_counts():
    tp, mask, Y, tr, te, folds, trailing = synthetic()
    vis, tr_count, last_vis = absence(mask)
    assert np.array_equal(tr_count, trailing)
    assert np.all(last_vis == T - 1 - trailing)
    c = absence_counts(vis, tr_count, tr, te)
    assert c["any_absent_step"] == int((trailing > 0).sum())
    assert c["all"]["at_least_k"][1] == c["any_absent_step"]
    assert sum(c["all"]["exactly_k"].values()) == c["any_absent_step"]


def test_persistent_vs_vanishing():
    tp, mask, Y, tr, te, folds, _ = synthetic()
    out = run(tp, mask, Y, "circular", tr, te, folds, points=(0, 1), n_boot=50, n_perm=100, log=lambda *a: None)
    keep, vanish = out["points"]
    for r in (keep, vanish):
        assert r["test"]["visible"]["mae"] < 10                                # both decode on visible steps
    assert keep["test"]["absent"]["mae"] < 10                                 # persistent: still decodable
    assert keep["pooled"]["absent_shuffled"]["p_mae"] < 0.05
    assert vanish["pooled"]["absent"]["mae"] > 45                             # vanishing: near chance (90 deg)
    assert vanish["pooled"]["absent_shuffled"]["p_mae"] > 0.05
    per_keep, per_vanish = keep["pooled"]["persistence"], vanish["pooled"]["persistence"]
    assert per_keep["0"]["mae"] < 10 and per_vanish["0"]["mae"] < 10          # lag 0 = last visible step
    assert all(per_keep[k]["mae"] < 10 for k in per_keep)
    assert all(per_vanish[k]["mae"] > 45 for k in per_vanish if k != "0")
    lo, hi = keep["test"]["absent"]["mae_ci"]
    assert lo <= keep["test"]["absent"]["mae"] <= hi


def test_scalar_target():
    tp, mask, Y, tr, te, folds, _ = synthetic()
    y = Y[:, :1] * 2 + 4                                                      # scalar stand-in for speed
    out = run(tp, mask, y, "scalar", tr, te, folds, points=(0, 1), n_boot=20, n_perm=50, log=lambda *a: None)
    keep, vanish = out["points"]
    assert keep["pooled"]["absent"]["r2"] > 0.8
    assert vanish["pooled"]["absent"]["r2"] < 0.3

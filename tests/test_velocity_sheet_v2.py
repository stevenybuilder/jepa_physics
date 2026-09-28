import numpy as np

from wm import timeman as tm
from wm import vsheet as vs


def cone_sheet(k=6, a=0.3, b=1.0):
    """Exact cone in R^k: axis along e0 with height v, ring in (e1, e2) of radius a + b v."""
    def P(th, v):
        th, v = np.radians(np.asarray(th, float)), np.asarray(v, float)
        out = np.zeros((len(th), k))
        out[:, 0] = v
        out[:, 1] = (a + b * v) * np.cos(th)
        out[:, 2] = (a + b * v) * np.sin(th)
        return out
    return P


def test_holdout_designs():
    d = np.repeat(np.arange(16), 8)
    s = np.tile(np.arange(8), 16)
    h, t, hd, hs = vs.holdout(d, s, "block2", 0)
    assert h.sum() == 4 and len(t) == 4 and hd == [1, 2] and hs == [2, 3]
    h, t, _, _ = vs.holdout(d, s, "block3", 1)
    assert h.sum() == 9 and len(t) == 9
    h, t, hd, hs = vs.holdout(d, s, "cross", 0)
    assert h.sum() == 2 * 8 + 2 * 16 - 4 and len(t) == 4
    # in the cross design no kept cell shares the target's direction bin or speed bin
    assert not (~h & np.isin(d, hd)).any() and not (~h & np.isin(s, hs)).any()


def test_cone_frame_recovers_axis_and_radius():
    P = cone_sheet()
    f = vs.ConeFrame(P)
    v = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(f.center(v)[:, 0], v, atol=1e-12)
    np.testing.assert_allclose(f.center(v)[:, 1:], 0, atol=1e-12)
    np.testing.assert_allclose(f.radius(v), 0.3 + v, atol=1e-12)
    u = f.unit(np.array([90.0]), np.array([2.0]))
    np.testing.assert_allclose(u[0, :3], [0, 0, 1], atol=1e-12)


def test_factorial_full_equals_sheet_difference_and_ablations():
    P = cone_sheet()
    f = vs.ConeFrame(P)
    ths, vs_, tht, vt = np.array([0.0, 30.0]), np.array([0.5, 1.0]), np.array([90.0, 200.0]), np.array([2.0, 1.5])
    full = vs.factorial_delta(f, ths, vs_, tht, vt)
    np.testing.assert_allclose(full, P(tht, vt) - P(ths, vs_), atol=1e-10)
    # radius frozen: the ring part keeps the source radius, the axis part still moves
    rf = vs.factorial_delta(f, ths, vs_, tht, vt, radius=False)
    exp = P(tht, vs_) - P(ths, vs_)
    exp[:, 0] += vt - vs_
    np.testing.assert_allclose(rf, exp, atol=1e-10)
    # on an exact cone the ring shape does not change with speed, so shape-frozen == full
    np.testing.assert_allclose(vs.factorial_delta(f, ths, vs_, tht, vt, shape=False), full, atol=1e-10)


def test_rescale_rows_and_paths():
    d = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = vs.rescale_rows(d, np.array([10.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [10, 1])
    path = np.stack([d * s for s in (0.0, 0.5, 1.0)], 1)
    out = vs.rescale_rows(path, np.array([10.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(out[:, -1], axis=1), [10, 1])
    np.testing.assert_allclose(out[:, 1], out[:, -1] / 2)


def test_min_norm_probe_edit_hits_target_with_min_norm():
    rng = np.random.default_rng(0)
    W, b = rng.normal(size=(3, 10)), rng.normal(size=3)
    z0, y = rng.normal(size=(5, 10)), rng.normal(size=(5, 3))
    dl = vs.min_norm_probe_edit(W, b, z0, y)
    np.testing.assert_allclose((z0 + dl) @ W.T + b, y, atol=1e-10)
    # min norm: the edit lies in the row space of W
    proj = dl @ np.linalg.pinv(W) @ W
    np.testing.assert_allclose(proj, dl, atol=1e-10)


def test_idw_cell_estimate():
    cd, cs = np.array([0, 2, 15, 5]), np.array([3, 3, 3, 3])
    C = np.array([[0.0], [2.0], [-1.0], [100.0]])
    est = vs.idw_cell_estimate(cd, cs, C, 1, 3, k=2)
    np.testing.assert_allclose(est, [1.0])            # bins 0 and 2 at distance 1 each
    est = vs.idw_cell_estimate(cd, cs, C, 0, 3, k=2)  # bin 15 wraps to distance 1, bin 0 distance 0 dominates
    assert abs(est[0]) < 1e-6


def test_clip_boot_matches_cluster_bootstrap_mean():
    rng = np.random.default_rng(1)
    v, g = rng.normal(size=200), rng.integers(0, 30, 200)
    a, b = vs.clip_boot(v, g), tm.cluster_bootstrap(v, g)
    assert abs(a["mean"] - b["mean"]) < 1e-12 and a["n_clips"] == b["n_clips"]
    assert abs(a["ci95"][0] - b["ci95"][0]) < 0.1 and abs(a["ci95"][1] - b["ci95"][1]) < 0.1

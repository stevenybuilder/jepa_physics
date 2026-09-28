import numpy as np

from wm import makelov as mk


def test_raw_directions_unit_and_scaling():
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((10, 4)))[0]
    sd = np.ones(10) * 3.0
    R = mk.raw_directions(Q, sd, 3)
    assert R.shape == (3, 10)
    np.testing.assert_allclose(np.linalg.norm(R, axis=1), 1.0)
    np.testing.assert_allclose(R, Q[:, :3].T, atol=1e-12)          # uniform sd leaves the direction unchanged
    sd2 = np.r_[np.ones(5), np.full(5, 10.0)]
    R2 = mk.raw_directions(np.eye(10)[:, [0, 7]], sd2)
    np.testing.assert_allclose(R2, np.eye(10)[[0, 7]])


def test_scaled_edits_norm():
    d = np.eye(3)[:2]
    E = mk.scaled_edits(d, [2.0, 5.0])
    np.testing.assert_allclose(np.linalg.norm(E, axis=2), [[2, 2], [5, 5]])


def test_wrap_deg():
    np.testing.assert_allclose(mk.wrap_deg([10, 190, -190, 350, 180]), [10, -170, 170, -10, 180])


def test_order_spearman_signs():
    assert mk.order_spearman(np.arange(8)[::-1]) == -1.0
    assert mk.order_spearman(np.arange(8)) == 1.0


def test_carrier_bootstrap_ci_contains_rho():
    rng = np.random.default_rng(1)
    M = np.linspace(5, 1, 12)[None] + rng.normal(0, 0.5, (16, 12))
    r = mk.carrier_bootstrap_spearman(M, n_boot=300)
    assert r["rho"] < -0.8 and r["ci95"][0] <= r["rho"] <= r["ci95"][1]


def test_permutation_p():
    assert mk.permutation_p(np.arange(20)[::-1], n_perm=500) < 0.01


def test_arc_clip_bootstrap():
    v = [np.array([1.0, 1.0, 3.0, 3.0]), np.array([10.0, 10.0])]
    g = [np.array([0, 0, 1, 1]), np.array([5, 6])]
    r = mk.arc_clip_bootstrap(v, g, n_boot=500)
    assert r["mean"] == 6.0 and r["n_arcs"] == 2
    assert 5.5 <= r["ci95"][0] <= 6.0 <= r["ci95"][1] <= 6.5
    same = mk.arc_clip_bootstrap([np.ones(4)], [np.arange(4)], n_boot=50)
    assert same["ci95"] == [1.0, 1.0]

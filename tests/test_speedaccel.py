import numpy as np

from wm import speedaccel as sa


def test_line_vector_recovers_planted_slope():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 4, 500)
    u = rng.standard_normal(16)
    X = y[:, None] * u + 0.01 * rng.standard_normal((500, 16))
    assert np.allclose(sa.line_vector(X, y), u, atol=0.01)


def test_joint_line_vectors_separate_collinear_parts():
    rng = np.random.default_rng(1)
    v, a = rng.uniform(0, 4, 800), rng.uniform(0, 10, 800)
    bv, ba = rng.standard_normal(8), rng.standard_normal(8)
    X = v[:, None] * bv + a[:, None] * ba
    B = sa.joint_line_vectors(X, np.column_stack([v, a]))
    assert np.allclose(B[0], bv) and np.allclose(B[1], ba)


def test_rate_vs_displacement_pure_cases():
    tau = (2 * np.arange(8) + 0.5) / 24
    b = np.ones(5)
    r = sa.rate_vs_displacement(np.tile(b, (8, 1)), tau)
    assert abs(r["rate_frac"] - 1) < 1e-9 and r["growth_frac"] < 1e-9
    d = np.arange(5.0)
    U = tau[:, None] * d[None]
    r = sa.rate_vs_displacement(U, tau)
    assert abs(r["rate_frac"] + r["growth_frac"] + r["resid_frac"] - 1) < 1e-9
    assert r["resid_frac"] < 1e-9 and r["growth_frac"] > 0.1


def test_disattenuated_cos_and_cos():
    assert abs(sa.cos([1, 0], [1, 1]) - np.sqrt(0.5)) < 1e-9
    assert abs(sa.disattenuated_cos(0.4, 0.8, 0.5) - 0.4 / np.sqrt(0.4)) < 1e-12
    assert np.isnan(sa.disattenuated_cos(0.4, -0.1, 0.5))


def test_spacing_r2_linear_vs_log():
    v = np.linspace(0.25, 4, 64)
    assert sa.spacing_r2(3 * v + 1, v)["r2_linear"] > 0.999
    s = sa.spacing_r2(np.log(v), v)
    assert s["r2_log"] > 0.999 and s["r2_linear"] < 0.95


def test_knn_ratio_and_nearest_real():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((300, 4))
    assert 0.5 < np.median(sa.knn_ratio(R[:50] + 1e-3, R, k=5)) < 1.5
    assert np.median(sa.knn_ratio(R[:50] + 20, R, k=5)) > 5
    x0, t = np.zeros((1, 3)), np.ones((1, 3))
    assert np.allclose(sa.nearest_real_R(t, x0, t), 1) and np.allclose(sa.nearest_real_R(x0, x0, t), 0)


def test_along_line_gain_and_bins():
    u = np.array([2.0, 0, 0])
    g, f = sa.along_line_gain(np.array([[4.0, 0, 0], [2.0, 2.0, 0]]), u, np.array([2.0, 1.0]))
    assert np.allclose(g, [1, 1]) and np.allclose(f, [1, 0.5])
    assert list(sa.angle_bins([0, 44, 46, 350], 8)) == [0, 1, 1, 0]
    assert np.allclose(sa.circ_diff_deg(350, 10), -20)

import numpy as np

from wm.motiongeom import (COORDS, captured_variance, circ_err_deg, coords_to_state, fit_linear_encoding,
                           fourier_features, holm, orth, overlap, principal_angles_deg, projector_strength,
                           residualise, union_dimension)


def test_principal_angles_and_overlap():
    D = 10
    e = np.eye(D)
    Q1 = e[:, [0, 1]]
    Q2 = orth(np.column_stack([e[:, 0], np.cos(0.3) * e[:, 1] + np.sin(0.3) * e[:, 2]]))
    a = principal_angles_deg(Q1, Q2)
    assert np.allclose(a, [0.0, np.degrees(0.3)], atol=1e-8)
    assert np.isclose(overlap(Q1, e[:, [2, 3]]), 0.0)
    assert np.isclose(overlap(Q1, Q1), 1.0)


def test_union_dimension_orthogonal_vs_shared():
    e = np.eye(6)
    u = union_dimension([e[:, [0, 1]], e[:, [2]], e[:, [3]]])
    assert u["sum_dims"] == 4 and np.isclose(u["participation_ratio"], 4.0) and u["rank_s2_gt_0.5"] == 4
    s = union_dimension([e[:, [0]], e[:, [0]]])
    assert np.isclose(s["participation_ratio"], 1.0) and s["rank_s2_gt_0.5"] == 1


def test_fit_linear_encoding_recovers_map():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 3))
    B = rng.normal(size=(3, 7))
    X = 2.0 + F @ B
    a, Bh = fit_linear_encoding(X, F)
    assert np.allclose(Bh, B, atol=1e-8) and np.allclose(a, 2.0, atol=1e-8)


def test_residualise_removes_component_and_keeps_norm():
    e = np.eye(4)
    d = np.array([[1.0, 1.0, 0.0, 0.0]])
    r = residualise(d, e[:, [0]])
    assert np.isclose(r[0, 0], 0.0) and np.isclose(np.linalg.norm(r), np.linalg.norm(d))
    r2 = residualise(d, e[:, [0]], renorm=False)
    assert np.allclose(r2, [[0, 1, 0, 0]])


def test_projector_strength_and_captured_variance():
    e = np.eye(3)
    X = np.array([[3.0, 4.0, 0.0], [-3.0, -4.0, 0.0]])
    assert np.allclose(projector_strength(X, e[:, [0, 1]]), [5.0, 5.0])
    assert np.isclose(captured_variance(X, e[:, [0]]), 9 / 25)


def test_coords_roundtrip():
    th = np.array([10.0, 200.0, 359.0])
    v = np.array([0.5, 1.0, 3.0])
    for name, f in COORDS.items():
        t2, v2 = coords_to_state(name, f(th, v))
        assert np.allclose(circ_err_deg(t2, th), 0, atol=1e-8), name
        assert np.allclose(v2, v), name
    assert fourier_features(th, 2).shape == (3, 4)


def test_holm():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_svd_ridge_matches_sklearn_ridge():
    from sklearn.linear_model import Ridge
    from wm.motiongeom import SVDRidge
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 20))
    Y = X[:, :2] @ rng.normal(size=(2, 3)) + 0.1 * rng.normal(size=(80, 3))
    m = SVDRidge(X, alphas=[3.0]).fit(Y)
    Z = (X - X.mean(0)) / X.std(0)
    sk = Ridge(alpha=3.0).fit(Z, Y)
    assert np.allclose(m.predict(X), sk.predict(Z), atol=1e-8)
    assert m.predict(X[:, :]).shape == (80, 3) and SVDRidge(X).fit(Y[:, 0]).predict(X).shape == (80,)


def test_oblique_residualise_zeroes_decoder_and_moves_along_encoder():
    from wm.motiongeom import oblique_residualise
    rng = np.random.default_rng(1)
    B = rng.normal(size=(6, 2))
    W = rng.normal(size=(2, 6))
    d = rng.normal(size=(5, 6))
    r = oblique_residualise(d, B, W, renorm=False)
    assert np.allclose(r @ W.T, 0, atol=1e-10)
    diff = d - r
    coef, *_ = np.linalg.lstsq(B, diff.T, rcond=None)
    assert np.allclose(B @ coef, diff.T, atol=1e-10)

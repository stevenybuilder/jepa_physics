"""Synthetic tests for wm.manifold and wm.geometry_checks: a planted ring (direction) and a planted line (speed)."""
import numpy as np
import pytest

from wm import geometry_checks as gc
from wm import manifold as mf

D, R, NOISE, PER_VALUE = 60, 5.0, 0.3, 20


def ring_data(seed=0, rotate=True):
    """64 directions on a circle of radius R in two directions of a D-dim space, plus isotropic noise.
    With rotate=True the ring plane is a random 2-D plane, not two coordinate axes."""
    rng = np.random.default_rng(seed)
    theta = np.repeat(np.arange(64) * 360.0 / 64, PER_VALUE)
    Q = np.linalg.qr(rng.standard_normal((D, 2)))[0] if rotate else np.eye(D)[:, :2]
    r = np.radians(theta)
    X = R * np.stack([np.cos(r), np.sin(r)], 1) @ Q.T + 3.0 + NOISE * rng.standard_normal((len(theta), D))
    return X, theta, Q


def line_data(seed=0):
    """64 speeds 0.25..4.0 on a straight line, evenly spaced in activation space, plus noise."""
    rng = np.random.default_rng(seed)
    v = np.repeat(np.linspace(0.25, 4.0, 64), PER_VALUE)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    X = np.outer(3.0 * v, u) + NOISE * rng.standard_normal((len(v), D))
    return X, v


@pytest.fixture(scope="module")
def ring():
    X, theta, Q = ring_data()
    pca = mf.fit_pca(X, k=16)
    cent = mf.centroids(pca.project(X), theta)
    return X, theta, Q, pca, cent, mf.fit_curve(cent, periodic=True)


def test_pca_roundtrip_and_complement():
    X, _, _ = ring_data()
    pca = mf.fit_pca(X[:800], k=8)
    np.testing.assert_allclose(pca.lift(pca.project(X)) + pca.complement(X), X, atol=1e-10)
    assert np.abs(pca.complement(X) @ pca.components.T).max() < 1e-8


def test_centroids_counts_and_spread(ring):
    _, _, _, _, cent, _ = ring
    assert len(cent["values"]) == 64 and (cent["count"] == PER_VALUE).all()
    # 16 PCA dims of N(0, 0.3^2) noise: spread ~ 0.3 * sqrt(16) = 1.2 (a little less after projection)
    assert 0.8 < np.median(cent["spread"]) < 1.4


def test_unsupervised_angle_matches_truth(ring):
    _, _, _, _, cent, curve = ring
    assert curve.coord_source == "unsupervised_angle"
    cmp = mf.compare_angles(mf.unsupervised_angle(cent["C"]), cent["values"])
    assert abs(cmp["circular_corr"]) > 0.99 and cmp["max_dev_deg"] < 5 and cmp["order_preserved"]
    cmp2 = mf.compare_angles(mf.unsupervised_angle(cent["C"], plane="centroid"), cent["values"])
    assert cmp2["max_dev_deg"] < 5


def test_scrambled_angle_is_flagged():
    rng = np.random.default_rng(1)
    cmp = mf.compare_angles(rng.uniform(0, 2 * np.pi, 64), np.arange(64) * 360 / 64)
    assert abs(cmp["circular_corr"]) < 0.4 and not cmp["order_preserved"]


def test_periodic_spline_closes(ring):
    curve = ring[-1]
    t0 = curve.coords[0]
    np.testing.assert_allclose(curve(t0), curve(t0 + 2 * np.pi), atol=1e-9)
    np.testing.assert_allclose(curve.spline(t0, 1), curve.spline(t0 + 2 * np.pi, 1), atol=1e-8)
    np.testing.assert_allclose(curve(curve.coords), curve.points, atol=1e-9)   # interpolates the centroids


def test_linear_cuts_the_centre_manifold_keeps_radius(ring):
    X, theta, Q, pca, cent, curve = ring
    full_c = {v: X[theta == v].mean(0) for v in (0.0, 180.0)}
    centre = X.mean(0)
    x = full_c[0.0]
    lin = mf.linear_path(x, full_c[0.0], full_c[180.0], K=21)
    man = mf.steer_to_value(x, pca, curve, 180.0, K=21, source_value=0.0)
    radius = lambda W: np.linalg.norm((W - centre) @ Q, axis=1)
    assert radius(lin)[10] < 0.1 * R                        # the straight line passes through the centre
    assert np.all(np.abs(radius(man) - R) < 0.1 * R)         # the spline path stays on the ring
    e_lin = mf.off_manifold_energy(lin, pca, curve, X)
    e_man = mf.off_manifold_energy(man, pca, curve, X)
    assert e_lin["to_curve"].mean() > 5 * e_man["to_curve"].mean()
    assert e_lin["to_nearest_real"].max() > e_man["to_nearest_real"].max()
    # both land on the target value
    ang = lambda w: np.degrees(np.arctan2(*((w - centre) @ Q)[::-1])) % 360
    assert abs(ang(man[-1]) - 180) < 3 and abs(ang(lin[-1]) - 180) < 3


def test_zero_length_paths_return_x(ring):
    X, _, _, pca, _, curve = ring
    x = X[5]
    np.testing.assert_allclose(mf.manifold_path(x, pca, curve, 1.0, 1.0, K=7), np.tile(x, (7, 1)), atol=1e-10)
    np.testing.assert_allclose(mf.linear_path(x, X[0], X[0], K=7), np.tile(x, (7, 1)))
    Xb = mf.manifold_path(X[:3], pca, curve, np.array([0.1, 1, 2]), np.array([0.1, 1, 2]), K=4)
    np.testing.assert_allclose(Xb, np.repeat(X[:3, None], 4, 1), atol=1e-10)


def test_manifold_path_keeps_complement(ring):
    X, _, _, pca, _, curve = ring
    W = mf.manifold_path(X[0], pca, curve, 0.0, 2.0, K=5, mode="replace")
    np.testing.assert_allclose(pca.complement(W), np.tile(pca.complement(X[0]), (5, 1)), atol=1e-9)
    np.testing.assert_allclose(pca.project(W), curve(np.linspace(0, 2, 5)), atol=1e-9)


def test_heldout_knots_reconstructed(ring):
    X, theta, Q, pca, cent, _ = ring
    keep = ~mf.heldout_mask(cent["values"], periodic=True)
    assert keep.sum() == 48
    curve = mf.fit_curve(cent, periodic=True, keep=keep)
    held = cent["values"][~keep]
    spline_pts = curve(curve.coord_of_value(held))
    line_pts = mf.piecewise_linear_point(curve, held)
    err_s = np.linalg.norm(spline_pts - cent["C"][~keep], axis=1).mean()
    err_l = np.linalg.norm(line_pts - cent["C"][~keep], axis=1).mean()
    noise = np.median(cent["spread"] / np.sqrt(cent["count"]))
    assert err_s < 2 * noise and err_l < 2 * noise          # dense knots: both at the noise floor
    # steering to an unseen value lands on the unseen centroid
    x = X[theta == 0.0].mean(0)
    x_star = mf.steer_to_value(x, pca, curve, held[5], source_value=0.0)[-1]
    assert np.linalg.norm(pca.project(x_star) - cent["C"][~keep][5]) < 3 * noise


def test_straight_line_gives_cubic_no_advantage():
    X, v = line_data()
    pca = mf.fit_pca(X, k=16)
    cent = mf.centroids(pca.project(X), v)
    curve = mf.fit_curve(cent, periodic=False)
    noise = cent["spread"] / np.sqrt(cent["count"])
    loo = gc.loo_reconstruction(curve, noise)
    assert loo["mean_log_ratio"] > -0.1                   # cubic no better than the line
    assert loo["line_err_over_noise"] < 2.5                # line error is at the noise floor
    far = gc.loo_reconstruction(curve, noise, stride=16)   # even with sparse knots the cubic gains nothing
    assert far["mean_log_ratio"] > -0.1 and far["line_err_over_noise"] < 2.5
    sc = gc.spread_vs_curvature(curve, cent["spread"], cent["count"])
    assert sc["bend_over_noise"] < 2.5
    ks = gc.knot_spacing(curve)
    assert ks["better"] == "linear" and ks["r2_linear"] > 0.99


def test_ring_checks_say_curved(ring):
    _, _, _, _, cent, curve = ring
    noise = gc.align(cent["spread"] / np.sqrt(cent["count"]), cent["values"], curve.values)
    near = gc.loo_reconstruction(curve, noise, stride=1)    # one spacing: the bend (0.5% of R) is below noise
    assert near["line_err_over_noise"] < 2.5
    far = gc.loo_reconstruction(curve, noise, stride=16)    # knots 90 degrees apart: the chord cuts inside the ring
    assert far["winner"] == "cubic" and far["mean_err_line"] > 2 * far["mean_err_cubic"]
    sc = gc.spread_vs_curvature(curve, gc.align(cent["spread"], cent["values"], curve.values),
                                gc.align(cent["count"], cent["values"], curve.values))
    assert sc["median_bend"] == pytest.approx(R * (1 - np.cos(2 * np.pi / 64)), abs=3 * sc["median_centroid_noise"])


def test_bf16_rounding():
    x = np.array([1.0, 1.0 + 2 ** -9, 1.0 + 3 * 2 ** -9, -2.5, 1e-3], np.float32)
    torch = pytest.importorskip("torch")
    ref = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
    np.testing.assert_array_equal(gc.bf16_round(x), ref)
    X, theta, _ = ring_data()
    pc = gc.precision_check(X, theta, periodic=True, k=16, stride=16)
    assert pc["winner_same"] and pc["fp32"]["winner"] == "cubic"


def test_shared_delta_fraction():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(D)
    assert gc.shared_delta_fraction(np.tile(d, (50, 1))) == pytest.approx(1.0)
    assert gc.shared_delta_fraction(rng.standard_normal((50, D))) < 0.1


def test_participation_and_angles(ring):
    X, theta, Q, pca, cent, _ = ring
    pr = gc.participation_check(pca.project(X), theta)
    assert 1.5 < pr["pr_centroids"] < 2.5 and pr["pr_residuals"] > 10
    C_full = np.array([X[theta == v].mean(0) for v in cent["values"]])
    assert gc.principal_angles_to_centroid_plane(Q, C_full)["max_deg"] < 5
    other = np.linalg.qr(np.random.default_rng(3).standard_normal((D, 2)))[0]
    assert gc.principal_angles_to_centroid_plane(other, C_full)["min_deg"] > 30


def test_controls_match_length(ring):
    curve = ring[-1]
    rng = np.random.default_rng(0)
    rnd = gc.random_smooth_curve(curve, rng)
    L = lambda c: mf.arc_length_table(c)[1][-1]
    assert L(rnd) == pytest.approx(L(curve), rel=0.02)
    np.testing.assert_allclose(rnd.points.mean(0), curve.points.mean(0), atol=1e-9)
    shuf = gc.shuffled_curve(curve, rng)
    assert L(shuf) > 3 * L(curve)                          # a shuffled loop zig-zags
    assert gc.loo_reconstruction(shuf)["mean_err_line"] > R


def test_readouts_and_isometry(ring):
    X, theta, _, pca, cent, curve = ring
    probe = mf.ProbeReadout(X[1::2], theta[1::2], periodic=True)
    assert np.mean(mf.value_error(probe.predict(X[::2]), theta[::2], True)) < 5
    x = X[theta == 0.0]
    x_star = mf.steer_to_value(x, pca, curve, 90.0, source_value=0.0)[:, -1]
    assert np.median(probe.score(x_star, x, 90.0)) > 0.8
    near = mf.NearestRealReadout(X[1::2], theta[1::2])
    assert np.median(near.score(x_star, x, 90.0)) > 0.5
    assert np.median(near.score(x, x, 90.0)) == 0.0
    pts = probe.raw(pca.lift(curve.points))
    assert mf.isometry(curve, pts) > 0.95
    lens = mf.path_lengths(mf.linear_path(x[0], x[0], x[0] + 1.0, 5), probe.raw)
    assert lens["activation"] == pytest.approx(np.sqrt(D), rel=1e-6)


def test_linear_extension_continues_end_tangent():
    X, v = line_data()
    cent = mf.centroids(gc.fit_pca(X, 8).project(X), v)
    for sp in ("interp", "smooth"):
        cub = mf.fit_curve(cent, False, spline=sp)
        lin = mf.fit_curve(cent, False, spline=sp, extend="linear")
        inside = np.linspace(0.3, 3.9, 7)
        np.testing.assert_allclose(lin(inside), cub(inside))
        hi, slope = 4.0, cub.spline(4.0, 1)
        np.testing.assert_allclose(lin(np.array([4.5, 5.0])), cub(hi) + np.array([[0.5], [1.0]]) * slope)
        np.testing.assert_allclose(lin.spline(np.array([0.0, 5.0]), 1), cub.spline(np.array([0.25, 4.0]), 1))


def test_goodfire_periodic_angle_scales_and_tests():
    th = np.radians(np.arange(64) * 360.0 / 64)
    ell = np.stack([4 * np.cos(th), 1.0 * np.sin(th), np.zeros(64)], 1) + 2.0
    g = mf.goodfire_periodic_angle(ell)
    assert not g["passes"] and g["rel_diff"] == pytest.approx(1 - 1 / 16, rel=1e-6)
    dev = np.degrees(mf.wrap_pi(g["angle"] - th))
    assert np.abs(dev - dev.mean()).max() < 1e-6          # variance scaling maps the ellipse back to the true angle
    assert mf.goodfire_periodic_angle(np.stack([3 * np.cos(th), 2.5 * np.sin(th)], 1))["passes"]


def _jumbled_ring():
    """Issue #213 case: 16 knots on a unit ring, values 0..337.5 in 22.5 steps, block 90-157.5 held out. The knot at
    180 (the block's upper value neighbour) sits at angle 220 deg, beyond the 191 deg knot of value 202.5, so the
    knot order in angle is not monotone in value across the block."""
    vals = np.arange(16) * 22.5
    keep = ~((vals >= 90) & (vals <= 157.5))
    ang = np.radians(vals.copy())
    ang[vals == 180.0] = np.radians(220.0)
    ang[vals == 202.5] = np.radians(191.0)
    C = np.stack([np.cos(ang), np.sin(ang), 0.1 * np.cos(2 * ang)], 1)[keep]
    curve = mf.Curve(spline=None, values=vals[keep], coords=ang[keep], points=C, periodic=True,
                     coord_source="unsupervised_angle", aim="arc")
    order = np.argsort(curve.coords)
    t = np.append(curve.coords[order], curve.coords[order][0] + 2 * np.pi)
    from scipy.interpolate import CubicSpline
    curve.spline = CubicSpline(t, np.vstack([C[order], C[order][:1]]), bc_type="periodic")
    curve.values, curve.coords, curve.points = curve.values[order], curve.coords[order], C[order]
    return curve


def test_value_neighbour_chord_on_nonmonotone_ring():
    curve = _jumbled_ring()
    idx = {float(v): i for i, v in enumerate(curve.values)}
    a, b, f = curve.value_neighbours([146.25, 67.5, 90.0])
    assert (curve.values[a[0]], curve.values[b[0]]) == (67.5, 180.0) and f[0] == pytest.approx(78.75 / 112.5)
    assert a[1] == b[1] == idx[67.5] and f[1] == 0.0                          # a knot value is its own neighbour
    p = mf.piecewise_linear_point(curve, 146.25)
    want = (1 - 0.7) * curve.points[idx[67.5]] + 0.7 * curve.points[idx[180.0]]
    np.testing.assert_allclose(p, want, atol=1e-12)                           # A.9 chord between value neighbours
    # the legacy coordinate-order polyline would have run through the foreign 202.5 knot (angle 191 < 200)
    legacy = mf.Curve(spline=curve.spline, values=curve.values, coords=curve.coords, points=curve.points,
                      periodic=True, coord_source="x", aim="coord")
    t = legacy.coord_of_value(146.25)
    assert 67.5 < np.degrees(t) < 191.0     # in angle order its polyline segment is 67.5 -> the foreign 202.5 knot
    P = np.vstack([curve.points, curve.points[:1]])
    tc = np.append(curve.coords, curve.coords[0] + 2 * np.pi)
    old = np.array([np.interp((t - tc[0]) % (2 * np.pi) + tc[0], tc, P[:, j]) for j in range(3)])
    assert np.linalg.norm(old - want) > 0.25                                  # the legacy polyline was misaimed
    np.testing.assert_array_equal(mf.piecewise_linear_point(legacy, 146.25), old)  # aim='coord' = legacy path
    np.testing.assert_allclose(mf.piecewise_linear_point(curve, curve.values), curve.points, atol=1e-12)


def test_arc_aim_fraction_along_fitted_curve():
    curve = _jumbled_ring()
    idx = {float(v): i for i, v in enumerate(curve.values)}
    ta, tb = curve.coords[idx[67.5]], curve.coords[idx[180.0]]
    for v in (90.0, 112.5, 146.25, 157.5):
        f = (v - 67.5) / 112.5
        t = curve.coord_of_value(v)
        assert 0 < mf.wrap_pi(t - ta) < mf.wrap_pi(tb - ta)                  # between the value neighbours
        assert mf.arc_length(curve, ta, t, n=4000) / mf.arc_length(curve, ta, tb, n=4000) == pytest.approx(f, abs=2e-3)
    np.testing.assert_allclose(curve.coord_of_value(curve.values), curve.coords % (2 * np.pi), atol=1e-12)
    # monotone ring: the chord fix reproduces the legacy coordinate-order polyline exactly
    X, theta, _ = ring_data(1)
    pca = mf.fit_pca(X, 8)
    cent = mf.centroids(pca.project(X), theta)
    keep = ~mf.heldout_mask(cent["values"], periodic=True)
    c2 = mf.fit_curve(cent, periodic=True, keep=keep, angle="labels")
    held = cent["values"][~keep]
    t = c2.coord_of_value(held)
    P = np.vstack([c2.points, c2.points[:1]])
    tc = np.append(c2.coords, c2.coords[0] + 2 * np.pi)
    tt = (t - tc[0]) % (2 * np.pi) + tc[0]
    legacy = np.stack([np.interp(tt, tc, P[:, j]) for j in range(P.shape[1])], -1)
    c2.aim = "arc"
    np.testing.assert_allclose(mf.piecewise_linear_point(c2, held), legacy, atol=1e-10)

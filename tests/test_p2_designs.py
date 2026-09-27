"""Part 2 evaluation design: held-out designs, sagitta, matched support, curvature controls, dose matching,
circular chart, planted-ring control. Synthetic ring/line fixtures from test_manifold."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wm import geometry_checks as gc
from wm import manifold as mf

from test_manifold import R, line_data, ring_data

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name[:-3], ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P2 = load_script("run_part2.py")
GRID = np.arange(64) * 360.0 / 64
SPEEDS = np.linspace(0.25, 4.0, 64)


def test_heldout_designs():
    m, info = mf.heldout_design(GRID, "scattered", periodic=True)
    assert m.sum() == 16 and info["design"] == "scattered"
    m, info = mf.heldout_design(GRID, "contiguous", periodic=True, seed=3)
    idx = np.flatnonzero(m)
    assert m.sum() == 8 and info["arc_width_deg"] == 45.0
    assert len(info["held_out_values"]) == 8
    start = info["block_start_index"]
    assert set(idx) == {(start + j) % 64 for j in range(8)}          # consecutive, may wrap round 0/360
    assert mf.heldout_design(GRID, "contiguous", True, seed=3)[1] == info  # the seed fixes the arc
    for seed in range(30):                                              # scalar block always interior
        m, info = mf.heldout_design(SPEEDS, "contiguous", periodic=False, seed=seed)
        idx = np.flatnonzero(m)
        assert m.sum() == 8 and np.all(np.diff(idx) == 1) and idx[0] >= 1 and idx[-1] <= 62
    m, info = mf.heldout_design(SPEEDS, "extrapolation", periodic=False)
    assert np.flatnonzero(m).tolist() == list(range(56, 64))
    with pytest.raises(ValueError):
        mf.heldout_design(GRID, "extrapolation", periodic=True)


def fit(X, y, periodic, keep=None, k=16):
    pca = mf.fit_pca(X if keep is None else X[np.isin(y, keep)], k)
    rows = slice(None) if keep is None else np.isin(y, keep)
    cent = mf.centroids(pca.project(X[rows]), y[rows])
    return pca, cent, mf.fit_curve(cent, periodic)


def quiet_ring(noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    th = np.repeat(GRID, 20)
    Q = np.linalg.qr(rng.standard_normal((60, 2)))[0]
    r = np.radians(th)
    return R * np.stack([np.cos(r), np.sin(r)], 1) @ Q.T + noise * rng.standard_normal((len(th), 60)), th


def test_sagitta_ring_contiguous_vs_line():
    X, th = quiet_ring()
    mask, info = mf.heldout_design(GRID, "contiguous", True, seed=0)
    pca, cent, curve = fit(X, th, True, keep=GRID[~mask])
    sag = mf.sagitta(curve, GRID[mask])
    mid = max(sag, key=lambda s: s["sagitta"])
    half_gap = np.radians(9 * 360 / 64 / 2)                            # 8 held out -> chord spans 9 spacings
    assert mid["sagitta"] == pytest.approx(R * (1 - np.cos(half_gap)), rel=0.3)
    assert all(s["chord_from"] not in info["held_out_values"] for s in sag)


def test_extrapolation_extends_the_line():
    Xl, v = line_data()
    mask, _ = mf.heldout_design(SPEEDS, "extrapolation", False)
    _, cent, curve = fit(Xl, v, False, keep=SPEEDS[~mask])
    top = SPEEDS[-1]
    assert curve.coord_of_value(top) == pytest.approx(top)            # value coordinate is not clamped
    p_far, p_last = mf.piecewise_linear_point(curve, top), curve.points[-1]
    assert np.linalg.norm(p_far - p_last) > 5 * 3.0 * (SPEEDS[1] - SPEEDS[0])   # moved past the last knot
    assert mf.sagitta(curve, [top])[0]["chord_to"] == pytest.approx(curve.values.max())


def test_chord_controls_share_endpoints_and_spacing():
    X, th, _ = ring_data()
    pca, cent, curve = fit(X, th, True)
    Z = pca.project(X[:5])
    Zs = mf.manifold_coords(Z, curve, np.zeros(5), np.full(5, np.pi * 0.9), K=21)
    proj, refl = mf.chord_coords(Zs)
    for W in (proj, refl):
        np.testing.assert_allclose(W[:, [0, -1]], Zs[:, [0, -1]], atol=1e-10)
    np.testing.assert_allclose(refl, 2 * proj - Zs, atol=1e-12)
    # projected lies on the chord, spaced like the spline's cumulative arc length
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Zs[0], axis=0), axis=1))])
    f = np.linalg.norm(proj[0] - proj[0, 0], axis=1) / np.linalg.norm(proj[0, -1] - proj[0, 0])
    np.testing.assert_allclose(f, s / s[-1], atol=1e-9)
    # the reflected bend sits on the other side of the chord from the spline
    mid = 10
    assert np.dot(Zs[0, mid] - proj[0, mid], refl[0, mid] - proj[0, mid]) < 0
    zero = mf.chord_coords(np.repeat(Z[:1, None], 4, 1))[0]
    np.testing.assert_allclose(zero, np.repeat(Z[:1, None], 4, 1))


@pytest.fixture(scope="module")
def ring_model():
    X, th, Q = ring_data()
    f = np.arange(len(th)) % 5
    d = {"X": X, "y": th, "periodic": True, "role": np.where(f < 3, "knot", np.where(f == 3, "probe", "test"))}
    m = P2.build(d, 16, "unsupervised", "contiguous", seed=0, n_controls=2)
    return X, th, Q, m


def test_matched_support_residual_identical(ring_model):
    X, th, _, m = ring_model
    x, src, tgt = X[th == 0.0][:6].astype(float), np.zeros(6), float(m["held"][3])
    pca = m["pca"]
    Z, resid = pca.project(x), pca.complement(x)
    arms = P2.subspace_arms(Z, src, tgt, m, K=9)
    assert set(arms) == {"manifold", "linear", "linear_dose_matched", "projected", "reflected", "manifold_transport"}
    for name, Zk in arms.items():
        assert Zk.shape == (6, 9, pca.components.shape[0])        # arms only carry in-subspace coordinates
        W = P2.compose(pca, Zk, resid)
        # the one residual array is added unchanged: bit-identical across arms
        np.testing.assert_array_equal(W, pca.lift(Zk) + resid[:, None, :])
        np.testing.assert_allclose(pca.complement(W), np.repeat(pca.complement(x)[:, None], 9, 1), atol=1e-9)
        np.testing.assert_allclose(W[:, 0], x, atol=1e-9)           # every arm starts at the clip
    # Goodfire's linear baseline erases the residual; its manifold arm keeps it
    gf = P2.goodfire_arms(x, src, tgt, m, K=9)
    np.testing.assert_allclose(pca.complement(gf["goodfire_manifold"]),
                               np.repeat(pca.complement(x)[:, None], 9, 1), atol=1e-9)
    assert np.abs(pca.complement(gf["goodfire_linear"][:, -1]) - pca.complement(x)).max() > 0.1


def test_linear_and_spline_targets_differ_by_sagitta(ring_model):
    X, th, _, m = ring_model
    x, src = X[th == 0.0][:3].astype(float), np.zeros(3)
    Z = m["pca"].project(x)
    for s in m["sagitta"]:
        arms = P2.subspace_arms(Z, src, s["value"], m, K=3)
        gap = np.linalg.norm(arms["manifold"][:, -1] - arms["linear"][:, -1], axis=1)
        np.testing.assert_allclose(gap, s["sagitta"], rtol=1e-6, atol=1e-9)   # source 0 is a kept knot


def test_dose_matched_arm(ring_model):
    X, th, _, m = ring_model
    x, src = X[th == 0.0][:4].astype(float), np.zeros(4)
    Z = m["pca"].project(x)
    arms = P2.subspace_arms(Z, src, 180.0 if 180.0 in m["held"] else float(m["held"][0]), m, K=11)
    dose = lambda Zk: np.linalg.norm(Zk - Z[:, None], axis=-1)
    np.testing.assert_allclose(dose(arms["linear_dose_matched"]), dose(arms["manifold"]), atol=1e-9)
    # same direction as the line
    dl = arms["linear"][:, -1] - Z
    dd = arms["linear_dose_matched"][:, -1] - Z
    cos = np.sum(dl * dd, 1) / np.linalg.norm(dl, axis=1) / np.linalg.norm(dd, axis=1)
    np.testing.assert_allclose(cos, 1.0, atol=1e-9)


def test_control_band_rank():
    b = P2.band([0.1, 0.2, 0.3, 0.4], 0.35, higher_better=True)
    assert b["spline_rank"] == 2 and b["frac_draws_spline_beats"] == 0.75 and b["n_draws"] == 4
    assert P2.band([0.1, 0.2], 0.05, higher_better=False)["spline_rank"] == 1


def test_rescue_harm_counts():
    cat = {"linear": {"nearest_real_R": np.array([0.1, 0.5, -0.2])},
           "manifold": {"nearest_real_R": np.array([0.3, 0.4, 0.1])}}
    rh = P2.rescue_harm(cat, ["manifold", "linear"])
    assert rh["manifold"]["vs_linear"] == {"improved": 2, "worsened": 1, "tied": 0,
                                           "mean_diff": pytest.approx(0.4 / 3)}
    assert rh["linear"]["vs_unsteered"] == {"improved": 2, "worsened": 1}


def test_circular_chart_matches_pc_angle():
    X, th, Q = ring_data()
    pca = mf.fit_pca(X, 16)
    cc = gc.chart_check(X, th, pca)
    assert cc["centroids_chart_vs_pc12"]["circular_corr"] > 0.99
    assert cc["centroids_chart_vs_pc12"]["mae_deg"] < 2
    assert max(cc["principal_angles_chart_vs_pc12_deg"]) < 2
    assert cc["chart_radius"] == pytest.approx(R, rel=0.05)
    assert cc["chart_vs_true_mae_deg"]["centroids"] < 2


def test_fisher_lee_evenly_spread_angles():
    t = np.radians(GRID)
    assert mf.fisher_lee(t + 0.7, t) == pytest.approx(1.0)
    assert mf.fisher_lee(-t, t) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    assert abs(mf.fisher_lee(t + 0.1 * rng.standard_normal(64), t)) > 0.98


@pytest.mark.parametrize("fixture", ["line", "ring"])
def test_planted_ring_recovered_when_large(fixture):
    X, y = line_data() if fixture == "line" else ring_data()[:2]
    pr = gc.planted_ring_control(X, y, radii=(0.5, 4.0), k=16, seed=7)
    small, big = pr["rows"]
    assert big["recovered"] and big["circular_corr"] > 0.95 and big["loo_winner"] == "cubic"
    assert not small["recovered"] and small["circular_corr"] < big["circular_corr"]
    assert big["ring_radius"] == pytest.approx(4.0 * pr["sd"])


def test_smoothing_spline_crosses_a_held_out_block():
    """With ~20 noisy clips per value an interpolating spline overshoots across a held-out block; the count-weighted
    smoothing spline (Goodfire B.1) does not, and on a ring it beats the chord."""
    X, th, _ = ring_data()
    rec = gc.heldout_reconstruction(X, th, True, k=16, seeds=(0, 1))
    c = rec["contiguous"]
    assert c["best"] == "smooth" and c["smooth"] < 0.8 * c["chord"] and c["interp"] > c["chord"]
    Xl, v = line_data()
    cl = gc.heldout_reconstruction(Xl, v, False, k=16, seeds=(0, 1))["contiguous"]
    assert cl["smooth"] < 1.2 * cl["chord"] and cl["interp"] > 1.5 * cl["smooth"]
    pca = mf.fit_pca(X, 16)
    curve = mf.fit_curve(mf.centroids(pca.project(X), th), True, spline="smooth")
    assert curve.kind == "smoothing"
    t0 = curve.coords[0]
    np.testing.assert_allclose(curve(t0), curve(t0 + 2 * np.pi), atol=1e-9)
    np.testing.assert_allclose(curve.spline(t0, 1), curve.spline(t0 + 2 * np.pi - 1e-12, 1), atol=1e-6)
    np.testing.assert_allclose(curve(curve.coords), curve.points, atol=1e-9)


def test_behaviour_manifold_eq9():
    X, th, _ = ring_data()
    pca = mf.fit_pca(X[::2], 16)
    curve = mf.fit_curve(mf.centroids(pca.project(X[::2]), th[::2]), True)
    ref = mf.fit_curve(mf.centroids(pca.project(X[1::2]), th[1::2]), True, angle="labels", spline="smooth")
    bm = mf.BehaviourManifold(X[1::2], th[1::2], True, curve=ref, pca=pca)             # literal Eq. 9
    assert bm.mu.shape == (128, 16)
    P = bm.F(X[th == 90.0])
    peak = np.degrees(ref.coords[0] + np.arange(128) * 2 * np.pi / 128)[P.mean(0).argmax()] % 360
    assert np.allclose(P.sum(1), 1) and abs((peak - 90 + 180) % 360 - 180) < 6          # peaked at the true value
    bm_sq = mf.BehaviourManifold(X[1::2], th[1::2], True, mode="centroid_sq")
    assert bm_sq.values[bm_sq.F(X[th == 90.0]).mean(0).argmax()] == 90.0
    np.testing.assert_allclose(np.linalg.norm(bm.grid, axis=1), 1.0, atol=1e-9)         # M_y stays on the sphere
    x = X[th == 0.0][:5]
    Z, resid = pca.project(x), pca.complement(x)
    src = np.zeros(5)
    Zs = mf.manifold_coords(Z, curve, curve.coord_of_value(src), curve.coord_of_value(np.full(5, 180.0)), 50)
    Zl = mf.linear_coords(Z, mf.piecewise_linear_point(curve, src), mf.piecewise_linear_point(curve, 180.0), 50)
    e_s = bm.bhattacharyya(pca.lift(Zs) + resid[:, None]).sum(1)
    e_l = bm.bhattacharyya(pca.lift(Zl) + resid[:, None]).sum(1)
    assert np.all(e_l > 5 * e_s)                              # Goodfire's E_BC ordering on a planted ring
    assert mf.isometry(curve, bm.geodesic_matrix(curve.values), "precomputed") > 0.95


def test_isometry_angular_vs_chord():
    X, th, _ = ring_data()
    pca = mf.fit_pca(X, 16)
    curve = mf.fit_curve(mf.centroids(pca.project(X), th), True)
    r = np.radians(curve.values)
    readout = np.stack([np.sin(r), np.cos(r)], 1)
    ang = mf.isometry(curve, readout, "angular")
    chord = mf.isometry(curve, readout, "euclidean")
    assert ang > 0.99 and ang > chord                          # geodesic vs geodesic is linear; chords bend


def test_transport_keeps_offset_outward():
    X, th, _ = ring_data()
    pca = mf.fit_pca(X, 16)
    curve = mf.fit_curve(mf.centroids(pca.project(X), th), True)
    c = curve.points.mean(0)
    p0 = curve(curve.coord_of_value(0.0))
    Z0 = (p0 + 0.5 * (p0 - c) / np.linalg.norm(p0 - c))[None]
    ta, tb = curve.coord_of_value(0.0), curve.coord_of_value(180.0)
    rad = lambda mode: np.linalg.norm(mf.manifold_coords(Z0, curve, ta, tb, 21, mode)[0] - c, axis=1)
    r_end = np.linalg.norm(curve(tb) - c)
    assert rad("transport")[-1] == pytest.approx(r_end + 0.5, abs=0.1)
    assert rad("shift")[-1] < r_end                           # rigid shift lands the offset on the inner side
    np.testing.assert_allclose(mf.manifold_coords(Z0, curve, ta, tb, 5, "transport")[0, 0], Z0[0], atol=1e-9)
    Xl, v = line_data()
    pl = mf.fit_pca(Xl, 8)
    cl = mf.fit_curve(mf.centroids(pl.project(Xl), v), False)
    with pytest.raises(ValueError):
        mf.manifold_coords(pl.project(Xl[:2]), cl, 1.0, 2.0, 3, "transport")


def test_endpoint_matched_random_control():
    X, th, _ = ring_data()
    pca = mf.fit_pca(X, 16)
    curve = mf.fit_curve(mf.centroids(pca.project(X), th), True)
    Z = pca.project(X[:6])
    Zs = mf.manifold_coords(Z, curve, np.zeros(6), np.linspace(0.5, 3.0, 6), K=30)
    rng = np.random.default_rng(0)
    L = lambda P: np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
    for _ in range(3):
        Zr = mf.endpoint_matched_random(Zs, rng.standard_normal((3, 16)))
        np.testing.assert_allclose(Zr[:, [0, -1]], Zs[:, [0, -1]], atol=1e-9)
        np.testing.assert_allclose(L(Zr), L(Zs), rtol=1e-4)
        assert np.abs(Zr - Zs).max() > 0.1                  # a different bend


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    b = P2.paired_bootstrap(1.0 + rng.standard_normal(400))
    assert b["ci95"][0] < b["mean"] < b["ci95"][1] and b["ci95"][0] > 0.8 and b["ci95"][1] < 1.2


def test_angle_acceptance_mirror_and_local_swaps():
    """A mirrored ring whose neighbouring knots swap by noise is recovered (sign recorded); a scrambled one is not."""
    rng = np.random.default_rng(0)
    t = np.radians(GRID)
    noisy = -t + np.radians(8.0) * rng.standard_normal(64)          # mirror + ~8 deg jitter: many local swaps
    C = np.zeros((64, 6))
    C[:, 0], C[:, 1] = np.cos(noisy), np.sin(noisy)
    C[:, 2:] = 0.01 * rng.standard_normal((64, 4))
    ch = mf.choose_angle_source(C, GRID)
    c = ch["checks"]["activation"]
    assert c["order_frac"] < 1.0 and c["order_coarse_preserved"]
    assert ch["angle"] == "unsupervised" and ch["orientation"] == -1
    C2 = C.copy()
    C2[:, :2] = C[rng.permutation(64), :2]
    assert mf.choose_angle_source(C2, GRID)["angle"] == "labels"


def test_planted_ring_reports_real_ring():
    X, th, _ = ring_data()
    pr = gc.planted_ring_control(X, th, radii=(1.0,), k=16, seed=7, periodic=True)
    assert pr["real_ring"]["radius"] == pytest.approx(R, rel=0.05)
    assert pr["rows"][0]["radius_over_real_ring"] == pytest.approx(pr["rows"][0]["ring_radius"] / pr["real_ring"]["radius"])


def test_single_provenance_module():
    from wm import common, provenance as pv
    assert common.provenance is pv.result_provenance and common.git_commit is pv.git_commit
    for block in (pv.result_provenance({"pool": "meanpool", "point": 3}), pv.provenance(seeds={"seed": 0}, layer=3)):
        assert {"commit", "dirty", "git_commit", "git_dirty_src_or_scripts", "split_sha256"} <= set(block)


def test_nuisance_fit_on_primary_applied_to_context():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"theta_degrees": rng.uniform(0, 360, n), "speed_mps": rng.uniform(1, 7, n),
                       "acceleration_mps2": 0.0, "start_x": rng.normal(size=n), "start_y": rng.normal(size=n),
                       "motion": np.where(np.arange(n) % 2, "velocity", "acceleration")})
    X = rng.normal(size=(n, 10)) + np.outer(df["speed_mps"], np.ones(10))
    train = np.arange(n) < 150
    model, R = gc.fit_nuisance(X, df, "direction", train)
    assert "acceleration_mps2" not in model["names"] and "speed_mps" in model["names"]   # constant in primary train
    assert np.abs(R[train].T @ df["speed_mps"].to_numpy()[train]).max() < 1e-6          # speed regressed out
    # context set: constant speed, no start_x column -> primary coefficients, start_x filled with the primary mean
    ctx = df.drop(columns=["start_x"]).assign(speed_mps=2.0)
    Xc = rng.normal(size=(n, 10))
    Rc, filled = gc.apply_nuisance(model, Xc, ctx)
    assert filled == ["start_x"]
    N = np.column_stack([np.ones(n), *[np.full(n, model["fill"][c]) if c == "start_x" else
                                        gc.nuisance_columns(ctx, "direction")[c] for c in model["names"]]])
    np.testing.assert_allclose(Rc, (Xc - model["mu"]) / model["sd"] - N @ model["B"], atol=1e-10)


def test_paired_bootstrap_clusters_by_clip():
    """Rows that share a clip are correlated: resampling clips gives a wider CI than resampling rows."""
    rng = np.random.default_rng(1)
    clip_effect = rng.standard_normal(30)
    ids = np.repeat(np.arange(30), 10)                       # each clip steered to 10 targets
    diff = clip_effect[ids] + 0.1 * rng.standard_normal(300)
    rows = P2.paired_bootstrap(diff)
    clips = P2.paired_bootstrap(diff, ids)
    assert clips["n_clips"] == 30 and rows["n_clips"] == 300
    width = lambda b: b["ci95"][1] - b["ci95"][0]
    assert width(clips) > 2 * width(rows)


def test_angle_max_dev_is_reported_not_required():
    """One knot 40 degrees off: accepted by corr + coarse order (max deviation recorded); a cap rejects it."""
    rng = np.random.default_rng(3)
    t = np.radians(GRID) + np.radians(2.0) * rng.standard_normal(64)
    t[10] += np.radians(40.0)
    C = np.zeros((64, 4))
    C[:, 0], C[:, 1] = np.cos(t), np.sin(t)
    ch = mf.choose_angle_source(C, GRID)
    assert ch["angle"] == "unsupervised" and ch["max_dev_deg"] > 30 and abs(ch["circular_corr"]) >= 0.9
    assert mf.choose_angle_source(C, GRID, max_dev_deg=30.0)["angle"] == "labels"

"""Pure helpers of the time-manifold and velocity-sheet experiments (wm.timeman, run_velocity_sheet)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wm import timeman as tm


def test_step_times_and_distance():
    tau = tm.step_times()
    assert tau[0] == pytest.approx(0.5 / 24) and np.allclose(np.diff(tau), 2 / 24)
    d = tm.step_distance(np.array([1.0, 0.0]), np.array([0.0, 2.0]))
    assert np.allclose(d[0], tau) and np.allclose(d[1], tau ** 2)


def test_step_position_moves_along_theta():
    p = tm.step_position(np.array([[0.5, -0.5]]), np.array([90.0]), np.array([2.0]), np.array([0.0]))
    assert np.allclose(p[0, :, 0], 0.5) and np.allclose(p[0, :, 1], -0.5 + 2 * tm.step_times())


def test_remove_clip_mean_and_slope():
    F = np.random.default_rng(0).standard_normal((3, 8, 5))
    assert np.allclose(tm.remove_clip_mean(F).mean(1), 0)
    coord = 2.0 * np.arange(8)[None] + np.array([[1.0], [-3.0]])
    assert np.allclose(tm.per_clip_slope(coord), 2.0)


def test_odometer_coord_is_identity_for_the_reference_clip():
    t = np.arange(8.0)
    assert np.allclose(tm.odometer_coord(tm.step_distance(np.array([2.0]), np.array([0.0])), ref_speed=2.0)[0], t)
    assert np.allclose(tm.odometer_coord(tm.step_distance(np.array([0.0]), np.array([5.0])), ref_accel=5.0)[0], t)
    # a clip twice as fast covers the reference's distance in half the time: slope 2 in steps
    fast = tm.odometer_coord(tm.step_distance(np.array([4.0]), np.array([0.0])), ref_speed=2.0)
    assert tm.per_clip_slope(fast)[0] == pytest.approx(2.0)


def test_clock_odometer_fit_separates_the_two():
    rng = np.random.default_rng(1)
    v = rng.uniform(0.3, 4, 200)
    odo = tm.odometer_coord(tm.step_distance(v, np.zeros(200)), ref_speed=2.0)
    t = np.broadcast_to(np.arange(8.0), odo.shape)
    clock = tm.clock_odometer_fit(t + 0.01 * rng.standard_normal(t.shape), t, odo)
    assert clock["b_clock"] == pytest.approx(1, abs=0.02) and abs(clock["c_odometer"]) < 0.02
    assert clock["r2_clock_only"] > clock["r2_odometer_only"]
    od = tm.clock_odometer_fit(odo, t, odo)
    assert od["c_odometer"] == pytest.approx(1, abs=1e-6) and abs(od["b_clock"]) < 1e-6


def test_nearest_coord():
    gt = np.linspace(0, 7, 701)
    pts = np.stack([np.cos(gt), np.sin(gt), gt], 1)
    q = np.stack([np.cos([1.0, 5.5]), np.sin([1.0, 5.5]), [1.0, 5.5]], 1) + 1e-3
    assert np.allclose(tm.nearest_coord(q, gt, pts), [1.0, 5.5], atol=0.02)


def test_cluster_bootstrap_groups():
    vals = np.r_[np.zeros(10), np.ones(10)]
    out = tm.cluster_bootstrap(vals, np.repeat(np.arange(2), 10), n_boot=200)
    assert out["mean"] == 0.5 and out["n_clips"] == 2 and out["ci95"][0] <= 0.5 <= out["ci95"][1]


def test_ring_radius_corrected():
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    C = 3.0 * np.stack([np.cos(th), np.sin(th)], 1)
    raw, cor = tm.ring_radius_corrected(C, noise=0.0)
    assert raw == pytest.approx(3.0) and cor == pytest.approx(3.0)
    assert tm.ring_radius_corrected(C, noise=np.sqrt(5.0))[1] == pytest.approx(2.0)


def test_cone_fit_recovers_scaled_rings_and_cylinder_does_not():
    rng = np.random.default_rng(2)
    n_dir, bands = 8, np.arange(4)
    u = rng.standard_normal((n_dir, 6))
    m = rng.standard_normal((4, 6))
    r = np.array([0.5, 1.0, 1.5, 2.0])
    band, dirb = np.repeat(bands, n_dir), np.tile(np.arange(n_dir), 4)
    C = m[band] + r[band, None] * u[dirb]
    fit = tm.cone_cylinder_fit(C, band, dirb, n_dir)
    assert np.allclose(tm.predict_sheet(fit, band, dirb, "cone"), C, atol=1e-6)
    assert not np.allclose(tm.predict_sheet(fit, band, dirb, "cylinder"), C, atol=1e-2)
    assert np.allclose(fit["r_cone"] / fit["r_cone"][0], r / r[0], atol=1e-6)


def test_angular_bins():
    assert tm.angular_bins([0, 22.4, 22.5, 359.9, 360], 16).tolist() == [0, 0, 1, 15, 0]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_velocity_sheet_bins_and_inputs():
    vs = _script("run_velocity_sheet")
    speed = np.repeat(np.linspace(0.25, 4.0, 64), 2)
    theta = np.tile([0.0, 90.0], 64)
    dbin, sbin = vs.bins(theta, speed)
    assert sbin.min() == 0 and sbin.max() == 7 and np.all(np.bincount(sbin) == 16)
    assert set(dbin.tolist()) == {0, 4}
    x = vs.sheet_inputs([90.0], [vs.V_MID + vs.V_HALF], 2.0)
    assert np.allclose(x, [[0.0, 1.0, 2.0]], atol=1e-12)

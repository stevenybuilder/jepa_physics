"""scripts/session3_speed_predictor.py --variable acceleration: pure helpers (acceleration twin metadata,
quadratic-fit acceleration from per-step positions, variable configuration)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("s3a", ROOT / "scripts" / "session3_speed_predictor.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)


def test_acceleration_twin_meta_and_trajectory():
    from wm.render_twin import trajectory_m
    meta = {"theta_degrees": 120.0, "speed_mps": 0.0, "acceleration_mps2": 2.0, "start_position_xy_m": [0.3, 0.1],
            "fps": 24, "frames": 16}
    tw = s3.acceleration_twin_meta(meta, 8.0)
    assert meta["acceleration_mps2"] == 2.0 and tw["acceleration_mps2"] == 8.0 and tw["speed_mps"] == 0.0
    a, b = trajectory_m(meta), trajectory_m(tw)
    assert np.allclose(a[0], b[0])                                   # same start
    assert np.allclose(b - b[0], 4 * (a - a[0]))                     # from rest: displacement scales with a


def _positions(acc, v0=0.0, theta=37.0, p0=(40.0, 200.0), steps=range(4, 8)):
    """Tubelet-averaged pixel centroids (frames 2t, 2t+1) of p0 + (v0 t + a t^2 / 2) u, y down."""
    u = np.array([np.cos(np.radians(theta)), -np.sin(np.radians(theta))])
    out = []
    for k in steps:
        t = np.array([2 * k, 2 * k + 1]) / 24.0
        s = (v0 * t + 0.5 * acc * t ** 2) * 32.0
        out.append((np.asarray(p0)[None] + s[:, None] * u[None]).mean(0))
    return np.array(out)


def test_accel_from_positions_recovers_acceleration():
    for acc in (0.25, 3.0, 10.0):
        for v0 in (0.0, 1.5):
            assert np.isclose(s3.accel_from_positions(_positions(acc, v0)), acc, atol=1e-9)
    batch = np.stack([_positions(a, theta=th) for a, th in ((1.0, 0.0), (5.0, 200.0))])
    assert np.allclose(s3.accel_from_positions(batch), [1.0, 5.0])
    # constant velocity -> zero acceleration; the speed helper still reads the speed
    lin = _positions(0.0, v0=2.0)
    assert abs(s3.accel_from_positions(lin)) < 1e-9
    assert np.isclose(s3.speed_from_positions(lin), 2.0)


def test_configure_switches_design_and_back():
    try:
        s3.configure("acceleration")
        assert s3.DATASET == "acceleration" and s3.POINTS == (12, 21)
        assert set(s3.CONDS) == {12, 21} and s3.CONDS[21] == ("own", "chord_norm", "natural_norm")
        assert s3.motion_from_positions(_positions(6.0)) == s3.accel_from_positions(_positions(6.0))
        assert s3.twin_meta({"acceleration_mps2": 1.0}, 2.5)["acceleration_mps2"] == 2.5
    finally:
        s3.configure("speed")
    assert s3.DATASET == "speed" and s3.POINTS == (12, 22) and s3.ART.name == "session3_speed"
    assert np.isclose(s3.motion_from_positions(_positions(0.0, v0=2.0)), 2.0)

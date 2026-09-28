"""scripts/session3_speed_predictor.py: pure helpers (speed twin metadata, null aim, displacement speed, rescaling)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("s3", ROOT / "scripts" / "session3_speed_predictor.py")
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)


def test_speed_twin_meta_and_trajectory():
    from wm.render_twin import trajectory_m
    meta = {"theta_degrees": 30.0, "speed_mps": 1.0, "acceleration_mps2": 0.0, "start_position_xy_m": [0.1, -0.2],
            "fps": 24, "frames": 16}
    tw = s3.speed_twin_meta(meta, 3.0)
    assert meta["speed_mps"] == 1.0 and tw["speed_mps"] == 3.0 and tw["theta_degrees"] == 30.0
    a, b = trajectory_m(meta), trajectory_m(tw)
    assert np.allclose(a[0], b[0])                                  # same start
    assert np.allclose((b - b[0]), 3 * (a - a[0]))                  # same heading, 3x displacement


def test_far_end_value():
    vals = np.linspace(0.25, 4.0, 64)
    assert s3.far_end_value(vals, vals[47:55]) == 0.25
    assert s3.far_end_value(vals, vals[2:10]) == 4.0


def test_speed_from_positions():
    v = 2.0                                                          # m/s -> px per tubelet = v * 32 * 2 / 24
    step = v * 32 * 2 / 24
    pos = np.stack([[10 + k * step * 0.6, 50 - k * step * 0.8] for k in range(4)])
    assert np.isclose(s3.speed_from_positions(pos), v)
    assert s3.speed_from_positions(np.stack([pos, pos])).shape == (2,)


def test_rescale_and_projection():
    d = np.array([[3.0, 4.0], [0.0, 0.0]])
    out, s = s3.rescale(d, np.array([10.0, 5.0]))
    assert np.allclose(np.linalg.norm(out[0]), 10.0) and np.allclose(out[1], 0) and s[1] == 0
    assert np.isclose(s3.projection_R(np.array([2.0, 1.0]), np.array([4.0, 0.0])), 0.5)
    r = s3.per_clip_ci([1.0, 2.0, 3.0, np.nan])
    assert r["n"] == 3 and np.isclose(r["mean"], 2.0) and np.isclose(r["se"], 1 / np.sqrt(3))

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "run_straightening", Path(__file__).resolve().parents[1] / "scripts" / "run_straightening.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)


def test_straight_line_is_zero():
    rng = np.random.default_rng(0)
    d, x0 = rng.standard_normal(1024), rng.standard_normal(1024)
    t = np.array([0, 0.5, 1.7, 2.0, 3.1, 4.0, 6.0, 9.0])[:, None]   # uneven speed along the line
    assert abs(rs.curvature(x0 + t * d)) < 1e-4


def test_uniform_circle_is_step_angle():
    for step in (np.pi / 8, 2 * np.pi / 7, 0.3):
        th = step * np.arange(8)
        X = np.zeros((8, 16))
        X[:, 3], X[:, 11] = 2.5 * np.cos(th), 2.5 * np.sin(th)
        assert abs(rs.curvature(X) - np.degrees(step)) < 1e-6


def test_batch_shapes_and_reversal_invariance():
    X = np.random.default_rng(1).standard_normal((5, 3, 8, 32))
    c = rs.curvature(X)
    assert c.shape == (5, 3)
    assert np.allclose(c, rs.curvature(X[..., ::-1, :]))


def test_isotropic_null_high_dim_near_90():
    X = np.cumsum(np.random.default_rng(2).standard_normal((200, 8, 1024)), 1)
    c = rs.null_curvature(X, np.random.default_rng(3))
    assert abs(c.mean() - 90) < 2

"""scripts/session2_pullback_ctxring.py: ring paths and ring distance on a synthetic labels-angle ring."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("pbc", ROOT / "scripts" / "session2_pullback_ctxring.py")
pbc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbc)
from wm import manifold as mf  # noqa: E402


def ring_curve():
    vals = np.arange(0, 360, 5.625)
    y = np.repeat(vals, 4)
    th = np.radians(y)
    Z = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    return mf.fit_curve(mf.centroids(Z, y), True, angle="labels", plane="activation", spline="interp")


def test_ring_paths_and_distance():
    c = ring_curve()
    sp, ch, stp = pbc.ring_paths(c, 45.0, 135.0, 20, 1.0)
    assert np.allclose(np.linalg.norm(sp[:, :2], axis=1), 1, atol=1e-3) and abs(abs(stp) - np.pi / 2) < 1e-6
    assert np.allclose(ch[[0, -1]], sp[[0, -1]])
    assert pbc.dist_to_ring(sp, c).max() < 1e-3
    assert abs(pbc.dist_to_ring(np.array([0.0, 0.0, 0.5]), c)[0] - np.sqrt(1.25)) < 1e-3
    # a 180-degree pair follows the requested direction
    a = pbc.ring_paths(c, 0.0, 180.0, 21, 1.0)[0][10]
    b = pbc.ring_paths(c, 0.0, 180.0, 21, -1.0)[0][10]
    assert np.sign(a[1]) == -np.sign(b[1]) and abs(a[1]) > 0.9

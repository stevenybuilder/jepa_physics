"""scripts/session2_pullback_goodfire_ctx.py: the ctx ring path follows the behaviour target's signed shift."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("pbx", ROOT / "scripts" / "session2_pullback_goodfire_ctx.py")
pbx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbx)
from wm import manifold as mf  # noqa: E402


def test_ctx_ring_path_direction():
    vals = np.arange(0, 360, 5.625)
    th = np.radians(np.repeat(vals, 4))
    c = mf.fit_curve(mf.centroids(np.stack([np.cos(th), np.sin(th)], 1), np.repeat(vals, 4)), True, angle="labels",
                     plane="activation", spline="interp")
    sp, ch, stp, fl = pbx.ctx_ring_path(c, mf, 45.0, 135.0, 90.0, 20)
    assert not fl and np.allclose(ch[[0, -1]], sp[[0, -1]]) and np.allclose(sp[-1], [np.cos(np.radians(135)),
                                                                                       np.sin(np.radians(135))], atol=1e-3)
    mids = [pbx.ctx_ring_path(c, mf, 0.0, 180.0, s, 21)[0][10] for s in (180.0, -180.0)]
    assert np.allclose(mids[0], [0, 1], atol=1e-3) and np.allclose(mids[1], [0, -1], atol=1e-3)

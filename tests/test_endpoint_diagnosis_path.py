"""Path stage of the endpoint diagnosis (scripts/run_endpoint_diagnosis_path.py) on a small synthetic ring."""
import importlib.util
from pathlib import Path

import numpy as np

from wm import manifold as mf

_p = Path(__file__).resolve().parents[1] / "scripts" / "run_endpoint_diagnosis_path.py"
_s = importlib.util.spec_from_file_location("edp", _p)
edp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(edp)


class _Ring:
    """Unit circle in the first two of D dims; coordinate = angle in radians; step = short way round."""
    periodic = True

    def __call__(self, t):
        t = np.asarray(t, float)
        out = np.zeros(t.shape + (6,))
        out[..., 0], out[..., 1] = np.sin(t) * 5, np.cos(t) * 5
        return out

    def step(self, ta, tb):
        return mf.wrap_pi(np.asarray(tb, float) - np.asarray(ta, float))


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    vals = np.arange(0, 360, 10.0)
    y = np.repeat(vals, 20)
    ring = _Ring()
    X = ring(np.radians(y)) + 0.3 * rng.standard_normal((len(y), 6))
    probe = mf.ProbeReadout(X, y, True)
    near = mf.NearestRealReadout(X, y)
    pf_C = np.array([X[y == v].mean(0) for v in vals])
    return ring, X, y, probe, near, vals, pf_C


def test_fncurve_walk_matches_manifold_coords():
    ring = _Ring()
    fc = edp.FnCurve(ring, ring)
    Z = np.zeros((3, 6))
    ta, tb = np.radians([0.0, 90.0, 350.0]), np.radians([170.0, 10.0, 20.0])
    a = mf.manifold_coords(Z, fc, ta, tb, 11, "shift")
    b = mf.manifold_coords(Z, ring, ta, tb, 11, "shift")
    assert np.allclose(a, b)
    assert np.allclose(a[:, -1], ring(tb) - ring(ta))            # endpoint = f(tb) - f(ta), wrap handled by step


def test_ring_path_keeps_radius_chord_collapses_and_monotone():
    ring, X, y, probe, near, vals, pf_C = _setup()
    src = np.array([0.0, 0.0, 30.0])
    tgt = 170.0
    x = ring(np.radians(src))
    K, s = 11, np.linspace(0, 1, 11)
    ta, tb = np.radians(src), np.radians(np.full(3, tgt))
    W_arc = mf.manifold_coords(x, edp.FnCurve(ring, ring), ta, tb, K, "shift")
    W_chord = mf.linear_coords(x, ring(ta), ring(tb), K)
    shift = ((tgt - src) + 180) % 360 - 180
    r_arc = edp.path_metrics(W_arc, x, src, tgt, s, shift, probe, near, vals, pf_C)
    r_ch = edp.path_metrics(W_chord, x, src, tgt, s, shift, probe, near, vals, pf_C)
    assert r_arc["radius"].shape == (3, K)
    assert (r_arc["radius_min"] > 0.8).all()                      # the arc stays on the ring
    assert (r_ch["radius_min"] < 0.35).all()                      # a near-180 chord passes the centre
    assert (r_arc["monotone_frac"] == 1).all()
    assert (r_arc["err_end"] < 10).all() and (r_ch["err_end"] < 10).all()
    assert r_arc["R_wp"].mean() > r_ch["R_wp"].mean()            # arc waypoints sit nearer the real centroid there

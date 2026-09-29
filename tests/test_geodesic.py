"""scripts/run_geodesic.py: Eq. 4 under G = I is the straight line; under a ring-shaped density energy it bends to the ring."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DTYPE = torch.get_default_dtype()
spec = importlib.util.spec_from_file_location("rg", ROOT / "scripts" / "run_geodesic.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)                      # the script sets torch's default dtype to float64 at import
torch.set_default_dtype(_DEFAULT_DTYPE)          # do not leak it into other test modules


@pytest.fixture(autouse=True)
def _float64_default():
    torch.set_default_dtype(torch.float64)       # as the script runs
    yield
    torch.set_default_dtype(_DEFAULT_DTYPE)


def _ring(n=400, r=1.0, noise=0.03, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    Z = np.zeros((n, dim))
    Z[:, 0], Z[:, 1] = r * np.cos(th), r * np.sin(th)
    return Z + noise * rng.standard_normal((n, dim))


def _arc_init(a, b, M, bump=None):
    s = np.linspace(0, 1, M)[:, None]
    P = a + s * (b - a)
    if bump is not None:
        P = P + np.sin(np.pi * s) * bump
    return P[None]


def test_flat_metric_recovers_chord():
    a, b = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    P0 = _arc_init(a, b, 20, bump=np.array([0.6, 0.6, 0.3, -0.2]))
    P, _ = rg.geodesic(P0, rg.FlatEnergy(), steps=100)
    chord = _arc_init(a, b, 20)
    dmean, dmax = rg.closest(P, chord)
    assert dmax[0] < 1e-3
    assert abs(rg.lg(P, rg.FlatEnergy(), None)[0] - np.linalg.norm(b - a)) < 1e-3


def test_ring_energy_bends_toward_ring():
    Z = _ring()
    e = rg.KNNEnergy(Z, None, k=5)
    th = np.radians([0.0, 100.0])
    a, b = np.array([np.cos(th[0]), np.sin(th[0]), 0, 0]), np.array([np.cos(th[1]), np.sin(th[1]), 0, 0])
    P0 = _arc_init(a, b, 32)
    P, _ = rg.geodesic(P0, e, steps=150)
    Pr = rg.resample(P, 50)
    chord = rg.resample(P0, 50)
    r_geo = np.linalg.norm(Pr[0, :, :2], axis=1)
    r_chord = np.linalg.norm(chord[0, :, :2], axis=1)
    assert r_geo.min() > r_chord.min() + 0.1            # bowed out toward the unit ring
    assert np.abs(r_geo - 1).max() < 0.2
    assert rg.lg(P, e, None)[0] < rg.lg(P0, e, None)[0]
    U = np.eye(4)[:2]
    frac, _ = rg.bend_stats(Pr, chord, rg.resample(P, 50), U)
    assert frac[0] > 0.9                                 # the bend lives in the ring plane

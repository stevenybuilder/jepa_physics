"""scripts/run_offtarget.py: an edit orthogonal to the speed probe's readout direction moves speed by 0; an edit along
it moves speed by the planted amount; the clip bootstrap covers the mean."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "run_offtarget", Path(__file__).resolve().parents[1] / "scripts" / "run_offtarget.py")
ot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ot)


def _speed_probe(seed=0, n=400, D=32):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D))
    w = rng.standard_normal(D)
    y = X @ w + 0.01 * rng.standard_normal(n)
    return ot.RidgeReadout(X, y, np.arange(n) % 5, "scalar"), X


def _readout_direction(read):
    """Raw-space gradient of the linear readout: W / sd (the standardiser divides by sd)."""
    return read.W[:, 0] / read.st.std


def test_orthogonal_edit_changes_speed_by_zero():
    read, X = _speed_probe()
    g = _readout_direction(read)
    rng = np.random.default_rng(1)
    e = rng.standard_normal((50, X.shape[1]))
    e -= np.outer(e @ g / (g @ g), g)                   # remove the readout direction
    x0 = X[:50]
    v = ot.offtarget(read, x0, x0 + 3.0 * e)
    assert np.abs(v).max() < 1e-9


def test_edit_along_readout_changes_speed_by_planted_amount():
    read, X = _speed_probe()
    g = _readout_direction(read)
    x0 = X[:50]
    v = ot.offtarget(read, x0, x0 + 0.5 * g / (g @ g))  # r(x + t g/|g|^2) - r(x) = t
    assert np.allclose(v, 0.5)


def test_cluster_boot_and_spread():
    v = np.r_[np.zeros(10), np.ones(10)]
    ids = np.r_[np.arange(10), np.arange(10, 20)]
    b = ot.cluster_boot(v, ids, n_boot=500)
    assert b["mean"] == 0.5 and b["ci95"][0] < 0.5 < b["ci95"][1] and b["n_clips"] == 20
    assert np.isclose(ot.pooled_within_sd(np.r_[0.0, 2.0, 10.0, 12.0], np.r_[0, 0, 1, 1]), np.sqrt(2.0))

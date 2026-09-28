"""Pure helpers of scripts/session2_fourier4_predictor.py: fold geometry, extrapolation deltas, forced choice."""
import importlib.util
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location(
    "f4", Path(__file__).resolve().parents[1] / "scripts" / "session2_fourier4_predictor.py")
f4 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f4)
from wm.motiongeom import fit_linear_encoding, fourier_features  # noqa: E402


def test_folds_are_32_values_and_exclude_held_arc():
    held = 303.75 + 5.625 * np.arange(8)
    for first, last in f4.FOLDS.values():
        v, vu = f4.fold_values(first, last)
        assert len(v) == 32 and np.all(np.diff(vu) > 0) and not np.isin(v, held).any()
    assert f4.in_span(0.0, 348.75, 163.125) and not f4.in_span(320.0, 348.75, 163.125)
    assert f4.unwrap_to(0.0, 348.75) == 360.0


def test_fourier_half_extrapolates_exactly_on_a_fourier_ring():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(4, 6))
    first, last = f4.FOLDS["A"]
    vals, vals_u = f4.fold_values(first, last)
    Zk = fourier_features(np.repeat(vals, 3), 2) @ B
    _, Bf = fit_linear_encoding(Zk, fourier_features(np.repeat(vals, 3), 2))
    cent = fourier_features(vals, 2) @ B
    src = np.array([150.0, 200.0])
    tgt = np.array([[303.75, 343.125], [315.0, 326.25]])
    Z = fourier_features(src, 2) @ B
    dd, v_e, gap = f4.ext_deltas_pca(Z, src, tgt, vals_u, cent, Bf, first)
    truth = fourier_features(tgt.ravel(), 2).reshape(2, 2, 4) @ B - Z[:, None]
    assert np.allclose(dd["fourier_half"], truth, atol=1e-8)
    assert np.allclose(v_e, 298.125) and np.allclose(gap, tgt - 298.125)
    # chord goes to the end knot for every target; the tangent line is exact at zero gap
    assert np.allclose(dd["chord_nearest"][:, 0], dd["chord_nearest"][:, 1])
    assert np.abs(dd["chord_nearest"] - truth).max() > 1e-3


def test_forced_choice():
    src = np.zeros((1, 4, 3))
    tw = np.ones((1, 4, 3))
    e = np.stack([np.full((4, 3), 0.6), np.full((4, 3), 0.4)])
    assert f4.forced_choice(e, tw, src).tolist() == [True, False]


def test_forced_choice_is_recovery_above_half():
    rng = np.random.default_rng(1)
    e, tw, src = rng.normal(size=(50, 4, 3)), rng.normal(size=(50, 4, 3)), rng.normal(size=(50, 4, 3))
    assert (f4.forced_choice(e, tw, src) == (f4.recovery_full(e, tw, src) > 0.5)).all()


def test_twin_identify():
    rng = np.random.default_rng(2)
    src = np.zeros((1, 1, 2, 3))
    tw = rng.normal(size=(1, 4, 2, 3))
    assert f4.twin_identify(0.1 * tw, tw, src).all()
    assert f4.twin_identify(np.repeat(tw[:, :1], 4, 1), tw, src).tolist() == [[True, False, False, False]]


def test_spline_tangent_is_exact_on_a_straight_ring_segment():
    first, last = f4.FOLDS["A"]
    vals, vals_u = f4.fold_values(first, last)
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.1, -0.2])
    cent = a + vals_u[:, None] * b                      # a straight line in the value: natural spline = the line
    src = np.array([200.0])
    tgt = np.array([[320.625]])
    dd, _, _ = f4.ext_deltas_pca(np.zeros((1, 3)), src, tgt, vals_u, cent, np.zeros((4, 3)), first)
    assert np.allclose(dd["spline_tangent"][0, 0], (320.625 - 200.0) * b)

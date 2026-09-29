"""Unit tests for the depth-fraction / peak / onset arithmetic of scripts/run_scaling_layerwise.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "run_scaling_layerwise", Path(__file__).resolve().parents[1] / "scripts" / "run_scaling_layerwise.py")
rsl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rsl)


def test_depth_fraction():
    assert rsl.depth_fraction(0, 24) == 0.0
    assert rsl.depth_fraction(8, 24) == pytest.approx(1 / 3)
    assert rsl.depth_fraction(16, 32) == 0.5
    assert rsl.depth_fraction(40, 40) == 1.0
    assert rsl.depth_fraction(33, 32) == 1.0      # post-LN point sits at 1.0


def test_curve_summary_peak_onset_and_postln_excluded():
    L = 4
    curve = [0.0, 0.5, 0.95, 1.0, 0.9, 5.0]       # last entry = post-LN point, must be ignored
    s = rsl.curve_summary(curve, L)
    assert s["peak"] == 3 and s["peak_score"] == 1.0
    assert s["peak_frac"] == 0.75
    assert s["onset"] == 2 and s["onset_frac"] == 0.5   # first point >= 0.9 * 1.0
    assert s["final_block_score"] == 0.9 and s["decline"] == pytest.approx(0.1)


def test_curve_summary_threshold_is_inclusive_and_nonpositive_max():
    s = rsl.curve_summary([0.0, 0.9, 1.0], 2)
    assert s["onset"] == 1
    s = rsl.curve_summary([-0.2, -0.1, -0.3], 2)
    assert s["onset"] is None and s["onset_frac"] is None and s["peak"] == 1


def test_curve_summary_matches_wm_availability():
    from wm.probes import availability
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = rng.uniform(-0.1, 1.0, 25)
        a, s = availability(c), rsl.curve_summary(c, 24)
        assert (a["onset"], a["peak"]) == (s["onset"], s["peak"])


def test_curve_summary_length_check():
    with pytest.raises(AssertionError):
        rsl.curve_summary([0.1, 0.2], 4)


def test_halves_zone():
    pts = [1, 4, 6, 7, 8, 9, 10, 12]
    cr = [0.7, -0.7, -0.7, -1.1, -1.4, 0.08, 0.08, 0.4]
    z = rsl.halves_zone(pts, cr, 24)
    assert z["collapse_points"] == [4, 6, 7, 8] and z["recovery_point"] == 9
    assert z["recovery_frac"] == 0.375 and z["collapse_frac"] == [4 / 24, 8 / 24]
    assert z["min_cross_point"] == 8
    assert rsl.halves_zone([1, 2], [0.1, 0.2], 24)["recovery_point"] is None

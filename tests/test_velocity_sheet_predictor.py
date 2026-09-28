import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_velocity_sheet_predictor import (circ_err, heading_from_positions, per_clip_ci,  # noqa: E402
                                          projection_fraction, rescale_to, signed_circ)


def test_heading_image_y_down():
    s = np.arange(4)[:, None]
    for th in (0.0, 30.0, 90.0, 200.0, 315.0):
        r = np.radians(th)
        pos = 128 + s * np.array([np.cos(r), -np.sin(r)]) * 5          # world y up -> row decreases
        assert abs(signed_circ(heading_from_positions(pos), th)) < 1e-9


def test_circ_and_signed():
    assert circ_err(350, 10) == 20 and circ_err(10, 350) == 20
    assert signed_circ(10, 350) == 20 and signed_circ(350, 10) == -20
    assert circ_err(0, 180) == 180


def test_rescale_to():
    d = np.array([[3.0, 4.0], [0.0, 0.0]])
    r, s = rescale_to(d, np.array([10.0, 5.0]))
    assert np.allclose(np.linalg.norm(r[0]), 10) and np.allclose(r[1], 0) and s[1] == 0


def test_projection_fraction():
    t = np.array([[1.0, -2.0]])
    assert np.allclose(projection_fraction(0.5 * t, t), 0.5)
    assert np.allclose(projection_fraction(-t, t), -1.0)


def test_per_clip_ci_drops_nan():
    r = per_clip_ci([1.0, np.nan, 3.0])
    assert r["n"] == 2 and r["mean"] == 2.0 and r["ci95"][0] <= 2.0 <= r["ci95"][1]

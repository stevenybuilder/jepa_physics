import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_temporal_contact as tc  # noqa: E402


def test_reflect_vertical_wall():
    assert np.allclose(tc.reflect([-1.0, 1.0], [1.0, 0.0]), [1.0, 1.0])


def test_bounce_trajectory_contact_and_out_angle():
    xy, th_out = tc.bounce_trajectory([0.0, 0.0], 180.0 - 30.0, 2.4, 5, (1.0, 0.0))
    assert np.allclose(xy[5], [0.0, 0.0])
    assert abs(th_out - 30.0) < 1e-9
    # before contact moving left-up, after contact moving right-up; constant speed
    assert xy[4, 0] > 0 and xy[6, 0] > 0
    assert np.allclose(np.linalg.norm(np.diff(xy, axis=0), axis=1), 0.1)


def test_tubelet_phase():
    assert tc.tubelet_phase(8).tolist() == [-1, -1, -1, -1, 1, 1, 1, 1]
    assert tc.tubelet_phase(5).tolist() == [-1, -1, 0, 1, 1, 1, 1, 1]


def test_turn_fraction():
    assert np.isclose(tc.turn_fraction(30.0, 0.0, 60.0), 0.5)
    assert np.isclose(tc.turn_fraction(350.0, 0.0, 300.0), 1.0 / 6.0)
    assert np.isclose(tc.turn_fraction(60.0, 0.0, 60.0), 1.0)


def test_ablate_and_tokens():
    h = np.zeros((2, 8 * 256, 3))
    fill = np.ones((8 * 256, 3))
    idx = tc.tubelet_tokens((1, 3))
    out = tc.ablate(h, idx, fill)
    assert out[:, 256:512].min() == 1 and out[:, 768:1024].min() == 1 and out[:, :256].max() == 0
    assert h.max() == 0 and len(idx) == 512


def test_sample_bounces_inside_frame():
    cfgs = tc.sample_bounces(20, seed=1)
    lim = 4.0 - (tc.R_PX + 4.0) / tc.PX_PER_M
    for c in cfgs:
        xy, th_out = tc.bounce_trajectory(c["contact_xy_m"], c["theta_in"], c["speed_mps"], c["k_b"], c["normal"])
        assert np.abs(xy).max() <= lim + 1e-9
        assert abs(th_out - c["theta_out"]) < 1e-9
        # incoming moves toward the wall, outgoing away from it
        n = np.array(c["normal"])
        assert (xy[c["k_b"] - 1] - xy[c["k_b"]]) @ n > 0 and (xy[c["k_b"] + 1] - xy[c["k_b"]]) @ n > 0

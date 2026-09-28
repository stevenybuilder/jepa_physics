import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_clock_stimuli import clock_meta, clock_trajectory_m  # noqa: E402

from wm.render_twin import trajectory_m


def _meta(v, a, th=30.0):
    return {"id": 0, "theta_degrees": th, "motion": "velocity" if a == 0 else "acceleration", "speed_mps": v,
            "acceleration_mps2": a, "magnitude": v if a == 0 else a, "start_position_xy_m": [0.3, -0.2],
            "fps": 24, "frames": 16}


def test_time_rescaling_scales_velocity_by_lambda_and_acceleration_by_lambda_squared():
    dt = 1 / 24
    for lam in (0.5, 1.0, 2.0):
        m = _meta(1.5, 2.0)
        x = clock_trajectory_m(m, lam)
        v0 = (x[1] - x[0]) / dt                                    # finite differences at nominal fps
        acc = (x[2] - 2 * x[1] + x[0]) / dt ** 2
        u = np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
        assert np.allclose(acc, lam ** 2 * 2.0 * u)
        assert np.allclose(v0, (lam * 1.5 + 0.5 * lam ** 2 * 2.0 * dt) * u)
        assert np.allclose(x[0], [0.3, -0.2])                      # same start, same direction
        # x_lam(t) = x(lam t) equals the nominal law with effective labels lam v, lam^2 a
        eff = clock_meta(m, lam, "t")
        assert np.isclose(eff["speed_mps"], lam * 1.5) and np.isclose(eff["acceleration_mps2"], lam ** 2 * 2.0)
        assert np.allclose(x, trajectory_m(eff))
        assert eff["speed_mps_nominal"] == 1.5 and eff["clock_lambda"] == lam


def test_in_frame_flag():
    assert clock_meta(_meta(1.0, 0.0), 1.0, "t")["in_frame_all_frames"]
    assert not clock_meta(_meta(14.0, 0.0), 2.0, "t")["in_frame_all_frames"]

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_contact_in_context as cc  # noqa: E402
import run_temporal_contact as tc  # noqa: E402


def test_configs_contact_in_context_and_inside_frame():
    cfgs = cc.sample_bounces_ctx(15, seed=3)
    lim = 4.0 - (tc.R_PX + 4.0) / tc.PX_PER_M
    assert sorted({c["k_b"] for c in cfgs}) == [2, 3, 4, 5, 6]
    for c in cfgs:
        k = c["k_b"]
        assert 2 <= k <= 6 and tc.tubelet_phase(k)[3] == 1          # last context tubelet (frames 6, 7) post-contact
        for kind in cc.KINDS:
            assert np.abs(cc.trajectory(c, kind)).max() <= lim + 1e-9
        b, si, so = (cc.trajectory(c, k_) for k_ in cc.KINDS)
        assert np.allclose(b[:k + 1], si[:k + 1]) and np.allclose(b[k:], so[k:])   # twins share the right halves
        n = np.array(c["normal"])
        assert (b[k - 1] - b[k]) @ n > 0 and (b[k + 1] - b[k]) @ n > 0


def test_straight_out_heading_is_theta_out():
    c = cc.sample_bounces_ctx(1, seed=0)[0]
    d = np.diff(cc.trajectory(c, "straight_out"), axis=0)
    assert np.allclose(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0, c["theta_out"])


def test_turn_fraction_endpoints_for_kinds():
    c = cc.sample_bounces_ctx(1, seed=0)[0]
    assert np.isclose(tc.turn_fraction(c["theta_in"], c["theta_in"], c["theta_out"]), 0.0)
    assert np.isclose(tc.turn_fraction(c["theta_out"], c["theta_in"], c["theta_out"]), 1.0)

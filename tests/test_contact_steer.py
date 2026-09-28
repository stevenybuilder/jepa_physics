import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_contact_in_context as cc  # noqa: E402
import run_contact_steer as cs  # noqa: E402


def test_disk_token_mask_covers_centre_patch():
    xy = np.zeros((16, 2))                      # disk at the image centre, pixel (128, 128)
    m = cs.disk_token_mask(xy).reshape(4, 16, 16)
    assert m[:, 8, 8].all() and m[:, 7, 7].all()         # 10.5 px disk at 128 touches patches 7 and 8
    assert m.sum(axis=(1, 2)).max() <= 9 and not m[:, 0, 0].any()


def test_donors_leave_one_out_and_far():
    cfgs = cc.sample_bounces_ctx(30, seed=2)
    same, far = cs.donors(cfgs)
    for i, c in enumerate(cfgs):
        assert i not in same[i] and i not in far[i]
        assert all(cs.geometry_class(cfgs[j]) == cs.geometry_class(c) for j in same[i])
        assert all(abs(cs.wrap_signed(cfgs[j]["theta_out"] - c["theta_out"])) >= cs.FAR_DEG for j in far[i])


def test_donor_mean_falls_back_to_last_tubelet():
    D = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    phase = np.array([[-1, -1, 1, 1], [1, 1, 1, 1]])
    v = cs.donor_mean(D, phase, [0, 1], 1)
    assert np.allclose(v, (D[0, 3] + D[1, 1]) / 2)

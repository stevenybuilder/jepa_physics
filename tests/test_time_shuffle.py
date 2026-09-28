import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "run_time_shuffle", Path(__file__).resolve().parents[1] / "scripts" / "run_time_shuffle.py")
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = ts.derangement(rng)
        assert sorted(p.tolist()) == list(range(8)) and (p != np.arange(8)).all()


def test_shuffle_tubelets_moves_frame_pairs():
    frames = np.arange(16)[:, None, None, None] * np.ones((16, 2, 2, 3), np.uint8)
    perm = np.array([7, 6, 5, 4, 3, 2, 1, 0])
    out = ts.shuffle_tubelets(frames, perm)
    assert out[:, 0, 0, 0].tolist() == [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1]


def test_scores_pure_positional_and_pure_content():
    rng = np.random.default_rng(1)
    perms = np.array([[ts.derangement(rng) for _ in range(3)] for _ in range(40)])
    slot = np.broadcast_to(np.arange(8), perms.shape).astype(float)
    pos = ts.slot_content_scores(slot + 0.05 * rng.standard_normal(slot.shape), perms, n_boot=50)
    assert pos["frac_nearer_slot"]["mean"] > 0.95 and abs(pos["coef_slot"]["mean"] - 1) < 0.05
    assert abs(pos["coef_content"]["mean"]) < 0.05
    con = ts.slot_content_scores(perms + 0.05 * rng.standard_normal(slot.shape), perms, n_boot=50)
    assert con["frac_nearer_content"]["mean"] > 0.95 and abs(con["coef_content"]["mean"] - 1) < 0.05
    mix = ts.slot_content_scores(0.5 * slot + 0.5 * perms, perms, n_boot=50)
    assert abs(mix["coef_slot"]["mean"] - 0.5) < 1e-6 and abs(mix["coef_content"]["mean"] - 0.5) < 1e-6

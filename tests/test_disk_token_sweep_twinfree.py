import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import session2_disk_token_sweep_twinfree as tf  # noqa: E402


def test_bin_members_exact_and_circular():
    y = np.array([0.0, 5.625, 354.375, 360.0 - 1e-9, 5.625])
    assert list(tf.bin_members(y, 5.625)) == [1, 4]
    assert list(tf.bin_members(y, 0.0)) == [0, 3]


def test_twinfree_dose_kinds():
    n_twin = np.array([1.0, 3.0, 5.0, 100.0])
    prof = np.array([2.0, 2.0, 8.0, 50.0])
    mask = np.array([True, True, True, False])
    assert np.allclose(tf.twinfree_dose(prof, n_twin, mask, "pertoken"), [1, 3, 5, 0])
    assert np.allclose(tf.twinfree_dose(prof, n_twin, mask, "uniform"), [3, 3, 3, 0])
    m = tf.twinfree_dose(prof, n_twin, mask, "matched")
    assert np.isclose(m[mask].mean(), n_twin[mask].mean()) and m[3] == 0       # same total dose as uniform
    assert np.allclose(m[mask] / m[mask].sum(), prof[mask] / prof[mask].sum())  # twin-free shape
    assert np.allclose(tf.twinfree_dose(prof, n_twin, mask, "raw"), [2, 2, 8, 0])
    assert np.allclose(tf.twinfree_dose(prof, n_twin, np.zeros(4, bool), "matched"), 0)


def test_matched_dose_ignores_twin_pattern():
    """Two twins with the same set mean but different per-token pattern give the same matched dose."""
    prof, mask = np.array([1.0, 2.0, 3.0]), np.ones(3, bool)
    a = tf.twinfree_dose(prof, np.array([2.0, 2.0, 2.0]), mask, "matched")
    b = tf.twinfree_dose(prof, np.array([0.0, 0.0, 6.0]), mask, "matched")
    assert np.allclose(a, b)

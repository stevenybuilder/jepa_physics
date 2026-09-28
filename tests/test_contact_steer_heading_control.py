import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_contact_steer_heading_control as hc  # noqa: E402


def test_match_norm_keeps_direction_and_takes_ref_norm():
    v, ref = np.array([3.0, 4.0]), np.array([0.0, 10.0])
    out = hc.match_norm(v, ref)
    assert np.isclose(np.linalg.norm(out), 10.0) and np.allclose(out / 10.0, v / 5.0)
    assert np.allclose(hc.match_norm(np.zeros(2), ref), 0.0)
    rows = hc.match_norm(np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[0.0, 3.0], [4.0, 0.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=-1), [3.0, 4.0])


def test_paired_difference_and_ci():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    r = hc.paired(a, a - 0.5)
    assert np.isclose(r["mean"], 0.5) and np.allclose(r["ci95"], 0.5) and r["frac_carriers_positive"] == 1.0
    r2 = hc.paired(np.array([0.0, 1.0, np.nan]), np.array([1.0, 1.0, 0.0]))
    assert r2["n"] == 2 and np.isclose(r2["mean"], -0.5) and r2["ci95"][0] <= -0.5 <= r2["ci95"][1]


def test_ratio_ci():
    r = hc.ratio_ci(np.array([2.0, 4.0]), np.array([1.0, 2.0]))
    assert np.isclose(r["ratio"], 2.0) and np.allclose(r["ci95"], 2.0)

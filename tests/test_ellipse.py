"""Shape estimators of scripts/run_ellipse.py: a planted circle gives b/a = 1 and no 2-theta power; a planted b/a = 0.5
ellipse is recovered by all three estimators (plane fit, eigenvalues, angle distortion); a saddle bend does not fool
the ring-plane estimators."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from wm import ellipse as el

ROOT = Path(__file__).resolve().parents[1]
TH = np.arange(64) * 2 * np.pi / 64


def run_ellipse():
    spec = importlib.util.spec_from_file_location("run_ellipse", ROOT / "scripts" / "run_ellipse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def planted(q, rot=0.6, phase=1.1, D=32, seed=0):
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    P = np.stack([3 * np.cos(TH + phase), 3 * q * np.sin(TH + phase)], 1) @ R.T + [1.0, -2.0]
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, 2)))[0]
    return P, P @ Q.T


def three_estimators(P, C):
    con = el.conic_fit(P)
    return {"conic": con["q"], "polar": el.polar_fit(P, con)["q"], "chart": el.chart_fit(P, TH)["q"],
            "eig": el.centroid_spectrum(C)["q_eig"], "k1_chart_full": el.chart_q_full(C, TH)[0]["q"],
            "angle_exact": el.angle_distortion(np.arctan2(*(P - P.mean(0))[:, ::-1].T), TH)["q_exact"],
            "angle_linear": el.angle_distortion(np.arctan2(*(P - P.mean(0))[:, ::-1].T), TH)["q_linear"]}


def test_circle_is_q1_with_no_2theta_power():
    P, C = planted(1.0)
    for name, q in three_estimators(P, C).items():
        assert q == pytest.approx(1.0, abs=1e-4), name
    rf = el.radius_fourier(C, TH)
    assert rf["p2_over_p0"] < 1e-12 and rf["q_from_r2"] == pytest.approx(1.0, abs=1e-6)
    assert el.harmonic_variance(C, TH)["share"][0] == pytest.approx(1.0)


def test_ellipse_half_recovered_by_all_estimators():
    P, C = planted(0.5)
    for name, q in three_estimators(P, C).items():
        assert q == pytest.approx(0.5, abs=0.02), name
    assert el.radius_fourier(C, TH)["q_from_r2"] == pytest.approx(0.5, abs=0.02)
    assert el.angle_distortion(np.arctan2(*(P - P.mean(0))[:, ::-1].T), TH)["r2_2theta"] > 0.95


def clips(q, bend=0.0, per=20, D=48, noise=0.3, seed=1):
    """64 values x per clips: ellipse (3, 3q) in dims 0-1, saddle bend * 3 cos 2 theta in dim 2, isotropic noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.degrees(TH), per)
    t = np.radians(y)
    X = noise * rng.standard_normal((len(y), D))
    X[:, 0] += 3 * np.cos(t)
    X[:, 1] += 3 * q * np.sin(t)
    X[:, 2] += bend * 3 * np.cos(2 * t)
    return X @ np.linalg.qr(rng.standard_normal((D, D)))[0], y


@pytest.mark.parametrize("q", [1.0, 0.5])
def test_noisy_clips_through_the_script(q):
    s = run_ellipse().analyse(*clips(q), k=16)[0]["summary"]
    for key in ("q_conic", "q_polar", "q_k1_chart_full_noise_corrected", "q_eig_top2_full_noise_corrected",
                "q_angle_exact"):
        assert s[key] == pytest.approx(q, abs=0.02), key
    if q == 1.0:
        assert s["p2_over_p0"] < 1e-3


def test_saddle_bend_contaminates_top2_but_not_ring_plane():
    s = run_ellipse().analyse(*clips(0.6, bend=0.9), k=16)[0]["summary"]
    assert s["k2_axis_rank"] == 2                                   # bend variance 0.81 * 4.5 > minor 0.36 * 4.5
    assert abs(s["q_eig_top2_full_noise_corrected"] - 0.6) > 0.2
    for key in ("q_conic", "q_k1_chart_full_noise_corrected", "q_angle_exact"):
        assert s[key] == pytest.approx(0.6, abs=0.03), key
    assert s["var_share_k2"] == pytest.approx(0.81 / (1 + 0.36 + 0.81), abs=0.03)

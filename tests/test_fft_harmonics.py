"""Planted rings for scripts/run_fft_harmonics.py: a circle is all k = 1; a saddle ring splits k = 1 / k = 2 by the
squared amplitudes."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "run_fft_harmonics", Path(__file__).resolve().parents[1] / "scripts" / "run_fft_harmonics.py")
fh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fh)

TH = np.radians(np.arange(64) * 360.0 / 64)


def _embed(C, D=64, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, C.shape[1])))
    return C @ Q.T


def test_planted_ring_is_k1():
    f = fh.harmonic_fractions(_embed(np.stack([np.cos(TH), np.sin(TH)], 1)))
    assert f["k1"] > 0.999 and f["k2"] < 1e-9 and f["k4plus"] < 1e-9


def test_planted_saddle_ring_ratio():
    b = 0.6                                              # saddle bend: third axis b * cos(2 theta)
    C = _embed(np.stack([np.cos(TH), 0.8 * np.sin(TH), b * np.cos(2 * TH)], 1))
    f = fh.harmonic_fractions(C)
    k1, k2 = (1 + 0.8 ** 2) / 2, b ** 2 / 2              # mean-square power of each harmonic
    assert np.isclose(f["k2"], k2 / (k1 + k2), atol=1e-9)
    assert np.isclose(f["k1"], k1 / (k1 + k2), atol=1e-9)
    assert f["k3"] < 1e-9


def test_spectrum_point_recovers_saddle_from_noisy_clips():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(64) * 5.625, 20)
    t = np.radians(y)
    X = _embed(np.stack([np.cos(t), np.sin(t), 0.7 * np.cos(2 * t)], 1), D=128) + 0.05 * rng.standard_normal((len(y), 128))
    r = fh.spectrum_point(X, y, n_boot=20, n_shuffle=3)
    expect = 0.7 ** 2 / 2 / (1 + 0.7 ** 2 / 2)
    assert r["ci95"]["k2"][0] - 0.02 < expect < r["ci95"]["k2"][1] + 0.02
    assert r["shuffle_noise_power_over_real"] < 0.1


def test_ramp_reference_is_sawtooth_not_pure_k1():
    r = fh.ramp_reference()
    assert 0.55 < r["k1"] < 0.65 and r["k2"] > 0.1

import numpy as np

from wm.metrics import circular_mae_deg, r2


def test_wraparound():
    t = np.radians(1.0)
    assert abs(circular_mae_deg(np.array([359.0]), np.sin([t]), np.cos([t])) - 2.0) < 1e-9
    t = np.radians(359.0)
    assert abs(circular_mae_deg(np.array([1.0]), np.sin([t]), np.cos([t])) - 2.0) < 1e-9


def test_exact_and_opposite():
    theta = np.linspace(0, 359, 50)
    r = np.radians(theta)
    assert circular_mae_deg(theta, np.sin(r), np.cos(r)) < 1e-9
    assert abs(circular_mae_deg(theta, -np.sin(r), -np.cos(r)) - 180.0) < 1e-6


def test_chance_is_90():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 360, 100_000)
    guess = rng.uniform(0, 2 * np.pi, 100_000)
    assert abs(circular_mae_deg(theta, np.sin(guess), np.cos(guess)) - 90.0) < 1.0


def test_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0
    assert abs(r2(y, np.full(4, y.mean()))) < 1e-12
    Y = np.stack([y, -y], axis=1)
    assert r2(Y, Y) == 1.0


def test_readout_radius():
    from wm.metrics import radius_summary, readout_radius
    P = np.array([[0.6, 0.8], [0.0, 0.5], [3.0, 4.0]])
    assert np.allclose(readout_radius(P), [1.0, 0.5, 5.0])
    s = radius_summary(P)
    assert s["median"] == 1.0 and abs(s["mean"] - 6.5 / 3) < 1e-12

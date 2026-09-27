"""scripts/run_cosine_tangent.py: step / tangent / chord cosines on planted geometry."""
import numpy as np

from test_p2_extras import load


def embed(P, D=12, seed=0):
    """Rotate low-dimensional points [n, K, p] into D dimensions with a random orthonormal frame."""
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))[0][:, :P.shape[-1]]
    return P @ Q.T + 3.0


def test_planted_circle_tangent_vs_chord():
    mod = load("run_cosine_tangent")
    arc = np.radians(90.0)
    for K in (51, 50):
        phi = np.linspace(0.0, arc, K)
        W = embed(np.stack([np.cos(phi), np.sin(phi)], -1)[None] * 2.0)
        chord = W[:, -1] - W[:, 0]
        c = mod.path_cosines(W, chord, chord)
        tc = c["tangent_chord"][0]
        np.testing.assert_allclose([tc[0], tc[-1]], np.cos(arc / 2), atol=1e-3)
        np.testing.assert_allclose(mod.midpoint(tc), 1.0, atol=1e-3)
        np.testing.assert_allclose(c["step_tangent"], c["tangent_chord"])
        np.testing.assert_allclose(c["tangent_net"], c["tangent_chord"], atol=1e-12)
        np.testing.assert_allclose([c["step_chord"][0], c["step_net"][0]], 1.0)
        assert np.all(np.diff(tc[: K // 2]) > 0)            # agreement rises from the ends to the middle


def test_straight_manifold_all_ones():
    mod = load("run_cosine_tangent")
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((4, 12)), rng.standard_normal((4, 12))
    s = np.linspace(0.0, 1.0, 50)[None, :, None] ** 2              # uneven spacing along the line
    W = a[:, None] + s * (b - a)[:, None]
    c = mod.path_cosines(W, 3.0 * (b - a), b - a)
    for q in c:
        np.testing.assert_allclose(c[q], 1.0, atol=1e-9)

"""wm.conceptor: isotropic conceptor, contrastive AND-NOT on orthogonal subspaces, and a planted-ring steer."""
import numpy as np

from wm import conceptor as cc


def test_isotropic_gaussian_gives_scaled_identity():
    rng = np.random.default_rng(0)
    d, s2, alpha = 8, 4.0, 0.5
    C = cc.conceptor(rng.normal(0, np.sqrt(s2), (20000, d)), alpha)
    np.testing.assert_allclose(C, s2 / (s2 + alpha ** -2) * np.eye(d), atol=0.02)


def test_contrastive_keeps_target_subspace_removes_source():
    rng = np.random.default_rng(1)
    d, alpha = 20, 1.0
    Xt = np.zeros((500, d)); Xt[:, :5] = rng.normal(0, 2.0, (500, 5))        # target varies in e0..e4
    Xs = np.zeros((500, d)); Xs[:, 5:10] = rng.normal(0, 2.0, (500, 5))      # source varies in e5..e9
    Ct, Cs = cc.conceptor(Xt, alpha), cc.conceptor(Xs, alpha)
    C = cc.contrastive(Ct, Cs, "jaeger")
    e = np.linalg.eigvalsh(C)
    assert e.min() > -1e-8 and e.max() < 1 + 1e-8                            # a valid conceptor
    np.testing.assert_allclose(np.diag(C)[:5], np.diag(Ct)[:5], atol=1e-6)   # target directions kept
    assert np.abs(C[5:, 5:]).max() < 1e-6                                     # source (and unused) directions removed
    # COAST's pinv form on singular conceptors keeps the source directions (why the runs use Jaeger's AND)
    assert np.abs(cc.contrastive(Ct, Cs, "pinv")[5:10, 5:10]).max() > 0.1


def test_steering_a_planted_ring_moves_the_readout_toward_target():
    rng = np.random.default_rng(2)
    d, r, noise = 16, 5.0, 0.2

    def clips(deg, n=40):
        t = np.radians(np.repeat(np.atleast_1d(deg), n))
        X = rng.normal(0, noise, (len(t), d))
        X[:, 0] += r * np.cos(t); X[:, 1] += r * np.sin(t)
        return X

    Xt = clips(np.arange(45, 136, 15))                                        # target condition: arc around 90 deg
    Xs = clips(0.0)                                                           # source condition: 0 deg
    C = cc.contrastive(cc.conceptor(Xt, 1.0), cc.conceptor(Xs, 1.0), "jaeger")
    assert min(C[0, 0], C[1, 1]) > 5 * np.diag(C)[2:].max()                   # the ring plane dominates C
    z = clips(0.0, 10)
    path = cc.aimed_path(z, C, Xt.mean(0), np.linspace(0, 1, 11))
    ang = np.degrees(np.arctan2(path[..., 1], path[..., 0]))                  # the planted readout
    assert np.all(np.abs(90 - ang[:, -1]) < np.abs(90 - ang[:, 0]) - 30)      # every clip at least 30 deg closer
    assert np.all(np.diff(ang.mean(0)) > 0)                                   # monotone toward 90 deg

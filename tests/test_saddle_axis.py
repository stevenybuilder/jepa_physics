"""scripts/run_saddle_axis.py: saddle-axis fit, projection and variance-matched random axes on a planted saddle ring."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sad", ROOT / "scripts" / "run_saddle_axis.py")
sad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sad)


def planted(k=12, m=64, a=3.0, b=2.0, h=1.2, phase=20.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((k, k)))[0]
    e1, e2, e3 = Q[:, 0], Q[:, 1], Q[:, 2]
    th = np.radians(np.arange(m) * 360.0 / m)
    mu = rng.standard_normal(k)
    C = (mu + a * np.cos(th)[:, None] * e1 + b * np.sin(th)[:, None] * e2
         + h * np.cos(2 * (th - np.radians(phase)))[:, None] * e3 + noise * rng.standard_normal((m, k)))
    return C, th, mu, (e1, e2, e3)


def test_fit_recovers_plane_axis_phase_and_share():
    C, th, mu, (e1, e2, e3) = planted()
    f = sad.fit_saddle(C, th)
    assert abs(abs(f["u"] @ e3) - 1) < 1e-3
    assert np.allclose(np.linalg.svd(f["R"].T @ np.c_[e1, e2], compute_uv=False), 1, atol=1e-3)
    assert abs(((f["phase_deg"] - 20.0 + 45) % 90) - 45) < 1.0          # phase mod 90 (sign of u is a convention)
    # centroid variance: ring (a² + b²) / 2 = 6.5, saddle h² / 2 = 0.72
    assert abs(f["share_u"] - 0.72 / 7.22) < 0.01
    assert f["rank_ratio_A2"] < 0.05 and f["u_harmonic_share_k2"] > 0.99
    assert abs(f["a1_axis_ratio"] - 2 / 3) < 0.01
    assert np.allclose(f["mu"], mu, atol=0.02)


def test_projection_removes_scales_and_flips_only_the_axis():
    C, th, mu, (e1, e2, e3) = planted(noise=0.0)
    f = sad.fit_saddle(C, th)
    for scale in (0.0, 2.0, -1.0):
        d, c = sad.rescale_axis(C, f["mu"], f["u"], scale)
        X = C + d
        assert np.allclose((X - f["mu"]) @ f["u"], scale * c, atol=1e-9)
        assert np.allclose((X - C) @ f["R"], 0, atol=1e-9)            # ring-plane coordinates untouched
    d0, _ = sad.rescale_axis(C, f["mu"], f["u"], 0.0)
    assert np.allclose((C + d0 - mu) @ e3, 0, atol=1e-6)               # removed = projected onto the ring plane
    W = np.random.default_rng(3).standard_normal((5, 4, C.shape[1]))
    Wf = sad.flatten_path(W, f["u"])
    assert np.allclose(Wf @ f["u"], 0) and np.allclose(W - Wf, (W @ f["u"])[..., None] * f["u"])


def test_variance_matched_axes():
    rng = np.random.default_rng(1)
    k = 16
    A = rng.standard_normal((400, k)) * np.linspace(3, 0.2, k)
    Sig = np.cov(A.T)
    C, th, _, _ = planted(k=k)
    f = sad.fit_saddle(C, th)
    Q = sad.complement_basis(np.c_[f["R"], f["u"]], k)
    assert Q.shape == (k, k - 3) and np.allclose(Q.T @ Q, np.eye(k - 3), atol=1e-9)
    lam = np.linalg.eigvalsh(Q.T @ Sig @ Q)
    target = float(np.sqrt(lam.min() * lam.max()))
    ax, p = sad.variance_matched_axes(Sig, Q, target, 20, seed=0)
    assert np.allclose(np.einsum("nk,kl,nl->n", ax, Sig, ax), target, rtol=1e-6)
    assert np.allclose(np.linalg.norm(ax, axis=1), 1)
    assert np.allclose(ax @ np.c_[f["R"], f["u"]], 0, atol=1e-9)
    assert len({round(v, 6) for v in (ax @ ax[0])}) > 10                # distinct draws
    try:
        sad.variance_matched_axes(Sig, Q, lam.max() * 2, 1)
        raise AssertionError("out-of-range target accepted")
    except ValueError:
        pass


def test_radial_rescale_keeps_angle_and_matched_planes():
    C, th, mu, (e1, e2, e3) = planted(noise=0.0)
    f = sad.fit_saddle(C, th)
    d, r = sad.rescale_plane(C, f["mu"], f["R"], 0.5)
    c0, c1 = (C - f["mu"]) @ f["R"], (C + d - f["mu"]) @ f["R"]
    assert np.allclose(c1, 0.5 * c0) and np.allclose(r, np.linalg.norm(c0, axis=1))
    assert np.allclose(d @ f["u"], 0, atol=1e-9)
    rng = np.random.default_rng(2)
    k = C.shape[1]
    Sig = np.cov((rng.standard_normal((500, k)) * np.linspace(3, 0.2, k)).T)
    Q = sad.complement_basis(np.c_[f["R"], f["u"]], k)
    lam = np.linalg.eigvalsh(Q.T @ Sig @ Q)
    tg = [float(np.sqrt(lam.min() * lam.max())), float(np.sqrt(lam.min() * lam.max())) * 0.8]
    Pl = sad.variance_matched_planes(Sig, np.c_[f["R"], f["u"]], tg, 3, seed=0)
    for B in Pl:
        assert np.allclose(B.T @ B, np.eye(2), atol=1e-9)
        assert np.allclose(np.diag(B.T @ Sig @ B), tg, rtol=1e-6)
        assert np.allclose(B.T @ np.c_[f["R"], f["u"]], 0, atol=1e-9)

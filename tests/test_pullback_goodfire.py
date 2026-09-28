"""scripts/session2_pullback_goodfire.py: causalab recap metrics and the Goodfire pullback on a synthetic ring."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pbg", ROOT / "scripts" / "session2_pullback_goodfire.py")
pbg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbg)


def arc(a0, a1, n, r=1.0):
    t = np.linspace(a0, a1, n)
    return np.stack([r * np.cos(t), r * np.sin(t), np.zeros(n)], 1)


def test_recap_metrics_on_known_curves():
    sp = arc(0, np.pi * 0.75, 20)
    ch = sp[:1] + np.linspace(0, 1, 20)[:, None] * (sp[-1:] - sp[:1])
    on = pbg.recap_metrics(arc(0.01, np.pi * 0.74, 20), sp)                         # reparameterised arc
    assert on["mean_closest_point_residual"] < 1e-3 and on["r_squared"] > 0.999
    lin = pbg.recap_metrics(ch, sp)
    sag = 1 - np.cos(np.pi * 0.375)                                                   # max chord-to-arc gap
    assert 0.3 * sag < lin["mean_closest_point_residual"] < sag and lin["r_squared"] < 0.9
    assert pbg.recap_metrics(ch, ch)["mean_closest_point_residual"] < 1e-9
    # closest point, not equal-t: a path bunched at the arc's start is still on the arc
    bunched = arc(0, np.pi * 0.2, 20)
    assert pbg.recap_metrics(bunched, sp)["mean_closest_point_residual"] < 3e-3
    s = pbg.score_triplet(sp, sp, ch, unit=2.0)
    assert s["resid_to_spline"] < 1e-9 and s["resid_to_chord"] > 0 and s["r2_spline"] > s["r2_chord"]


def test_pullback_recovers_ring_on_synthetic_readout():
    """Toy: 3-d activations, readout P = first two coords, p = softmax(kappa <P, u_k>) over 32 values (the readout
    family used on the real data). Behaviour centroids from clips on the unit ring; target = sqrt-space spline
    geodesic. Top-2 coordinates replaced by the free path, coordinate 3 kept per carrier. From the chord init,
    L-BFGS must pull the path onto the ring (the chord's interior gives too-flat distributions)."""
    rng = np.random.default_rng(0)
    vals = np.arange(32) * 360.0 / 32
    U = np.stack([np.sin(np.radians(vals)), np.cos(np.radians(vals))], 1)
    kappa = 6.0
    clips = np.repeat(vals, 10)
    P = np.stack([np.sin(np.radians(clips)), np.cos(np.radians(clips))], 1)          # clips on the unit ring
    kap, _ = pbg.fit_kappa(P + 0.3 * rng.standard_normal(P.shape), np.searchsorted(vals, clips), U)
    assert 1 < kap < 400
    pd = pbg.readout_dist(P, U, kappa)
    B = np.stack([pd[clips == v].mean(0) for v in vals])
    spl = pbg.sqrt_spline(vals, B)
    src, shift = 45.0, 135.0
    ts = np.linspace(0, 1, pbg.K)
    tgt = torch.tensor(pbg.behaviour_target(spl, src, shift, ts))
    ring = np.stack([np.sin(np.radians(src + ts * shift)), np.cos(np.radians(src + ts * shift))], 1)
    chord = ring[:1] + ts[:, None] * (ring[-1:] - ring[:1])
    car = torch.tensor(rng.standard_normal((16, 3)))                                  # 16 carriers
    Ut = torch.tensor(U)
    V = torch.tensor(chord).clone().requires_grad_(True)

    def closure_fn():
        h = car[None].repeat(pbg.K, 1, 1).clone()
        h[..., :2] = V[:, None, :]                                                   # replacement of the top coords
        logp = torch.log_softmax(kappa * (h[..., :2] @ Ut.T), -1)
        dh2 = 1 - (torch.exp(0.5 * logp) * tgt[:, None].sqrt()).sum(-1)
        loss = dh2.mean(1).sum()
        loss.backward()
        return loss.detach(), dh2.mean(1).tolist()

    res = pbg.optimise(V, closure_fn, log=lambda s: None)
    assert res["optimiser"] == "lbfgs_strong_wolfe" and res["history"][-1] < 0.2 * res["history"][0]
    s = pbg.score_triplet(V.detach().numpy(), ring, chord, unit=1.0)
    assert s["resid_to_spline"] < 0.5 * s["resid_to_chord"]
    assert s["r2_spline"] > 0.9
    assert s["resid_to_spline"] < s["chord_baseline_resid_to_spline"]

"""CPU checks of wm.repair: hook bookkeeping on a toy block with the HF VJEPA2Layer structure, and the projection
removal used by the patching test (scripts/run_repair_attribution.py)."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.repair import SublayerTap, remove_along  # noqa: E402

D, N, B = 8, 6, 3


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x, position_mask=None, head_mask=None, output_attentions=False):
        w = torch.softmax(x @ x.transpose(1, 2), -1)          # token mixing, like attention
        return (self.lin(w @ x),)


class Layer(nn.Module):
    """Same forward as transformers 4.56.2 VJEPA2Layer (drop_path = identity)."""
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attention = Attn()
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, h, position_mask=None, head_mask=None, output_attentions=False):
        r = h
        h = self.attention(self.norm1(h))[0] + r
        r = h
        h = self.mlp(self.norm2(h)) + r
        return (h,)


def toy():
    torch.manual_seed(0)
    return nn.ModuleList([Layer() for _ in range(4)])


def run(layers, h, start=0):
    for layer in layers[start:]:
        h = layer(h)[0]
    return h


def test_remove_along():
    torch.manual_seed(1)
    x, ref = torch.randn(B, N, D), torch.randn(B, N, D)
    u = torch.nn.functional.normalize(torch.randn(B, D), dim=1)
    y = remove_along(x, ref, u)
    assert torch.allclose(((y - ref) * u[:, None]).sum(-1), torch.zeros(B, N), atol=1e-6)
    orth = lambda z: z - (z * u[:, None]).sum(-1, keepdim=True) * u[:, None]   # noqa: E731
    assert torch.allclose(orth(y - ref), orth(x - ref), atol=1e-6)
    assert torch.allclose(remove_along(ref, ref, u), ref)


def test_bookkeeping_matches_manual_forward():
    layers = toy()
    h = torch.randn(B, N, D)
    with torch.no_grad(), SublayerTap(layers, [1, 2, 3, 4]) as t:
        out = run(layers, h)
    hh = h
    with torch.no_grad():
        for b, layer in enumerate(layers, start=1):
            a = layer.attention(layer.norm1(hh))[0]
            mid = hh + a
            m = layer.mlp(layer.norm2(mid))
            hh = mid + m
            assert torch.allclose(t.rec[("attn", b)]["sub_mp"], a.mean(1), atol=1e-6)
            assert torch.allclose(t.rec[("attn", b)]["resid_mp"], mid.mean(1), atol=1e-6)
            assert torch.allclose(t.rec[("mlp", b)]["sub_mp"], m.mean(1), atol=1e-6)
            assert torch.allclose(t.rec[("mlp", b)]["resid_mp"], hh.mean(1), atol=1e-6)
    assert torch.allclose(out, hh)
    assert t.stacked("resid_mp").shape == (B, 8, D)
    assert t.sites()[:3] == [("attn", 1), ("mlp", 1), ("attn", 2)]
    for layer in layers:          # hooks removed on exit
        assert not layer._forward_hooks and not layer.attention._forward_hooks
        assert not layer.norm2._forward_pre_hooks and not layer.mlp._forward_hooks


def test_survival_decomposes_into_sublayer_contributions():
    layers = toy()
    h = torch.randn(B, N, D)
    delta = torch.randn(B, D)
    u = delta / delta.norm(dim=1, keepdim=True)
    with torch.no_grad():
        with SublayerTap(layers, [2, 3, 4], keep_tokens=True) as c:
            run(layers, h, 1)
        with SublayerTap(layers, [2, 3, 4], clean=c.tokens, u=u) as e:
            run(layers, h + delta[:, None], 1)
    surv_prev = torch.ones(B)
    for b in (2, 3, 4):
        for k in ("attn", "mlp"):
            dR = e.rec[(k, b)]["resid_mp"] - c.rec[(k, b)]["resid_mp"]
            dS = e.rec[(k, b)]["sub_mp"] - c.rec[(k, b)]["sub_mp"]
            surv = (dR * delta).sum(1) / (delta ** 2).sum(1)
            contrib = (dS * delta).sum(1) / (delta ** 2).sum(1)
            assert torch.allclose(surv, surv_prev + contrib, atol=1e-5)   # residual stream is additive
            surv_prev = surv
            assert e.rec[(k, b)]["sub_tok_cos"].shape == (B,)


def test_patch_removes_only_the_edit_direction():
    layers = toy()
    h = torch.randn(B, N, D)
    delta = torch.randn(B, D)
    u = delta / delta.norm(dim=1, keepdim=True)
    with torch.no_grad():
        with SublayerTap(layers, [2, 3], keep_tokens=True) as c:
            run(layers, h, 1)
        with SublayerTap(layers, [2, 3], clean=c.tokens, u=u, patch={("attn", 2), ("attn", 3)}) as p:
            run(layers, h + delta[:, None], 1)
        # zero edit + patch = the clean run exactly
        with SublayerTap(layers, [2, 3], clean=c.tokens, u=u, patch={("attn", 2), ("mlp", 2)}) as z:
            out0 = run(layers, h.clone(), 1)
    for b in (2, 3):
        dS = p.rec[("attn", b)]["sub_mp"] - c.rec[("attn", b)]["sub_mp"]
        assert torch.allclose((dS * u).sum(1), torch.zeros(B), atol=1e-5)
        assert torch.allclose(p.rec[("attn", b)]["sub_tok_cos"], torch.zeros(B), atol=1e-5)
    # attention-only patch: attention adds nothing along u, so the edit's survival after attn equals before it
    dR_in = (h + delta[:, None]).mean(1) - h.mean(1)
    dR_mid = p.rec[("attn", 2)]["resid_mp"] - c.rec[("attn", 2)]["resid_mp"]
    assert torch.allclose((dR_mid * u).sum(1), (dR_in * u).sum(1), atol=1e-5)
    assert torch.allclose(z.rec[("mlp", 3)]["resid_mp"], c.rec[("mlp", 3)]["resid_mp"], atol=1e-6)
    assert torch.allclose(out0, run(layers, h, 1), atol=1e-6)

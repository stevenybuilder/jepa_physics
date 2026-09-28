"""Sublayer attribution of the repair of an activation edit (REPORT §4.5: point-12 edits are undone within four blocks).

Hooks on the HF V-JEPA 2 encoder blocks (transformers 4.56.2 VJEPA2Layer.forward: h_mid = h + attn(norm1(h)),
h_out = h_mid + mlp(norm2(h_mid))), identified by 1-indexed block number b (= enc.layer[b - 1]; the output of block b is
our point b, see wm.propagate):

  ("attn", b)  forward hook on layer.attention  -> sublayer output a (tokens), optional patch
               forward pre-hook on layer.norm2  -> residual after attention h_mid
  ("mlp", b)   forward hook on layer.mlp        -> sublayer output m (tokens), optional patch
               forward hook on the layer        -> residual after the MLP h_out (= point b)

Per site the tap records the token-mean of the residual and of the sublayer output ([B, D], same summation order as
wm.propagate._mean). With a clean reference (the same rows run without the edit, keep_tokens=True) it also records
token-level statistics of the edit-induced change against the per-row unit edit direction u [B, D].

Patch ("remove the sublayer's contribution to the edit subspace"): for each patched site the sublayer output becomes
x - ((x - x_clean) . u) u per token, i.e. the change the sublayer makes relative to the clean run is kept except its
component along u.
"""
import torch

from wm.propagate import _mean


def remove_along(x, ref, u):
    """x, ref [B, N, D]; u [B, D] unit rows. Remove from x the component of (x - ref) along u, token by token."""
    uu = u[:, None, :].to(x.dtype)
    coef = ((x - ref) * uu).sum(-1, keepdim=True)
    return x - coef * uu


class SublayerTap:
    """Context manager registering the hooks on `layers` (the encoder's nn.ModuleList) for blocks `blocks`.

    keep_tokens: store the token tensors (clean run) in self.tokens[(kind, b)] = dict(sub=, resid=).
    clean: a previous tap's .tokens, enables change statistics against u (and is required for patching).
    u: [B, D] unit edit directions (tensor), one per row.
    patch: set of (kind, b) sites whose sublayer output change along u is removed.
    """

    def __init__(self, layers, blocks, keep_tokens=False, clean=None, u=None, patch=()):
        self.layers, self.blocks = layers, sorted(int(b) for b in blocks)
        self.keep_tokens, self.clean, self.u, self.patch = keep_tokens, clean, u, set(patch)
        assert not self.patch or (clean is not None and u is not None), "patching needs a clean reference and u"
        self.rec, self.tokens, self._pending, self._handles = {}, {}, {}, []

    # ------------------------------------------------------------ bookkeeping
    def _sub(self, kind, b, x):
        site = (kind, b)
        if site in self.patch:
            x = remove_along(x, self.clean[site]["sub"], self.u)
        r = self.rec.setdefault(site, {})
        r["sub_mp"] = _mean(x).float().cpu()
        if self.keep_tokens:
            self.tokens.setdefault(site, {})["sub"] = x.detach().clone()
        if self.clean is not None and self.u is not None:
            d = (x - self.clean[site]["sub"]).float()
            u = self.u[:, None, :].float()
            n = d.norm(dim=-1)
            r["sub_tok_cos"] = ((d * u).sum(-1) / n.clamp_min(1e-12)).mean(1).cpu()
            r["sub_tok_rms"] = n.pow(2).mean(1).sqrt().cpu()
        return x

    def _resid(self, kind, b, h):
        site = (kind, b)
        r = self.rec.setdefault(site, {})
        r["resid_mp"] = _mean(h).float().cpu()
        r["resid_tok_norm_mean"] = h.float().norm(dim=-1).mean(1).cpu()
        if self.keep_tokens:
            self.tokens.setdefault(site, {})["resid"] = h.detach().clone()
        if self.clean is not None and self.u is not None:
            d = (h - self.clean[site]["resid"]).float()
            r["resid_tok_rms_change"] = d.norm(dim=-1).pow(2).mean(1).sqrt().cpu()

    # ------------------------------------------------------------ hooks
    def __enter__(self):
        for b in self.blocks:
            layer = self.layers[b - 1]

            def attn_hook(mod, args, out, b=b):
                a = self._sub("attn", b, out[0])
                return (a,) + tuple(out[1:])

            def norm2_pre(mod, args, b=b):
                self._resid("attn", b, args[0])

            def mlp_hook(mod, args, out, b=b):
                return self._sub("mlp", b, out)

            def layer_hook(mod, args, out, b=b):
                self._resid("mlp", b, out[0])

            self._handles += [layer.attention.register_forward_hook(attn_hook),
                              layer.norm2.register_forward_pre_hook(norm2_pre),
                              layer.mlp.register_forward_hook(mlp_hook),
                              layer.register_forward_hook(layer_hook)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def sites(self):
        return [(k, b) for b in self.blocks for k in ("attn", "mlp")]

    def stacked(self, key):
        """[B, n_sites, ...] of one recorded quantity over self.sites() (NaN-free sites only)."""
        return torch.stack([self.rec[s][key] for s in self.sites()], dim=1)

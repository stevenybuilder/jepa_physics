"""The V-JEPA 2 predictor as a behavioural readout of an activation edit (spec §6 item 5b, Part 2 design note §7 rung 3).

What the checkpoint contains (facebook/vjepa2-vitl-fpc64-256, model.safetensors, 587 tensors): the 24-block encoder
(384 tensors + embeddings + final LN) AND the trained predictor: 12 blocks of width 384, 12 heads (192 tensors),
predictor_embeddings (1024 -> 384), 10 mask tokens, predictor LayerNorm and proj (384 -> 1024). So the predictor is
used; no fallback is needed (the fallback, the encoder's own future-step tokens, is still returned as `real future`).
Only mask token 0 is trained (norms 0.769, then nine exact zeros); HF's predictor forward hard-codes mask_index=1,
so predict_future re-implements the forward with mask_index=0 by default (mask_index=1 reproduces HF exactly). On 4
direction clips the index-0 forecast had lower MSE to the clip's own real future than index 1 on all 4.

Why the context is encoded alone. In transformers 4.56.2 VJEPA2Model.forward runs self.encoder on ALL frames with no
mask, and only then VJEPA2Predictor.forward calls apply_masks(encoder_hidden_states, context_mask) (modeling_vjepa2.py
line 695). Encoder attention is bidirectional, so every "context" token of a 16-frame encoding has already attended
to the future frames. Here the first 8 frames (4 tubelets, tokens 0..1023) are encoded on their own; RoPE positions
come from token indices, so tubelets 0..3 get the same positions as in the full clip. The predictor then receives
context_mask = arange(1024) (a no-op gather on the 1024 context tokens) and target_mask = arange(1024, 2048), i.e. it
forecasts tubelets 4..7 (frames 9..16), and its output is in the encoder's post-LN token space (V-JEPA 2 regresses
layer-normed target-encoder tokens).

Readouts per clip: predicted future tokens [1024, D] and their mean per future time step [4, D]; the encoder's
actual future tokens (post-LN, tokens 1024..2047 of the full 16-frame encoding) for the real clip and for its
counterfactual twin (wm.render_twin), both token-level and pooled per step.
"""
import numpy as np
import torch

from wm.data import N_STEPS
from wm.propagate import prefix, suffix

CTX_FRAMES = 8
CTX_STEPS = CTX_FRAMES // 2          # 4 tubelets
TOK_PER_STEP = 256
N_CTX = CTX_STEPS * TOK_PER_STEP     # 1024
N_ALL = N_STEPS * TOK_PER_STEP       # 2048


def masks_for(batch, device):
    ctx = torch.arange(N_CTX, device=device).unsqueeze(0).repeat(batch, 1)
    tgt = torch.arange(N_CTX, N_ALL, device=device).unsqueeze(0).repeat(batch, 1)
    return [ctx], [tgt]


MASK_INDEX = 0   # the only trained mask token (see module docstring)


@torch.no_grad()
def predict_future(model, ctx_final, mask_index=MASK_INDEX):
    """Run the predictor on post-LN context tokens [B, 1024, D]; returns predicted tokens for tubelets 4..7
    [B, 1024, D] in (t, h, w) order (t = 4..7).

    Same computation as VJEPA2Predictor.forward (embed, sort by position, 12 blocks with RoPE from the position
    ids, LN, unsort, keep the target tokens, proj) except that the mask token index is a parameter:
    HF's forward calls embeddings(..) with its default mask_index=1, and in this checkpoint mask tokens 1..9 are
    all-zero (untrained; norms 0.769, 0, 0, ...), so HF's default would forecast from a zero query token.
    mask_index=1 reproduces model.predictor(...) exactly (checked in tests/test_session2.py)."""
    assert ctx_final.shape[1] == N_CTX, ctx_final.shape
    P = model.predictor
    cm, tm = masks_for(ctx_final.shape[0], ctx_final.device)
    h, pos = P.embeddings(ctx_final, cm, tm, mask_index=mask_index)
    order = torch.argsort(pos, dim=1)
    h, pos, _ = P.sort_tokens(h, pos, order, None)
    for layer in P.layer:
        h = layer(h, pos, None, False)[0]
    h = P.layernorm(h)
    h = P.unsort_tokens(h, order)[:, N_CTX:]
    return P.proj(h)


def pool_steps(tokens):
    """[B, n_steps * 256, D] -> [B, n_steps, D]: mean over the 256 patches of each time step."""
    B, N, D = tokens.shape
    return tokens.reshape(B, N // TOK_PER_STEP, TOK_PER_STEP, D).mean(dim=2)


@torch.no_grad()
def context_prefix(model, pixel_values_full, save_points):
    """Encode frames 1..8 only. pixel_values_full [B, 16, 3, H, W] (or already [B, 8, ...]).
    Returns (saved {point: [B, 1024, D]}, ctx_final [B, 1024, D], ctx_meanpool [B, 26, D])."""
    pv = pixel_values_full[:, :CTX_FRAMES]
    return prefix(model, pv, save_points)


@torch.no_grad()
def edited_prediction(model, ctx_h, point, delta, mask_index=MASK_INDEX):
    """Add delta (raw space, [D] or [B, D]) to every context token at `point`, finish the encoder, run the predictor.
    ctx_h: context residual stream at `point` [B, 1024, D]. Returns (pred tokens [B, 1024, D], ctx meanpool at points
    point..25 [B, 26 - point, D])."""
    r = suffix(model, ctx_h, point, delta, return_final=True)
    return predict_future(model, r["final"], mask_index), r["meanpool"]


@torch.no_grad()
def real_future(final_full):
    """Encoder's own future tokens from a full 16-frame encoding (post-LN) [B, 2048, D] -> [B, 1024, D]."""
    return final_full[:, N_CTX:]


def recovery(z_edit, z_src, z_tgt, eps=1e-12):
    """Probe-free recovery R = <z_edit - z_src, z_tgt - z_src> / ||z_tgt - z_src||^2 over all trailing dims.
    1 = the edit moved the forecast all the way to the target (in projection), 0 = no movement along it.
    Accepts torch or numpy, leading batch dims broadcast."""
    if torch.is_tensor(z_edit):
        a, b = (z_edit - z_src).flatten(1), (z_tgt - z_src).flatten(1)
        return ((a * b).sum(1) / (b * b).sum(1).clamp_min(eps)).cpu().numpy()
    a = np.asarray(z_edit - z_src, float).reshape(len(z_edit), -1)
    b = np.asarray(z_tgt - z_src, float).reshape(len(z_edit), -1)
    return (a * b).sum(1) / np.maximum((b * b).sum(1), eps)


def disk_token_index(mask_steps):
    """Bool token mask [n_steps * 256] from a disk mask [n_steps, 16, 16] (same (t, h, w) order as the tokens)."""
    return np.asarray(mask_steps, bool).reshape(-1)


@torch.no_grad()
def readout(model, pixel_values_src, point, deltas, pixel_values_twin=None):
    """Convenience wrapper for one clip: context-only encoding, edits at `point`, predictor, and the real futures.

    pixel_values_src [1, 16, 3, H, W]; deltas [E, D] raw space; pixel_values_twin [1, 16, 3, H, W] or None.
    Returns dict(pred_src [1024, D], pred_edit [E, 1024, D], pred_edit_pooled [E, 4, D], pred_src_pooled [4, D],
    real_future_src [1024, D] (+ pooled), and for the twin real_future_twin, pred_twin (+ pooled), R_real, R_pred).
    """
    saved, ctx_final, _ = context_prefix(model, pixel_values_src, [point])
    z_src = predict_future(model, ctx_final)
    _, final_full, _ = prefix(model, pixel_values_src, [])
    out = {"pred_src": z_src[0], "pred_src_pooled": pool_steps(z_src)[0],
           "real_future_src": real_future(final_full)[0], "real_future_src_pooled": pool_steps(real_future(final_full))[0]}
    d = torch.as_tensor(np.asarray(deltas, np.float32), device=ctx_final.device)
    z_edit, _ = edited_prediction(model, saved[point].expand(len(d), -1, -1), point, d)
    out.update({"pred_edit": z_edit, "pred_edit_pooled": pool_steps(z_edit)})
    if pixel_values_twin is not None:
        _, twin_final, _ = prefix(model, pixel_values_twin, [])
        _, twin_ctx_final, _ = context_prefix(model, pixel_values_twin, [])
        z_twin = predict_future(model, twin_ctx_final)
        zt_real = real_future(twin_final)
        out.update({"real_future_twin": zt_real[0], "real_future_twin_pooled": pool_steps(zt_real)[0],
                    "pred_twin": z_twin[0], "pred_twin_pooled": pool_steps(z_twin)[0],
                    "R_pred": recovery(z_edit, z_src.expand_as(z_edit), z_twin.expand_as(z_edit)),
                    "R_real": recovery(z_edit, z_src.expand_as(z_edit), zt_real.expand_as(z_edit))})
    return out

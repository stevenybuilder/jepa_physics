"""Propagate an activation edit through the rest of the frozen V-JEPA 2 encoder (spec §6 item 5a).

Point convention (same as the stored activations, wm.extract): point 0 = patch embedding, point l (1..24) = the
residual stream after block l, point 25 = final LayerNorm of block 24's output. "Edit at point L" adds a vector
delta (raw activation space, d = 1024) to EVERY token of the residual stream at point L, then runs blocks L+1..24
and the final LayerNorm. Edits made in the train-standardised coordinates of Part 1 are mapped back first:
delta_raw = delta_std * sd_train (to_raw_delta).

The encoder is run block by block with exactly the calls VJEPA2Encoder.forward makes (layer(h, None, None, False)),
so adding a zero delta is the same computation as wm.extract.encode (a forward hook on block L that adds delta
would be equivalent; the explicit loop lets one prefix pass at point L serve many edits). Mean-pooled activations
use wm.extract.pool, so Δ = 0 reproduces the stored `meanpool` up to device / batch-size float noise.
"""
import time

import numpy as np
import torch

from wm.extract import N_POINTS, WIDTH, pool, preprocess

LAST_POINT = N_POINTS - 1        # 25: final LayerNorm


def to_raw_delta(delta_std, sd_train):
    """A displacement in train-standardised coordinates -> the same displacement in raw activation space."""
    return np.asarray(delta_std, float) * np.asarray(sd_train, float)


@torch.no_grad()
def prefix(model, pixel_values, save_points):
    """Run the encoder on pixel_values [B, T, 3, 256, 256] and return {point: hidden [B, N, D]} for each point in
    save_points (0..24); other states are not kept (memory). Also returns the post-LN tokens [B, N, D] and the
    mean-pooled activations at every point [B, 1 + n_blocks + 1, D] (numpy; [B, 26, D] for 16 frames)."""
    enc = model.encoder
    h = enc.embeddings(pixel_values)
    save_points = set(int(p) for p in save_points)
    saved, pooled = {}, []
    if 0 in save_points:
        saved[0] = h
    pooled.append(_mean(h))
    for i, layer in enumerate(enc.layer):
        h = layer(h, None, None, False)[0]
        if i + 1 in save_points:
            saved[i + 1] = h
        pooled.append(_mean(h))
    final = enc.layernorm(h)
    pooled.append(_mean(final))
    return saved, final, torch.stack(pooled, dim=1).cpu().numpy()


def _mean(x):
    """Mean over tokens, summed in the order wm.extract.pool uses ((t, h*w) reshape, then mean over both), so a
    zero edit reproduces extract.encode + pool bit for bit on the same device and batch."""
    B, N, D = x.shape
    if N % 256 == 0:
        return x.reshape(B, N // 256, 256, D).mean(dim=(1, 2))
    return x.mean(dim=1)


@torch.no_grad()
def suffix(model, h, point, delta=None, masks=None, return_final=False):
    """Add delta to every token of h (the residual stream at `point`, [B, N, D]) and run the remaining blocks and
    the final LayerNorm.

    delta: None, a tensor/array [D] (same for all rows) or [B, D] (one per row).
    masks: optional bool [B, 8, 16, 16] disk masks -> also returns timepool and diskpool (wm.extract.pool).
    Returns dict(meanpool [B, 26 - point, D] numpy float32 for points point..25; with masks also timepool,
    diskpool [B, 26 - point, 8, D]; with return_final the post-LN tokens [B, N, D] as a tensor).
    """
    if not 0 <= point <= 24:
        raise ValueError("edit point must be 0 (embedding) .. 24 (block 24 output); the post-LN point is not a residual")
    enc = model.encoder
    if delta is not None:
        d = torch.as_tensor(np.asarray(delta, np.float32) if not torch.is_tensor(delta) else delta,
                            dtype=h.dtype, device=h.device)
        h = h + (d.view(1, 1, -1) if d.dim() == 1 else d[:, None, :])
    out_mean, out_time, out_disk = [], [], []

    def record(x):
        if masks is None:
            out_mean.append(_mean(x).cpu().numpy())
        else:
            m, t, dsk = pool([x], masks)
            out_mean.append(m[:, 0])
            out_time.append(t[:, 0])
            out_disk.append(dsk[:, 0])

    record(h)
    for layer in enc.layer[point:]:
        h = layer(h, None, None, False)[0]
        record(h)
    final = enc.layernorm(h)
    record(final)
    res = {"meanpool": np.stack(out_mean, axis=1).astype(np.float32)}
    if masks is not None:
        res["timepool"] = np.stack(out_time, axis=1)
        res["diskpool"] = np.stack(out_disk, axis=1)
    if return_final:
        res["final"] = final
    return res


def propagate(model, frames, point, deltas, device=None, batch_size=16, masks=None):
    """Edit one clip at `point` with each of several deltas and return the later-layer pooled activations.

    frames: uint8 [16, 256, 256, 3] (one clip, as wm.data.decode returns).
    deltas: [E, D] raw-space edits, or [E, K, D] waypoint edits (flattened, then reshaped back).
    Returns meanpool [E, 26 - point, D] (or [E, K, 26 - point, D]); index 0 is the edited point itself, whose
    meanpool is exactly the unedited meanpool + delta. With masks ([8, 16, 16] bool) also diskpool/timepool.
    """
    device = device or next(model.parameters()).device
    deltas = np.asarray(deltas, np.float32)
    shape = deltas.shape
    flat = deltas.reshape(-1, WIDTH)
    saved, _, _ = prefix(model, preprocess([frames]).to(device), [point])
    h0 = saved[point]
    outs = {}
    for s in range(0, len(flat), batch_size):
        d = flat[s:s + batch_size]
        m = None if masks is None else torch.as_tensor(np.broadcast_to(masks, (len(d),) + masks.shape)).to(device)
        r = suffix(model, h0.expand(len(d), -1, -1), point, d, m)
        for k, v in r.items():
            outs.setdefault(k, []).append(v)
    return {k: np.concatenate(v).reshape(shape[:-1] + v[0].shape[1:]) for k, v in outs.items()}


def propagate_batch(model, frames_list, point, deltas, device=None, masks=None):
    """One edit per clip: frames_list of B clips, deltas [B, D]. Returns suffix() output for the batch."""
    device = device or next(model.parameters()).device
    saved, _, _ = prefix(model, preprocess(frames_list).to(device), [point])
    m = None if masks is None else torch.as_tensor(np.asarray(masks)).to(device)
    return suffix(model, saved[point], point, np.asarray(deltas, np.float32), m)


def timed(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return out, time.time() - t0

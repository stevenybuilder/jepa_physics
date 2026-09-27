"""Frozen V-JEPA 2 ViT-L feature extraction (spec section 2).

Hook points, verified in transformers 4.56.2 `models/vjepa2/modeling_vjepa2.py`:
  VJEPA2Encoder.forward appends `hidden_states` to all_hidden_states *before* each of the 24
  blocks, then applies the final LayerNorm and appends once more. So with
  output_hidden_states=True there are 25 entries:
    [0]      patch embedding (Conv3d tubelet embed; RoPE is applied inside attention, so there is
             no additive position embedding)
    [1..23]  residual stream after blocks 1..23
    [24]     layernorm(block 24 output) == last_hidden_state
  Block 24's raw (pre-LN) output is not in that tuple, so a forward hook on encoder.layer[23]
  captures it. The 26 stored points are: [emb, block1, ..., block24 (pre-LN), final post-LN].

Token order, verified in the same file: VJEPA2Embeddings permutes [B,T,C,H,W] -> [B,C,T,H,W];
the Conv3d gives [B,1024,8,16,16] and `.flatten(2).transpose(1, 2)` makes the token index
t*256 + h*16 + w, i.e. (t, h, w) row-major. The RoPE position ids (get_position_ids) decode
the index the same way (tokens_per_frame = 256, tokens_per_row = 16).

Preprocessing: uint8 frames / 255, ImageNet mean/std, no resize, no crop. This deliberately
differs from the HF video processor, which resizes the short side to 292 and centre-crops 256.

Second model (objective axis, `kind='videomae'`): MCG-NJU/videomae-large, the self-supervised VideoMAE ViT-L
(Kinetics-400 masked-autoencoding pretrain, no fine-tune; the checkpoint is VideoMAEForPreTraining and its
`videomae.*` encoder weights, including the final `videomae.layernorm`, load into VideoMAEModel; the pixel decoder is
dropped). The physics paper's App. C used the VideoMAE-v2 family (OpenGVLab/VideoMAEv2-*, remote code); this is the
closest checkpoint with a native transformers class. Input differences, handled explicitly:
  - 224x224 input, patch 16, tubelet 2, 16 frames -> 8 x 14 x 14 = 1,568 tokens (the paper's 8 x 196 setup).
    Our clips are 256x256, so frames are RESIZED (not cropped) 256 -> 224 with bilinear + antialias, on [0, 1]
    floats, before ImageNet mean/std. The HF VideoMAEImageProcessor (resize short side 224, PIL bilinear, centre
    crop 224) does the same geometry on a square clip; it is not used so the path stays in torch and fp32.
  - Token order: VideoMAEPatchEmbeddings permutes [B,T,C,H,W] -> [B,C,T,H,W], Conv3d -> [B,1024,8,14,14],
    `.flatten(2).transpose(1, 2)`: index t*196 + h*14 + w, (t, h, w) row-major, as for V-JEPA.
  - Position: fixed sinusoidal table ADDED to the patch embedding inside VideoMAEEmbeddings, so point 0 is
    patch embed + position (V-JEPA's point 0 has no position; it uses RoPE in attention).
  - Hidden states: VideoMAEEncoder has 24 blocks and no internal norm; VideoMAEModel applies `layernorm` after
    block 24 (use_mean_pooling=False in this config). Forward hooks on `embeddings` and every `encoder.layer[i]`
    give 25 tensors; `last_hidden_state` is the 26th. So VideoMAE-L also has 26 points with the SAME layout:
    [emb(+pos), block1, ..., block24 (pre-LN), final post-LN] -- point k = output of block k for both models.
  - Disk mask on the 14x14 grid: the red-channel disk test (wm.data.disk_pixels rule) on the resized frames,
    any-pixel per 16x16 patch, any over the tubelet's two frames -> diskmask [8, 14, 14].
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import VideoMAEModel, VJEPA2Config, VJEPA2Model

from wm.data import (DISK_CHANNEL, DISK_THRESH, GRID, N_FRAMES, N_STEPS, PATCH, SIZE, TUBELET, decode, disk_mask,
                     frame_hash)

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_TOKENS = N_STEPS * GRID * GRID   # 2048
N_POINTS = 26
WIDTH = 1024
CHUNK = 64
POINT_NAMES = ["emb"] + [f"block{i}" for i in range(1, 25)] + ["final_ln"]

VIDEOMAE_ID = "MCG-NJU/videomae-large"
VM_SIZE = 224
VM_GRID = VM_SIZE // PATCH                 # 14
VM_TOKENS = N_STEPS * VM_GRID * VM_GRID    # 1568
VM_POINT_NAMES = ["emb_plus_pos"] + [f"block{i}" for i in range(1, 25)] + ["final_ln"]
MODEL_IDS = {"vjepa2": MODEL_ID, "random": MODEL_ID, "videomae": VIDEOMAE_ID}
PREPROCESS_NOTE = {
    "vjepa2": "decode 16 frames (PyAV rgb24) -> /255 -> ImageNet mean/std; no resize, no crop",
    "videomae": ("decode 16 frames (PyAV rgb24) -> /255 -> RESIZE 256->224 (torch bilinear, antialias=True, "
                 "align_corners=False; no crop) -> ImageNet mean/std; 8x14x14 = 1568 tokens"),
}
PREPROCESS_NOTE["random"] = PREPROCESS_NOTE["vjepa2"]


def grid_of(kind):
    """Spatial token grid side for a model kind: 16 (V-JEPA, 256 px) or 14 (VideoMAE, 224 px)."""
    return VM_GRID if kind == "videomae" else GRID


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_precision():
    """fp32 everywhere; no TF32 on Ampere+ (the Conv3d patch embed would use it by default).
    The flags are no-ops on CPU; they are set regardless so the index records the same values."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def precision_flags():
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def load_model(kind, device):
    """kind = 'vjepa2' (pretrained), 'random' (same architecture, torch.manual_seed(0) init) or 'videomae'
    (MCG-NJU/videomae-large encoder)."""
    if kind == "videomae":
        model = VideoMAEModel.from_pretrained(VIDEOMAE_ID, dtype=torch.float32)
        model = model.to(device=device, dtype=torch.float32).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        c = model.config
        assert len(model.encoder.layer) == 24 and c.hidden_size == WIDTH and model.layernorm is not None
        assert (c.image_size, c.patch_size, c.tubelet_size, c.num_frames) == (VM_SIZE, PATCH, TUBELET, N_FRAMES)
        return model
    if kind == "vjepa2":
        model = VJEPA2Model.from_pretrained(MODEL_ID, dtype=torch.float32)
    elif kind == "random":
        config = VJEPA2Config.from_pretrained(MODEL_ID)
        torch.manual_seed(0)
        model = VJEPA2Model(config)
    else:
        raise ValueError(kind)
    model = model.to(device=device, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert len(model.encoder.layer) == 24 and model.config.hidden_size == WIDTH
    return model


def preprocess(frames_batch):
    """list of uint8 [16,256,256,3] -> float32 tensor [B,16,3,256,256] (the layout
    pixel_values_videos expects: batch, frames, channels, height, width)."""
    x = torch.from_numpy(np.stack(frames_batch)).float() / 255.0          # [B,T,H,W,C]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 1, 1, 3)
    x = ((x - mean) / std).permute(0, 1, 4, 2, 3).contiguous()           # [B,T,C,H,W]
    assert x.shape[1:] == (N_FRAMES, 3, SIZE, SIZE), x.shape
    return x


def resize_videomae(x):
    """float [B,T,H,W,C] in [0, 1] at 256 px -> [B,T,224,224,C]: bilinear with antialias, no crop."""
    B, T = x.shape[:2]
    y = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, SIZE, SIZE)
    y = torch.nn.functional.interpolate(y, size=(VM_SIZE, VM_SIZE), mode="bilinear", align_corners=False,
                                        antialias=True)
    return y.reshape(B, T, 3, VM_SIZE, VM_SIZE).permute(0, 1, 3, 4, 2)


def preprocess_videomae(frames_batch):
    """list of uint8 [16,256,256,3] -> float32 [B,16,3,224,224] (VideoMAEModel's pixel_values layout)."""
    x = resize_videomae(torch.from_numpy(np.stack(frames_batch)).float() / 255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 1, 1, 3)
    x = ((x - mean) / std).permute(0, 1, 4, 2, 3).contiguous()
    assert x.shape[1:] == (N_FRAMES, 3, VM_SIZE, VM_SIZE), x.shape
    return x


def disk_mask_videomae(frames, thresh=DISK_THRESH):
    """Bool [8, 14, 14]: the disk_pixels rule (red channel > thresh and > the other two) applied to the frames after
    the same 256->224 resize the model sees; a patch is True if any disk pixel is in it in either tubelet frame."""
    x = resize_videomae(torch.from_numpy(frames[None]).float())[0].numpy()           # [16,224,224,3], 0..255
    c = x[..., DISK_CHANNEL]
    others = np.delete(x, DISK_CHANNEL, axis=-1)
    pix = (c > thresh) & (c > others[..., 0]) & (c > others[..., 1])
    per_frame = pix.reshape(N_FRAMES, VM_GRID, PATCH, VM_GRID, PATCH).any(axis=(2, 4))
    return per_frame.reshape(N_STEPS, TUBELET, VM_GRID, VM_GRID).any(axis=1)


@torch.no_grad()
def encode_videomae(model, pixel_values):
    """VideoMAE forward, all tokens (no masking). Returns (points, out): 26 tensors [B,1568,1024] =
    [embeddings output (patch + sinusoidal position), block 1..24 outputs, final layernorm = last_hidden_state]."""
    captured = [None] * 25
    hooks = [model.embeddings.register_forward_hook(lambda m, i, o: captured.__setitem__(0, o))]
    for k, layer in enumerate(model.encoder.layer):
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, k=k: captured.__setitem__(k + 1, o[0] if isinstance(o, tuple) else o)))
    try:
        out = model(pixel_values=pixel_values)
    finally:
        for h in hooks:
            h.remove()
    assert all(c is not None for c in captured)
    points = captured + [out.last_hidden_state]
    assert len(points) == N_POINTS and points[0].shape[1:] == (VM_TOKENS, WIDTH), points[0].shape
    return points, out


@torch.no_grad()
def encode(model, pixel_values):
    """Forward pass. Returns (points, out): points is a list of 26 tensors [B,2048,1024]."""
    captured = {}
    hook = model.encoder.layer[-1].register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("block24", output[0]))
    try:
        out = model(pixel_values_videos=pixel_values, output_hidden_states=True, skip_predictor=True)
    finally:
        hook.remove()
    hs = out.hidden_states
    assert len(hs) == 25, len(hs)
    assert hs[0].shape[1:] == (N_TOKENS, WIDTH), hs[0].shape
    points = list(hs[:24]) + [captured["block24"], hs[24]]
    assert len(points) == N_POINTS
    return points, out


def pool(points, masks, grid=GRID):
    """points: 26 x [B,T*grid*grid,1024]; masks: bool tensor [B,8,grid,grid] (grid 16 for V-JEPA, 14 for VideoMAE).

    Returns float32 numpy arrays meanpool [B,26,1024], timepool [B,26,8,1024],
    diskpool [B,26,8,1024] (NaN at time steps where the mask is empty).
    """
    B = points[0].shape[0]
    m = masks.reshape(B, N_STEPS, grid * grid, 1).to(points[0].dtype)     # [B,8,256,1] (V-JEPA)
    n_disk = m.sum(dim=2)                                                   # [B,8,1]
    mean, time_, disk = [], [], []
    for x in points:
        x = x.reshape(B, N_STEPS, grid * grid, WIDTH)                       # (t, h*w) order
        mean.append(x.mean(dim=(1, 2)))
        time_.append(x.mean(dim=2))
        d = (x * m).sum(dim=2) / n_disk.clamp(min=1)
        disk.append(torch.where(n_disk > 0, d, torch.full_like(d, float("nan"))))
    to_np = lambda xs: torch.stack(xs, dim=1).cpu().numpy()                # noqa: E731
    return to_np(mean), to_np(time_), to_np(disk)


# ---------------------------------------------------------------------------------------------
# Chunked, resumable extraction


def chunk_arrays(kind):
    """Names and per-clip shapes/dtypes stored for each model kind."""
    arrays = {"meanpool": ((N_POINTS, WIDTH), np.float32)}
    if kind in ("vjepa2", "videomae"):
        g = grid_of(kind)
        arrays.update({
            "timepool": ((N_POINTS, N_STEPS, WIDTH), np.float16),
            "diskpool": ((N_POINTS, N_STEPS, WIDTH), np.float16),
            "diskmask": ((N_STEPS, g, g), np.bool_),
        })
    return arrays


def chunk_ok(out_dir, c, ids, kind):
    """True if chunk c exists with the expected ids and array shapes."""
    meta_path = out_dir / f"chunk_{c:04d}.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    if meta["ids"] != [int(i) for i in ids]:
        return False
    for name, (shape, dtype) in chunk_arrays(kind).items():
        path = out_dir / f"chunk_{c:04d}_{name}.npy"
        if not path.exists():
            return False
        a = np.load(path, mmap_mode="r")
        if a.shape != (len(ids),) + shape or a.dtype != dtype:
            return False
    return True


def save_npy(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    tmp.rename(path)


def run_extraction(df, kind, out_dir, batch_size=8, device=None):
    """Extract every row of df (manifest order) into chunk files under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    set_precision()
    ids = [int(i) for i in df["id"]]
    videos = dict(zip(ids, df["video"]))
    chunks = [ids[s:s + CHUNK] for s in range(0, len(ids), CHUNK)]
    model = None
    t_run = time.time()

    for c, chunk_ids in enumerate(chunks):
        if chunk_ok(out_dir, c, chunk_ids, kind):
            print(f"chunk {c}: exists, skipping")
            continue
        if model is None:
            model = load_model(kind, device)
        store = {name: [] for name in chunk_arrays(kind)}
        hashes, seconds = {}, {}
        for s in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[s:s + batch_size]
            t0 = time.time()
            frames = [decode(videos[i]) for i in batch_ids]
            for i, f in zip(batch_ids, frames):
                hashes[i] = frame_hash(f)
            if kind == "videomae":
                masks = [disk_mask_videomae(f) for f in frames]
                points, _ = encode_videomae(model, preprocess_videomae(frames).to(device))
            else:
                masks = [disk_mask(f)[0] for f in frames]
                points, _ = encode(model, preprocess(frames).to(device))
            mean, tpool, dpool = pool(points, torch.from_numpy(np.stack(masks)).to(device), grid_of(kind))
            store["meanpool"].append(mean.astype(np.float32))
            if kind in ("vjepa2", "videomae"):
                store["timepool"].append(tpool.astype(np.float16))
                store["diskpool"].append(dpool.astype(np.float16))
                store["diskmask"].append(np.stack(masks))
            per_clip = (time.time() - t0) / len(batch_ids)
            for i in batch_ids:
                seconds[i] = per_clip
                print(f"chunk {c} clip {i}: {per_clip:.2f} s")

        for name, (shape, dtype) in chunk_arrays(kind).items():
            a = np.concatenate(store[name])
            assert a.shape == (len(chunk_ids),) + shape and a.dtype == dtype, (name, a.shape, a.dtype)
            save_npy(out_dir / f"chunk_{c:04d}_{name}.npy", a)
        meta = {"ids": chunk_ids, "frame_hash": {str(i): hashes[i] for i in chunk_ids},
                "seconds_per_clip": {str(i): seconds[i] for i in chunk_ids}}
        (out_dir / f"chunk_{c:04d}.json").write_text(json.dumps(meta))

    write_index(out_dir, ids, len(chunks), kind, device, time.time() - t_run)


def write_index(out_dir, ids, n_chunks, kind, device, run_seconds):
    hashes, seconds = {}, {}
    for c in range(n_chunks):
        meta = json.loads((out_dir / f"chunk_{c:04d}.json").read_text())
        hashes.update(meta["frame_hash"])
        seconds.update(meta["seconds_per_clip"])
    index = {
        "model_id": MODEL_IDS[kind], "model": kind, "ids": ids, "n_chunks": n_chunks, "chunk_size": CHUNK,
        "points": VM_POINT_NAMES if kind == "videomae" else POINT_NAMES, "arrays": {k: [list(s), np.dtype(d).name] for k, (s, d) in chunk_arrays(kind).items()},
        "frame_hash": hashes, "seconds_per_clip": seconds, "last_run_seconds": run_seconds,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "forward_dtype": "float32", **precision_flags(),
        "preprocess": PREPROCESS_NOTE[kind],
    }
    if kind == "videomae":
        index.update({"n_tokens": VM_TOKENS, "token_grid": [N_STEPS, VM_GRID, VM_GRID], "input_size": VM_SIZE,
                      "n_hidden_state_points": N_POINTS,
                      "points_note": ("25 hooked tensors (embeddings output incl. sinusoidal position, blocks 1..24; "
                                      "no norm inside the encoder) + VideoMAEModel.layernorm(block 24) = 26 points, "
                                      "same layout as V-JEPA: point k = output of block k, point 25 = post-LN"),
                      "diskmask_note": "disk_pixels rule on the 224-px resized frames; any pixel per 16-px patch, "
                                       "any over the tubelet's 2 frames -> [8,14,14]"})
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


def merge(out_dir, manifest_ids):
    """Concatenate chunks into {name}.npy with N = len(manifest_ids), in manifest id order."""
    out_dir = Path(out_dir)
    index = json.loads((out_dir / "index.json").read_text())
    kind, ids = index["model"], index["ids"]
    assert ids == [int(i) for i in manifest_ids], "chunks do not cover the manifest in id order"
    for c in range(index["n_chunks"]):
        assert chunk_ok(out_dir, c, ids[c * CHUNK:(c + 1) * CHUNK], kind), f"chunk {c} missing or bad"
    for name, (shape, dtype) in chunk_arrays(kind).items():
        a = np.concatenate([np.load(out_dir / f"chunk_{c:04d}_{name}.npy") for c in range(index["n_chunks"])])
        assert a.shape == (len(ids),) + shape and a.dtype == dtype, (name, a.shape)
        save_npy(out_dir / f"{name}.npy", a)
        print(f"{name}: {a.shape} {a.dtype}")
    (out_dir / "ids.json").write_text(json.dumps(ids))

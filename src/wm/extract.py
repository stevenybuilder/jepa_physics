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
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import VJEPA2Config, VJEPA2Model

from wm.data import GRID, N_FRAMES, N_STEPS, SIZE, decode, disk_mask, frame_hash

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_TOKENS = N_STEPS * GRID * GRID   # 2048
N_POINTS = 26
WIDTH = 1024
CHUNK = 64
POINT_NAMES = ["emb"] + [f"block{i}" for i in range(1, 25)] + ["final_ln"]


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
    """kind = 'vjepa2' (pretrained) or 'random' (same architecture, torch.manual_seed(0) init)."""
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


def pool(points, masks):
    """points: 26 x [B,2048,1024]; masks: bool tensor [B,8,16,16].

    Returns float32 numpy arrays meanpool [B,26,1024], timepool [B,26,8,1024],
    diskpool [B,26,8,1024] (NaN at time steps where the mask is empty).
    """
    B = points[0].shape[0]
    m = masks.reshape(B, N_STEPS, GRID * GRID, 1).to(points[0].dtype)     # [B,8,256,1]
    n_disk = m.sum(dim=2)                                                   # [B,8,1]
    mean, time_, disk = [], [], []
    for x in points:
        x = x.reshape(B, N_STEPS, GRID * GRID, WIDTH)                       # (t, h*w) order
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
    if kind == "vjepa2":
        arrays.update({
            "timepool": ((N_POINTS, N_STEPS, WIDTH), np.float16),
            "diskpool": ((N_POINTS, N_STEPS, WIDTH), np.float16),
            "diskmask": ((N_STEPS, GRID, GRID), np.bool_),
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
            masks = [disk_mask(f)[0] for f in frames]
            for i, f in zip(batch_ids, frames):
                hashes[i] = frame_hash(f)
            points, _ = encode(model, preprocess(frames).to(device))
            mean, tpool, dpool = pool(points, torch.from_numpy(np.stack(masks)).to(device))
            store["meanpool"].append(mean.astype(np.float32))
            if kind == "vjepa2":
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
        "model_id": MODEL_ID, "model": kind, "ids": ids, "n_chunks": n_chunks, "chunk_size": CHUNK,
        "points": POINT_NAMES, "arrays": {k: [list(s), np.dtype(d).name] for k, (s, d) in chunk_arrays(kind).items()},
        "frame_hash": hashes, "seconds_per_clip": seconds, "last_run_seconds": run_seconds,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "forward_dtype": "float32", **precision_flags(),
        "preprocess": "decode 16 frames (PyAV rgb24) -> /255 -> ImageNet mean/std; no resize, no crop",
    }
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

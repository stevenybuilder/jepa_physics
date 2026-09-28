"""Per-object pools of the relational clips (binding test): for every clip and all 26 points, the mean over the tokens
covering disk 1 (orange, wm.data.disk_pixels rule), disk 2 (blue: blue > 128 and > red, green) and background (tokens
touching neither disk), over all 8 token time steps. A token belongs to a disk if any disk pixel falls in its 16 x 16
patch in either frame of its tubelet (the wm.data.disk_mask rule). Same model load / preprocessing / fp32 as
wm.extract (V-JEPA 2 and the seed-0 random-init ViT-L).

Output {out}/{model}/objpool.npy float32 [N, 26, 3, 1024] (pool order: disk1, disk2, background), ntok.npy int
[N, 3] (tokens per pool over the 8 steps), n_shared_tokens.npy [N] (tokens in both disk masks; expected 0), index.json.

  python scripts/extract_relational_objpool.py --out artifacts/activations/stimuli_relational_objpool
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import GRID, N_FRAMES, N_STEPS, PATCH, PROJECT_ROOT, TUBELET, decode, disk_pixels, load_table  # noqa: E402
from wm.extract import MODEL_ID, WIDTH, encode, load_model, pick_device, preprocess, set_precision  # noqa: E402


def blue_pixels(frames, thresh=128):
    x = frames.astype(np.int16)
    return (x[..., 2] > thresh) & (x[..., 2] > x[..., 0]) & (x[..., 2] > x[..., 1])


def token_mask(pix):
    per_frame = pix.reshape(N_FRAMES, GRID, PATCH, GRID, PATCH).any(axis=(2, 4))
    return per_frame.reshape(N_STEPS, TUBELET, GRID, GRID).any(axis=1).reshape(-1)      # [2048], (t, h, w) order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "activations" / "stimuli_relational_objpool"))
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    set_precision()
    dev = pick_device()
    df = load_table("speed", root=PROJECT_ROOT / "artifacts" / "stimuli" / "relational")
    ids = df["id"].tolist()
    for kind in ("vjepa2", "random"):
        t0 = time.time()
        model = load_model(kind, dev)
        pools, ntok, shared = [], [], []
        for s in range(0, len(df), a.batch_size):
            frames = [decode(v) for v in df["video"].iloc[s:s + a.batch_size]]
            m1 = np.stack([token_mask(disk_pixels(f)) for f in frames])
            m2 = np.stack([token_mask(blue_pixels(f)) for f in frames])
            shared.append((m1 & m2).sum(1))
            masks = np.stack([m1, m2 & ~m1, ~(m1 | m2)], 1).astype(np.float32)             # [B, 3, 2048]
            ntok.append(masks.sum(2).astype(np.int32))
            M = torch.from_numpy(masks).to(dev)
            points, _ = encode(model, preprocess(frames).to(dev))
            per = [torch.einsum("bpt,btd->bpd", M, x) / M.sum(2, keepdim=True).clamp(min=1) for x in points]
            pools.append(torch.stack(per, 1).cpu().numpy().astype(np.float32))                # [B, 26, 3, 1024]
            print(f"{kind} {s + len(frames)}/{len(df)}", flush=True)
        out = Path(a.out) / kind
        out.mkdir(parents=True, exist_ok=True)
        P = np.concatenate(pools)
        assert P.shape == (len(df), 26, 3, WIDTH)
        np.save(out / "objpool.npy", P)
        np.save(out / "ntok.npy", np.concatenate(ntok))
        np.save(out / "n_shared_tokens.npy", np.concatenate(shared))
        (out / "ids.json").write_text(json.dumps(ids))
        (out / "index.json").write_text(json.dumps({
            "model_id": MODEL_ID, "model": kind, "init_seed": 0 if kind == "random" else None, "n": len(df),
            "pools": ["disk1_orange", "disk2_blue", "background"], "shape": list(P.shape), "dtype": "float32",
            "mask_rule": "token = any disk pixel in its patch in either tubelet frame; disk1 = red>128 & > G,B; "
                         "disk2 = blue>128 & > R,G (minus disk1 tokens); background = neither",
            "seconds": time.time() - t0, "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "torch": torch.__version__}, indent=1))
        print(f"{kind}: done in {time.time() - t0:.1f} s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

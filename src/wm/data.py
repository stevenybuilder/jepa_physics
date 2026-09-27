"""Loading, decoding and QA for the supplied clips.

The data directory is read-only: nothing in this module writes under DATA_ROOT.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "vjepa-physics-takehome-4E00" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS = ("direction", "speed", "acceleration")

N_FRAMES, SIZE, PATCH, TUBELET = 16, 256, 16, 2
GRID = SIZE // PATCH           # 16 patches per side
N_STEPS = N_FRAMES // TUBELET  # 8 token time steps

# Disk segmentation. DATA.md says "blue disk", but the decoded frames show an ORANGE disk
# (core RGB ~ (235, 117, 33)) on a dark grey background (median RGB (29, 31, 28)); checked with
# both PyAV and ffmpeg. So the colour test uses the red channel. Red-channel histogram on three
# clips (speed/0, direction/900, acceleration/1535): background + codec noise all below 64,
# disk core in 208-240, anti-aliased edge sparse in between. 128 is the half-way point between
# background and disk core; it gives a stable ~350 px/frame (disk radius ~10.6 px) on all three.
DISK_CHANNEL = 0   # red
DISK_THRESH = 128


def load_table(dataset, root=None):
    """One row per manifest id, with labels from metadata.json.

    label     = primary target: theta_degrees for direction, magnitude for speed/acceleration.
    label_idx = rank (0..63) of the label value among the dataset's distinct values.
    root: a directory holding manifest.jsonl + videos/ in the same schema (e.g. artifacts/stimuli/hard) to read
    instead of DATA_ROOT / dataset; `dataset` then only picks the label column, and the 64-value check is skipped.
    """
    override = root is not None
    root = Path(root) if override else DATA_ROOT / dataset
    rows = []
    with open(root / "manifest.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            meta = json.loads((root / entry["metadata"]).read_text())
            assert meta["id"] == entry["id"], (dataset, entry)
            rows.append({
                "id": entry["id"],
                "video": str(root / entry["video"]),
                "theta_degrees": meta["theta_degrees"],
                "speed_mps": meta["speed_mps"],
                "acceleration_mps2": meta["acceleration_mps2"],
                "magnitude": meta.get("magnitude", np.nan),  # direction metadata has no magnitude
                "motion": meta["motion"],
                "start_x": meta["start_position_xy_m"][0],
                "start_y": meta["start_position_xy_m"][1],
            })
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    assert df["id"].is_unique
    df["label"] = df["theta_degrees"] if dataset == "direction" else df["magnitude"]
    assert df["label"].notna().all(), dataset
    values, df["label_idx"] = np.unique(df["label"].to_numpy(), return_inverse=True)
    assert override or len(values) == 64, (dataset, len(values))
    return df


def decode(video_path):
    """All frames as uint8 RGB, shape [16, 256, 256, 3]."""
    with av.open(str(video_path)) as container:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    assert frames.shape == (N_FRAMES, SIZE, SIZE, 3) and frames.dtype == np.uint8, (video_path, frames.shape)
    return frames


def frame_hash(frames):
    """sha256 of the decoded frame bytes."""
    return hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()


def disk_pixels(frames, thresh=DISK_THRESH):
    """Bool [16, 256, 256]: pixel belongs to the disk."""
    x = frames.astype(np.int16)
    c = x[..., DISK_CHANNEL]
    others = np.delete(x, DISK_CHANNEL, axis=-1)
    return (c > thresh) & (c > others[..., 0]) & (c > others[..., 1])


def disk_mask(frames, thresh=DISK_THRESH):
    """Disk mask on the token grid.

    Returns (mask [8, 16, 16] bool, counts [16] int): a patch is True if any disk pixel falls
    in it in either frame of its tubelet; counts = disk pixels per frame.
    """
    pix = disk_pixels(frames, thresh)
    counts = pix.reshape(N_FRAMES, -1).sum(axis=1)
    per_frame = pix.reshape(N_FRAMES, GRID, PATCH, GRID, PATCH).any(axis=(2, 4))     # [16,16,16]
    mask = per_frame.reshape(N_STEPS, TUBELET, GRID, GRID).any(axis=1)                # [8,16,16]
    assert mask.shape == (N_STEPS, GRID, GRID)
    return mask, counts


def qa(dataset):
    """Decode every clip and write results/qa_{dataset}.json. Reports, never filters."""
    df = load_table(dataset)
    per_clip, hashes = [], {}
    for row in df.itertuples():
        frames = decode(row.video)
        mask, counts = disk_mask(frames)
        h = frame_hash(frames)
        hashes.setdefault(h, []).append(int(row.id))
        per_clip.append({
            "id": int(row.id),
            "frame_hash": h,
            "visible_frac": float((counts > 0).mean()),
            "min_pixels": int(counts.min()),
            "max_pixels": int(counts.max()),
            "steps_with_disk_patch": int(mask.any(axis=(1, 2)).sum()),
        })

    vis = np.array([c["visible_frac"] for c in per_clip])
    out_of_frame = [c["id"] for c in per_clip if c["min_pixels"] == 0]
    dup_groups = [ids for ids in hashes.values() if len(ids) > 1]
    label_of = dict(zip(df["id"], df["label"]))
    report = {
        "dataset": dataset,
        "n_clips": len(df),
        "disk_channel": "red", "disk_thresh": DISK_THRESH,
        "visible_frac": {"mean": float(vis.mean()), "min": float(vis.min()),
                         "n_all_frames_visible": int((vis == 1).sum()),
                         "n_no_frame_visible": int((vis == 0).sum())},
        "n_clips_any_frame_disk_absent": len(out_of_frame),
        "clips_any_frame_disk_absent": out_of_frame,
        "n_unique_frame_hashes": len(hashes),
        "duplicate_groups": dup_groups,
        "duplicate_groups_with_conflicting_labels": [
            ids for ids in dup_groups if len({label_of[i] for i in ids}) > 1],
        "label_value_counts": {str(k): int(v) for k, v in sorted(Counter(df["label"]).items())},
        "motion_counts": {str(k): int(v) for k, v in Counter(df["motion"]).items()},
        "per_clip": per_clip,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"qa_{dataset}.json").write_text(json.dumps(report, indent=1))
    return report

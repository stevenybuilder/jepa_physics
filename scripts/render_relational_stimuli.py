"""Relational-motion clip set (ideas brief E4): two disks moving along x, balanced over common and relative velocity.

Design: common velocity c in {-3..3} m/s x relative velocity v_rel = v1 - v2 in {-6..6} m/s (91 cells) x 7 clips
per cell = 637 clips. v1 = c + v_rel / 2, v2 = c - v_rel / 2 (|v| <= 6 m/s). Disk 1 orange (the validated renderer's
colour), disk 2 blue; which disk is on top is random per clip, so identity is carried by colour alone. Rows are
fixed per clip: the top disk's y ~ U[0.75, 3.0] m, the bottom disk's y ~ U[-3.0, -0.75] m (>= 1.5 m = 48 px apart
centre to centre, > 2 r = 21 px, so the disks never overlap or collide). Start x ~ U[-4, 4] m per disk, redrawn
until the disk is fully inside the frame in all 16 frames (rounded centre in [11, 244] px); rejections counted.
Renderer = wm.render_twin (flat dark background, r = 10.5 px, integer-rounded centre, MPEG-4 Part 2, GOP 12).

Writes artifacts/stimuli/relational/{manifest.jsonl, videos/scene_XXXX/{video.mp4, metadata.json}} (schema of the
other sets; load_table(..., root=...) reads it: theta_degrees / speed_mps / start_position_xy_m describe disk 1,
magnitude = v_rel), splits/split_relational.json (wm.splits.make_split with label = (c, v_rel) cell: 80/20 stratified
by cell, 5 stratified folds in train, identical-frame units merged) and results/p5_relational_stimuli_validation.json.

  python scripts/render_relational_stimuli.py [--out artifacts/stimuli] [--skip-render]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, SIZE, decode, disk_pixels, frame_hash, load_table  # noqa: E402
from wm.render_twin import FPS, RenderParams, disk_coverage, to_pixels, write_mp4  # noqa: E402

NAME = "relational"
COMMON = [float(c) for c in range(-3, 4)]
VREL = [float(v) for v in range(-6, 7)]
N_PER_CELL = 7
N_FRAMES = 16
BLUE = (41, 116, 231)                      # disk 2; orange disk 1 = RenderParams().fg
LO_PX, HI_PX = 11.0, SIZE - 12.0           # rounded centre col in [11, 244]: disk (r 10.5) fully inside 256 px
Y_BAND = (0.75, 3.0)


def cell_of(c, v_rel):
    return COMMON.index(c) * len(VREL) + VREL.index(v_rel)


def x_track(x0, v):
    return x0 + v * np.arange(N_FRAMES) / FPS


def in_frame(x0, v, y):
    px = to_pixels(np.stack([x_track(x0, v), np.full(N_FRAMES, y)], 1))
    col, row = np.round(px[:, 0]), np.round(px[:, 1])
    return bool(((col >= LO_PX) & (col <= HI_PX) & (row >= LO_PX) & (row <= HI_PX)).all())


def layout(seed=0):
    """637 metadata dicts (id = cell * 7 + k) and the rejection count."""
    rng = np.random.default_rng(seed)
    metas, rejected, i = [], 0, 0
    for c in COMMON:
        for vr in VREL:
            v1, v2 = c + vr / 2, c - vr / 2
            for _ in range(N_PER_CELL):
                top1 = bool(rng.integers(2))
                ya, yb = rng.uniform(*Y_BAND), -rng.uniform(*Y_BAND)
                y1, y2 = (ya, yb) if top1 else (yb, ya)
                xs = []
                for v, y in ((v1, y1), (v2, y2)):
                    while True:
                        x0 = rng.uniform(-4.0, 4.0)
                        if in_frame(x0, v, y):
                            break
                        rejected += 1
                    xs.append(x0)
                gap0 = xs[0] - xs[1]
                t_cross = (-gap0 / vr) if vr != 0 and -gap0 / vr > 0 else None
                metas.append({"id": i, "motion": "velocity", "fps": FPS, "frames": N_FRAMES,
                              "theta_degrees": 0.0 if v1 >= 0 else 180.0, "speed_mps": abs(v1),
                              "acceleration_mps2": 0.0, "magnitude": vr, "primary_label": "v_rel_mps",
                              "start_position_xy_m": [float(xs[0]), float(y1)],
                              "v1_mps": v1, "v2_mps": v2, "v_rel_mps": vr, "common_mps": c,
                              "closing_speed_mps": abs(vr), "cell": cell_of(c, vr),
                              "disk1": {"colour_rgb": list(RenderParams().fg), "start_xy_m": [float(xs[0]), float(y1)],
                                        "vx_mps": v1},
                              "disk2": {"colour_rgb": list(BLUE), "start_xy_m": [float(xs[1]), float(y2)],
                                        "vx_mps": v2},
                              "disk1_on_top": top1,
                              "x_gap0_m": float(gap0),
                              "x_crossing_time_s": None if t_cross is None else float(t_cross),
                              "x_crossing_in_clip": bool(t_cross is not None and t_cross <= (N_FRAMES - 1) / FPS),
                              "contact": "never (vertical offset >= 1.5 m); x_crossing_time_s = time the x "
                                         "coordinates coincide (the time-to-contact analogue), null if they diverge"})
                i += 1
    return metas, rejected


def tracks_px(meta):
    """Rounded pixel centres [2, 16, 2] (col, row) of disk 1 and disk 2."""
    out = []
    for d in ("disk1", "disk2"):
        x0, y = meta[d]["start_xy_m"]
        px = to_pixels(np.stack([x_track(x0, meta[d]["vx_mps"]), np.full(N_FRAMES, y)], 1))
        out.append(np.round(px))
    return np.stack(out)


def render(meta, p=None):
    """uint8 [16, 256, 256, 3] before the codec."""
    p = p or RenderParams()
    bg = np.asarray(p.bg, np.float32)
    fgs = [np.asarray(p.fg, np.float32), np.asarray(BLUE, np.float32)]
    tr = tracks_px(meta)
    out = np.empty((N_FRAMES, SIZE, SIZE, 3), np.uint8)
    for k in range(N_FRAMES):
        img = np.broadcast_to(bg, (SIZE, SIZE, 3)).copy()
        for d in range(2):
            cov = disk_coverage(tr[d, k], p.radius, p.supersample)[..., None]
            img += cov * (fgs[d] - bg)
        out[k] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return out


def blue_pixels(frames, thresh=128):
    x = frames.astype(np.int16)
    return (x[..., 2] > thresh) & (x[..., 2] > x[..., 0]) & (x[..., 2] > x[..., 1])


def write_set(out_root, metas, p):
    root = Path(out_root) / NAME
    (root / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    for m in metas:
        scene = root / "videos" / f"scene_{m['id']:04d}"
        scene.mkdir(exist_ok=True)
        write_mp4(render(m, p), str(scene / "video.mp4"), p.bit_rate, p.gop)
        (scene / "metadata.json").write_text(json.dumps({**m, "stimulus_set": NAME}, indent=2))
        rows.append({"id": m["id"], "video": f"videos/scene_{m['id']:04d}/video.mp4",
                     "metadata": f"videos/scene_{m['id']:04d}/metadata.json"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def check_set(root, metas, p):
    """Decoded clip vs pre-codec render; centroid of each colour mask vs the metadata track; frame hashes."""
    mae, errs, missing, hashes = [], {1: [], 2: []}, 0, {}
    for m in metas:
        dec = decode(root / "videos" / f"scene_{m['id']:04d}" / "video.mp4")
        hashes[m["id"]] = frame_hash(dec)
        mae.append(float(np.abs(dec.astype(np.int16) - render(m, p).astype(np.int16)).mean()))
        tr = tracks_px(m) + 0.5
        for d, mask in ((1, disk_pixels(dec)), (2, blue_pixels(dec))):
            for k in range(N_FRAMES):
                if not mask[k].any():
                    missing += 1
                    continue
                yy, xx = np.nonzero(mask[k])
                errs[d].append(float(np.hypot(xx.mean() + 0.5 - tr[d - 1, k, 0], yy.mean() + 0.5 - tr[d - 1, k, 1])))
    return hashes, {"codec_mae_vs_precodec_mean": float(np.mean(mae)), "codec_mae_vs_precodec_max": float(np.max(mae)),
                    "centroid_err_px_median": {f"disk{d}": float(np.median(e)) for d, e in errs.items()},
                    "centroid_err_px_max": {f"disk{d}": float(np.max(e)) for d, e in errs.items()},
                    "n_disk_frames_not_found": missing, "n_disk_frames": 2 * N_FRAMES * len(metas)}


def make_relational_split(root, hashes):
    from wm.splits import make_split, validate_split
    df = load_table("speed", root=root)
    cells = {m["id"]: m["cell"] for m in (json.loads((root / r).read_text()) for r in
                                           (json.loads(l)["metadata"] for l in open(root / "manifest.jsonl")))}
    df["label"] = df["id"].map(cells)
    split = make_split(df, hashes)
    validate_split("relational", split, df=df)
    split["grouping"] = ("label = (c, v_rel) cell index (91 cells); 80/20 stratified by cell, 5 stratified folds by "
                         "cell inside train (seed 0), identical-frame units merged (wm.splits.make_split)")
    return split


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "stimuli"))
    ap.add_argument("--skip-render", action="store_true", help="reuse written clips; redo checks and split")
    a = ap.parse_args(argv)
    from wm.probes import write_json
    p = RenderParams()
    metas, rejected = layout()
    root = Path(a.out) / NAME
    if not a.skip_render:
        write_set(a.out, metas, p)
    hashes, check = check_set(root, metas, p)
    split = make_relational_split(root, hashes)
    sp = PROJECT_ROOT / "splits" / "split_relational.json"
    sp.write_text(json.dumps({"relational": split}, indent=1))
    v1 = np.array([m["v1_mps"] for m in metas]); v2 = np.array([m["v2_mps"] for m in metas])
    c = (v1 + v2) / 2; vr = v1 - v2
    report = {"set": NAME, "n_clips": len(metas), "n_cells": len(COMMON) * len(VREL), "clips_per_cell": N_PER_CELL,
              "common_mps": COMMON, "v_rel_mps": VREL, "start_rejections": rejected,
              "start_rule": "x0 ~ U[-4, 4] m per disk, redrawn until the rounded centre stays in [11, 244] px in all "
                            "16 frames; numpy default_rng(0)",
              "corr": {"c_vrel": float(np.corrcoef(c, vr)[0, 1]), "v1_v2": float(np.corrcoef(v1, v2)[0, 1])},
              "n_x_crossing_in_clip": int(sum(m["x_crossing_in_clip"] for m in metas)),
              "renderer": {"radius": p.radius, "disk1_rgb": list(p.fg), "disk2_rgb": list(BLUE), "bg": list(p.bg),
                           "round_centre": p.round_centre, "codec": "MPEG-4 Part 2, GOP 12 (wm.render_twin.write_mp4)"},
              **check, "split_counts": split["counts"]}
    write_json(PROJECT_ROOT / "results" / "p5_relational_stimuli_validation.json", report, sp)
    print(json.dumps({k: report[k] for k in ("n_clips", "start_rejections", "corr", "codec_mae_vs_precodec_mean",
                                             "centroid_err_px_median", "n_disk_frames_not_found", "split_counts")},
                     indent=1))


if __name__ == "__main__":
    main()

"""Counterfactual-twin renderer for the direction clips (spec §6 item 5b).

Reverse-engineered from the supplied videos (checked on decoded frames, see validate()):
  - 256 x 256, 16 frames, 24 fps; frame k is the scene at t = k / 24 s (frame 0 = start position).
  - background flat RGB (29, 31, 28) far from the disk (std 0 over a 56 x 60 corner, identical across frames
    and clips); disk interior ~ (232, 114, 41) after decoding (orange, not blue as DATA.md says).
  - world -> pixel: 32 px per metre, origin at pixel (128, 128), y up:  col = 128 + 32 x,  row = 128 - 32 y.
    The centre is rounded to the nearest integer pixel (centroids of clip 0 step 87 -> 88 -> 90 at 1.33 px/frame).
  - motion along u = (cos theta, sin theta): p(t) = p0 + (v t + a t^2 / 2) u (velocity clips a = 0, acceleration
    clips start at rest, v = 0).
  - disk radius ~10.5 px with an anti-aliased edge (~350 px above the red > 128 threshold per frame).
  - codec: MPEG-4 Part 2 ('mp4v', Simple Profile), yuv420p, GOP 12 (I at frames 0 and 12), no B-frames,
    ~40 kbit/s (4.2 KB per clip). The 2 x 2 chroma subsampling and 8 x 8 DCT ringing around the disk are part of
    what V-JEPA sees, so the twin goes through the same codec (encode with PyAV, decode back) by default.

The radius and colours are fit on supplied clips by fit_params(); validate() re-renders clips from their metadata
and reports per-frame mean absolute pixel error and disk-mask IoU. If agreement is poor, use the mask-level twin
(twin_mask) as the readout target instead of pixels.
"""
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np

from wm.data import N_FRAMES, SIZE, decode, disk_pixels, load_table

FPS = 24
PX_PER_M = 32.0
ORIGIN = 128.0


@dataclass
class RenderParams:
    radius: float = 10.5
    fg: tuple = (231, 116, 41)     # RGB fed to the codec, from fit_params (decoded disk median (232, 114, 41))
    bg: tuple = (29, 32, 30)       # from fit_params; decodes to exactly (29, 31, 28) through rgb24 -> yuv420p -> rgb24
    round_centre: bool = True
    supersample: int = 8
    codec: bool = True
    bit_rate: int = 40000
    gop: int = 12


def load_meta(dataset, clip_id):
    """metadata.json of one clip (read only)."""
    from wm.data import DATA_ROOT
    return json.loads((DATA_ROOT / dataset / "videos" / f"scene_{int(clip_id):04d}" / "metadata.json").read_text())


def trajectory_m(meta, theta_deg=None):
    """World positions [16, 2] (metres) of the disk centre; theta_deg overrides the clip's direction."""
    th = np.radians(meta["theta_degrees"] if theta_deg is None else theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    t = np.arange(meta.get("frames", N_FRAMES)) / float(meta.get("fps", FPS))
    s = meta["speed_mps"] * t + 0.5 * meta["acceleration_mps2"] * t ** 2
    return np.asarray(meta["start_position_xy_m"], float)[None] + s[:, None] * u[None]


def to_pixels(xy):
    """World metres [.., 2] -> pixel (col, row) [.., 2], unrounded."""
    xy = np.asarray(xy, float)
    return np.stack([ORIGIN + PX_PER_M * xy[..., 0], ORIGIN - PX_PER_M * xy[..., 1]], axis=-1)


def disk_coverage(centre, radius, ss=8):
    """Fraction of each pixel covered by a disk (supersampled ss x ss). Pixel (r, c) spans [c, c+1) x [r, r+1) with
    its centre at (c + 0.5, r + 0.5); the disk centre (col, row) is given in pixel-index units, so a centre of
    (87, 110) sits on the middle of pixel (110, 87)."""
    cx, cy = centre[0] + 0.5, centre[1] + 0.5
    r0, r1 = int(max(np.floor(cy - radius - 1), 0)), int(min(np.ceil(cy + radius + 1), SIZE))
    c0, c1 = int(max(np.floor(cx - radius - 1), 0)), int(min(np.ceil(cx + radius + 1), SIZE))
    cov = np.zeros((SIZE, SIZE), np.float32)
    if r0 >= r1 or c0 >= c1:
        return cov
    off = (np.arange(ss) + 0.5) / ss
    ys = (np.arange(r0, r1)[:, None] + off[None]).reshape(-1)
    xs = (np.arange(c0, c1)[:, None] + off[None]).reshape(-1)
    inside = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= radius ** 2
    cov[r0:r1, c0:c1] = inside.reshape(r1 - r0, ss, c1 - c0, ss).mean(axis=(1, 3))
    return cov


def render_rgb(centres_px, p: RenderParams):
    """uint8 [16, 256, 256, 3] before any codec."""
    fg, bg = np.asarray(p.fg, np.float32), np.asarray(p.bg, np.float32)
    out = np.empty((len(centres_px), SIZE, SIZE, 3), np.uint8)
    for k, c in enumerate(centres_px):
        c = np.round(c) if p.round_centre else np.asarray(c, float)
        cov = disk_coverage(c, p.radius, p.supersample)[..., None]
        out[k] = np.clip(np.round(bg + cov * (fg - bg)), 0, 255).astype(np.uint8)
    return out


def write_mp4(frames, dest, bit_rate=40000, gop=12):
    """Encode uint8 [T, 256, 256, 3] with MPEG-4 Part 2 (yuv420p, GOP 12, no B-frames) to a path or file object."""
    with av.open(dest, mode="w", format="mp4") as out:
        st = out.add_stream("mpeg4", rate=FPS)
        st.width = st.height = SIZE
        st.pix_fmt = "yuv420p"
        st.codec_context.bit_rate = int(bit_rate)   # measured inert at 256 x 256 (5.1 KB per clip at 20k-200k)
        st.codec_context.gop_size = gop
        st.codec_context.max_b_frames = 0
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def codec_roundtrip(frames, bit_rate=40000, gop=12):
    """Encode in memory with the supplied clips' codec and decode back: uint8 [16, 256, 256, 3]."""
    buf = io.BytesIO()
    write_mp4(frames, buf, bit_rate, gop)
    buf.seek(0)
    with av.open(buf, mode="r") as inp:
        dec = np.stack([f.to_ndarray(format="rgb24") for f in inp.decode(video=0)])
    assert dec.shape == frames.shape, (dec.shape, frames.shape)
    return dec


def render(meta, theta_deg=None, params=None):
    """Frames of the clip described by meta (or its twin at theta_deg): uint8 [16, 256, 256, 3]."""
    p = params or RenderParams()
    frames = render_rgb(to_pixels(trajectory_m(meta, theta_deg)), p)
    return codec_roundtrip(frames, p.bit_rate, p.gop) if p.codec else frames


def write_twin(meta, theta_deg, path, params=None):
    """Render the twin, write it as an MP4 with the supplied clips' codec, and return the frames decoded from that
    file (what the model will see; decoding is deterministic across PyAV builds, encoding need not be)."""
    p = params or RenderParams()
    frames = render_rgb(to_pixels(trajectory_m(meta, theta_deg)), p)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, str(path), p.bit_rate, p.gop)
    return decode(path)


def twin_mask(meta, theta_deg=None, params=None):
    """Mask-level twin: bool [16, 256, 256], pixel covered by at least half of the disk (no colour, no codec)."""
    p = params or RenderParams()
    cs = to_pixels(trajectory_m(meta, theta_deg))
    return np.stack([disk_coverage(np.round(c) if p.round_centre else c, p.radius, p.supersample) >= 0.5 for c in cs])


# ---------------------------------------------------------------- fit and validation

def yuv_roundtrip(rgb):
    """What a flat RGB colour decodes to after rgb24 -> yuv420p -> rgb24 (swscale, as in the codec path)."""
    f = av.VideoFrame.from_ndarray(np.tile(np.asarray(rgb, np.uint8), (16, 16, 1)), format="rgb24")
    return f.reformat(format="yuv420p").reformat(format="rgb24").to_ndarray()[8, 8].astype(int)


def calibrate_colour(target, span=4):
    """Input RGB (within +-span per channel of target + 1) whose colour-conversion round trip lands closest (L1) to
    the decoded target colour. The conversion loses about 1 unit per channel, so rendering the decoded colour
    directly would shift the whole background."""
    target = np.asarray(target, int)
    best, best_err = tuple(int(v) for v in target), None
    rng = range(-span + 1, span + 2)
    for d0 in rng:
        for d1 in rng:
            for d2 in rng:
                c = np.clip(target + np.array([d0, d1, d2]), 0, 255)
                err = int(np.abs(yuv_roundtrip(c) - target).sum())
                if best_err is None or err < best_err:
                    best, best_err = tuple(int(v) for v in c), err
                    if err == 0:
                        return best
    return best


def iou(a, b):
    """Per-frame IoU of two bool stacks [T, H, W]; frames where both are empty count as 1."""
    inter = (a & b).reshape(len(a), -1).sum(1)
    union = (a | b).reshape(len(a), -1).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def compare(real, fake):
    """Per-frame MAE (all pixels, uint8 units), MAE inside the union of disk bounding boxes, and disk-mask IoU."""
    real_i, fake_i = real.astype(np.int16), fake.astype(np.int16)
    mae = np.abs(real_i - fake_i).reshape(len(real), -1).mean(1)
    mr, mf = disk_pixels(real), disk_pixels(fake)
    local = []
    for k in range(len(real)):
        rr, cc = np.nonzero(mr[k] | mf[k])
        if len(rr) == 0:
            local.append(0.0)
            continue
        sl = (slice(max(rr.min() - 4, 0), rr.max() + 5), slice(max(cc.min() - 4, 0), cc.max() + 5))
        local.append(float(np.abs(real_i[k][sl] - fake_i[k][sl]).mean()))
    return {"mae": mae, "mae_disk_box": np.array(local), "iou": iou(mr, mf)}


def pick_clips(dataset="direction", n=20, seed=0):
    df = load_table(dataset)
    idx = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


def fit_params(dataset="direction", n=8, seed=1, radii=(10.0, 10.25, 10.5, 10.75, 11.0), base=None):
    """Grid-fit the radius and the centre convention (rounded vs sub-pixel) on n clips disjoint from the default
    validation draw (different seed), minimising the mean disk-box MAE after the codec round trip. Colours are the
    medians of decoded disk-interior and background pixels over those clips."""
    base = base or RenderParams()
    df = load_table(dataset)
    val_ids = set(pick_clips(dataset, 20, 0)["id"])       # never fit on the default validation clips
    pool = df[~df["id"].isin(val_ids)]
    rows = pool.iloc[np.sort(np.random.default_rng(seed).choice(len(pool), size=n, replace=False))]
    reals = [decode(v) for v in rows["video"]]
    metas = [load_meta(dataset, i) for i in rows["id"]]
    fg_dec = np.median(np.concatenate([f[disk_pixels(f)] for f in reals]), axis=0)
    corner = np.concatenate([f[:, 200:, :40].reshape(-1, 3) for f in reals])
    bg_dec = np.median(corner, axis=0)
    fg, bg = calibrate_colour(fg_dec), calibrate_colour(bg_dec)
    grid = []
    for rc in (True, False):
        for r in radii:
            p = RenderParams(**{**asdict(base), "radius": r, "round_centre": rc,
                                "fg": tuple(int(v) for v in fg), "bg": tuple(int(v) for v in bg)})
            s = [compare(real, render(m, params=p)) for real, m in zip(reals, metas)]
            grid.append({"radius": r, "round_centre": rc,
                         "mae_disk_box": float(np.mean([x["mae_disk_box"].mean() for x in s])),
                         "iou_mean": float(np.mean([x["iou"].mean() for x in s]))})
    best = min(grid, key=lambda g: g["mae_disk_box"])
    p = RenderParams(**{**asdict(base), "radius": best["radius"], "round_centre": best["round_centre"],
                        "fg": tuple(int(v) for v in fg), "bg": tuple(int(v) for v in bg)})
    return p, {"fit_clip_ids": [int(i) for i in rows["id"]], "grid": grid, "best": best,
               "decoded_fg": fg_dec.tolist(), "decoded_bg": bg_dec.tolist(), "input_fg": list(fg), "input_bg": list(bg)}


def validate(dataset="direction", n=20, seed=0, params=None):
    """Re-render n supplied clips from their metadata, with and without the codec round trip, and compare with the
    decoded originals. Returns a JSON-able report."""
    p = params or RenderParams()
    rows = pick_clips(dataset, n, seed)
    out = {"params": asdict(p), "clip_ids": [int(i) for i in rows["id"]], "n_clips": n, "per_clip": []}
    agg = {"codec": {"mae": [], "mae_disk_box": [], "iou": []}, "no_codec": {"mae": [], "mae_disk_box": [], "iou": []}}
    for cid, video in zip(rows["id"], rows["video"]):
        real, meta = decode(video), load_meta(dataset, cid)
        rec = {"id": int(cid), "theta": meta["theta_degrees"], "motion": meta["motion"]}
        for name, codec in (("codec", True), ("no_codec", False)):
            c = compare(real, render(meta, params=RenderParams(**{**asdict(p), "codec": codec})))
            for k in agg[name]:
                agg[name][k].append(c[k])
            rec[name] = {"mae_mean": float(c["mae"].mean()), "mae_disk_box_mean": float(c["mae_disk_box"].mean()),
                         "iou_min": float(c["iou"].min()), "iou_mean": float(c["iou"].mean())}
        m_real, m_twin = disk_pixels(real), twin_mask(meta, params=p)
        rec["mask_twin_iou_mean"] = float(iou(m_real, m_twin).mean())
        out["per_clip"].append(rec)
    for name, a in agg.items():
        iou_all = np.concatenate(a["iou"])
        out[name] = {"frame_mae_mean": float(np.concatenate(a["mae"]).mean()),
                     "frame_mae_p95": float(np.percentile(np.concatenate(a["mae"]), 95)),
                     "disk_box_mae_mean": float(np.concatenate(a["mae_disk_box"]).mean()),
                     "iou_mean": float(iou_all.mean()), "iou_min": float(iou_all.min()),
                     "frac_frames_iou_gt_0.95": float((iou_all > 0.95).mean()),
                     "n_frames": int(len(iou_all))}
    out["mask_twin_iou_mean"] = float(np.mean([r["mask_twin_iou_mean"] for r in out["per_clip"]]))
    out["verdict"] = ("pixel twin usable" if out["codec"]["frac_frames_iou_gt_0.95"] >= 0.9
                      else "pixel agreement poor: use the mask-level twin (twin_mask) as the readout target")
    return out


def main(argv=None):
    import argparse
    from wm.probes import write_json
    ap = argparse.ArgumentParser(description="Fit and validate the counterfactual-twin renderer")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results" / "session2_renderer_validation.json"))
    a = ap.parse_args(argv)
    p, fit = fit_params()
    rep = validate(n=a.n, params=p)
    rep["fit"] = fit
    write_json(a.out, rep)
    print(json.dumps({k: rep[k] for k in ("codec", "no_codec", "mask_twin_iou_mean", "verdict")}, indent=1))
    print("params", asdict(p))
    return rep


if __name__ == "__main__":
    main()

"""Two new clip sets for the stimulus-difficulty control (spec §6.2 / §7 follow-up), written to artifacts/stimuli/.

  paper_layout  392 clips in the physics paper's layout: 8 directions (0, 45, ..., 315 deg) x 7 speeds (1..7 m/s) x
                7 start positions (drawn once, uniform in [-2, 2]^2, seed 0), constant speed, rendered with the
                validated counterfactual-twin renderer (wm.render_twin: flat dark background, orange disk r = 10.5 px,
                integer-rounded centre, MPEG-4 Part 2 codec).
  hard          the same 392 trajectories with a static textured grey floor (fixed smooth value-noise texture,
                +-20 % luminance contrast around grey 110), the disk radius halved (5.25 px, sub-pixel centre) and a
                fake directional light on the disk (radial luminance falloff centred 0.45 r up-left of the disk
                centre). A cheap proxy for the paper's Kubric regime (shaded sphere ~15-20 px on a grey floor).

Each set: artifacts/stimuli/{set}/manifest.jsonl + videos/scene_XXXX/{video.mp4, metadata.json}, the same schema as
data/*/manifest.jsonl, readable with wm.data.load_table("direction", root="artifacts/stimuli/{set}").

Validation (results/session2_stimuli_validation.json): the renderer's 20-clip agreement with the supplied clips
(wm.render_twin.validate), and for every written clip the codec loss (decoded vs pre-codec frames) and the disk
centroid recovered from the decoded video vs the metadata trajectory. Example frames: figures/stimuli_{set}_examples.png.

  python scripts/render_hard_stimuli.py [--out artifacts/stimuli] [--no-validate-renderer]

Render-seed replicates of the hard set (--hard-seed N, N >= 1): only the hard set, written as hard_seed{N}, with the start
positions drawn from default_rng(N) and the floor texture from texture_seed 1 + N (seed 0 = the original hard set);
design, disk, shading, resolution and codec unchanged. Validation -> results/session2_stimuli_validation_hard_seed{N}.json.

  python scripts/render_hard_stimuli.py --hard-seed 1
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, disk_pixels, load_table  # noqa: E402
from wm.render_twin import (RenderParams, SIZE, codec_roundtrip, disk_coverage, render_rgb, to_pixels,  # noqa: E402
                            trajectory_m, validate, write_mp4)

DIRECTIONS = [float(d) for d in range(0, 360, 45)]
SPEEDS = [float(s) for s in range(1, 8)]
N_STARTS = 7
HARD = {"floor_grey": 110.0, "floor_contrast": 0.20, "radius": 5.25, "light_offset_r": 0.45,
        "shade_max": 1.2, "shade_min": 0.3, "shade_falloff_r": 2.0, "texture_seed": 1, "round_centre": False}


def layout(seed=0):
    """392 metadata dicts, id = (direction_index * 7 + speed_index) * 7 + start_index."""
    starts = np.random.default_rng(seed).uniform(-2.0, 2.0, size=(N_STARTS, 2))
    metas, i = [], 0
    for th in DIRECTIONS:
        for v in SPEEDS:
            for p in starts:
                metas.append({"id": i, "theta_degrees": th, "motion": "velocity", "speed_mps": v,
                              "acceleration_mps2": 0.0, "magnitude": v, "primary_label": "theta_degrees",
                              "start_position_xy_m": [float(p[0]), float(p[1])], "fps": 24, "frames": 16})
                i += 1
    return metas, starts


def hard_seed_config(seed):
    """(set name, layout seed, renderer config) of hard-set render seed `seed`; seed 0 reproduces the original set."""
    return ("hard" if seed == 0 else f"hard_seed{seed}"), seed, {**HARD, "texture_seed": HARD["texture_seed"] + seed}


def value_noise(seed, size=SIZE, octaves=(8, 16, 32, 64)):
    """Fixed smooth texture in [-1, 1]: sum of bilinearly upsampled random grids, amplitude halving per octave."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size))
    for k, g in enumerate(octaves):
        grid = rng.uniform(-1, 1, (g + 1, g + 1))
        x = np.linspace(0, g, size)
        i0 = np.minimum(np.floor(x).astype(int), g - 1)
        f = x - i0
        rows = grid[i0] * (1 - f)[:, None] + grid[i0 + 1] * f[:, None]
        tex += 0.5 ** k * (rows[:, i0] * (1 - f)[None] + rows[:, i0 + 1] * f[None])
    return tex / np.abs(tex).max()


def hard_floor(cfg=HARD):
    t = value_noise(cfg["texture_seed"])
    return np.repeat((cfg["floor_grey"] * (1 + cfg["floor_contrast"] * t))[..., None], 3, axis=-1)   # [H, W, 3]


def render_hard(meta, floor, fg=(231, 116, 41), cfg=HARD):
    """uint8 [16, 256, 256, 3] before the codec: shaded small disk composited over the textured floor."""
    fg = np.asarray(fg, float)
    r = cfg["radius"]
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] + 0.5
    out = np.empty((16, SIZE, SIZE, 3), np.uint8)
    light = cfg["light_offset_r"] * r * np.array([-1.0, -1.0]) / np.sqrt(2)          # up-left in image coordinates
    for k, c in enumerate(to_pixels(trajectory_m(meta))):
        c = np.round(c) if cfg["round_centre"] else c
        cov = disk_coverage(c, r, 8)[..., None]
        lx, ly = c[0] + 0.5 + light[0], c[1] + 0.5 + light[1]
        dist = np.hypot(xx - lx, yy - ly) / (cfg["shade_falloff_r"] * r)
        shade = np.clip(cfg["shade_max"] - (cfg["shade_max"] - cfg["shade_min"]) * dist, cfg["shade_min"],
                        cfg["shade_max"])[..., None]
        disk = np.clip(fg * shade, 0, 255)
        out[k] = np.clip(np.round(floor * (1 - cov) + disk * cov), 0, 255).astype(np.uint8)
    return out


def write_set(out_root, name, metas, frames_fn, params):
    root = Path(out_root) / name
    (root / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    for m in metas:
        scene = root / "videos" / f"scene_{m['id']:04d}"
        scene.mkdir(exist_ok=True)
        write_mp4(frames_fn(m), str(scene / "video.mp4"), params.bit_rate, params.gop)
        (scene / "metadata.json").write_text(json.dumps({**m, "stimulus_set": name}, indent=2))
        rows.append({"id": m["id"], "video": f"videos/scene_{m['id']:04d}/video.mp4",
                     "metadata": f"videos/scene_{m['id']:04d}/metadata.json"})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def centroid_check(dec, expected_px, disk_fn):
    """Centroid of the recovered disk pixels vs the expected centre, over frames where the disk is fully in view."""
    errs, n_vis, n_full = [], 0, 0
    for k in range(len(dec)):
        m = disk_fn(dec[k], k)
        c = expected_px[k] + 0.5
        inside = 12 <= c[0] <= SIZE - 12 and 12 <= c[1] <= SIZE - 12
        n_vis += bool(m.any())
        if inside:
            n_full += 1
            if m.any():
                yy, xx = np.nonzero(m)
                errs.append(float(np.hypot(xx.mean() + 0.5 - c[0], yy.mean() + 0.5 - c[1])))
    return errs, n_vis, n_full


def check_set(root, name, pre_fn, disk_fn, params, round_centre):
    df = load_table("direction", root=root)
    mae, errs, vis, full, missing = [], [], 0, 0, 0
    for row in df.itertuples():
        meta = json.loads((Path(root) / "videos" / f"scene_{row.id:04d}" / "metadata.json").read_text())
        dec = decode(row.video)
        pre = pre_fn(meta)
        mae.append(float(np.abs(dec.astype(np.int16) - pre.astype(np.int16)).mean()))
        px = to_pixels(trajectory_m(meta))
        px = np.round(px) if round_centre else px
        e, v, f = centroid_check(dec, px, disk_fn)
        errs += e
        vis += v
        full += f
        missing += f - len(e)
    n_frames = 16 * len(df)
    return {"set": name, "n_clips": len(df), "label_values": sorted(df["label"].unique().tolist()),
            "codec_mae_vs_precodec_mean": float(np.mean(mae)), "codec_mae_vs_precodec_max": float(np.max(mae)),
            "centroid_err_px_median": float(np.median(errs)), "centroid_err_px_p95": float(np.percentile(errs, 95)),
            "centroid_err_px_max": float(np.max(errs)), "n_frames_centre_inside_12px": full,
            "n_frames_inside_disk_not_found": missing, "frac_frames_disk_visible": vis / n_frames,
            "frac_frames_fully_in_view": full / n_frames}


def example_png(root, path, ids=(0, 100, 250, 391), steps=(0, 5, 10, 15), zoom=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_table("direction", root=root).set_index("id")
    ncol = len(steps) + (1 if zoom else 0)
    fig, axes = plt.subplots(len(ids), ncol, figsize=(2.2 * ncol, 2.2 * len(ids)))
    for r, i in enumerate(ids):
        fr = decode(df.loc[i, "video"])
        meta = json.loads((Path(root) / "videos" / f"scene_{i:04d}" / "metadata.json").read_text())
        for c, k in enumerate(steps):
            axes[r, c].imshow(fr[k])
            axes[r, c].set_title(f"id {i} f{k}  {meta['theta_degrees']:.0f}deg {meta['speed_mps']:.0f}m/s", fontsize=6)
        if zoom:
            cx, cy = np.clip(np.round(to_pixels(trajectory_m(meta))[0]).astype(int), 16, SIZE - 16)
            axes[r, -1].imshow(fr[0][cy - 16:cy + 16, cx - 16:cx + 16], interpolation="nearest")
            axes[r, -1].set_title("frame 0, 32 px crop", fontsize=6)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main_hard_seed(a):
    """One render-seed replicate of the hard set (fresh starts and floor texture, everything else identical)."""
    from wm.probes import write_json
    params = RenderParams()
    name, lseed, cfg = hard_seed_config(a.hard_seed)
    metas, starts = layout(lseed)
    floor = hard_floor(cfg)
    floor_dec = codec_roundtrip(np.repeat(np.round(floor).astype(np.uint8)[None], 16, 0)).astype(np.int16)
    hd_pre = lambda m: render_hard(m, floor, params.fg, cfg)                                 # noqa: E731
    root = write_set(a.out, name, metas, hd_pre, params)
    print("written:", root)
    hard_disk = lambda fr, k: np.abs(fr.astype(np.int16) - floor_dec[k]).sum(-1) > 60     # noqa: E731
    report = {"render_seed": a.hard_seed,
              "layout": {"directions_deg": DIRECTIONS, "speeds_mps": SPEEDS, "starts_m": starts.round(4).tolist(),
                         "start_rule": f"7 starts uniform in [-2, 2]^2, numpy default_rng({lseed}), shared by every "
                                       "direction x speed cell; constant speed; 16 frames at 24 fps",
                         "n_clips": len(metas)},
              name: {"renderer": {**cfg, "fg": list(params.fg), "codec": "same MPEG-4 Part 2 path"},
                     **check_set(root, name, hd_pre, hard_disk, params, False)}}
    png = Path(a.figures) / f"stimuli_{name}_examples.png"
    example_png(root, png)
    report[name]["examples_png"] = str(png.relative_to(PROJECT_ROOT)) if png.is_relative_to(PROJECT_ROOT) else str(png)
    out = Path(a.results).with_name(f"session2_stimuli_validation_{name}.json")
    write_json(out, report)
    print(json.dumps({q: report[name][q] for q in ("codec_mae_vs_precodec_mean", "centroid_err_px_median",
                                                  "centroid_err_px_p95", "frac_frames_fully_in_view",
                                                  "n_frames_inside_disk_not_found")}, indent=1), "->", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "stimuli"))
    ap.add_argument("--no-validate-renderer", action="store_true", help="skip the 20-clip renderer re-validation")
    ap.add_argument("--results", default=str(PROJECT_ROOT / "results" / "session2_stimuli_validation.json"))
    ap.add_argument("--figures", default=str(PROJECT_ROOT / "figures"))
    ap.add_argument("--hard-seed", type=int, default=None, help="render only hard-set replicate N (>= 1) as hard_seedN")
    a = ap.parse_args(argv)
    if a.hard_seed is not None:
        assert a.hard_seed >= 1, "seed 0 is the original hard set"
        return main_hard_seed(a)
    from wm.probes import write_json
    params = RenderParams()
    metas, starts = layout()
    floor = hard_floor()
    floor_dec = codec_roundtrip(np.repeat(np.round(floor).astype(np.uint8)[None], 16, 0)).astype(np.int16)

    pl_pre = lambda m: render_rgb(to_pixels(trajectory_m(m)), params)                      # noqa: E731
    hd_pre = lambda m: render_hard(m, floor, params.fg)                                      # noqa: E731
    roots = {"paper_layout": write_set(a.out, "paper_layout", metas, lambda m: pl_pre(m), params),
             "hard": write_set(a.out, "hard", metas, lambda m: hd_pre(m), params)}
    print("written:", {k: str(v) for k, v in roots.items()})

    hard_disk = lambda fr, k: np.abs(fr.astype(np.int16) - floor_dec[k]).sum(-1) > 60     # noqa: E731
    report = {"layout": {"directions_deg": DIRECTIONS, "speeds_mps": SPEEDS, "starts_m": starts.round(4).tolist(),
                         "start_rule": "7 starts uniform in [-2, 2]^2, numpy default_rng(0), shared by every "
                                       "direction x speed cell; constant speed; 16 frames at 24 fps",
                         "n_clips": len(metas)},
              "paper_layout": {"renderer": asdict(params),
                               **check_set(roots["paper_layout"], "paper_layout", pl_pre,
                                           lambda fr, k: disk_pixels(fr[None])[0], params, True)},
              "hard": {"renderer": {**HARD, "fg": list(params.fg), "codec": "same MPEG-4 Part 2 path"},
                       **check_set(roots["hard"], "hard", hd_pre, hard_disk, params, False)},
              "disk_mask_note": ("wm.data.disk_pixels (red > 128 and red > green, blue) finds the disk in paper_layout; "
                                 "in the hard set the shaded rim can fall below it, so the hard-set centroid check uses "
                                 "|decoded - decoded floor| summed over channels > 60, and extract.py's diskmask/diskpool "
                                 "on the hard set is approximate")}
    if not a.no_validate_renderer:
        v = validate(n=20, params=params)
        report["renderer_validation_vs_supplied"] = {k: v[k] for k in ("codec", "no_codec", "mask_twin_iou_mean",
                                                                          "verdict", "clip_ids")}
    for name, root in roots.items():
        png = Path(a.figures) / f"stimuli_{name}_examples.png"
        example_png(root, png)
        report[name]["examples_png"] = str(png.relative_to(PROJECT_ROOT)) if png.is_relative_to(PROJECT_ROOT) else str(png)
    write_json(a.results, report)
    print(json.dumps({k: {q: report[k][q] for q in ("codec_mae_vs_precodec_mean", "centroid_err_px_median",
                                                     "centroid_err_px_p95", "frac_frames_fully_in_view",
                                                     "n_frames_inside_disk_not_found")}
                      for k in ("paper_layout", "hard")}, indent=1))
    if "renderer_validation_vs_supplied" in report:
        print("renderer vs supplied:", json.dumps(report["renderer_validation_vs_supplied"]["codec"]))


if __name__ == "__main__":
    main()

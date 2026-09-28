"""Time-rescaled paper_layout trajectories (ideas brief E3, "change the clock").

Caveat: at the fixed 24 fps a time-rescaled clip IS the clip of speed lam v (accel lam^2 a); many are bit-identical to
paper_layout clips, so these sets test probe linearity / extrapolation, not a clock (see scripts/run_clock_test.py).

x_lam(t) = x(lam t): same start, direction and frame count (16 frames at the nominal 24 fps), positions sampled at
lam * t. Velocity scales by lam, acceleration by lam^2, direction is unchanged. Rendered with the validated
paper_layout renderer (wm.render_twin: flat background, orange disk r = 10.5 px, rounded centre, MPEG-4 Part 2).

  clock_lam{0.5,2}        the 392 paper_layout clips (8 directions x 7 speeds 1..7 m/s x 7 starts, default_rng(0))
                          at lam = 0.5, 2 (lam = 1 is artifacts/stimuli/paper_layout itself).
  clock_acc_lam{0.5,1,2}  the same 8 x 7 x 7 layout with constant acceleration from rest (the supplied acceleration
                          set's law, trajectory_m with speed 0), a = 1..7 m/s^2, at lam = 0.5, 1, 2.

Metadata: speed_mps / acceleration_mps2 / magnitude are the EFFECTIVE labels (lam v, lam^2 a), so wm.data.load_table
reads effective labels; *_nominal hold the source labels; clock_lambda, source_set, source_id; in-frame flags
(disk fully inside the 256 px frame at every frame; centre inside at every frame). No clip is dropped: out-of-frame
clips are flagged and the analysis reports both all clips and the in-frame subset.

  python scripts/render_clock_stimuli.py [--sets clock_lam0.5 clock_lam2 clock_acc_lam0.5 clock_acc_lam1 clock_acc_lam2]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_hard_stimuli import layout  # noqa: E402
from wm.data import N_FRAMES, PROJECT_ROOT, SIZE  # noqa: E402
from wm.render_twin import FPS, RenderParams, render_rgb, to_pixels, write_mp4  # noqa: E402

ACCELS = [float(a) for a in range(1, 8)]
SETS = {"clock_lam0.5": ("velocity", 0.5), "clock_lam2": ("velocity", 2.0),
        "clock_acc_lam0.5": ("acceleration", 0.5), "clock_acc_lam1": ("acceleration", 1.0),
        "clock_acc_lam2": ("acceleration", 2.0)}


def clock_trajectory_m(meta, lam):
    """World positions [16, 2] of the NOMINAL trajectory sampled at lam * t: x_lam(t) = x(lam t)."""
    th = np.radians(meta["theta_degrees"])
    u = np.array([np.cos(th), np.sin(th)])
    t = lam * np.arange(meta.get("frames", N_FRAMES)) / float(meta.get("fps", FPS))
    s = meta["speed_mps"] * t + 0.5 * meta["acceleration_mps2"] * t ** 2
    return np.asarray(meta["start_position_xy_m"], float)[None] + s[:, None] * u[None]


def nominal_layout(motion):
    """392 nominal metadata dicts in the paper_layout id order; acceleration swaps speeds 1..7 for a = 1..7 from rest."""
    metas, _ = layout(0)
    if motion == "velocity":
        return metas
    return [{**m, "motion": "acceleration", "speed_mps": 0.0, "acceleration_mps2": ACCELS[(m["id"] // 7) % 7],
             "magnitude": ACCELS[(m["id"] // 7) % 7]} for m in metas]


def clock_meta(m, lam, name, radius=RenderParams.radius):
    px = np.round(to_pixels(clock_trajectory_m(m, lam)))
    full = ((px >= radius) & (px <= SIZE - 1 - radius)).all(1)
    centre = ((px >= 0) & (px <= SIZE - 1)).all(1)
    return {**m, "speed_mps": lam * m["speed_mps"], "acceleration_mps2": lam ** 2 * m["acceleration_mps2"],
            "magnitude": lam * m["magnitude"] if m["motion"] == "velocity" else lam ** 2 * m["magnitude"],
            "speed_mps_nominal": m["speed_mps"], "acceleration_mps2_nominal": m["acceleration_mps2"],
            "magnitude_nominal": m["magnitude"], "clock_lambda": lam, "source_set": "paper_layout",
            "source_id": m["id"], "in_frame_all_frames": bool(full.all()),
            "centre_in_frame_all_frames": bool(centre.all()), "n_frames_fully_in_frame": int(full.sum()),
            "stimulus_set": name}


def _write_one(job):
    root, m, lam = job
    scene = Path(root) / "videos" / f"scene_{m['id']:04d}"
    scene.mkdir(parents=True, exist_ok=True)
    p = RenderParams()
    write_mp4(render_rgb(to_pixels(clock_trajectory_m(m, lam)), p), str(scene / "video.mp4"), p.bit_rate, p.gop)
    return m["id"]


def write_clock_set(out_root, name, workers=8):
    motion, lam = SETS[name]
    root = Path(out_root) / name
    nominal = nominal_layout(motion)
    with Pool(workers) as pool:
        pool.map(_write_one, [(str(root), m, lam) for m in nominal], chunksize=8)
    rows, metas = [], []
    for m in nominal:
        cm = clock_meta(m, lam, name)
        (root / "videos" / f"scene_{m['id']:04d}" / "metadata.json").write_text(json.dumps(cm, indent=2))
        rows.append({"id": m["id"], "video": f"videos/scene_{m['id']:04d}/video.mp4",
                     "metadata": f"videos/scene_{m['id']:04d}/metadata.json"})
        metas.append(cm)
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"set": name, "motion": motion, "clock_lambda": lam, "n_clips": len(metas),
            "n_in_frame_all_frames": sum(c["in_frame_all_frames"] for c in metas),
            "n_centre_in_frame_all_frames": sum(c["centre_in_frame_all_frames"] for c in metas)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "stimuli"))
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    for name in a.sets:
        print(json.dumps(write_clock_set(a.out, name, a.workers)))


if __name__ == "__main__":
    main()

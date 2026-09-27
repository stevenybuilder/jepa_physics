import json

import numpy as np
import pandas as pd

from wm.pixels import clip_features, run_baseline


def _disk_clip(theta_deg, speed, r=10, size=256, t=16):
    """Orange disk on dark grey moving from the centre along theta at `speed` px/frame."""
    yy, xx = np.mgrid[:size, :size]
    frames = np.full((t, size, size, 3), 30, np.uint8)
    th = np.radians(theta_deg)
    centres = [(128 + speed * k * np.cos(th), 128 - speed * k * np.sin(th)) for k in range(t)]
    for k, (cx, cy) in enumerate(centres):
        frames[k][(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = (235, 117, 33)
    return frames, np.array(centres)


def test_clip_features_centroid_and_downsample():
    frames, centres = _disk_clip(30.0, 4.0)
    gray, cen = clip_features(frames)
    assert gray.shape == (16, 32, 32) and gray.dtype == np.uint8
    assert np.abs(cen - centres).max() < 0.5
    assert gray[0].max() > gray[0].min() + 50           # the disk survives the 8×8 block mean
    blank = np.full((16, 256, 256, 3), 30, np.uint8)
    assert np.isnan(clip_features(blank)[1]).all()      # no disk -> NaN centroid


def test_trajectory_baseline_decodes_direction_on_synthetic_clips(tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    theta = rng.uniform(0, 360, n)
    speed = np.full(n, 4.0)                              # fixed speed: displacement is linear in (cos, sin)
    feats = [clip_features(_disk_clip(a, s)[0]) for a, s in zip(theta, speed)]
    gray, cen = np.stack([f[0] for f in feats]), np.stack([f[1] for f in feats])
    df = pd.DataFrame({"id": range(n), "theta_degrees": theta, "speed_mps": speed, "acceleration_mps2": 0.0,
                       "motion": np.where(np.arange(n) % 2, "velocity", "acceleration")})
    te = np.arange(64, n)
    tr = np.arange(64)
    folds = tr % 5
    out = run_baseline("synthetic", "direction", "trajectory", gray, cen, tmp_path, df, (tr, te, folds))
    row = out["layers"][0]
    assert row["point"] == -1 and row["cv_mean"] > 0.95 and row["test_mae"] < 5.0
    assert set(row["by_motion"]) == {"acceleration", "velocity"}
    saved = json.loads((tmp_path / "p1a_synthetic_direction_trajectory.json").read_text())
    assert saved["layers"][0]["cv_acc15_mean"] == row["cv_acc15_mean"]
    px = run_baseline("synthetic", "direction", "pixels", gray, cen, tmp_path, df, (tr, te, folds))["layers"][0]
    assert px["dim"] == 63 and np.isfinite(px["cv_mean"])   # PCA capped at n_train − 1

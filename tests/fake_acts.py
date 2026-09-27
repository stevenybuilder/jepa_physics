"""A tiny fake meanpool.npy with a planted, layer-dependent signal, for the end-to-end test."""
import numpy as np

from wm.data import load_table


def write_fake_meanpool(root, dataset, d=32, seed=0):
    """[N, 26, d] float32 in manifest order. Direction ramps up around point 8, magnitude from point 2;
    each is carried by two noisy copies (distinct noise levels), plus isotropic noise."""
    rng = np.random.default_rng(seed)
    df = load_table(dataset)
    n = len(df)
    th = np.radians(df["theta_degrees"].to_numpy(float))
    mag = (df["speed_mps"] + df["acceleration_mps2"]).to_numpy(float)
    mag = (mag - mag.mean()) / mag.std()
    code = np.stack([np.sin(th), np.cos(th), np.sin(th), np.cos(th), mag, mag], axis=1)
    code += rng.standard_normal(code.shape) * np.array([0.1, 0.1, 0.3, 0.3, 0.1, 0.3])
    basis = np.linalg.qr(rng.standard_normal((d, code.shape[1])))[0]
    acts = np.zeros((n, 26, d), np.float32)
    for p in range(26):
        s_dir = 1 / (1 + np.exp(-(p - 8) / 1.5))
        s_mag = 1 / (1 + np.exp(-(p - 2) / 1.0))
        strength = np.array([s_dir] * 4 + [s_mag] * 2)
        acts[:, p] = (code * strength) @ basis.T + 0.3 * rng.standard_normal((n, d))
    out = root / dataset / "vjepa2"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "meanpool.npy", acts)
    return out / "meanpool.npy"


def write_fake_diskpool(root, dataset, nan_rows=(), d=32, seed=0):
    """[N, 26, 8, d] float16 disk pool from the fake meanpool signal (+ small per-step noise); rows in
    nan_rows are NaN at every step (disk out of frame throughout)."""
    mp = np.load(write_fake_meanpool(root, dataset, d, seed))
    rng = np.random.default_rng(seed + 1)
    dp = (mp[:, :, None, :] + 0.05 * rng.standard_normal((mp.shape[0], 26, 8, d))).astype(np.float16)
    dp[list(nan_rows)] = np.nan
    np.save(root / dataset / "vjepa2" / "diskpool.npy", dp)
    return root / dataset / "vjepa2" / "diskpool.npy"

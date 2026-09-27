"""Pixel and hand-crafted-trajectory baselines for the Part 1 probe curves (spec §6 item 2).

Two feature sets per clip, probed with the exact ridge / α-CV / metrics of wm.probes on the same split:
  pixels     : 16 grayscale frames at 32×32 (8×8 block mean of 256²) + the 15 frame differences
               (16·1024 + 15·1024 = 31744 numbers), PCA-256 fit on the train clips only.
  trajectory : disk centroid (x_t, y_t) per frame from the red-channel disk mask (wm.data.disk_pixels,
               red > 128), 32 numbers, + its 15 differences (30). NaN where the disk is out of frame;
               the train-standardiser sets those to the train mean.
Both are then z-scored with train statistics (the one coordinate system of wm.probes).
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from wm.data import N_FRAMES, PROJECT_ROOT, SIZE, decode, disk_pixels, load_table
from wm.probes import (ALPHAS, RESULTS, Standardizer, cv_select_alpha, fit_ridge, predict,
                       radius_summary, result_name, split_rows, targets, write_json)

PIX_ROOT = PROJECT_ROOT / "artifacts" / "pixels"
LOW = 32
N_PCA = 256
LUMA = np.array([0.299, 0.587, 0.114])


def clip_features(frames):
    """frames uint8 [16, 256, 256, 3] -> (gray32 uint8 [16, 32, 32], centroid float32 [16, 2] (x, y) px)."""
    t, h, w, _ = frames.shape
    k = h // LOW
    gray = frames.astype(np.float32) @ LUMA.astype(np.float32)
    small = gray.reshape(t, LOW, k, LOW, k).mean(axis=(2, 4))
    pix = disk_pixels(frames)
    counts = pix.sum(axis=(1, 2))
    ys, xs = np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (pix.sum(axis=1) * xs).sum(axis=1) / counts
        cy = (pix.sum(axis=2) * ys).sum(axis=1) / counts
    return np.clip(np.rint(small), 0, 255).astype(np.uint8), np.stack([cx, cy], 1).astype(np.float32)


def _decode_one(path):
    return clip_features(decode(path))


def load_pixels(dataset, workers=8, root=None):
    """Cached (gray32 [N, 16, 32, 32] uint8, centroid [N, 16, 2] float32) in manifest id order.
    First call decodes every clip with a process pool and writes artifacts/pixels/{dataset}{,_centroid}.npy."""
    root = Path(root or PIX_ROOT)
    f_gray, f_cen = root / f"{dataset}.npy", root / f"{dataset}_centroid.npy"
    df = load_table(dataset)
    if not (f_gray.exists() and f_cen.exists()):
        with Pool(workers) as pool:
            out = pool.map(_decode_one, df["video"].tolist(), chunksize=8)
        root.mkdir(parents=True, exist_ok=True)
        np.save(f_gray, np.stack([o[0] for o in out]))
        np.save(f_cen, np.stack([o[1] for o in out]))
    gray, cen = np.load(f_gray), np.load(f_cen)
    assert gray.shape == (len(df), N_FRAMES, LOW, LOW) and cen.shape == (len(df), N_FRAMES, 2)
    return gray, cen


def pixel_matrix(gray):
    """[N, 16·1024 + 15·1024] float32: frames in [0, 1] then frame differences."""
    x = gray.astype(np.float32) / 255.0
    n = len(x)
    return np.concatenate([x.reshape(n, -1), np.diff(x, axis=1).reshape(n, -1)], axis=1)


def trajectory_matrix(cen):
    """[N, 32 + 30]: centroid (x, y) per frame, normalised by the frame size, then its differences."""
    c = cen.astype(np.float64) / SIZE
    n = len(c)
    return np.concatenate([c.reshape(n, -1), np.diff(c, axis=1).reshape(n, -1)], axis=1)


def features(feature_set, gray, cen, tr, te, n_pca=N_PCA):
    """Train-standardised (X_train, X_test) for one feature set. PCA (pixels) is fit on train rows only."""
    if feature_set == "pixels":
        X = pixel_matrix(gray)
        pca = PCA(n_components=min(n_pca, len(tr) - 1), svd_solver="randomized", random_state=0).fit(X[tr])
        Xtr, Xte = pca.transform(X[tr]), pca.transform(X[te])
    elif feature_set == "trajectory":
        X = trajectory_matrix(cen)
        Xtr, Xte = X[tr], X[te]
    else:
        raise ValueError(feature_set)
    st = Standardizer().fit(Xtr)
    return st.transform(Xtr).astype(np.float64), st.transform(Xte).astype(np.float64)


def probe_row(Xtr, Xte, Ytr, Yte, folds, kind, score_fn, motion_tr=None, motion_te=None):
    """One results row in the layer_sweep schema (point = frac = -1): α by 5-fold CV, refit on train, test once."""
    cv = cv_select_alpha(Xtr, Ytr, folds, ALPHAS, score_fn)
    W, b = fit_ridge(Xtr, Ytr, cv["alpha"])
    Pte = predict(Xte, W, b)
    test = score_fn(Yte, Pte)
    row = {"point": -1, "frac": -1, "post_ln": False,
           "alpha": cv["alpha"], "cv_mean": cv["cv_mean"], "cv_sd": cv["cv_sd"],
           "cv_mae_mean": cv["cv_mae_mean"], "cv_mae_sd": cv["cv_mae_sd"],
           "test_r2": test["r2"], "test_mae": test["mae"], "n_train": len(Ytr), "n_test": len(Yte),
           "dim": int(Xtr.shape[1])}
    if kind == "circular":
        row["test_radius"] = radius_summary(Pte)
        row["cv_acc15_mean"] = float(np.mean([f["acc15"] for f in cv["fold_scores"]]))
        row["test_acc15"] = test["acc15"]
    if motion_tr is not None and len(set(motion_tr)) > 1:
        row["by_motion"] = {}
        for m in sorted(set(motion_tr)):
            fs = [score_fn(Ytr[(folds == k) & (motion_tr == m)], cv["oof"][(folds == k) & (motion_tr == m)])
                  for k in np.unique(folds)]
            r2s = [s["r2"] for s in fs]
            t = score_fn(Yte[motion_te == m], Pte[motion_te == m])
            row["by_motion"][m] = {"cv_mean": float(np.mean(r2s)), "cv_sd": float(np.std(r2s, ddof=1)),
                                   "cv_mae_mean": float(np.mean([s["mae"] for s in fs])),
                                   "test_r2": t["r2"], "test_mae": t["mae"]}
    return row, Pte


def run_baseline(dataset, variable, feature_set, gray=None, cen=None, results_dir=None, df=None, split=None):
    """Probe one variable with one feature set; writes results/p1a_{dataset}_{variable}_{feature_set}.json.
    df/split (tr, te, folds) default to load_table and splits/split_v1.json."""
    df = load_table(dataset) if df is None else df
    if gray is None or cen is None:
        gray, cen = load_pixels(dataset)
    tr, te, folds = split_rows(dataset, df) if split is None else split
    Y, kind, score_fn = targets(df, variable)
    Xtr, Xte = features(feature_set, gray, cen, tr, te)
    motion = df["motion"].to_numpy()
    row, Pte = probe_row(Xtr, Xte, Y[tr], Y[te], folds, kind, score_fn, motion[tr], motion[te])
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": feature_set, "model": feature_set,
           "shuffled": False, "alphas": ALPHAS.tolist(), "n_train": len(tr), "n_test": len(te),
           "features": {"pixels": f"16x{LOW}x{LOW} grayscale + 15 frame diffs, PCA-{N_PCA} fit on train",
                        "trajectory": "disk centroid (x,y)/256 per frame (red>128 mask) + 15 diffs"}[feature_set],
           "layers": [row]}
    if kind == "circular":
        out["test_theta"] = df["theta_degrees"].to_numpy()[te].tolist()
        out["test_pred"] = {"-1": Pte.round(4).tolist()}
    write_json(Path(results_dir or RESULTS) / (result_name("p1a", dataset, variable, feature_set) + ".json"), out)
    return out

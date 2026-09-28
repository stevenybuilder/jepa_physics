"""Is the direction ring an object-bound code or a scene-wide code?

Pools per clip, derived exactly from the stored per-time-step pools (no per-patch tokens exist on disk):
  scene      = meanpool.npy, the mean of all 8 x 256 tokens (stored, fp32).
  object     = mean of the disk tokens: sum_t p_t * diskpool_t / sum_t p_t, with p_t = diskmask[t].sum()
               (count-weighted over steps; Part 1's diskpool probes used an unweighted nanmean over steps).
  background = mean of the non-disk tokens: (256 * sum_t timepool_t - sum_t p_t * diskpool_t) / (2048 - sum_t p_t).
  The identity  scene = (P_obj * object + P_bg * background) / 2048  holds up to the fp16 storage of timepool/diskpool.
Random-init ViT-L stores meanpool/timepool only (no diskpool), so its control is the scene pool alone.

Per pool (direction set, split_v1: probe = train folds 0-4 -> test; geometry = train rows; held-out arcs = knot folds):
  probe      ridge (wm.probes recipe: train-standardised, alpha by the 5 train folds), test R2 on (sin, cos), circular
             MAE, acc15; all 26 points. CI: 200-draw bootstrap of test clips over fixed predictions.
  geometry   points 8/12/22: b/a of the ring-plane conic (run_ellipse q_conic), noise-corrected k=1 chart b/a in 1024-D,
             noise-corrected harmonic variance shares k=1 / k=2 (the cos 2theta saddle share), |circular corr| of the
             label-free atan2 angle in the centroid plane and in the activation-PC plane, participation ratios;
             chart_check at the point estimate. CI: 200-draw bootstrap of train clips with refit.
  heldout    points 8/12/22: the 16 contiguous 45-degree blocks of results/arcs (seeds 1-16), run_part2.build recipe
             (knot folds 0-2, PCA-64, label-free angle, count-weighted smoothing spline); mean distance of the spline
             and of the raw-centroid chord to the held-out centroids; ratio spline/chord. CI: 200-draw bootstrap of
             knot clips with refit.
  transfer   probe fit on pool A (A's train standardisation), applied to pool B's test rows standardised with B's own
             train statistics; all 26 points; test-clip bootstrap CI.
  timerev    each pool's forward probe applied to the same pool of the time-reversed clips (their own train
             standardisation): a genuine motion code decodes theta + 180; a static trajectory/position-set code does not.
Speed (speed set): probe and transfer as a scalar comparison.

Writes results/p5_object_vs_scene_direction.json and figures/fig_object_vs_scene.png.
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_object_vs_scene.py [--n-boot 200] [--workers 8]
"""
import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm import ellipse as el  # noqa: E402
from wm import geometry_checks as gc  # noqa: E402
from wm import manifold as mf  # noqa: E402
from wm import probes as pr  # noqa: E402
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.provenance import provenance, sha256_file  # noqa: E402

ACT = PROJECT_ROOT / "artifacts" / "activations"
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
GEOM_POINTS = (8, 12, 22)
ARC_SEEDS = tuple(range(1, 17))
POOLS = ("scene", "object", "background")
N_POINTS, GRID2 = 26, 256
KNOT_FOLDS = (0, 1, 2)


# ---------------------------------------------------------------- pools

def derive_pools(timepool, diskpool, mask):
    """timepool, diskpool [N, T, D] (diskpool NaN where the step has no disk patch), mask [N, T, G, G] bool.
    Returns dict(scene_recon, object, background [N, D] float64, n_obj, n_bg [N])."""
    tp = np.asarray(timepool, np.float64)
    dp = np.nan_to_num(np.asarray(diskpool, np.float64), nan=0.0)
    p = np.asarray(mask).reshape(mask.shape[0], mask.shape[1], -1).sum(-1).astype(np.float64)   # [N, T]
    g = np.asarray(mask).shape[2] * np.asarray(mask).shape[3]
    n_tok = g * tp.shape[1]
    dsum = (p[..., None] * dp).sum(1)
    tsum = g * tp.sum(1)
    n_obj = p.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        obj = dsum / n_obj[:, None]
    obj[n_obj == 0] = np.nan
    bg = (tsum - dsum) / (n_tok - n_obj)[:, None]
    return {"scene_recon": tsum / n_tok, "object": obj, "background": bg, "n_obj": n_obj, "n_bg": n_tok - n_obj}


def load_pools(dataset, point, model_dir="vjepa2"):
    d = ACT / dataset / model_dir
    scene = np.asarray(np.load(d / "meanpool.npy", mmap_mode="r")[:, point], np.float64)
    if not (d / "diskpool.npy").exists():
        return {"scene": scene}
    tp = np.load(d / "timepool.npy", mmap_mode="r")[:, point]
    dp = np.load(d / "diskpool.npy", mmap_mode="r")[:, point]
    mk = np.load(d / "diskmask.npy")
    out = derive_pools(tp, dp, mk)
    rel = np.abs(out["scene_recon"] - scene).max() / np.abs(scene).max()
    return {"scene": scene, "object": out["object"], "background": out["background"],
            "_identity_rel_err": float(rel), "_n_obj_mean": float(out["n_obj"].mean())}


# ---------------------------------------------------------------- probe + transfer

def ci(draws):
    draws = np.asarray(draws, float)
    return [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def fit_probe(Xtr, Ytr, folds, score_fn):
    st = pr.Standardizer().fit(Xtr)
    Z = st.transform(Xtr)
    cv = pr.cv_select_alpha(Z, Ytr, folds, pr.ALPHAS, score_fn)
    W, b = pr.fit_ridge(Z, Ytr, cv["alpha"])
    return {"st": st, "W": W, "b": b, "alpha": cv["alpha"], "cv_mean": cv["cv_mean"]}


def apply_probe(probe, Xte, Xtr_target=None):
    """Predict with probe on Xte; Xte is standardised with the TARGET pool's own train stats when Xtr_target given."""
    st = pr.Standardizer().fit(Xtr_target) if Xtr_target is not None else probe["st"]
    return pr.predict(st.transform(Xte), probe["W"], probe["b"])


def boot_scores(Y, P, score_fn, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Y)
    keys = list(score_fn(Y, P))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        s = score_fn(Y[i], P[i])
        for k in keys:
            draws[k].append(s[k])
    return {k: ci(v) for k, v in draws.items()}


def transfer_matrix(pools, tr, te, Ytr, Yte, folds, score_fn, n_boot=0, probes=None):
    """pools: dict name -> X [N, D]. Entry [a][b]: probe fit on pool a, tested on pool b (b's own train standardisation)."""
    names = list(pools)
    probes = probes or {a: fit_probe(pools[a][tr], Ytr, folds, score_fn) for a in names}
    M = {}
    for a in names:
        M[a] = {}
        for b in names:
            P = apply_probe(probes[a], pools[b][te], None if a == b else pools[b][tr])
            s = score_fn(Yte, P)
            if n_boot:
                s["ci"] = boot_scores(Yte, P, score_fn, n_boot)
            M[a][b] = s
    return M, probes


# ---------------------------------------------------------------- geometry

def fast_pca(X, k=64):
    """mf.fit_pca via the n x n Gram matrix (n << D): same subspace, explained variances and projections up to the sign
    of each component (every quantity used here is sign-invariant); ~20x faster than the D-wide SVD on CPU."""
    X = np.asarray(X, np.float64)
    mean = X.mean(0)
    Xc = X - mean
    n = X.shape[0]
    k = min(k, n - 1, X.shape[1])
    w, U = eigh(Xc @ Xc.T, subset_by_index=[n - k, n - 1])
    w, U = w[::-1].copy(), np.ascontiguousarray(U[:, ::-1])
    s = np.sqrt(np.clip(w, 1e-300, None))
    return mf.PCA(mean=mean, components=(U.T @ Xc) / s[:, None], explained=w / (X.shape[0] - 1))


def ring_geometry(X, y, k=64, fast=False):
    """Shape numbers of run_ellipse.analyse (same estimators) plus the label-free angle check and participation."""
    X, y = np.asarray(X, np.float64), np.asarray(y, float)
    pca = (fast_pca if fast else mf.fit_pca)(X, k)
    Z = pca.project(X)
    cent = mf.centroids(Z, y)
    values, count, C = cent["values"], cent["count"], cent["C"]
    th = np.radians(values)
    idx = np.searchsorted(values, y)
    A64 = el.chart_fit(C, th)["A"]
    ring_basis = np.linalg.svd(A64, full_matrices=False)[0][:, :2]
    con = el.conic_fit((C - C.mean(0)) @ ring_basis)
    Xc = np.array([X[idx == i].mean(0) for i in range(len(values))])
    R = X - Xc[idx]
    noise_trace = float((R ** 2).sum() / (len(X) - len(values)) * np.mean(1 / count))
    harm = el.harmonic_variance(Xc, th, noise_trace=noise_trace)
    chq, _ = el.chart_q_full(Xc, th, noise_trace=noise_trace)
    choice = mf.choose_angle_source(C, values)
    part = gc.participation_check(Z, y)
    return {"b_over_a_conic_ring_plane": con["q"],
            "b_over_a_k1_chart_full_noise_corrected": chq["q_noise_corrected"],
            "var_share_k1": harm["share_noise_corrected"][0], "saddle_share_k2": harm["share_noise_corrected"][1],
            "circ_corr_centroid_plane": abs(choice["checks"]["centroid"]["circular_corr"]),
            "circ_corr_activation_plane": abs(choice["checks"]["activation"]["circular_corr"]),
            "angle_source": choice["angle"] if choice["angle"] == "labels" else f"unsupervised/{choice['plane']}",
            "pr_centroids": part["pr_centroids"], "pr_residuals": part["pr_residuals"]}, pca


GEOM_KEYS = ("b_over_a_conic_ring_plane", "b_over_a_k1_chart_full_noise_corrected", "var_share_k1", "saddle_share_k2",
             "circ_corr_centroid_plane", "circ_corr_activation_plane", "pr_centroids", "pr_residuals")


def heldout_arcs(X, y, seeds=ARC_SEEDS, k=64, fast=False):
    """run_part2.build on knot rows X, y: per seed, PCA/centroids/smoothing spline on the kept values; distance of the
    spline and of the raw-centroid chord to the held-out values' centroids (same rows, never fit)."""
    X, y = np.asarray(X, np.float64), np.asarray(y, float)
    values = np.unique(y)
    sp, ch, labels_fallback = [], [], 0
    for s in seeds:
        mask, _ = mf.heldout_design(values, "contiguous", True, seed=s)
        held = values[mask]
        keep = ~np.isin(y, held)
        pca = (fast_pca if fast else mf.fit_pca)(X[keep], k)
        cent = mf.centroids(pca.project(X[keep]), y[keep])
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        labels_fallback += choice["angle"] == "labels"
        curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline="smooth")
        raw = mf.raw_knot_curve(curve, cent)
        truth = np.array([pca.project(X[y == v]).mean(0) for v in held])
        sp.append(np.linalg.norm(curve(curve.coord_of_value(held)) - truth, axis=1).mean())
        ch.append(np.linalg.norm(mf.piecewise_linear_point(raw, held) - truth, axis=1).mean())
    sp, ch = np.array(sp), np.array(ch)
    return {"spline_err": float(sp.mean()), "chord_err": float(ch.mean()), "ratio_spline_over_chord": float(sp.mean() / ch.mean()),
            "frac_arcs_spline_better": float(np.mean(sp < ch)), "n_labels_fallback": int(labels_fallback)}


HELD_KEYS = ("spline_err", "chord_err", "ratio_spline_over_chord", "frac_arcs_spline_better")


def _geom_point(args):
    """Point estimates with the pipeline's own PCA (mf.fit_pca), plus chart_check."""
    X, y, Xk, yk = args
    g0, pca = ring_geometry(X, y)
    h0 = heldout_arcs(Xk, yk)
    chart = gc.chart_check(X, y, pca, "centroid" if g0["angle_source"].endswith("centroid") else "activation")
    g0["chart_check"] = {"chart_vs_true_mae_deg": chart["chart_vs_true_mae_deg"],
                         "centroids_chart_vs_pc12_circ_corr": chart["centroids_chart_vs_pc12"]["circular_corr"],
                         "chart_radius": chart["chart_radius"]}
    return g0, h0


def _geom_boot(args):
    """n bootstrap draws (clips resampled with replacement, everything refit); Gram-matrix PCA for speed."""
    X, y, Xk, yk, n, seed = args
    rng = np.random.default_rng(seed)
    gd, hd = {q: [] for q in GEOM_KEYS}, {q: [] for q in HELD_KEYS}
    for _ in range(n):
        i = rng.integers(0, len(X), len(X))
        g, _ = ring_geometry(X[i], y[i], fast=True)
        j = rng.integers(0, len(Xk), len(Xk))
        h = heldout_arcs(Xk[j], yk[j], fast=True)
        for q in GEOM_KEYS:
            gd[q].append(g[q])
        for q in HELD_KEYS:
            hd[q].append(h[q])
    return gd, hd


def geometry_all(jobs, n_boot, workers, chunk=10):
    """jobs: key -> (X, y, Xk, yk). Returns key -> (geometry dict with ci, heldout dict with ci)."""
    keys = list(jobs)
    tasks = [(key, c) for key in keys for c in range(0, n_boot, chunk)]
    with ProcessPoolExecutor(workers) as ex:
        pts = dict(zip(keys, ex.map(_geom_point, [jobs[k] for k in keys])))
        boots = list(ex.map(_geom_boot, [(*jobs[key], min(chunk, n_boot - c), 1000 * keys.index(key) + c)
                                         for key, c in tasks]))
    out = {}
    for key in keys:
        g0, h0 = pts[key]
        gd, hd = {q: [] for q in GEOM_KEYS}, {q: [] for q in HELD_KEYS}
        for (k2, _), (g, h) in zip(tasks, boots):
            if k2 == key:
                for q in GEOM_KEYS:
                    gd[q] += g[q]
                for q in HELD_KEYS:
                    hd[q] += h[q]
        out[key] = ({**g0, "ci": {q: ci(v) for q, v in gd.items()}}, {**h0, "ci": {q: ci(v) for q, v in hd.items()}})
    return out


def principal_angles(X_a, X_b, y):
    """Principal angles (deg) between the circular-chart planes of two pools, each fit on train-standardised rows."""
    A = [gc.fit_circular_chart((X - X.mean(0)) / (X.std(0) + 1e-6), y)["A"] for X in (X_a, X_b)]
    from scipy.linalg import subspace_angles
    return np.sort(np.degrees(subspace_angles(A[0], A[1]))).tolist()


# ---------------------------------------------------------------- CPU re-encode with token-level pools

CPU_POOL_DIR = ACT / "object_scene_cpu"
SUBSAMPLE = {"direction_sub": (4, 2)}     # train clips per value, test clips per value (stratified, seed 0)
CPU_POOLS = ("object", "ring", "far", "background", "scene")


def random_subset(set_name):
    """(dataset table, row positions) of the clips re-encoded on CPU."""
    if set_name == "hard":
        df = load_table("direction", root=PROJECT_ROOT / "artifacts" / "stimuli" / "hard")
        return df, np.arange(len(df))
    df = load_table("direction")
    tr, te, _ = pr.split_rows("direction", df)
    n_tr, n_te = SUBSAMPLE[set_name]
    rng = np.random.default_rng(0)
    th = df["theta_degrees"].to_numpy()
    rows = []
    for idx, n in ((tr, n_tr), (te, n_te)):
        for v in np.unique(th):
            cand = idx[th[idx] == v]
            rows += rng.choice(cand, size=min(n, len(cand)), replace=False).tolist()
    return df, np.sort(np.array(rows))


def ring_far_masks(mask):
    """mask [B, T, G, G] bool (disk patches). ring = non-disk patches within one patch (3 x 3 neighbourhood, same token
    time step) of a disk patch; far = everything else that is not disk. Returns (ring, far) bool, same shape."""
    m = np.asarray(mask, bool)
    pad = np.pad(m, ((0, 0), (0, 0), (1, 1), (1, 1)))
    G = m.shape[-1]
    dil = np.zeros_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            dil |= pad[..., dy:dy + G, dx:dx + G]
    ring = dil & ~m
    return ring, ~dil


def token_pools(tokens, mask):
    """tokens [B, T*G*G, D] ((t, h, w) order), mask [B, T, G, G] -> dict of [B, D] pools and token counts."""
    import torch
    ring, far = ring_far_masks(mask)
    out, counts = {}, {}
    x = tokens.double()
    for name, mk in (("object", mask), ("ring", ring), ("far", far), ("background", ~np.asarray(mask, bool))):
        w = torch.from_numpy(np.asarray(mk).reshape(len(mk), -1).astype(np.float64))
        counts[name] = w.sum(1)
        out[name] = (torch.einsum("bn,bnd->bd", w, x) / counts[name].clamp(min=1)[:, None]).float().numpy()
    out["scene"] = x.mean(1).float().numpy()
    return out, {k: v.numpy() for k, v in counts.items()}


def extract_cpu(set_name, model_kind, threads=4, batch=2, chunk=16):
    """Re-encode clips on CPU (fp32; wm.extract.load_model: 'random' = torch.manual_seed(0) init, 'vjepa2' = pretrained)
    and store token-level pools per chunk (resumable): object, ring, far, background, scene [n, 26, 1024] float32."""
    import torch
    from wm.data import decode, disk_mask
    from wm.extract import encode, load_model, preprocess, set_precision
    torch.set_num_threads(threads)
    set_precision()
    df, rows = random_subset(set_name)
    out = CPU_POOL_DIR / f"{set_name}_{model_kind}"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(model_kind, "cpu")
    for c0 in range(0, len(rows), chunk):
        f = out / f"chunk_{c0:05d}.npz"
        if f.exists():
            continue
        acc = {k: [] for k in CPU_POOLS + tuple(f"n_{k}" for k in CPU_POOLS[:4])}
        for b0 in range(c0, min(c0 + chunk, len(rows)), batch):
            r = rows[b0:min(b0 + batch, c0 + chunk, len(rows))]
            frames = [decode(df["video"].iloc[i]) for i in r]
            masks = np.stack([disk_mask(fr)[0] for fr in frames])                   # [B, 8, 16, 16]
            points, _ = encode(model, preprocess(frames))
            per_point = [token_pools(x, masks) for x in points]
            for k in CPU_POOLS:
                acc[k].append(np.stack([pp[0][k] for pp in per_point], 1))
            for k in CPU_POOLS[:4]:
                acc[f"n_{k}"].append(per_point[0][1][k])
            print(f"{set_name}/{model_kind} clips {b0}-{b0 + len(r) - 1} of {len(rows)}", flush=True)
        np.savez(f, rows=rows[c0:c0 + chunk], **{k: np.concatenate(v) for k, v in acc.items()})
    merge_cpu(set_name, model_kind)


def merge_cpu(set_name, model_kind):
    out = CPU_POOL_DIR / f"{set_name}_{model_kind}"
    parts = [np.load(f) for f in sorted(out.glob("chunk_*.npz"))]
    keys = ("rows",) + CPU_POOLS + tuple(f"n_{k}" for k in CPU_POOLS[:4])
    arr = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    df, rows = random_subset(set_name)
    assert np.array_equal(arr["rows"], rows), "CPU pools incomplete"
    for k in keys[1:]:
        np.save(out / f"{k}.npy", arr[k])
    stored_dir = ACT / ("stimuli_hard" if set_name == "hard" else "direction") / model_kind
    stored = np.load(stored_dir / "meanpool.npy", mmap_mode="r")[rows]
    info = {"model": model_kind, "rows": rows.tolist(), "ids": [int(i) for i in df["id"].iloc[rows]],
            "scene_vs_stored_meanpool_max_abs": float(np.abs(arr["scene"] - stored).max()),
            "stored_scale_max_abs": float(np.abs(stored).max()),
            "mean_tokens_per_clip": {k: float(arr[f"n_{k}"].mean()) for k in CPU_POOLS[:4]},
            "ring_rule": "non-disk patches in the 3 x 3 patch neighbourhood of a disk patch, same token time step"}
    if model_kind == "vjepa2":
        ds = "stimuli_hard" if set_name == "hard" else "direction"
        bg = np.stack([load_pools(ds, p)["background"][rows] for p in range(N_POINTS)], 1)
        info["background_vs_derived_from_stored_pools_max_abs"] = float(np.abs(arr["background"] - bg).max())
    (out / "index.json").write_text(json.dumps(info))
    print(set_name, model_kind, "parity:", {k: v for k, v in info.items() if "max_abs" in k})


# ---------------------------------------------------------------- binding index by depth

EMERGENCE = {"paper_middle_third_points": [8, 16], "our_perpatch_onset_points": [8, 9]}
PAPER_TABLE1 = {"file": "refs/physics_paper.txt", "lines": "88-90 (Table 1, row \"Object-centric state slots\")",
                "text": "Direction becomes spatially redundant across patches post-Physics Emergence Zone (App. C.5)"}
UNITS = ("object pool = mean over ALL disk tokens of the clip (per-step masked, count-weighted over the 8 token time "
         "steps). NOT comparable with Part 1's diskpool probes (unweighted nanmean of per-step disk means; e.g. its "
         "0.994) nor with the per-position curves of p1a_perpatch (time-pooled single spatial positions)")
PAPER_QUOTE = {"file": "refs/physics_paper.txt", "lines": "795-798",
               "text": "Future interpretability work will need to investigate why this is the case; we hypothesize it may "
                       "be related to feature object binding, in which velocity information is most \"bound\" to the "
                       "corresponding object. At the end of the network, information is catered to the optimization "
                       "objective of predicting the next frame in latent space, not necessarily to preserving "
                       "object-level information."}


def binding_index(r2_obj, r2_bg):
    """(R2_object - R2_background) / R2_object: 0 = the background pool decodes as well as the disk tokens,
    1 = only the disk tokens decode."""
    return (r2_obj - r2_bg) / r2_obj


def paired_binding(Y, P_obj, P_bg, score_fn, n_boot, seed=0):
    """Binding index on R2 and on the error (1 - err_obj / err_bg, err = circular MAE or MAE), with a paired
    test-clip bootstrap over the two pools' fixed predictions."""
    def both(i):
        so, sb = score_fn(Y[i], P_obj[i]), score_fn(Y[i], P_bg[i])
        return binding_index(so["r2"], sb["r2"]), 1 - so["mae"] / sb["mae"]
    all_ = np.arange(len(Y))
    b_r2, b_err = both(all_)
    rng = np.random.default_rng(seed)
    draws = np.array([both(rng.integers(0, len(Y), len(Y))) for _ in range(n_boot)])
    return {"binding_r2": float(b_r2), "binding_r2_ci": ci(draws[:, 0]),
            "binding_err": float(b_err), "binding_err_ci": ci(draws[:, 1])}


def set_loader(set_name, model):
    """(pool loader point -> dict, table, tr, te, folds, variable, score set) for one clip set and model."""
    if set_name in ("direction", "speed"):
        df = load_table(set_name)
        tr, te, folds = pr.split_rows(set_name, df)
        return (lambda pt: load_pools(set_name, pt)), df, tr, te, folds, set_name
    if set_name == "hard":
        df = load_table("direction", root=PROJECT_ROOT / "artifacts" / "stimuli" / "hard")
        tr, te, folds = pr.split_rows("direction", df, PROJECT_ROOT / "splits" / "split_hard.json")
        if model == "vjepa2":
            return (lambda pt: load_pools("stimuli_hard", pt)), df, tr, te, folds, "direction"
        df, _ = random_subset("hard")
    else:                                               # direction_sub: stratified subsample, positions re-indexed
        df_full = load_table("direction")
        tr_f, te_f, folds_f = pr.split_rows("direction", df_full)
        _, rows = random_subset(set_name)
        pos = {r: i for i, r in enumerate(rows)}
        fold_of = dict(zip(tr_f, folds_f))
        tr = np.array([pos[r] for r in tr_f if r in pos])
        te = np.array([pos[r] for r in te_f if r in pos])
        folds = np.array([fold_of[r] for r in tr_f if r in pos])
        df = df_full.iloc[rows].reset_index(drop=True)
        if model == "vjepa2":
            def load(pt):
                P = load_pools("direction", pt)
                return {k: (v[rows] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
            return load, df, tr, te, folds, "direction"
    d = CPU_POOL_DIR / f"{set_name}_{model.replace('_cpu', '')}"
    arr = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in CPU_POOLS}
    return (lambda pt: {k: np.asarray(arr[k][:, pt], np.float64) for k in CPU_POOLS}), df, tr, te, folds, "direction"


LOWDATA_N, LOWDATA_RESAMPLES = 64, 20
CEILING = 0.98       # both pools above this R2: the full-data binding index is not interpretable (flagged)


def ridge_small(X, Y, folds, score_fn, alphas=pr.ALPHAS):
    """wm.probes recipe (train standardisation, alpha by fold-mean R2, refit) in the dual/thin-SVD form for n << D."""
    st = pr.Standardizer().fit(X)
    Z = st.transform(X)

    def fit(Zt, Yt):
        mx, my = Zt.mean(0), Yt.mean(0)
        U, S, Vt = np.linalg.svd(Zt - mx, full_matrices=False)
        R = U.T @ (Yt - my)
        return [(Vt.T @ (R * (S / (S ** 2 + a))[:, None]), mx, my) for a in alphas]

    per_fold = {k: fit(Z[folds != k], Y[folds != k]) for k in np.unique(folds)}
    scores = [np.mean([score_fn(Y[folds == k], (Z[folds == k] - mx) @ W + my)["r2"]
                       for k, fits in per_fold.items() for (W, mx, my) in [fits[i]]]) for i in range(len(alphas))]
    W, mx, my = fit(Z, Y)[int(np.argmax(scores))]
    return lambda Xn: (st.transform(Xn) - mx) @ W + my


def _binding_point(args):
    set_name, model, point, n_boot = args
    load, df, tr, te, folds, variable = set_loader(set_name, model)
    Y, kind, score_fn = pr.targets(df, variable)
    pools = {k: v for k, v in load(point).items() if k in CPU_POOLS}
    names = list(pools)
    probes = {k: fit_probe(pools[k][tr], Y[tr], folds, score_fn) for k in names}
    preds = {k: apply_probe(probes[k], pools[k][te]) for k in names}
    row = {"point": point, **{f"r2_{k}": score_fn(Y[te], preds[k])["r2"] for k in names},
           **{f"err_{k}": score_fn(Y[te], preds[k])["mae"] for k in names},
           **paired_binding(Y[te], preds["object"], preds["background"], score_fn, n_boot)}
    for k in ("ring", "far"):
        if k in pools:
            b = paired_binding(Y[te], preds["object"], preds[k], score_fn, n_boot)
            row.update({f"{q}_{k}": v for q, v in b.items()})
    want_ci = set_name not in ("direction", "speed")        # those two have transfer CIs in res["direction"/"speed"]
    M, _ = transfer_matrix(pools, tr, te, Y[tr], Y[te], folds, score_fn, n_boot if want_ci else 0, probes)
    row["transfer_r2"] = {a: {b: M[a][b]["r2"] for b in M[a]} for a in M}
    row["transfer_err"] = {a: {b: M[a][b]["mae"] for b in M[a]} for a in M}
    if want_ci:
        row["transfer_r2_ci"] = {a: {b: M[a][b]["ci"]["r2"] for b in M[a]} for a in M}
    # low-data regime: 64 train clips (5 folds among them), 20 resamples, full test set
    rng = np.random.default_rng(point)
    low = {k: {"r2": [], "err": []} for k in names}
    bi_r2, bi_err = [], []
    for _ in range(LOWDATA_RESAMPLES):
        sub = rng.choice(len(tr), LOWDATA_N, replace=False)
        f_sub = np.arange(LOWDATA_N) % 5
        sc = {}
        for k in names:
            P = ridge_small(pools[k][tr][sub], Y[tr][sub], f_sub, score_fn)(pools[k][te])
            sc[k] = score_fn(Y[te], P)
            low[k]["r2"].append(sc[k]["r2"])
            low[k]["err"].append(sc[k]["mae"])
        bi_r2.append(binding_index(sc["object"]["r2"], sc["background"]["r2"]))
        bi_err.append(1 - sc["object"]["mae"] / sc["background"]["mae"])
    row["lowdata"] = {"n_train": LOWDATA_N, "resamples": LOWDATA_RESAMPLES,
                      **{f"r2_{k}": float(np.mean(v["r2"])) for k, v in low.items()},
                      **{f"err_{k}": float(np.mean(v["err"])) for k, v in low.items()},
                      "binding_r2": float(np.mean(bi_r2)), "binding_r2_range_2p5_97p5": ci(bi_r2),
                      "binding_err": float(np.mean(bi_err)), "binding_err_range_2p5_97p5": ci(bi_err)}
    return row


def binding_curve(set_name, model, n_boot, workers=4):
    """Per point: probe R2 / error per pool, binding index of background (and ring / far where token-level pools exist)
    against the object pool with paired CIs, the cross-pool transfer matrix, and a 64-clip low-data regime."""
    _, df, tr, te, folds, variable = set_loader(set_name, model)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_binding_point, [(set_name, model, p, n_boot) for p in range(N_POINTS)]))
    for row in rows:
        print(f"binding {set_name}/{model} point {row['point']:2d}: R2 obj {row['r2_object']:.3f} bg "
              f"{row['r2_background']:.3f} BI {row['binding_r2']:.3f} [{row['binding_r2_ci'][0]:.3f}, "
              f"{row['binding_r2_ci'][1]:.3f}] BI_err {row['binding_err']:.3f} low-data BI {row['lowdata']['binding_r2']:.3f}",
              flush=True)
    return {"set": set_name, "model": model, "variable": variable, "n_train": int(len(tr)), "n_test": int(len(te)),
            "pools": [k for k in CPU_POOLS if f"r2_{k}" in rows[0]], "rows": rows}


def zone_summary(curve):
    """Mean binding index over the early (1-7), emergence (8-16) and late (17-24) points, and its peak point."""
    r = curve["rows"]
    band = lambda a, b, q: float(np.mean([x[q] for x in r if a <= x["point"] <= b]))   # noqa: E731
    out = {}
    for q in ("binding_r2", "binding_err", "binding_r2_ring", "binding_r2_far"):
        if q in r[0]:
            out[q] = {"point0_floor": r[0][q], "early_1_7": band(1, 7, q), "emergence_8_16": band(8, 16, q),
                      "late_17_24": band(17, 24, q),
                      "peak_point_1_24": int(max((x for x in r if 1 <= x["point"] <= 24), key=lambda x: x[q])["point"])}
    out["scene_r2"] = {"emergence_8_16": band(8, 16, "r2_scene"), "late_17_24": band(17, 24, "r2_scene")}
    lo = [x["lowdata"] for x in r]
    lband = lambda a, b, q: float(np.mean([x[q] for x, y in zip(lo, r) if a <= y["point"] <= b]))   # noqa: E731
    out["lowdata_binding_r2"] = {"point0_floor": lo[0]["binding_r2"], "early_1_7": lband(1, 7, "binding_r2"),
                                 "emergence_8_16": lband(8, 16, "binding_r2"), "late_17_24": lband(17, 24, "binding_r2")}
    out["lowdata_binding_err"] = {"point0_floor": lo[0]["binding_err"], "early_1_7": lband(1, 7, "binding_err"),
                                  "emergence_8_16": lband(8, 16, "binding_err"), "late_17_24": lband(17, 24, "binding_err")}
    out["points_at_ceiling"] = [x["point"] for x in r if min(x["r2_object"], x["r2_background"]) > CEILING]
    tb = lambda a, b: float(np.mean([x["transfer_r2"]["object"]["background"] for x in r if a <= x["point"] <= b]))  # noqa: E731
    out["transfer_object_to_background_r2"] = {"early_1_7": tb(1, 7), "emergence_8_16": tb(8, 16), "late_17_24": tb(17, 24)}
    return out


CPU_SETS = (("hard", "random"), ("hard", "vjepa2"), ("direction_sub", "vjepa2"), ("direction_sub", "random"))


def run_binding(a):
    path = RESULTS / "p5_object_vs_scene_direction.json"
    res = {} if a.cache_only else json.loads(path.read_text())
    B = {"paper_hypothesis": PAPER_QUOTE, "paper_table1": PAPER_TABLE1, "emergence_zone": EMERGENCE,
         "units": UNITS,
         "definition": "binding index = (R2_object - R2_other) / R2_object per point, other = background (all non-disk "
                       "tokens), ring or far; binding_err = 1 - err_object / err_other (circular MAE for direction, MAE "
                       "for speed); CIs: paired 200-draw test-clip bootstrap over fixed predictions. Point 0 is the "
                       "floor: the 2-frame tubelet lets disk tokens carry local displacement already at the embedding",
         "curves": {}}
    sets = [("direction", "vjepa2"), ("speed", "vjepa2"), ("hard", "vjepa2")]
    sets += [(sn, f"{m}_cpu") for sn, m in CPU_SETS if (CPU_POOL_DIR / f"{sn}_{m}" / "index.json").exists()]
    cache = Path(os.environ.get("P5_CACHE", "/tmp/p5_binding_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    only = set(a.sets or [])
    for set_name, model in sets:
        cf = cache / f"{set_name}_{model}_b{a.n_boot}.json"
        if cf.exists():
            c = json.loads(cf.read_text())
        elif only and set_name not in only:
            continue
        else:
            c = binding_curve(set_name, model, a.n_boot, a.workers)
            cf.write_text(json.dumps(c, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        if a.cache_only:
            continue
        c["zones"] = zone_summary(c)
        if model.endswith("_cpu"):
            info = json.loads((CPU_POOL_DIR / f"{set_name}_{model[:-4]}" / "index.json").read_text())
            info.pop("rows")
            info["ids"] = f"{len(info['ids'])} clips (artifacts/activations/object_scene_cpu/{set_name}_{model[:-4]})"
            c["extraction"] = info
        B["curves"][f"{set_name}/{model}"] = c
    if a.cache_only:
        return
    res["binding"] = B
    done = {k for k in B["curves"]}
    missing = [f"{sn}/{m}_cpu" for sn, m in CPU_SETS if f"{sn}/{m}_cpu" not in done]
    res["not_computable"] = [
        "GPU re-extraction of the random-init model with the disk mask on the full sets: not run (instruction: CPU "
        "only). Instead both models were re-encoded on CPU (fp32, same wm.extract code path, parity vs stored meanpool "
        "recorded) for the hard set (392 clips) and a stratified direction subsample (4 train + 2 test clips per "
        "value = 384), with token-level object / ring / far pools" + (f"; NOT finished this run: {missing}" if missing else ""),
        "ring / far split on the full direction and speed sets: needs per-token arrays, which are not stored (the "
        "p1a_perpatch memmaps were time-pooled [N, P, 256, 1024] and stayed on the GPU box); only the CPU re-encoded "
        "sets have it",
        "attribution: whether direction in non-disk tokens is written by attention from the disk tokens or reflects "
        "position-dependent tokens averaged over a disk-dependent complement set cannot be separated from pooled "
        "tokens; the ring/far split and the time-reversal check are partial controls"]
    res["summary_binding"] = [f"{k}: " + json.dumps(v["zones"]) for k, v in B["curves"].items()]
    path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n".join(res["summary_binding"]))
    plot(res, FIGURES / "fig_object_vs_scene.png")


# ---------------------------------------------------------------- run

def split_setup(dataset, variable):
    df = load_table(dataset)
    Y, kind, score_fn = pr.targets(df, variable)
    tr, te, folds = pr.split_rows(dataset, df)
    return df, Y, kind, score_fn, tr, te, folds


def probe_block(pools, tr, te, folds, Ytr, Yte, score_fn, n_boot):
    out, probes, preds = {}, {}, {}
    for name, X in pools.items():
        p = fit_probe(X[tr], Ytr, folds, score_fn)
        P = apply_probe(p, X[te])
        s = score_fn(Yte, P)
        out[name] = {**s, "alpha": p["alpha"], "cv_r2": p["cv_mean"], "ci": boot_scores(Yte, P, score_fn, n_boot)}
        probes[name], preds[name] = p, P
    return out, probes


def run(a):
    n_boot = a.n_boot
    res = {"question": "is the direction ring object-bound (disk tokens) or scene-wide (background tokens)?",
           "config": {"n_boot": n_boot, "geom_points": list(GEOM_POINTS), "arc_seeds": list(ARC_SEEDS), "k": 64,
                      "pools": {"scene": "stored meanpool (all 2048 tokens)",
                                "object": "disk-patch tokens, count-weighted over the 8 steps (diskmask any-pixel rule)",
                                "background": "non-disk tokens, (256*sum_t timepool - sum_t p_t*diskpool)/(2048 - sum p_t)",
                                "random_scene": "random-init ViT-L meanpool (no diskpool stored -> no object/background)"},
                      "ci": "2.5-97.5 percentile; probe/transfer/timerev: test-clip bootstrap over fixed predictions; "
                            "geometry: train-clip bootstrap with refit; heldout: knot-clip bootstrap with refit. Point "
                            "estimates use mf.fit_pca; bootstrap draws use the Gram-matrix PCA (same subspace, component "
                            "signs arbitrary; the label-free smoothing spline is slightly sign-sensitive, so bootstrap "
                            "spreads include that ambiguity)",
                      "splits": "split_v1: probe train folds 0-4 -> test; geometry = train rows; arcs = knot folds 0-2"},
           "not_computable": [
               "near-disk ring (1 patch around) vs far background: no per-patch token arrays exist on disk (the "
               "p1a_perpatch memmaps lived on the GPU box only; artifacts/ holds per-step pools); the background pool "
               "therefore includes the near-disk patches",
               "object/background pools for the random-init encoder: its extraction stored meanpool/timepool only "
               "(no diskpool), so the random control covers the scene pool only",
               "per-patch attention attribution: whether background direction info is written by attention from disk "
               "tokens or reflects position-dependent tokens averaged over a disk-dependent complement set cannot be "
               "separated from pooled tokens; the time-reversal check is the partial control"]}
    # ---- sha256
    files = [f"{ds}/{m}/{f}.npy" for ds in ("direction", "speed") for m, fs in
             (("vjepa2", ("meanpool", "timepool", "diskpool", "diskmask")), ("random", ("meanpool",))) for f in fs]
    files += [f"direction/vjepa2_timerev/{f}.npy" for f in ("meanpool", "timepool", "diskpool", "diskmask")]
    box = dict(line.split()[::-1] for line in (ACT / "sha256_box.txt").read_text().splitlines() if line.strip())
    sha = {}
    for f in files:
        h = sha256_file(ACT / f) if not a.skip_sha else None
        sha[f] = {"sha256": h, "box_sha256": box.get(f), "match_box": None if f not in box or h is None else h == box[f]}
    res["provenance"] = {**provenance(seeds={"bootstrap": 0}, pool="scene/object/background", k=64),
                         "activation_sha256": sha,
                         "gpu_session1_verification": json.loads((PROJECT_ROOT / "artifacts" / "gpu_session1.json")
                                                                 .read_text())["verification"]["sha256"]}

    # ---- direction
    df, Y, kind, score_fn, tr, te, folds = split_setup("direction", "direction")
    Ytr, Yte = Y[tr], Y[te]
    theta = df["theta_degrees"].to_numpy(float)
    fold_all = np.full(len(df), -1)
    fold_all[tr] = folds
    knot = np.isin(fold_all, KNOT_FOLDS)
    Y_rev = np.stack([np.sin(np.radians(theta + 180)), np.cos(np.radians(theta + 180))], 1)
    D = {"probe": {p: [] for p in POOLS + ("random_scene",)}, "transfer": {}, "timerev": {}, "chart_plane_angles": {},
         "identity": {}, "geometry": {}, "heldout": {}}
    jobs = {}
    for point in range(N_POINTS):
        pools = load_pools("direction", point)
        D["identity"][str(point)] = {"scene_vs_recon_max_rel_err": pools.pop("_identity_rel_err"),
                                "mean_disk_tokens_per_clip": pools.pop("_n_obj_mean")}
        rnd = load_pools("direction", point, "random")["scene"]
        pb, probes = probe_block({**pools, "random_scene": rnd}, tr, te, folds, Ytr, Yte, score_fn, n_boot)
        for name, s in pb.items():
            D["probe"][name].append({"point": point, **s})
        M, _ = transfer_matrix(pools, tr, te, Ytr, Yte, folds, score_fn, n_boot, {q: probes[q] for q in POOLS})
        D["transfer"][str(point)] = M
        rev = load_pools("direction", point, "vjepa2_timerev")
        D["timerev"][str(point)] = {}
        for name in POOLS:
            P = apply_probe(probes[name], rev[name][te], rev[name][tr])
            ang = np.degrees(np.arctan2(P[:, 0], P[:, 1]))
            e_fwd = np.abs((ang - theta[te] + 180) % 360 - 180)
            e_rev = np.abs((ang - theta[te]) % 360 - 180)
            flip = (e_rev < e_fwd).astype(float)
            rng = np.random.default_rng(0)
            bd = [flip[rng.integers(0, len(flip), len(flip))].mean() for _ in range(n_boot)]
            D["timerev"][str(point)][name] = {"frac_decoded_closer_to_theta_plus_180": float(flip.mean()), "ci": ci(bd),
                                        "mae_vs_theta_plus_180": float(e_rev.mean()),
                                        "mae_vs_theta": float(e_fwd.mean()),
                                        "r2_vs_reversed_target": score_fn(Y_rev[te], P)["r2"]}
        if point in GEOM_POINTS:
            D["chart_plane_angles"][str(point)] = {f"{p}_vs_{q}": principal_angles(pools[p][tr], pools[q][tr], theta[tr])
                                              for i, p in enumerate(POOLS) for q in POOLS[i + 1:]}
            for name, X in {**pools, "random_scene": rnd}.items():
                jobs[(point, name)] = (X[tr], theta[tr], X[knot], theta[knot])
        print(f"direction point {point:2d}: " + "  ".join(f"{n} R2={pb[n]['r2']:.3f}" for n in pb), flush=True)

    outs = geometry_all(jobs, n_boot, a.workers)
    for (point, name), (g, h) in outs.items():
        D["geometry"].setdefault(str(point), {})[name] = g
        D["heldout"].setdefault(str(point), {})[name] = h
    res["direction"] = D

    # ---- speed
    df, Y, kind, score_fn, tr, te, folds = split_setup("speed", "speed")
    S = {"probe": {p: [] for p in POOLS + ("random_scene",)}, "transfer": {}, "identity": {}}
    for point in range(N_POINTS):
        pools = load_pools("speed", point)
        S["identity"][str(point)] = {"scene_vs_recon_max_rel_err": pools.pop("_identity_rel_err"),
                                "mean_disk_tokens_per_clip": pools.pop("_n_obj_mean")}
        rnd = load_pools("speed", point, "random")["scene"]
        pb, probes = probe_block({**pools, "random_scene": rnd}, tr, te, folds, Y[tr], Y[te], score_fn, n_boot)
        for name, s in pb.items():
            S["probe"][name].append({"point": point, **s})
        S["transfer"][str(point)], _ = transfer_matrix(pools, tr, te, Y[tr], Y[te], folds, score_fn, n_boot,
                                                  {q: probes[q] for q in POOLS})
        print(f"speed point {point:2d}: " + "  ".join(f"{n} R2={pb[n]['r2']:.3f}" for n in pb), flush=True)
    res["speed"] = S

    # ---- parity with stored results
    par = {}
    p1 = json.loads((RESULTS / "p1a_direction_direction_meanpool.json").read_text())["layers"]
    par["probe_scene_vs_p1a_meanpool_test_r2_max_abs_diff"] = float(max(
        abs(r["test_r2"] - s["r2"]) for r, s in zip(p1, D["probe"]["scene"])))
    ell = json.loads((RESULTS / "p2_ellipse_direction.json").read_text())["per_point"]
    par["geometry_scene_vs_p2_ellipse"] = {
        str(p): {"q_conic": [D["geometry"][str(p)]["scene"]["b_over_a_conic_ring_plane"], ell[str(p)]["summary"]["q_conic"]],
                 "var_share_k2": [D["geometry"][str(p)]["scene"]["saddle_share_k2"], ell[str(p)]["summary"]["var_share_k2"]]}
        for p in GEOM_POINTS}
    res["parity"] = par
    res["summary"] = summary(res)
    out = RESULTS / "p5_object_vs_scene_direction.json"
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\n".join(res["summary"]))
    plot(res, FIGURES / "fig_object_vs_scene.png")
    return res


def summary(res):
    D, S = res["direction"], res["speed"]
    lines = []
    for p in GEOM_POINTS:
        pr_ = {n: D["probe"][n][p] for n in D["probe"]}
        lines.append(f"point {p} direction test R2: " + ", ".join(
            f"{n} {v['r2']:.3f} [{v['ci']['r2'][0]:.3f}, {v['ci']['r2'][1]:.3f}] MAE {v['mae']:.1f} deg"
            for n, v in pr_.items()))
        g = D["geometry"][str(p)]
        lines.append(f"point {p} b/a: " + ", ".join(f"{n} {g[n]['b_over_a_conic_ring_plane']:.2f}" for n in g) +
                     "; saddle share: " + ", ".join(f"{n} {g[n]['saddle_share_k2']:.2f}" for n in g) +
                     "; circ corr (centroid plane): " + ", ".join(f"{n} {g[n]['circ_corr_centroid_plane']:.3f}" for n in g))
        h = D["heldout"][str(p)]
        lines.append(f"point {p} held-out spline/chord: " + ", ".join(f"{n} {h[n]['ratio_spline_over_chord']:.3f}"
                                                                      for n in h))
        M = D["transfer"][str(p)]
        lines.append(f"point {p} transfer R2 (row=fit, col=test): " + "; ".join(
            f"{a}->{b} {M[a][b]['r2']:.3f}" for a in M for b in M[a]))
        lines.append(f"point {p} time-reversal flip frac: " + ", ".join(
            f"{n} {v['frac_decoded_closer_to_theta_plus_180']:.3f}" for n, v in D["timerev"][str(p)].items()))
        lines.append(f"point {p} speed test R2: " + ", ".join(f"{n} {S['probe'][n][p]['r2']:.3f}" for n in S["probe"]))
    return lines


# ---------------------------------------------------------------- figure

COLORS = {"scene": "#2a78d6", "object": "#eb6834", "background": "#1baf7a", "random_scene": "#8a8a85"}
LABELS = {"scene": "scene (all tokens)", "object": "object (disk tokens)", "background": "background (non-disk)",
          "random_scene": "random-init, scene"}


SET_COLORS = {"direction": "#2a78d6", "hard": "#eb6834", "speed": "#1baf7a", "direction_sub": "#2a78d6"}
SET_LABELS = {"direction/vjepa2": "direction set, V-JEPA 2", "speed/vjepa2": "speed set, V-JEPA 2",
              "hard/vjepa2": "hard set, V-JEPA 2", "hard/random_cpu": "hard set, random-init",
              "direction_sub/random_cpu": "direction subsample, random-init",
              "direction_sub/vjepa2_cpu": "direction subsample, V-JEPA 2", "hard/vjepa2_cpu": "hard set, V-JEPA 2 (CPU)"}


def _zone(ax):
    ax.axvspan(8, 16, color="0.93", zorder=0, lw=0)
    ax.axvspan(8, 9, color="0.85", zorder=0, lw=0)
    ax.spines[["top", "right"]].set_visible(False)


def _band(ax, rows, q, color, ls, label):
    x = [r["point"] for r in rows]
    ax.plot(x, [r[q] for r in rows], color=color, lw=2, ls=ls, label=label)
    if f"{q}_ci" in rows[0]:
        ax.fill_between(x, [r[f"{q}_ci"][0] for r in rows], [r[f"{q}_ci"][1] for r in rows], color=color, alpha=0.15,
                        lw=0)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = res["direction"]
    curves = res.get("binding", {}).get("curves", {})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    xlab = "stored point (0 = embedding, 24 = block 24, 25 = final LN)"

    ax = axes[0, 0]
    for key in ("direction/vjepa2", "hard/vjepa2", "speed/vjepa2", "hard/random_cpu", "direction_sub/random_cpu"):
        if key in curves:
            st, model = key.split("/")
            _band(ax, curves[key]["rows"], "binding_r2", SET_COLORS[st], "--" if "random" in model else "-",
                  SET_LABELS[key])
    _zone(ax)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("binding index (R²_object − R²_background) / R²_object")
    ax.set_title("A  Object-boundness by depth (grey: paper's middle third; dark: our per-patch onset 8-9)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    for key in ("direction/vjepa2", "hard/vjepa2", "speed/vjepa2"):
        if key in curves:
            rows = curves[key]["rows"]
            color = SET_COLORS[key.split("/")[0]]
            _band(ax, rows, "binding_err", color, "-", f"{SET_LABELS[key]}: 1 − err_obj/err_bg (full train)")
            x = [r["point"] for r in rows]
            ax.plot(x, [r["lowdata"]["binding_r2"] for r in rows], color=color, lw=1.5, ls=":",
                    label=f"{SET_LABELS[key]}: R² index, 64 train clips")
    _zone(ax)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel(xlab)
    ax.set_ylabel("binding index (off-ceiling readouts)")
    ax.set_title("B  Error-based index (solid) and low-data R² index (dotted)", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[0, 2]
    for n, rows in D["probe"].items():
        x = [r["point"] for r in rows]
        ax.plot(x, [r["r2"] for r in rows], color=COLORS[n], lw=2, ls="--" if n == "random_scene" else "-",
                label=LABELS[n])
        ax.fill_between(x, [r["ci"]["r2"][0] for r in rows], [r["ci"]["r2"][1] for r in rows], color=COLORS[n],
                        alpha=0.18, lw=0)
    _zone(ax)
    ax.set_xlabel(xlab)
    ax.set_ylabel("held-out test R² on (sin θ, cos θ)")
    ax.set_title("C  Direction probe by token pool (direction set)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1, 0]
    names = list(D["geometry"][str(GEOM_POINTS[0])])
    w = 0.8 / len(names)
    for mi, metric in enumerate(("b_over_a_conic_ring_plane", "saddle_share_k2")):
        for pi, p in enumerate(GEOM_POINTS):
            x0 = mi * (len(GEOM_POINTS) + 0.8) + pi
            for ni, n in enumerate(names):
                g = D["geometry"][str(p)][n]
                v, lo_hi = g[metric], g["ci"][metric]
                ax.errorbar(x0 + (ni - (len(names) - 1) / 2) * w, v, yerr=[[v - lo_hi[0]], [lo_hi[1] - v]], fmt="o",
                            ms=6, color=COLORS[n], capsize=0, lw=1.5, label=LABELS[n] if (mi, pi) == (0, 0) else None)
    ticks = [mi * (len(GEOM_POINTS) + 0.8) + pi for mi in range(2) for pi in range(len(GEOM_POINTS))]
    ax.set_xticks(ticks, [f"{t}\npt {p}" for t in ("b/a", "saddle") for p in GEOM_POINTS], fontsize=8)
    ax.axvline(len(GEOM_POINTS) - 0.1, color="0.8", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("value (95% bootstrap CI)")
    ax.set_title("D  Ring shape: axis ratio b/a | cos 2θ saddle variance share", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    M = D["transfer"]["12"]
    pools = list(M)
    A = np.array([[M[a][b]["r2"] for b in pools] for a in pools])
    im = ax.imshow(A, cmap="Blues", vmin=min(0, A.min()), vmax=1)
    for i in range(len(pools)):
        for j in range(len(pools)):
            lo, hi = M[pools[i]][pools[j]]["ci"]["r2"]
            ax.text(j, i, f"{A[i, j]:.3f}\n[{lo:.3f}, {hi:.3f}]", ha="center", va="center", fontsize=8,
                    color="white" if A[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(pools)), pools)
    ax.set_yticks(range(len(pools)), pools)
    ax.set_xlabel("tested on pool (own train standardisation)")
    ax.set_ylabel("probe fit on pool")
    ax.set_title("E  Cross-pool transfer, point 12, direction set (test R²)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 2]
    for n in POOLS:
        x = list(range(N_POINTS))
        v = [D["timerev"][str(p)][n]["frac_decoded_closer_to_theta_plus_180"] for p in x]
        lo = [D["timerev"][str(p)][n]["ci"][0] for p in x]
        hi = [D["timerev"][str(p)][n]["ci"][1] for p in x]
        ax.plot(x, v, color=COLORS[n], lw=2, label=LABELS[n])
        ax.fill_between(x, lo, hi, color=COLORS[n], alpha=0.18, lw=0)
    _zone(ax)
    ax.axhline(0.5, color="0.5", lw=0.8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlab)
    ax.set_ylabel("fraction of reversed clips decoded nearer θ + 180°")
    ax.set_title("F  Time-reversal: forward probe on reversed clips", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--skip-sha", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--extract-cpu", nargs=2, metavar=("SET", "MODEL"), default=None,
                   help="CPU re-encode SET (hard | direction_sub) with MODEL (random | vjepa2): token-level pools")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--binding", action="store_true", help="add the binding-index-by-depth curves to the results JSON")
    p.add_argument("--sets", nargs="*", default=None, help="binding: only compute these sets (others from cache)")
    p.add_argument("--cache-only", action="store_true", help="binding: compute + cache curves, do not touch the JSON")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    if args.binding:
        run_binding(args)
    elif args.extract_cpu:
        extract_cpu(args.extract_cpu[0], args.extract_cpu[1], args.threads)
    elif args.plot_only:
        plot(json.loads((RESULTS / "p5_object_vs_scene_direction.json").read_text()), FIGURES / "fig_object_vs_scene.png")
    else:
        run(args)

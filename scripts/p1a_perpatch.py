"""Per-patch direction probes across layers (the physics paper's App. C.5 / Fig. 18 diagnostic).

The paper's emergence zone for direction is a per-patch claim: mean-pooled probes do moderately well early, per-patch
probes become reliable only at the zone (its layers 7 -> 8 = our points 8 -> 9), and a probe fit on one half of the frame
transfers to the other half only from there on. Part 1 used mean-pooled probes only; this tests the per-patch reading.

  extract  box, GPU. wm.extract.encode (16 frames, 256 px, no crop, fp32, TF32 off), time-pool over the 8 token time
           steps -> fp16 memmap [N, P, 256, 1024] (P = chosen points, 256 = 16 x 16 spatial positions, index h*16 + w)
           plus a JSON sidecar (ids, points, frame hashes, timing, meanpool parity against the stored meanpool.npy).
  probe    box, CPU. Closed-form ridge exactly as wm.probes (train-standardised features, targets (sin, cos), alpha by
           the split's 5 train folds over ALPHAS, refit on all train, score test once):
             meanpool   one probe on the position mean (= our Part 1 meanpool), for parity and a same-rule baseline
             perpos     one probe per spatial position (own standardiser and own CV alpha per position)
             pooled     one probe on every (train clip, position) sample; folds by clip; scored per test position
             halves     pooled probe fit on columns 0-7 (left) or 8-15 (right) of train clips, scored on both halves of
                        test clips: within-half vs cross-half
           Onsets: first point with metric >= 90% of its max over the sampled points; 95% CI from a 200-draw bootstrap of
           TEST clips over fixed test predictions (no refit).
  figures  Mac. fig1g (curves) and fig1h (heatmaps) from the results JSONs.

  python scripts/p1a_perpatch.py extract --set direction --model vjepa2 --out /workspace/wm/artifacts/perpatch/direction_vjepa2.f16
  python scripts/p1a_perpatch.py probe   --set direction --model vjepa2 --memmap ... --commit <sha>
  python scripts/p1a_perpatch.py figures
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402

POINTS_FULL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 24]
POINTS_STIM = [1, 4, 6, 7, 8, 9, 10, 12, 16, 22]
HEATMAP_POINTS = [4, 7, 8, 9, 12, 22]
GRID, NPOS, WIDTH, NSTEP = 16, 256, 1024, 8
SETS = {"direction": (None, "splits/split_v1.json"),
        "paper_layout": ("artifacts/stimuli/paper_layout", "splits/split_paper_layout.json"),
        "hard": ("artifacts/stimuli/hard", "splits/split_hard.json")}
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
N_BOOT = 200


def table(name):
    root = SETS[name][0]
    return load_table("direction", root=None if root is None else PROJECT_ROOT / root)


def result_path(name, model, subset=None):
    suffix = ("" if name == "direction" else f"_{name}") + (f"_{subset}" if subset else "")
    return RESULTS / f"p1a_perpatch_direction_{model}{suffix}.json"


# ================================================================ extract (GPU)

def extract(a):
    import torch
    from wm.data import decode, frame_hash
    from wm.extract import encode, load_model, pick_device, preprocess, set_precision

    points = a.points or (POINTS_FULL if a.set == "direction" else POINTS_STIM)
    df = table(a.set)
    ids = [int(i) for i in df["id"]]
    videos = list(df["video"])
    N, P = len(ids), len(points)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    side = out.with_suffix(".json")
    mm = np.lib.format.open_memmap(out, mode="r+" if out.exists() else "w+", dtype=np.float16,
                                   shape=(N, P, NPOS, WIDTH))
    done_path = out.with_suffix(".done.npy")
    done = np.load(done_path) if done_path.exists() else np.zeros(N, bool)
    set_precision()
    device = pick_device()
    model = load_model(a.model, device)
    hashes = {}
    t0 = time.time()
    gpu_s = 0.0
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)
    starts = [s for s in range(0, N, a.batch_size) if not done[s:s + a.batch_size].all()]
    fut = {s: [pool.submit(decode, videos[i]) for i in range(s, min(s + a.batch_size, N))] for s in starts[:2]}
    for j, s in enumerate(starts):
        frames = [f.result() for f in fut.pop(s)]
        if j + 2 < len(starts):
            s2 = starts[j + 2]
            fut[s2] = [pool.submit(decode, videos[i]) for i in range(s2, min(s2 + a.batch_size, N))]
        for k, f in enumerate(frames):
            hashes[ids[s + k]] = frame_hash(f)
        tg = time.time()
        pts, _ = encode(model, preprocess(frames).to(device))
        B = len(frames)
        tp = torch.stack([pts[p].reshape(B, NSTEP, NPOS, WIDTH).mean(dim=1) for p in points], dim=1)  # [B,P,256,1024]
        tp = tp.half().cpu().numpy()
        gpu_s += time.time() - tg
        mm[s:s + B] = tp
        done[s:s + B] = True
        if j % 10 == 0 or j == len(starts) - 1:
            mm.flush()
            np.save(done_path, done)
            print(f"{a.set}/{a.model}: {int(done.sum())}/{N} clips, {time.time() - t0:.0f}s", flush=True)
        del pts
    mm.flush()
    np.save(done_path, done)
    info = json.loads(side.read_text()) if side.exists() else {"frame_hash": {}}
    info["frame_hash"].update({str(k): v for k, v in hashes.items()})
    # parity: position mean of the stored time-pool vs Part 1's stored meanpool (fp32), where available
    ref_dir = {"direction": "direction", "paper_layout": "stimuli_paper_layout", "hard": "stimuli_hard"}[a.set]
    ref = PROJECT_ROOT / "artifacts" / "activations" / ref_dir / a.model / "meanpool.npy"
    parity = None
    if ref.exists():
        refm = np.load(ref, mmap_mode="r")
        rows = np.arange(0, N, max(1, N // 50))
        mine = np.asarray(mm[rows], np.float32).mean(axis=2)                     # [r, P, 1024]
        theirs = np.asarray(refm[rows][:, points], np.float32)
        rel = np.linalg.norm(mine - theirs, axis=-1) / np.linalg.norm(theirs, axis=-1)
        parity = {"reference": str(ref.relative_to(PROJECT_ROOT)), "n_clips_checked": len(rows),
                  "max_rel_err_per_point": rel.max(axis=0).round(6).tolist()}
        print("parity max rel err per point:", parity["max_rel_err_per_point"])
    info.update({"set": a.set, "model": a.model, "ids": ids, "points": points, "shape": [N, P, NPOS, WIDTH],
                 "dtype": "float16", "memmap": str(out),
                 "pooling": "mean over the 8 token time steps; positions kept (16x16, index h*16 + w, w = column)",
                 "preprocess": "wm.extract: 16 frames, 256 px, ImageNet mean/std, no resize/crop, fp32 forward, TF32 off",
                 "random_init": "VJEPA2Config + torch.manual_seed(0) (wm.extract.load_model)" if a.model == "random" else None,
                 "gpu_seconds_forward": gpu_s + info.get("gpu_seconds_forward", 0.0),
                 "wall_seconds_last_run": time.time() - t0, "batch_size": a.batch_size,
                 "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                 "meanpool_parity": parity, "complete": bool(done.all()),
                 "location_note": ("RAM-backed /dev/shm: volatile, lost on reboot (box disk too small for all caches)"
                                   if str(out).startswith("/dev/shm") else "persistent box disk")})
    side.write_text(json.dumps(info, indent=1))
    print(f"done: {side}")


# ================================================================ probe (CPU)

_G = {}


def _init_worker():
    from threadpoolctl import threadpool_limits
    _G["tl"] = threadpool_limits(1)


def _perpos(task):
    """One (point, spatial position): wm.probes recipe (standardise on train, CV alpha on the 5 folds, refit, score
    test). Workers read the memmap themselves (page cache), so the pool is forked before the parent touches BLAS."""
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, ALPHAS
    pi, pos = task
    if "mm" not in _G:
        _G["mm"] = np.load(_G["memmap"], mmap_mode="r")
    tr, te, Ytr, Yte, folds, score_fn = (_G[k] for k in ("tr", "te", "Ytr", "Yte", "folds", "score_fn"))
    Xp = np.asarray(_G["mm"][:, pi, pos], np.float64)
    st = Standardizer().fit(Xp[tr])
    Xtr, Xte = st.transform(Xp[tr]), st.transform(Xp[te])
    cv = cv_select_alpha(Xtr, Ytr, folds, ALPHAS, score_fn)
    W, b = fit_ridge(Xtr, Ytr, cv["alpha"])
    P = predict(Xte, W, b)
    s = score_fn(Yte, P)
    return pi, pos, cv["alpha"], cv["cv_mean"], s["r2"], s["mae"], P.astype(np.float32)


def _dev():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gram_stats(Z, Y, chunk=16384):
    """Z [m, d] float32 torch (standardised samples), Y [m, 2] float32 torch. Chunked fp32 matmuls accumulated in
    float64 (the box's numpy BLAS is slow in float64 on this AMD CPU, so the pooled probes run on the GPU)."""
    import torch
    d = Z.shape[1]
    S = torch.zeros(d, dtype=torch.float64, device=Z.device)
    G = torch.zeros(d, d, dtype=torch.float64, device=Z.device)
    XtY = torch.zeros(d, Y.shape[1], dtype=torch.float64, device=Z.device)
    for s in range(0, Z.shape[0], chunk):
        x, y = Z[s:s + chunk], Y[s:s + chunk]
        S += x.double().sum(0)
        G += (x.T @ x).double()
        XtY += (x.T @ y).double()
    return {"n": Z.shape[0], "S": S.cpu().numpy(), "G": G.cpu().numpy(), "XtY": XtY.cpu().numpy(),
            "Sy": Y.double().sum(0).cpu().numpy()}


def solve_from_stats(parts, alphas):
    """Centred ridge path from summed stats: returns (U, lam, R, xm, ym) with W_a = U (R / (lam + a))."""
    n = sum(p["n"] for p in parts)
    S, G = sum(p["S"] for p in parts), sum(p["G"] for p in parts)
    XtY, Sy = sum(p["XtY"] for p in parts), sum(p["Sy"] for p in parts)
    xm, ym = S / n, Sy / n
    Gc = G - n * np.outer(xm, xm)
    XtYc = XtY - n * np.outer(xm, ym)
    lam, U = np.linalg.eigh(Gc)
    return U, lam, U.T @ XtYc, xm, ym


def pooled_probe(Xfit, Yfit, folds, evals, score_fn, alphas):
    """Pooled-patch ridge. Xfit [n, P, d]: every (clip, position) is a sample, standardised with the fit samples'
    statistics; alpha by fold-mean R2 with folds by clip. Grams on the GPU (fp32 chunks, float64 sums), eigh in float64.
    evals: {name: X [m, Q, d]}; returns alpha, cv curve, and {name: predictions [m, Q, 2]}."""
    import torch
    dev = _dev()
    n, P, d = Xfit.shape
    Xg = torch.from_numpy(np.ascontiguousarray(Xfit, dtype=np.float32)).to(dev).reshape(-1, d)
    S = torch.zeros(d, dtype=torch.float64, device=dev)
    SS = torch.zeros(d, dtype=torch.float64, device=dev)
    for s in range(0, Xg.shape[0], 16384):
        x = Xg[s:s + 16384].double()
        S += x.sum(0)
        SS += (x * x).sum(0)
    mean = S / Xg.shape[0]
    std = torch.sqrt(torch.clamp(SS / Xg.shape[0] - mean ** 2, min=0)).clamp(min=1e-6)
    mean32, std32 = mean.float(), std.float()
    Xg.sub_(mean32).div_(std32)                                                   # Z, in place
    Yg = torch.from_numpy(np.repeat(Yfit, P, axis=0).astype(np.float32)).to(dev)
    sample_fold = np.repeat(folds, P)
    ks = np.unique(folds)
    idx = {k: torch.from_numpy(np.flatnonzero(sample_fold == k)).to(dev) for k in ks}
    stats = {k: gram_stats(Xg[idx[k]], Yg[idx[k]]) for k in ks}
    curve = np.zeros(len(alphas))
    for k in ks:
        U, lam, R, xm, ym = solve_from_stats([stats[j] for j in ks if j != k], alphas)
        ZU = ((Xg[idx[k]] - torch.from_numpy(xm).float().to(dev)) @ torch.from_numpy(U).float().to(dev)).double()
        Yk = np.repeat(Yfit[folds == k], P, axis=0)
        for i, al in enumerate(alphas):
            Pk = (ZU @ torch.from_numpy(R / (lam + al)[:, None]).to(dev)).cpu().numpy() + ym
            curve[i] += score_fn(Yk, Pk)["r2"] / len(ks)
        del ZU
    best = int(np.argmax(curve))
    U, lam, R, xm, ym = solve_from_stats(list(stats.values()), alphas)
    W = U @ (R / (lam + alphas[best])[:, None])
    b = ym - xm @ W
    Wg = torch.from_numpy(W).float().to(dev)
    preds = {}
    for name, Xe in evals.items():
        m, Q, _ = Xe.shape
        Ze = (torch.from_numpy(np.ascontiguousarray(Xe, dtype=np.float32)).to(dev).reshape(-1, d) - mean32) / std32
        preds[name] = ((Ze @ Wg).double().cpu().numpy() + b).reshape(m, Q, -1).astype(np.float32)
        del Ze
    del Xg, Yg
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return float(alphas[best]), curve.tolist(), preds


def r2_per_pos(Y, P):
    """Y [n, 2], P [n, Q, 2] -> R2 per position [Q] (mean over sin, cos columns)."""
    ss_res = ((Y[:, None, :] - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).mean(-1)


def r2_pooled(Y, P):
    """R2 over all (clip, position) samples of P [n, Q, 2]."""
    ss_res = ((Y[:, None, :] - P) ** 2).sum((0, 1))
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) * P.shape[1]
    return float((1 - ss_res / ss_tot).mean())


def cmae(Y, P):
    """Circular MAE (deg) over all (clip, position) samples."""
    th = np.degrees(np.arctan2(Y[:, 0], Y[:, 1]))[:, None]
    ph = np.degrees(np.arctan2(P[..., 0], P[..., 1]))
    return float(np.mean(np.abs((ph - th + 180) % 360 - 180)))


def cmae_per_pos(Y, P):
    th = np.degrees(np.arctan2(Y[:, 0], Y[:, 1]))[:, None]
    ph = np.degrees(np.arctan2(P[..., 0], P[..., 1]))
    return np.mean(np.abs((ph - th + 180) % 360 - 180), axis=0)


def summarise(r2):
    r2 = np.asarray(r2)
    return {"mean_r2": float(r2.mean()), "median_r2": float(np.median(r2)), "frac_ge_0.5": float((r2 >= 0.5).mean()),
            "frac_ge_0.8": float((r2 >= 0.8).mean()), "best_r2": float(r2.max()), "best_pos": int(r2.argmax()),
            "worst_r2": float(r2.min())}


def onset_of(curve):
    curve = np.asarray(curve, float)
    if curve.max() <= 0:
        return None
    return int(np.argmax(curve >= 0.9 * curve.max()))


def jump(points, curve):
    d = np.diff(np.asarray(curve, float))
    i = int(np.argmax(d))
    return {"largest_jump": float(d[i]), "from_point": points[i], "to_point": points[i + 1],
            "second_largest": float(np.sort(d)[-2]) if len(d) > 1 else None,
            "note": "difference between consecutive SAMPLED points (points 10+ are 2-3 blocks apart)"}


def probe(a):
    import multiprocessing as mp
    from threadpoolctl import threadpool_limits
    from wm.probes import ALPHAS, targets, split_rows, Standardizer, cv_select_alpha, fit_ridge, predict
    from wm.provenance import sha256_file

    side = json.loads(Path(a.memmap).with_suffix(".json").read_text())
    assert side["complete"], "extraction incomplete"
    points = side["points"]
    mm = np.load(a.memmap, mmap_mode="r")
    df = table(a.set)
    assert [int(i) for i in df["id"]] == side["ids"]
    split_file = PROJECT_ROOT / SETS[a.set][1]
    Y, kind, score_fn = targets(df, "direction")
    tr, te, folds = split_rows("direction", df, split_file)
    if a.subset == "constvel":   # the paper's Fig. 18 uses its constant-velocity dataset only
        assert a.set == "direction"
        vel = df["motion"].to_numpy() == "velocity"
        keep = vel[tr]
        tr, folds, te = tr[keep], folds[keep], te[vel[te]]
    Ytr, Yte = Y[tr], Y[te]
    left = np.array([p for p in range(NPOS) if p % GRID < GRID // 2])
    right = np.array([p for p in range(NPOS) if p % GRID >= GRID // 2])
    rng = np.random.default_rng(0)
    boot = [rng.integers(0, len(te), len(te)) for _ in range(N_BOOT)]
    t_all = time.time()
    # per-position probes: fork the pool before any BLAS call in the parent, queue every (point, position) task
    _G.update(memmap=a.memmap, tr=tr, te=te, Ytr=Ytr, Yte=Yte, folds=folds, score_fn=score_fn)
    pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker)
    pending = pool.map_async(_perpos, [(pi, pos) for pi in range(len(points)) for pos in range(NPOS)], chunksize=4)
    pooled_rows = []
    with threadpool_limits(a.parent_threads):
        for pi, point in enumerate(points):
            t0 = time.time()
            X = np.ascontiguousarray(mm[:, pi]).astype(np.float32)                    # [N, 256, 1024]
            Xm = X.mean(axis=1).astype(np.float64)
            st = Standardizer().fit(Xm[tr])
            cvm = cv_select_alpha(st.transform(Xm[tr]), Ytr, folds, ALPHAS, score_fn)
            W, b = fit_ridge(st.transform(Xm[tr]), Ytr, cvm["alpha"])
            Pm = predict(st.transform(Xm[te]), W, b)
            al_all, curve_all, pr = pooled_probe(X[tr], Ytr, folds, {"test": X[te]}, score_fn, ALPHAS)
            al_L, curve_L, prL = pooled_probe(X[tr][:, left], Ytr, folds,
                                              {"L": X[te][:, left], "R": X[te][:, right]}, score_fn, ALPHAS)
            al_R, curve_R, prR = pooled_probe(X[tr][:, right], Ytr, folds,
                                              {"R": X[te][:, right], "L": X[te][:, left]}, score_fn, ALPHAS)
            pooled_rows.append((cvm, Pm, al_all, curve_all, pr, al_L, curve_L, prL, al_R, curve_R, prR))
            print(f"{a.set}/{a.model} point {point:2d}: meanpool/pooled/halves done in {time.time() - t0:.0f}s "
                  f"(pooled all-sample R2 {r2_pooled(Yte, pr['test']):.3f})", flush=True)
            del X
    allres = pending.get()
    pool.close()
    pool.join()
    print(f"per-position probes done ({time.time() - t_all:.0f}s since start)", flush=True)
    rows, cache = [], {"meanpool": [], "perpos": [], "pooled": [], "LL": [], "LR": [], "RR": [], "RL": []}
    for pi, point in enumerate(points):
        cvm, Pm, al_all, curve_all, pr, al_L, curve_L, prL, al_R, curve_R, prR = pooled_rows[pi]
        sm = score_fn(Yte, Pm)
        res = sorted([r for r in allres if r[0] == pi], key=lambda r: r[1])
        assert [r[1] for r in res] == list(range(NPOS))
        pp_alpha = np.array([r[2] for r in res])
        pp_cv = np.array([r[3] for r in res])
        pp_r2 = np.array([r[4] for r in res])
        pp_mae = np.array([r[5] for r in res])
        Ppp = np.stack([r[6] for r in res], axis=1)                              # [n_te, 256, 2]
        Ppool = pr["test"]
        pool_r2 = r2_per_pos(Yte, Ppool)
        halves = {
            "left_fit": {"alpha": al_L, "cv_curve_max": max(curve_L),
                         "within_r2": r2_pooled(Yte, prL["L"]), "within_mae": cmae(Yte, prL["L"]),
                         "cross_r2": r2_pooled(Yte, prL["R"]), "cross_mae": cmae(Yte, prL["R"]),
                         "within_mean_pos_r2": float(r2_per_pos(Yte, prL["L"]).mean()),
                         "cross_mean_pos_r2": float(r2_per_pos(Yte, prL["R"]).mean())},
            "right_fit": {"alpha": al_R, "cv_curve_max": max(curve_R),
                          "within_r2": r2_pooled(Yte, prR["R"]), "within_mae": cmae(Yte, prR["R"]),
                          "cross_r2": r2_pooled(Yte, prR["L"]), "cross_mae": cmae(Yte, prR["L"]),
                          "within_mean_pos_r2": float(r2_per_pos(Yte, prR["R"]).mean()),
                          "cross_mean_pos_r2": float(r2_per_pos(Yte, prR["L"]).mean())}}
        cross = (halves["left_fit"]["cross_r2"] + halves["right_fit"]["cross_r2"]) / 2
        within = (halves["left_fit"]["within_r2"] + halves["right_fit"]["within_r2"]) / 2
        row = {"point": point, "frac": min(point, 24) / 24,
               "meanpool": {"alpha": cvm["alpha"], "cv_mean": cvm["cv_mean"], "test_r2": sm["r2"], "test_mae": sm["mae"]},
               "perpos": {**summarise(pp_r2), "mean_mae": float(pp_mae.mean()), "median_mae": float(np.median(pp_mae)),
                          "best_mae": float(pp_mae.min()), "mean_cv_r2": float(pp_cv.mean()),
                          "r2": pp_r2.round(4).tolist(), "mae": pp_mae.round(2).tolist(),
                          "alpha": pp_alpha.tolist(), "cv_r2": pp_cv.round(4).tolist()},
               "pooled": {**summarise(pool_r2), "alpha": al_all, "cv_mean": max(curve_all),
                          "pooled_r2_all_samples": r2_pooled(Yte, Ppool), "mae_all_samples": cmae(Yte, Ppool),
                          "r2": pool_r2.round(4).tolist(), "mae": cmae_per_pos(Yte, Ppool).round(2).tolist()},
               "halves": {**halves, "cross_r2_mean": cross, "within_r2_mean": within,
                          "cross_minus_within": cross - within,
                          "cross_mae_mean": (halves["left_fit"]["cross_mae"] + halves["right_fit"]["cross_mae"]) / 2,
                          "within_mae_mean": (halves["left_fit"]["within_mae"] + halves["right_fit"]["within_mae"]) / 2}}
        rows.append(row)
        for k, v in (("meanpool", Pm[:, None, :]), ("perpos", Ppp), ("pooled", Ppool), ("LL", prL["L"]),
                     ("LR", prL["R"]), ("RR", prR["R"]), ("RL", prR["L"])):
            cache[k].append(v.astype(np.float32))
        print(f"{a.set}/{a.model} point {point:2d}: meanpool test R2 {sm['r2']:.3f} (Part1 alpha rule, a={cvm['alpha']:.3g}) | "
              f"perpos mean {pp_r2.mean():.3f} med {np.median(pp_r2):.3f} >=.5 {(pp_r2 >= .5).mean():.2f} best {pp_r2.max():.3f} | "
              f"pooled mean {pool_r2.mean():.3f} | cross {cross:.3f} within {within:.3f}", flush=True)

    # --- curves, onsets, bootstrap over test clips
    def metric_curves(idx):
        Yb = Yte[idx]
        out = {"meanpool_r2": [], "perpos_mean_r2": [], "perpos_frac_ge_0.5": [], "pooled_mean_r2": [],
               "pooled_frac_ge_0.5": [], "cross_half_r2": [], "within_half_r2": []}
        for i in range(len(points)):
            out["meanpool_r2"].append(r2_pooled(Yb, cache["meanpool"][i][idx]))
            pp = r2_per_pos(Yb, cache["perpos"][i][idx])
            out["perpos_mean_r2"].append(float(pp.mean()))
            out["perpos_frac_ge_0.5"].append(float((pp >= 0.5).mean()))
            po = r2_per_pos(Yb, cache["pooled"][i][idx])
            out["pooled_mean_r2"].append(float(po.mean()))
            out["pooled_frac_ge_0.5"].append(float((po >= 0.5).mean()))
            out["cross_half_r2"].append((r2_pooled(Yb, cache["LR"][i][idx]) + r2_pooled(Yb, cache["RL"][i][idx])) / 2)
            out["within_half_r2"].append((r2_pooled(Yb, cache["LL"][i][idx]) + r2_pooled(Yb, cache["RR"][i][idx])) / 2)
        return out

    curves = metric_curves(np.arange(len(te)))
    boots = [metric_curves(idx) for idx in boot]
    onsets = {}
    for name, c in curves.items():
        o = onset_of(c)
        bo = [onset_of(b[name]) for b in boots]
        bo_pts = np.array([points[x] for x in bo if x is not None])
        onsets[name] = {"onset": None if o is None else points[o],
                        "onset_frac": None if o is None else min(points[o], 24) / 24,
                        "ci95": None if len(bo_pts) == 0 else [int(np.percentile(bo_pts, 2.5)), int(np.percentile(bo_pts, 97.5))],
                        "boot_draws_without_onset": int(sum(x is None for x in bo)),
                        "max": float(max(c)), "argmax_point": points[int(np.argmax(c))],
                        **jump(points, c)}
    part1 = None
    if a.set == "direction":
        p1f = RESULTS / f"p1a_direction_direction_meanpool{'' if a.model == 'vjepa2' else '_' + a.model}.json"
        if p1f.exists():
            p1 = json.loads(p1f.read_text())
            part1 = {"file": p1f.name, "cv_onset": p1["availability"]["onset"], "cv_onset_ci": p1["availability"].get("onset_ci"),
                     "cv_mean_by_point": {str(r["point"]): r["cv_mean"] for r in p1["layers"] if r["point"] in points},
                     "test_r2_by_point": {str(r["point"]): r["test_r2"] for r in p1["layers"] if r["point"] in points}}
    wall = time.time() - t_all
    out = {"set": a.set, "subset": a.subset, "model": a.model, "variable": "direction", "target": "(sin theta, cos theta)",
           "n_train": len(tr), "n_test": len(te), "points": points, "curves": curves, "onsets": onsets,
           ("part1_meanpool_all_clips" if a.subset else "part1_meanpool"): part1,   # Part 1 = all 1,500 clips
           "layers": rows,
           "left_positions": "columns 0-7 of the 16x16 grid (128 positions); right = columns 8-15",
           "methods": {
               "probe": "closed-form ridge as wm.probes: train-standardised features, targets (sin, cos), alpha by fold-mean R2 on the split's 5 train folds over ALPHAS = logspace(-2, 4, 13), refit on all train, test scored once",
               "meanpool": "position mean of the stored time-pool (= Part 1 meanpool up to fp16 storage), same recipe",
               "perpos": "one probe per spatial position, own standardiser and own CV alpha (no alpha sharing)",
               "pooled": "one probe on all (train clip, position) samples (n_train x 256), standardised over those samples, folds by clip; test scored per position",
               "halves": "pooled probe fit on the left (right) 128 positions of train clips; within = same half of test clips, cross = other half; R2 over all (clip, position) samples of that half; cross_half_r2 = mean of left->right and right->left",
               "r2": "mean over the sin and cos columns of per-column R2 on test clips",
               "onset": "first SAMPLED point with metric >= 90% of its max over the sampled points; 95% CI from a 200-draw bootstrap of test clips over fixed test predictions (Part 1's CV onset used train-fold OOF predictions; the meanpool_r2 curve here gives the same-rule test-set baseline)"},
           "memmap": {"path": side["memmap"], "shape": side["shape"], "dtype": side["dtype"], "pooling": side["pooling"],
                      "location_note": side.get("location_note")},
           "extraction": {k: side.get(k) for k in ("gpu_seconds_forward", "wall_seconds_last_run", "batch_size", "gpu",
                                                     "meanpool_parity", "random_init", "preprocess")},
           "probe_wall_seconds": wall,
           "provenance": {"split_file": SETS[a.set][1], "split_sha256": sha256_file(split_file), "commit": a.commit,
                          "points": points, "pooling": side["pooling"], "positions": "16x16 (no spatial pooling)",
                          "host": "vast 53030966 RTX 4060 Ti (extract GPU; per-position probes CPU float64; pooled/half probes GPU fp32 grams with float64 sums, float64 eigh)",
                          "subset": ("constant-velocity clips only (motion == 'velocity'), split rows filtered, same folds"
                                     if a.subset == "constvel" else "all clips of the set"),
                          "time_averaging": ("per-position features are averaged over the 8 token time steps before probing "
                                             "(one 1024-vector per spatial position per clip); the paper does not state whether "
                                             "its per-patch probes use per-token or time-averaged features"),
                          "half_frame_design": ("the half-frame test fits ONE pooled probe over the 128 positions of one half "
                                                "(columns 0-7 or 8-15; every (clip, position) is a sample) and scores it on the "
                                                "other half's positions of test clips; the paper does not state its half-frame "
                                                "probe design"),
                          "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed_bootstrap": 0}}
    path = Path(a.results_dir) / result_path(a.set, a.model, a.subset).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({wall:.0f}s)")
    for name, o in onsets.items():
        print(f"  onset {name}: {o['onset']} {o['ci95']} | max {o['max']:.3f} | jump {o['largest_jump']:.3f} {o['from_point']}->{o['to_point']}")


# ================================================================ folds (CPU + GPU): 5-fold refit on the train clips

def ridge_dual_predict(Xfit, Yfit, Xev, alpha):
    """fit_ridge(Xfit, Yfit, alpha) then predict Xev, via the n x n kernel (n < d here): identical to the primal
    solve (XcᵀXc + αI)⁻¹ XcᵀYc, one small eigh."""
    xm, ym = Xfit.mean(axis=0), Yfit.mean(axis=0)
    Xc = Xfit - xm
    mu, V = np.linalg.eigh(Xc @ Xc.T)
    return (Xev - xm) @ Xc.T @ V @ ((V.T @ (Yfit - ym)) / (mu + alpha)[:, None]) + ym


def _perpos_fold(task):
    """One (point, position, outer fold k): fit rows = train clips outside fold k, standardised on them; alpha by
    fold-mean R2 over the remaining 4 folds (wm.paperscale.dual_cv = wm.probes.cv_select_alpha, dual solver); refit on
    all fit rows; predict fold k."""
    from wm.paperscale import dual_cv
    from wm.probes import ALPHAS, Standardizer
    pi, pos, k = task
    if "mm" not in _G:
        _G["mm"] = np.load(_G["memmap"], mmap_mode="r")
    tr, Ytr, folds, score_fn = (_G[x] for x in ("tr", "Ytr", "folds", "score_fn"))
    Xp = np.asarray(_G["mm"][tr, pi, pos], np.float64)
    fit, ev = folds != k, folds == k
    st = Standardizer().fit(Xp[fit])
    Xf, Xe = st.transform(Xp[fit]), st.transform(Xp[ev])
    cv = dual_cv(Xf, Ytr[fit], folds[fit], score_fn, ALPHAS)
    return pi, pos, k, cv["alpha"], ridge_dual_predict(Xf, Ytr[fit], Xe, cv["alpha"]).astype(np.float32)


def fold_curves(points, Yev, pp, pool, LL, LR, RR, RL, Pm):
    """Per-point metrics of one outer fold from its predictions (lists over points; pp/pool [m, 256, 2], halves
    [m, 128, 2], Pm [m, 2]) and the onsets (90%-of-max rule over the sampled points) of each curve."""
    c = {"meanpool_r2": [r2_pooled(Yev, x[:, None, :]) for x in Pm],
         "perpos_mean_r2": [float(r2_per_pos(Yev, x).mean()) for x in pp],
         "pooled_mean_r2": [float(r2_per_pos(Yev, x).mean()) for x in pool],
         "cross_half_r2": [(r2_pooled(Yev, a) + r2_pooled(Yev, b)) / 2 for a, b in zip(LR, RL)],
         "within_half_r2": [(r2_pooled(Yev, a) + r2_pooled(Yev, b)) / 2 for a, b in zip(LL, RR)]}
    on = {k: (None if onset_of(v) is None else points[onset_of(v)]) for k, v in c.items()}
    return c, on


def probe_folds(a):
    import multiprocessing as mp
    from threadpoolctl import threadpool_limits
    from wm.probes import ALPHAS, Standardizer, cv_select_alpha, fit_ridge, predict, split_rows, targets
    from wm.provenance import sha256_file

    path = Path(a.results_dir) / "p1a_perpatch_hard_folds.json"
    out = json.loads(path.read_text()) if path.exists() else {
        "design": ("5-fold refit on each set's train clips (splits/split_<set>.json folds): per outer fold k, fit on "
                   "the other 4 folds, alpha re-chosen by fold-mean R2 over those 4 folds (per position for perpos; by "
                   "clip for pooled / halves), score fold k; the stored test clips are not used. Probes, "
                   "standardisation and metrics as the probe stage (per-position R2 = mean over sin, cos columns)"),
        "onset_rule": "first SAMPLED point with the fold's metric >= 90% of its max over the sampled points",
        "sets": {}}
    for name in a.sets:
        mmp = str(Path(a.memmap_dir) / f"{name}_vjepa2.npy")
        side = json.loads(Path(mmp).with_suffix(".json").read_text())
        assert side["complete"]
        points = side["points"]
        mm = np.load(mmp, mmap_mode="r")
        df = table(name)
        assert [int(i) for i in df["id"]] == side["ids"]
        split_file = PROJECT_ROOT / SETS[name][1]
        Y, kind, score_fn = targets(df, "direction")
        tr, te, folds = split_rows("direction", df, split_file)
        Ytr = Y[tr]
        ks = np.unique(folds)
        left = np.array([p for p in range(NPOS) if p % GRID < GRID // 2])
        right = np.array([p for p in range(NPOS) if p % GRID >= GRID // 2])
        t0 = time.time()
        _G.clear()
        _G.update(memmap=mmp, tr=tr, Ytr=Ytr, folds=folds, score_fn=score_fn)
        pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker)
        pending = pool.map_async(_perpos_fold, [(pi, pos, k) for pi in range(len(points)) for pos in range(NPOS)
                                                for k in ks], chunksize=8)
        store = {k: {x: [] for x in ("Pm", "pool", "LL", "LR", "RR", "RL")} for k in ks}
        alphas = {k: {"meanpool": [], "pooled": [], "left": [], "right": []} for k in ks}
        with threadpool_limits(a.parent_threads):
            for pi, point in enumerate(points):
                X = np.ascontiguousarray(mm[tr, pi]).astype(np.float32)             # [n_train, 256, 1024]
                Xm = X.mean(axis=1).astype(np.float64)
                for k in ks:
                    fit, ev = folds != k, folds == k
                    st = Standardizer().fit(Xm[fit])
                    cvm = cv_select_alpha(st.transform(Xm[fit]), Ytr[fit], folds[fit], ALPHAS, score_fn)
                    W, b = fit_ridge(st.transform(Xm[fit]), Ytr[fit], cvm["alpha"])
                    store[k]["Pm"].append(predict(st.transform(Xm[ev]), W, b))
                    al, _, pr = pooled_probe(X[fit], Ytr[fit], folds[fit], {"ev": X[ev]}, score_fn, ALPHAS)
                    aL, _, prL = pooled_probe(X[fit][:, left], Ytr[fit], folds[fit],
                                              {"L": X[ev][:, left], "R": X[ev][:, right]}, score_fn, ALPHAS)
                    aR, _, prR = pooled_probe(X[fit][:, right], Ytr[fit], folds[fit],
                                              {"R": X[ev][:, right], "L": X[ev][:, left]}, score_fn, ALPHAS)
                    store[k]["pool"].append(pr["ev"])
                    store[k]["LL"].append(prL["L"]), store[k]["LR"].append(prL["R"])
                    store[k]["RR"].append(prR["R"]), store[k]["RL"].append(prR["L"])
                    for key, v in (("meanpool", cvm["alpha"]), ("pooled", al), ("left", aL), ("right", aR)):
                        alphas[k][key].append(v)
                print(f"{name} point {point}: meanpool/pooled/halves x {len(ks)} folds ({time.time() - t0:.0f}s)",
                      flush=True)
                del X
        res = pending.get()
        pool.close()
        pool.join()
        print(f"{name}: per-position probes done ({time.time() - t0:.0f}s)", flush=True)
        per_fold = []
        for k in ks:
            ev = folds == k
            pp = []
            for pi in range(len(points)):
                r = sorted([x for x in res if x[0] == pi and x[2] == k], key=lambda x: x[1])
                assert [x[1] for x in r] == list(range(NPOS))
                pp.append(np.stack([x[4] for x in r], axis=1))
            c, on = fold_curves(points, Ytr[ev], pp, store[k]["pool"], store[k]["LL"], store[k]["LR"],
                                store[k]["RR"], store[k]["RL"], store[k]["Pm"])
            pp_alpha = np.array([[x[3] for x in res if x[0] == pi and x[2] == k] for pi in range(len(points))])
            per_fold.append({"fold": int(k), "n_fit": int((~ev).sum()), "n_eval": int(ev.sum()), "curves": c,
                             "onsets": on, "alpha": {**alphas[k], "perpos_median": np.median(pp_alpha, 1).tolist()}})
        summ = {}
        for m in per_fold[0]["curves"]:
            arr = np.array([f["curves"][m] for f in per_fold])
            ons = [f["onsets"][m] for f in per_fold]
            summ[m] = {"mean": arr.mean(0).tolist(), "sd": arr.std(0, ddof=1).tolist(), "onset_per_fold": ons,
                       "onset_of_fold_mean_curve": None if onset_of(arr.mean(0)) is None else points[onset_of(arr.mean(0))]}
        out["sets"][name] = {"points": points, "n_train": len(tr), "fold_sizes": np.bincount(folds).tolist(),
                             "per_fold": per_fold, "summary": summ, "wall_seconds": time.time() - t0,
                             "provenance": {"split_file": SETS[name][1], "split_sha256": sha256_file(split_file),
                                            "memmap": mmp, "commit": a.commit, "workers": a.workers,
                                            "parent_threads": a.parent_threads,
                                            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path} [{name}] ({time.time() - t0:.0f}s)")
        for m, v in summ.items():
            print(f"  {m}: onsets {v['onset_per_fold']} | " + " ".join(f"{p}:{mu:.3f}±{sd:.3f}" for p, mu, sd in
                                                                     zip(points, v["mean"], v["sd"])))


# ================================================================ figures (Mac)

def figures(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = {"vjepa2": ("direction", "vjepa2", None, "V-JEPA 2 ViT-L — direction set (1,500 clips)"),
              "random": ("direction", "random", None, "random-init ViT-L — direction set"),
              "constvel": ("direction", "vjepa2", "constvel", "V-JEPA 2 — constant-velocity clips only (750)"),
              "hard": ("hard", "vjepa2", None, "V-JEPA 2 — hard render set (392 clips, 8 directions)"),
              "paper_layout": ("paper_layout", "vjepa2", None, "V-JEPA 2 — paper-layout set (392 clips)")}
    res, titles = {}, {}
    for key, (st, m, sub, title) in panels.items():
        if result_path(st, m, sub).exists():
            res[key], titles[key] = json.loads(result_path(st, m, sub).read_text()), title
    colors = {"meanpool_r2": "#6b7280", "perpos_mean_r2": "#2563eb", "perpos_frac_ge_0.5": "#16a34a",
              "cross_half_r2": "#dc2626", "pooled_mean_r2": "#9333ea"}
    labels = {"meanpool_r2": "mean-pooled probe R²", "perpos_mean_r2": "per-position probes: mean R²",
              "perpos_frac_ge_0.5": "per-position probes: fraction of positions R² ≥ 0.5",
              "pooled_mean_r2": "pooled-patch probe: mean per-position R²",
              "cross_half_r2": "half-frame probe: cross-half R²"}
    ncol = min(3, len(res))
    nrow = -(-len(res) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.2 * nrow), sharey=True, squeeze=False)
    for ax in axes.ravel()[len(res):]:
        ax.axis("off")
    for ax, (m, r) in zip(axes.ravel(), res.items()):
        x = [min(p, 24) / 24 for p in r["points"]]
        for k in colors:
            ax.plot(x, r["curves"][k], marker="o", ms=3.5, lw=1.8 if k != "meanpool_r2" else 2.4, color=colors[k],
                    label=labels[k], ls="--" if k == "meanpool_r2" else "-")
        ax.axvspan(8 / 24, 9 / 24, color="#fde68a", alpha=0.5, lw=0, label="paper's transition (its layers 7→8 = points 8→9)")
        ax.set_title(titles[m], fontsize=9.5)
        ax.set_xlabel("layer fraction (block / 24)")
        lo = min(min(r["curves"][k]) for k in colors)
        ax.set_ylim(max(-0.3, lo - 0.05) if lo < -0.1 else -0.1, 1.02)
        if lo < -0.3:
            ax.text(0.02, 0.97, f"cross-half R² dips to {lo:.2f} (clipped)", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color=colors["cross_half_r2"])
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(alpha=0.25)
    for row in axes:
        row[0].set_ylabel("test R² / fraction")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Per-patch direction probes vs layer (paper App. C.5 / Fig. 18 analogue); dashed = mean-pooled probe, "
                 "same test clips", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1g_perpatch_direction.png", dpi=160)
    plt.close(fig)

    res = {k: v for k, v in res.items() if k in ("vjepa2", "random", "hard")}
    fig, axes = plt.subplots(len(res), len(HEATMAP_POINTS), figsize=(2.3 * len(HEATMAP_POINTS) + 0.8, 2.5 * len(res)),
                             squeeze=False)
    for row, (m, r) in zip(axes, res.items()):
        by = {L["point"]: L for L in r["layers"]}
        for ax, p in zip(row, HEATMAP_POINTS):
            if p not in by:
                ax.axis("off")
                continue
            im = ax.imshow(np.array(by[p]["perpos"]["r2"]).reshape(GRID, GRID), vmin=0, vmax=1, cmap="viridis")
            name = {"vjepa2": "V-JEPA 2", "random": "random init", "hard": "V-JEPA 2, hard set"}[m]
            ax.set_title(f"{name} pt {p}\nmean {by[p]['perpos']['mean_r2']:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="per-position test R²")
    fig.suptitle("Per-position direction probe R² on the 16×16 grid (paper Fig. 18c analogue)", fontsize=10)
    fig.savefig(FIGURES / "fig1h_perpatch_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("extract", "probe", "folds", "figures"))
    ap.add_argument("--sets", nargs="+", default=["hard"], choices=tuple(SETS))
    ap.add_argument("--memmap-dir", default="/workspace/wm/artifacts/perpatch")
    ap.add_argument("--set", default="direction", choices=tuple(SETS))
    ap.add_argument("--model", default="vjepa2", choices=("vjepa2", "random"))
    ap.add_argument("--points", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--memmap", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--parent-threads", type=int, default=24)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--subset", default=None, choices=(None, "constvel"))
    ap.add_argument("--results-dir", default=str(RESULTS))
    a = ap.parse_args()
    {"extract": extract, "probe": probe, "folds": probe_folds, "figures": figures}[a.stage](a)

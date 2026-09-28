"""Session 3: steer SPEED at point 22 (and point 12 for propagation) and read the edit with the predictor's forecast,
as session 2 did for direction (REPORT §4.5, §7 next step for the scalars).

Design (speed set, constant-velocity clips, 64 speeds 0.25-4 m/s; splits/split_v1.json roles):
  held-out targets  the contiguous block of p2_steer_speed_speed_L19_contiguous.json (seed 0, 3.048-3.464 m/s); at
                    points 12 and 22 the curve is rebuilt from knot clips (folds 0-2) at the kept values only
  arms (PCA-64 of knot clips, additive, residual kept, as Part 2)
    spline        count-weighted smoothing spline (the Part 2 default), source value -> target value
    chord         straight line between the SMOOTHED knots (polyline through curve.points; Part 2 `linear`)
    linear_raw    straight line between the RAW kept centroids (Part 2 `linear_raw`, Goodfire's chord)
    null          the spline edit toward the far end of the range from the held-out block (0.25 m/s)
    unedited      no edit (no forward pass)
  norm conditions (as session2_norm_matched): own norm; ||delta|| := ||chord||; ||delta|| := ||twin_meanpool_L -
                  src_meanpool_L|| (natural twin change, full-clip meanpool). Point 12: own + natural.
  twins           each carrier re-rendered at each target speed (same start, theta, background; render_twin with
                  metadata speed_mps replaced), encoded, forecast by the predictor.
Stages:
  plan     CPU, local -> ART/plan.npz, plan.json, twins/*.mp4, centroids_{speed,twins}.npy, renderer validation
  extract  box, GPU. All 1536 speed clips: frames 1-8 encoded alone, predictor (mask token 0) -> native/pred_pooled_all.npy
  forward  box, GPU. Per 8-carrier group (fcntl lock on /tmp/wm_gpu.lock per group): source full + context-only
           encodings, twins, edits at 22 (predictor x 3 conditions, full-clip suffix own norm) and 12 (predictor x
           own/natural, full-clip suffix own norm) -> forward/group_*.npz
  score    CPU, local -> results/session3_speed_predictor.json, figures/fig_session3_speed.png

  python scripts/session3_speed_predictor.py plan|extract|forward|score [--out DIR] [--variable speed|acceleration]

--variable acceleration (the acceleration analogue; the speed path is unchanged): acceleration set (clips start at rest,
  64 accelerations 0.25-10 m/s^2), held-out block of p2_steer_acceleration_acceleration_L21_contiguous.json (seed 0,
  7.52-8.61 m/s^2), edit points 21 (the Part 2 point, three norm conditions) and 12 (own + natural); twins re-rendered
  with acceleration_mps2 replaced (render_twin draws p0 + (v t + a t^2 / 2) u for any clip); the displacement readout
  is a quadratic fit to the four read-out forecast positions (accel_from_positions). Unlike the speed run, every
  readout (probes, calibration, propagation readers) is fit on probe clips OUTSIDE the held-out block, so no fitting
  step sees a clip at a target value. Off-target: direction change and the change in the forecast's mean speed over
  steps 4-7 (which physics couples to acceleration: a twin's expected change is delta_a * t_mid).
"""
import argparse
import fcntl
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, disk_pixels, frame_hash, load_table  # noqa: E402

RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
ARMS = ("spline", "chord", "linear_raw", "null")
VARIABLES = {
    "speed": {"dataset": "speed", "col": "speed_mps", "hi": 22, "art": "session3_speed",
              "ref": "p2_steer_speed_speed_L19_contiguous.json", "result": "session3_speed_predictor.json",
              "fig": "fig_session3_speed.png", "unit": "m/s", "excl_held_readouts": False},
    "acceleration": {"dataset": "acceleration", "col": "acceleration_mps2", "hi": 21, "art": "session3_acceleration",
                     "ref": "p2_steer_acceleration_acceleration_L21_contiguous.json",
                     "result": "session3_acceleration_predictor.json", "fig": "fig_session3_acceleration.png",
                     "unit": "m/s^2", "excl_held_readouts": True},
}


def configure(variable):
    """Set the module-level design constants for one variable (speed = the original session-3 design)."""
    global VAR, CFG, DATASET, POINTS, CONDS, ART
    VAR, CFG = variable, VARIABLES[variable]
    DATASET = CFG["dataset"]
    POINTS = (12, CFG["hi"])
    CONDS = {CFG["hi"]: ("own", "chord_norm", "natural_norm"), 12: ("own", "natural_norm")}
    ART = PROJECT_ROOT / "artifacts" / CFG["art"]


configure("speed")
FPS, FRAMES_PER_STEP, PX_PER_M = 24.0, 2, 32.0
D = 1024
LOCK = "/tmp/wm_gpu.lock"


# ================================================================ pure helpers (tests/test_session3_speed.py)

def speed_twin_meta(meta, speed):
    """Clip metadata with speed_mps replaced (same start, theta, acceleration, frames): the speed twin."""
    return {**meta, "speed_mps": float(speed)}


def acceleration_twin_meta(meta, acc):
    """Clip metadata with acceleration_mps2 replaced (same start, theta, initial speed, frames): the acceleration twin."""
    return {**meta, "acceleration_mps2": float(acc)}


def twin_meta(meta, value):
    return (speed_twin_meta if VAR == "speed" else acceleration_twin_meta)(meta, value)


def far_end_value(values, held):
    """The end of the value range farther from the held-out block's centre (the null's aim)."""
    values = np.sort(np.asarray(values, float))
    c = float(np.mean(held))
    return float(values[0] if abs(c - values[0]) >= abs(values[-1] - c) else values[-1])


def speed_from_positions(pos, px_per_m=PX_PER_M, fps=FPS, frames_per_step=FRAMES_PER_STEP):
    """Per-tubelet disk positions [..., S, 2] (px) -> speed (m/s) from the first-to-last displacement:
    ||p_last - p_first|| / (S - 1) px per tubelet step, one step = frames_per_step / fps s."""
    pos = np.asarray(pos, float)
    step_px = np.linalg.norm(pos[..., -1, :] - pos[..., 0, :], axis=-1) / (pos.shape[-2] - 1)
    return step_px / px_per_m * fps / frames_per_step


def accel_from_positions(pos, px_per_m=PX_PER_M, fps=FPS, frames_per_step=FRAMES_PER_STEP):
    """Per-tubelet disk positions [..., S, 2] (px, S >= 3) -> acceleration (m/s^2) along the motion: least-squares
    fit p(t) = c0 + c1 t + c2 t^2 / 2 per coordinate (t = step * frames_per_step / fps), acceleration vector c2
    projected on the unit first-to-last displacement. A per-tubelet two-frame average shifts positions by a constant
    (a dt^2 / 8) and leaves c2 unchanged."""
    pos = np.asarray(pos, float)
    S = pos.shape[-2]
    t = np.arange(S) * frames_per_step / fps
    A = np.stack([np.ones(S), t, t ** 2 / 2], 1)                                  # [S, 3]
    c2 = np.einsum("s,...sc->...c", np.linalg.pinv(A)[2], pos)                   # [..., 2] px/s^2
    u = pos[..., -1, :] - pos[..., 0, :]
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-12)
    return np.sum(c2 * u, -1) / px_per_m


def motion_from_positions(pos):
    """The displacement readout of the configured variable (speed: first-to-last; acceleration: quadratic fit)."""
    return (speed_from_positions if VAR == "speed" else accel_from_positions)(pos)


def rescale(delta, ref_norm, eps=1e-6):
    """delta [..., D] rescaled to ||delta|| = ref_norm [...]; a (near-)zero delta stays zero."""
    delta = np.asarray(delta, float)
    n = np.linalg.norm(delta, axis=-1)
    s = np.where(n > eps, np.asarray(ref_norm, float) / np.maximum(n, eps), 0.0)
    return delta * s[..., None], s


def projection_R(a, b, axes=-1):
    """<a, b> / ||b||^2 over `axes` (recovery along the reference change b)."""
    return np.sum(a * b, axis=axes) / np.maximum(np.sum(b * b, axis=axes), 1e-12)


def per_clip_ci(v, n=1000, seed=0):
    """Mean, per-clip SE and a clip-bootstrap 95% CI of per-carrier values (NaNs dropped)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "se": None, "ci95": None, "n": 0}
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def tubelet_centroids(frames):
    """[8, 2] pixel centroid (x, y) per tubelet (session2_extras.centroids); NaN where the disk is absent."""
    pix = disk_pixels(frames)
    out = np.full((8, 2), np.nan)
    for t in range(8):
        yy, xx = np.nonzero(pix[2 * t:2 * t + 2].any(0))
        if len(xx):
            out[t] = xx.mean(), yy.mean()
    return out


@contextmanager
def gpu_lock(path=LOCK):
    """Exclusive flock(2) on the shared GPU lock file (same lock as `flock /tmp/wm_gpu.lock <cmd>`)."""
    with open(path, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t0
        try:
            yield waited
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ================================================================ plan (CPU)

def build_point(d, y, held, carriers, tg, k=64):
    """Edit deltas [C, A, T, D] (own norm) at one point, plus the geometry numbers."""
    from wm import manifold as mf
    X = d["X"].astype(np.float64)
    knot = (d["role"] == "knot") & ~np.isin(y, held)
    pca = mf.fit_pca(X[knot], k)
    cent = mf.centroids(pca.project(X[knot]), y[knot])
    curve = mf.fit_curve(cent, False, spline="smooth")
    raw = mf.raw_knot_curve(curve, cent)
    v_far = far_end_value(np.unique(y), held)
    Z = pca.project(X[carriers])
    src = y[carriers]
    C, T = tg.shape
    out = np.zeros((C, len(ARMS), T, D), np.float32)
    for j in range(T):
        sp = mf.manifold_coords(Z, curve, curve.coord_of_value(src), curve.coord_of_value(tg[:, j]), 2)[:, -1]
        out[:, 0, j] = pca.lift_delta(sp - Z)
        out[:, 1, j] = pca.lift_delta(mf.piecewise_linear_point(curve, tg[:, j]) - mf.piecewise_linear_point(curve, src))
        out[:, 2, j] = pca.lift_delta(mf.piecewise_linear_point(raw, tg[:, j]) - mf.piecewise_linear_point(raw, src))
        nl = mf.manifold_coords(Z, curve, curve.coord_of_value(src), curve.coord_of_value(np.full(C, v_far)), 2)[:, -1]
        out[:, 3, j] = pca.lift_delta(nl - Z)
    hv = np.asarray(held, float)
    sag = np.linalg.norm(curve(curve.coord_of_value(hv)) - mf.piecewise_linear_point(curve, hv), axis=-1)
    sag_raw = np.linalg.norm(curve(curve.coord_of_value(hv)) - mf.piecewise_linear_point(raw, hv), axis=-1)
    geo = {"n_knot_clips": int(knot.sum()), "v_far_null": v_far,
           "pca_var_explained_frac": float(pca.explained.sum() / np.var(X[knot], axis=0, ddof=1).sum()),
           "sagitta_spline_vs_smoothed_chord_pca_units": sag.round(4).tolist(),
           "sagitta_spline_vs_raw_chord_pca_units": sag_raw.round(4).tolist(),
           "centroid_noise_median_pca_units": float(np.median(cent["spread"] / np.sqrt(cent["count"]))),
           "delta_norm_median": {a: float(np.median(np.linalg.norm(out[:, i], axis=-1))) for i, a in enumerate(ARMS)},
           "rel_diff_spline_vs_chord_median": float(np.median(np.linalg.norm(out[:, 0] - out[:, 1], axis=-1)
                                                             / np.maximum(np.linalg.norm(out[:, 1], axis=-1), 1e-12))),
           "rel_diff_chord_vs_linear_raw_median": float(np.median(np.linalg.norm(out[:, 1] - out[:, 2], axis=-1)
                                                                  / np.maximum(np.linalg.norm(out[:, 1], axis=-1), 1e-12)))}
    return out, geo


def plan(args):
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, load_sweep, predict
    from wm.render_twin import load_meta, validate, write_twin
    out = Path(args.out)
    (out / "twins").mkdir(parents=True, exist_ok=True)
    d0 = load_inputs(DATASET, POINTS[0])
    df, y = d0["df"], d0["y"]
    values = np.unique(y)
    mask, hinfo = mf.heldout_design(values, "contiguous", False, seed=0)
    ref = json.loads((RES / CFG["ref"]).read_text())["holdout"]
    assert np.allclose(values[mask], ref["held_out_values"]), f"held-out block differs from the Part 2 run {CFG['ref']}"
    held = values[mask]
    rng = np.random.default_rng(args.seed)
    test = np.flatnonzero(d0["role"] == "test")
    carriers = np.sort(rng.choice(test, size=args.n_carriers, replace=False))
    tg = np.stack([rng.choice(held[~np.isclose(held, y[c])], size=args.n_targets, replace=False) for c in carriers])
    C, T = tg.shape
    arrays = {"carrier_rows": carriers, "carrier_ids": df["id"].to_numpy()[carriers], "targets": tg}
    full = np.load(PROJECT_ROOT / f"artifacts/activations/{DATASET}/vjepa2/meanpool.npy", mmap_mode="r")
    arrays["carrier_stored_meanpool"] = np.asarray(full[carriers], np.float32)
    info = {"variable": VAR, "dataset": DATASET, "points": list(POINTS), "arms": list(ARMS), "conditions": {str(k): v for k, v in CONDS.items()},
            "holdout": hinfo, "held_values": held.tolist(), "n_carriers": C, "n_targets": T, "seed": args.seed,
            "carrier_pool": f"test clips (split_v1 fold -1), any {VAR}; targets = 4 held-out values != own {VAR}",
            "readouts_exclude_held_values": CFG["excl_held_readouts"],
            "per_point": {}}
    sweep = load_sweep(DATASET, DATASET)
    for L in POINTS:
        d = load_inputs(DATASET, L)
        dl, geo = build_point(d, y, held, carriers, tg)
        arrays[f"deltas_L{L}"] = dl
        X = d["X"].astype(np.float64)
        probe = d["role"] == "probe"
        if CFG["excl_held_readouts"]:
            probe = probe & ~np.isin(y, held)
        st = Standardizer().fit(X[probe])
        W, b = fit_ridge(st.transform(X[probe]), y[probe], sweep["layers"][L]["alpha"])
        rd = lambda Z: predict(st.transform(Z.reshape(-1, D)), W, b).reshape(Z.shape[:-1])   # noqa: E731
        geo["same_point_probe_err_to_target"] = {a: float(np.abs(rd(X[carriers][:, None] + dl[:, i]) - tg).mean())
                                                 for i, a in enumerate(ARMS)}
        geo["same_point_probe_err_unedited"] = float(np.abs(rd(X[carriers])[:, None] - tg).mean())
        info["per_point"][str(L)] = geo
        print(f"L{L}: {json.dumps(geo['same_point_probe_err_to_target'])} unedited {geo['same_point_probe_err_unedited']:.3f}")
    # twins (speed or acceleration replaced), rendered here and shipped as MP4 (the box decodes them: its PyAV cannot change the encoding)
    hashes, cen_tw, vis = [], np.zeros((C, T, 8, 2)), []
    for i, c in enumerate(carriers):
        meta = load_meta(DATASET, int(df["id"].iloc[c]))
        for j in range(T):
            fr = write_twin(twin_meta(meta, tg[i, j]), None, out / "twins" / f"twin_{i:04d}_{j}.mp4")
            hashes.append(frame_hash(fr))
            cen_tw[i, j] = tubelet_centroids(fr)
            vis.append(float((disk_pixels(fr).reshape(16, -1).sum(1) > 0).mean()))
    arrays["twin_frame_hash"] = np.array(hashes).reshape(C, T)
    np.save(out / "centroids_twins.npy", cen_tw)
    info["twins"] = {"dir": "twins/twin_{carrier:04d}_{target}.mp4", "visible_frac_mean": float(np.mean(vis))}
    cache = out / f"centroids_{DATASET}.npy"
    if not cache.exists():
        np.save(cache, np.stack([tubelet_centroids(decode(v)) for v in df["video"]]))
    val = validate(DATASET, n=20, seed=0)
    info[f"renderer_validation_{DATASET}"] = {k: val[k] for k in ("codec", "mask_twin_iou_mean", "verdict")}
    np.savez(out / "plan.npz", **arrays)
    (out / "plan.json").write_text(json.dumps(info, indent=1))
    print("plan written", out, "renderer IoU", val["codec"]["iou_mean"], val["codec"]["iou_min"])


def load_plan(out):
    z = np.load(Path(out) / "plan.npz")
    return {k: z[k] for k in z.files}, json.loads((Path(out) / "plan.json").read_text())


# ================================================================ GPU stages

def _model():
    import torch
    from wm.extract import load_model, pick_device, set_precision
    torch.set_num_threads(1)                                      # shared, quota-throttled box CPU: no OMP spinning
    set_precision()
    device = pick_device()
    return load_model("vjepa2", device), device, torch


def extract(args):
    from wm.extract import preprocess
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    out = Path(args.out) / "native"
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(DATASET)
    model, device, torch = _model()
    t_dec = time.time()
    frames = [decode(v) for v in df["video"]]                      # CPU decode outside the GPU lock
    print(f"decoded {len(frames)} clips in {time.time() - t_dec:.0f}s", flush=True)
    with gpu_lock() as waited:
        res = np.zeros((len(df), 4, D), np.float32)
        t0 = time.time()
        for s in range(0, len(df), 16):
            pv = preprocess(frames[s:s + 16]).to(device)
            _, cf, _ = prefix(model, pv[:, :8], [])
            res[s:s + len(pv)] = pool_steps(predict_future(model, cf, 0)).cpu().numpy()
        secs = time.time() - t0
    np.save(out / "pred_pooled_all.npy", res)
    (out / "extract_info.json").write_text(json.dumps({"ids": df["id"].tolist(), "gpu_seconds": secs,
                                                       "lock_wait_s": waited, "batch_size": 16, "mask_index": 0,
                                                       "gpu": torch.cuda.get_device_name(0)}, indent=1))
    print("extract done", secs)


def forward(args):
    from wm.extract import preprocess
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix, suffix
    arrays, info = load_plan(args.out)
    fdir = Path(args.out) / "forward"
    fdir.mkdir(parents=True, exist_ok=True)
    df = load_table(DATASET)
    rows, tg = arrays["carrier_rows"], arrays["targets"]
    C, T = tg.shape
    A = len(ARMS)
    groups = [list(range(s, min(s + args.group, C))) for s in range(0, C, args.group)]
    if VAR != "speed":      # carriers are sorted by row = by value: stride groups so any finished prefix spans all values
        ng = len(groups)
        groups = [list(range(g, C, ng)) for g in range(ng)]
    model, device, torch = _model()
    bs = args.batch_size
    log = []
    only = {int(x) - 1 for x in args.groups.split(",")} if getattr(args, "groups", None) else None
    for g, idx in enumerate(groups):
        path = fdir / f"group_{g:03d}.npz"
        if path.exists() or (only is not None and g not in only):     # --groups: shard across boxes, same groups
            continue
        G = len(idx)
        frames = [decode(df["video"].iloc[rows[i]]) for i in idx]
        twins = [decode(Path(args.out) / "twins" / f"twin_{i:04d}_{j}.mp4") for i in idx for j in range(T)]
        assert [frame_hash(f) for f in twins] == [str(h) for h in arrays["twin_frame_hash"][idx].reshape(-1)]
        with gpu_lock() as waited:
            t0 = time.time()
            pv = preprocess(frames).to(device)
            saved, _, mp_src = prefix(model, pv, list(POINTS))
            ctx_saved, cf, _ = prefix(model, pv[:, :8], list(POINTS))
            src_pp = pool_steps(predict_future(model, cf, 0)).cpu().numpy()
            tw_mp, tw_pp = [], []
            for s in range(0, len(twins), bs):
                tpv = preprocess(twins[s:s + bs]).to(device)
                tw_mp.append(prefix(model, tpv, [])[2])
                tw_pp.append(pool_steps(predict_future(model, prefix(model, tpv[:, :8], [])[1], 0)).cpu().numpy())
            tw_mp = np.concatenate(tw_mp).reshape(G, T, 26, D)
            tw_pp = np.concatenate(tw_pp).reshape(G, T, 4, D)
            res = {"idx": np.array(idx), "src_meanpool": mp_src, "src_pred_pooled": src_pp, "twin_meanpool": tw_mp,
                   "twin_pred_pooled": tw_pp}
            for L in POINTS:
                dl = arrays[f"deltas_L{L}"][idx].astype(np.float64)                          # [G, A, T, D]
                nat = np.linalg.norm(tw_mp[:, :, L].astype(np.float64) - mp_src[:, None, L], axis=-1)   # [G, T]
                ref = {"own": np.linalg.norm(dl, axis=-1), "chord_norm": np.repeat(
                    np.linalg.norm(dl[:, 1], axis=-1)[:, None], A, 1), "natural_norm": np.repeat(nat[:, None], A, 1)}
                conds = CONDS[L]
                Dc = np.stack([rescale(dl, ref[c])[0] for c in conds], 1).astype(np.float32)   # [G, K, A, T, D]
                res[f"scale_L{L}"] = np.stack([rescale(dl, ref[c])[1] for c in conds], 1).astype(np.float32)
                res[f"natural_norm_L{L}"] = nat.astype(np.float32)
                pp = np.zeros((G, len(conds), A, T, 4, D), np.float32)
                jobs = [(a, k, i, j) for a in range(G) for k in range(len(conds)) for i in range(A) for j in range(T)]
                for s in range(0, len(jobs), bs):
                    ch = jobs[s:s + bs]
                    ca = torch.as_tensor([c[0] for c in ch], device=device)
                    dd = torch.as_tensor(np.stack([Dc[a, k, i, j] for a, k, i, j in ch]), device=device)
                    z, _ = edited_prediction(model, ctx_saved[L][ca], L, dd, 0)
                    p = pool_steps(z).cpu().numpy()
                    for q, (a, k, i, j) in enumerate(ch):
                        pp[a, k, i, j] = p[q]
                res[f"edit_pred_pooled_L{L}"] = pp
                em = np.zeros((G, A, T, 26 - L, D), np.float32)                               # full clip, own norm
                jobs = [(a, i, j) for a in range(G) for i in range(A) for j in range(T)]
                for s in range(0, len(jobs), bs):
                    ch = jobs[s:s + bs]
                    ca = torch.as_tensor([c[0] for c in ch], device=device)
                    dd = torch.as_tensor(np.stack([Dc[a, 0, i, j] for a, i, j in ch]), device=device)
                    r = suffix(model, saved[L][ca], L, dd)
                    for q, (a, i, j) in enumerate(ch):
                        em[a, i, j] = r["meanpool"][q]
                res[f"edit_meanpool_L{L}"] = em
            torch.cuda.synchronize()
            secs = time.time() - t0
        res["gpu_seconds"] = np.array([secs])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        log.append({"group": g, "gpu_seconds": secs, "lock_wait_s": waited})
        print(f"group {g + 1}/{len(groups)} {secs:.0f}s gpu (waited {waited:.0f}s)", flush=True)
    (fdir / f"timing_{int(time.time())}.json").write_text(json.dumps(
        {"groups": log, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch_size": bs}, indent=1))


# ================================================================ score (CPU)

def score(args):
    from run_session2 import angle_of, load_groups, sha256_file, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, load_sweep, predict, score as pscore, targets
    out = Path(args.out)
    arrays, info = load_plan(out)
    F = load_groups(out / "forward")
    idx = F["idx"]
    rows, tg = arrays["carrier_rows"][idx], arrays["targets"][idx]
    C, T = tg.shape
    df = load_table(DATASET)
    y = df[CFG["col"]].to_numpy(float)
    th = df["theta_degrees"].to_numpy(float)
    d0 = load_inputs(DATASET, 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    assert not probe[rows].any() and test[rows].all()
    all_probe = probe
    if CFG["excl_held_readouts"]:                                  # no readout sees a clip at a held-out value
        probe = probe & ~np.isin(y, info["held_values"])
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Zall = np.load(out / "native" / "pred_pooled_all.npy").astype(np.float64)       # [1536, 4, D]
    cen = np.load(out / f"centroids_{DATASET}.npy")                                 # [1536, 8, 2]
    cen_tw = np.load(out / "centroids_twins.npy")[idx]                              # [C, T, 8, 2]

    def fit(X, Yt, fl, kind):
        st = Standardizer().fit(X)
        cv = cv_select_alpha(st.transform(X), Yt, fl, score_fn=lambda a, b: pscore(a, b, kind))
        W, b = fit_ridge(st.transform(X), Yt, cv["alpha"])
        return (st, W, b), cv

    def apply(pr, X):
        st, W, b = pr
        sh = X.shape
        return predict(st.transform(X.reshape(-1, sh[-1])), W, b).reshape(sh[:-1] + (-1,))

    # ---- readouts fit on the predictor's own unedited forecasts of probe clips (folds 3-4), scored on test clips
    pos, pos_rows = {}, {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        pos[k], cv = fit(Zall[probe & ok, k], cen[probe & ok, t], folds5[ok[probe]], "linear")
        e = np.linalg.norm(apply(pos[k], Zall[test & ok, k]) - cen[test & ok, t], axis=1)
        pos_rows[str(t)] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "test_px_mean": float(e.mean())}
    read_pos = lambda Z: np.stack([apply(pos[k], Z[..., k, :]) for k in range(4)], -2)   # noqa: E731
    disp_all = motion_from_positions(read_pos(Zall))                                    # uncalibrated
    cal = np.polyfit(disp_all[probe], y[probe], 1)                                      # linear calibration, probe clips
    read_disp = lambda Z: np.polyval(cal, motion_from_positions(read_pos(Z)))           # noqa: E731
    read_vbar = lambda Z: speed_from_positions(read_pos(Z))                             # noqa: E731  mean speed, m/s
    sp_dir, cvs = fit(Zall.mean(1)[probe], y[probe, None], folds5, "linear")
    read_direct = lambda Z: apply(sp_dir, Z.mean(-2))[..., 0]                           # noqa: E731
    Yd = targets(df, "direction")[0]
    dirp, cvd = fit(Zall.mean(1)[probe], Yd[probe], folds5, "circular")
    read_dir = lambda Z: angle_of(apply(dirp, Z.mean(-2)))                              # noqa: E731
    true_disp = motion_from_positions(cen[:, 4:8])
    readouts = {
        "displacement": {"what": ("per-step disk position probes (ridge, one per forecast tubelet 4-7) -> |p7 - p4| / 3 "
                                  "px per tubelet -> m/s" if VAR == "speed" else
                                  "per-step disk position probes (ridge, one per forecast tubelet 4-7) -> least-squares "
                                  "quadratic p(t) = c0 + c1 t + c2 t^2/2, c2 projected on the p4->p7 direction -> m/s^2")
                                 + ", then a linear calibration fit on probe clips' forecasts",
                         "position_probes": pos_rows, "calibration_a_b": cal.tolist(),
                         "test_mae_mps": float(np.abs(read_disp(Zall[test]) - y[test]).mean()),
                         "test_r2": pscore(y[test, None], read_disp(Zall[test])[:, None], "linear")["r2"],
                         f"true_centroid_{VAR}_test_mae_mps": float(np.abs(true_disp[test] - y[test]).mean())},
        "direct": {"what": f"ridge {VAR} probe on the step-mean pooled forecast", "alpha": cvs["alpha"],
                   "cv_r2": cvs["cv_mean"], "test_mae_mps": float(np.abs(read_direct(Zall[test]) - y[test]).mean()),
                   "test_r2": pscore(y[test, None], read_direct(Zall[test])[:, None], "linear")["r2"]},
        "direction": {"what": "ridge [sin, cos] probe on the step-mean forecast (off-target check)", "alpha": cvd["alpha"],
                      "test_circ_mae_deg": float(wrap(read_dir(Zall[test]) - th[test]).mean())},
        "fit_rows": f"{int(probe.sum())} probe clips (folds 3-4), alpha by 5-fold CV inside them; test = {int(test.sum())} clips"}
    if VAR != "speed":
        readouts["fit_rows"] = (f"{int(probe.sum())} of {int(all_probe.sum())} probe clips (folds 3-4), held-out-block "
                                f"values excluded; alpha by 5-fold CV inside them; test = {int(test.sum())} clips")
        readouts["displacement"]["true_centroid_test_r2"] = pscore(y[test, None], true_disp[test][:, None], "linear")["r2"]
        readouts["units_note"] = "*_mps fields of this readout block are in m/s^2 (key names kept from the speed run)"
    parity = {"src_pred_vs_native_extract_rel_maxabs": float(np.abs(F["src_pred_pooled"] - Zall[rows]).max()
                                                              / np.abs(Zall[rows]).max()),
              "src_meanpool_vs_stored_rel_maxabs": float(np.abs(F["src_meanpool"] - arrays["carrier_stored_meanpool"][idx]).max()
                                                         / np.abs(arrays["carrier_stored_meanpool"][idx]).max())}
    src = y[rows]
    Zs, Zt = F["src_pred_pooled"].astype(np.float64), F["twin_pred_pooled"].astype(np.float64)
    ps, pt = read_pos(Zs), read_pos(Zt)
    s0 = {"displacement": read_disp(Zs), "direct": read_direct(Zs)}
    stw = {"displacement": read_disp(Zt), "direct": read_direct(Zt)}
    a0, atw = read_dir(Zs), read_dir(Zt)
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                                  # noqa: E731
    dtrue = tg - src[:, None]
    twin_ref = {}
    for r in ("displacement", "direct"):
        twin_ref[r] = {"unedited_err_to_target": per_clip_ci(np.abs(s0[r][:, None] - tg).mean(1)),
                       "twin_forecast_err_to_target": per_clip_ci(np.abs(stw[r] - tg).mean(1)),
                       f"twin_forecast_R_{VAR}": per_clip_ci(projection_R(stw[r] - s0[r][:, None], dtrue)),
                       "unedited_err_to_true": per_clip_ci(np.abs(s0[r] - src))}
    vs0, vtw = read_vbar(Zs), read_vbar(Zt)
    if VAR != "speed":
        t_mid = float(np.mean((2 * np.arange(4, 8) + 0.5) / FPS))
        twin_ref["twin_mean_speed_change_mps"] = per_clip_ci((vtw - vs0[:, None]).mean(1))
        twin_ref["physical_mean_speed_change_mps"] = per_clip_ci((dtrue * t_mid).mean(1))
    twin_ref["twin_dir_change_deg"] = per_clip_ci(wrap(atw - a0[:, None]).mean(1))
    twin_ref["twin_px_to_twin_true"] = per_clip_ci(np.nanmean(dist(pt, cen_tw[:, :, 4:8]), (1, 2)))
    twin_ref["unedited_px_to_twin_true"] = per_clip_ci(np.nanmean(dist(ps[:, None], cen_tw[:, :, 4:8]), (1, 2)))
    v_far = {L: info["per_point"][str(L)]["v_far_null"] for L in POINTS}

    def arm_metrics(Z, L):                                                               # Z [C, T, 4, D]
        m, pc = {}, {}
        pe = read_pos(Z)
        for r, fn in (("displacement", read_disp), ("direct", read_direct)):
            se = fn(Z)
            pc[r] = np.abs(se - tg).mean(1)
            m[f"{r}_err_to_target"] = per_clip_ci(pc[r])
            m[f"{r}_err_to_twin_forecast"] = per_clip_ci(np.abs(se - stw[r]).mean(1))
            m[f"{r}_R_{VAR}"] = per_clip_ci(projection_R(se - s0[r][:, None], dtrue))
            m[f"{r}_err_to_far_end"] = per_clip_ci(np.abs(se - v_far[L]).mean(1))
        m["R_token_twin_forecast"] = per_clip_ci(projection_R(Z - Zs[:, None], Zt - Zs[:, None], (-1, -2)).mean(1))
        m["px_to_twin_true"] = per_clip_ci(np.nanmean(dist(pe, cen_tw[:, :, 4:8]), (1, 2)))
        m["px_to_twin_forecast"] = per_clip_ci(np.nanmean(dist(pe, pt), (1, 2)))
        m["dir_change_deg"] = per_clip_ci(wrap(read_dir(Z) - a0[:, None]).mean(1))
        if VAR != "speed":
            m["mean_speed_change_mps"] = per_clip_ci((read_vbar(Z) - vs0[:, None]).mean(1))
        return m, pc

    pred, paired = {}, {}
    for L in POINTS:
        P = F[f"edit_pred_pooled_L{L}"].astype(np.float64)                               # [C, K, A, T, 4, D]
        sc = F[f"scale_L{L}"]
        pred[str(L)], paired[str(L)] = {}, {}
        for k, cond in enumerate(CONDS[L]):
            rowc, pcs = {}, {}
            for i, arm in enumerate(ARMS):
                rowc[arm], pcs[arm] = arm_metrics(P[:, k, i], L)
                rowc[arm]["scale_median"] = float(np.median(sc[:, k, i]))
                rowc[arm]["applied_norm_over_natural_median"] = float(np.median(
                    np.linalg.norm(arrays[f"deltas_L{L}"][idx][:, i], axis=-1) * sc[:, k, i] / F[f"natural_norm_L{L}"]))
            pred[str(L)][cond] = rowc
            paired[str(L)][cond] = {f"{a}_minus_{b}_{r}_err": per_clip_ci(pcs[a][r] - pcs[b][r])
                                    for a, b in (("spline", "chord"), ("spline", "linear_raw"), ("chord", "linear_raw"),
                                                 ("null", "chord"))
                                    for r in ("displacement", "direct")}
    # ---- propagation through the encoder (full-clip edits, own norm), speed probe refit per read point
    full = np.load(PROJECT_ROOT / f"artifacts/activations/{DATASET}/vjepa2/meanpool.npy", mmap_mode="r")
    sweep = load_sweep(DATASET, DATASET)
    readers = {}
    for p in range(min(POINTS), 26):
        Xp = np.asarray(full[:, p], np.float64)
        st = Standardizer().fit(Xp[probe])
        readers[p] = (st, *fit_ridge(st.transform(Xp[probe]), y[probe], sweep["layers"][p]["alpha"]))
    rp = lambda X, p: apply(readers[p], X)[..., 0]                                       # noqa: E731
    src_mp, tw_mp = F["src_meanpool"].astype(np.float64), F["twin_meanpool"].astype(np.float64)
    prop = {}
    for L in POINTS:
        E = F[f"edit_meanpool_L{L}"].astype(np.float64)                                 # [C, A, T, 26-L, D]
        pts = list(range(L, 26))
        row = {"read_points": pts,
               "unedited_err_to_target": [per_clip_ci(np.abs(rp(src_mp[:, p], p)[:, None] - tg).mean(1))["mean"] for p in pts],
               "twin_err_to_target": [per_clip_ci(np.abs(rp(tw_mp[:, :, p], p) - tg).mean(1))["mean"] for p in pts]}
        for i, arm in enumerate(ARMS):
            row[arm] = {"err_to_target": [per_clip_ci(np.abs(rp(E[:, i, :, q], p) - tg).mean(1)) for q, p in enumerate(pts)],
                        f"R_{VAR}": [per_clip_ci(projection_R(rp(E[:, i, :, q], p) - rp(src_mp[:, p], p)[:, None], dtrue))
                                    for q, p in enumerate(pts)],
                        "R_act_twin_change": [per_clip_ci(projection_R(E[:, i, :, q] - src_mp[:, None, p],
                                                                       tw_mp[:, :, p] - src_mp[:, None, p]).mean(1))
                                              for q, p in enumerate(pts)]}
        prop[str(L)] = row
    # ---- provenance: GPU seconds + cost
    gpu_s = float(F["gpu_seconds"].sum())
    ext = json.loads((out / "native" / "extract_info.json").read_text())
    cost = json.loads(args.cost) if args.cost else {}
    gpu_total = gpu_s + ext["gpu_seconds"]
    res = {"design": {k: info[k] for k in ("points", "arms", "conditions", "holdout", "held_values", "n_targets",
                                           "carrier_pool", "twins", f"renderer_validation_{DATASET}")},
           "n_carriers": int(C), "geometry": info["per_point"], "readouts": readouts, "parity": parity,
           "twin_reference": twin_ref, "predictor": pred, "paired": paired, "propagation": prop,
           "compute": {"gpu_seconds_forward": gpu_s, "gpu_seconds_extract": ext["gpu_seconds"],
                       "gpu_seconds_total": gpu_total, "box": "vast 53030966 RTX 4060 Ti", **cost,
                       "cost_usd": gpu_total / 3600 * cost["dph_total"] if "dph_total" in cost else None},
           "keys": {
               "predictor.<point>.<condition>.<arm>": "forecast of the edited context (frames 1-8, edit on every "
                   "context token at the point), mean over 4 targets per carrier; per_clip_ci = mean, per-clip SE, "
                   "1000-draw clip bootstrap 95% CI, n carriers",
               "<readout>_err_to_target": "|speed read from the forecast - target speed| (m/s)",
               "<readout>_err_to_twin_forecast": "|speed read from the edited forecast - speed read from the twin's own "
                                                 "forecast| (m/s): counterfactual-twin agreement",
               "<readout>_R_speed": "per carrier sum_t (s_edit - s_unedited)(target - source) / sum_t (target - source)^2",
               "<readout>_err_to_far_end": "|read speed - 0.25| (the null's aim)",
               "R_token_twin_forecast": "<Z_edit - Z_src, Z_twin - Z_src> / ||Z_twin - Z_src||^2 over the 4 pooled "
                                        "forecast steps (probe-free)",
               "px_to_twin_true / px_to_twin_forecast": "native position probes vs the twin's true disk centroids / "
                                                        "vs the positions read from the twin's own forecast",
               "dir_change_deg": "off-target: |direction read from edited forecast - from unedited forecast|",
               "paired.<point>.<cond>.<a>_minus_<b>_<readout>_err": "paired per-carrier difference of err_to_target",
               "propagation.<point>": "full 16-frame clip edited at the point (own norm), meanpool at every later "
                                      "point read by a speed ridge probe refit there on probe clips' stored meanpool "
                                      "(Part 1 sweep alpha); R_act_twin_change = projection on the twin's real change",
               "twin_reference": "the predictor's forecast from the rendered twin's own context (what a perfect edit "
                                 "could reach) and the unedited forecast"}}
    if (out / "forward_provenance.json").exists():                 # forward sharded across boxes (--groups)
        res["compute"]["group_provenance"] = json.loads((out / "forward_provenance.json").read_text())
        res["compute"]["box"] = "RTX 4060 Ti x3 (vast 53030966 extract + groups 1-6; 53252443, 53255819 shards)"
    if VAR != "speed":
        u = CFG["unit"]
        res["keys"].update({
            "<readout>_err_to_target": f"|acceleration read from the forecast - target acceleration| ({u})",
            "<readout>_err_to_twin_forecast": f"|acceleration read from the edited forecast - from the twin's own forecast| ({u})",
            "<readout>_R_acceleration": "per carrier sum_t (a_edit - a_unedited)(target - source) / sum_t (target - source)^2",
            "<readout>_err_to_far_end": f"|read acceleration - {v_far[POINTS[1]]}| (the null's aim)",
            "mean_speed_change_mps": "off-target/coupled: forecast mean speed over steps 4-7 (|p7 - p4| / 3 from the "
                                     "position probes, m/s) edited minus unedited; physics predicts delta_a * t_mid for a "
                                     "true acceleration change (twin_reference.physical_mean_speed_change_mps)",
            "propagation.<point>": "full 16-frame clip edited at the point (own norm), meanpool at every later point read "
                                   "by an acceleration ridge probe refit there on probe clips' stored meanpool outside the "
                                   "held-out block (Part 1 sweep alpha); R_act_twin_change = projection on the twin's change"})
        res["variable"] = VAR
    write(RES / CFG["result"], res, stage=f"session3_{VAR}_score", seeds={"plan": info["seed"],
          "bootstrap": 0, "cv_folds": 0}, plan_sha256=sha256_file(out / "plan.npz"))
    figure(res)
    return res


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"spline": "#3b6fb6", "chord": "#d9822b", "linear_raw": "#6aa84f", "null": "#999999"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    labels, x = [], 0
    hi = POINTS[1]
    for L, conds in ((hi, CONDS[hi]), (12, CONDS[12])):
        for cond in conds:
            for i, arm in enumerate(ARMS):
                r = res["predictor"][str(L)][cond][arm]["displacement_err_to_target"]
                ax[0].bar(x + i * 0.2, r["mean"], 0.2, color=cols[arm], label=arm if x == 0 else None,
                          yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], capsize=2)
            labels.append((x + 0.3, f"pt {L}\n{cond.replace('_norm', '')}"))
            x += 1
    tr = res["twin_reference"]["displacement"]
    ax[0].axhline(tr["unedited_err_to_target"]["mean"], ls="--", c="k", lw=1, label="unedited")
    ax[0].axhline(tr["twin_forecast_err_to_target"]["mean"], ls=":", c="k", lw=1, label="twin's own forecast")
    ax[0].set_xticks([p for p, _ in labels], [s for _, s in labels], fontsize=8)
    ax[0].set_ylabel(f"{VAR} error to target ({CFG['unit']}), displacement readout")
    ax[0].set_title(f"Forecast {VAR} error (95% clip-bootstrap CI)")
    ax[0].legend(fontsize=7)
    for i, arm in enumerate(ARMS):
        r = res["predictor"][str(hi)]["natural_norm"][arm]
        ax[1].bar(i, r["R_token_twin_forecast"]["mean"], color=cols[arm])
        ax[1].text(i, r["R_token_twin_forecast"]["mean"], f"{r['dir_change_deg']['mean']:.1f}°", ha="center",
                   va="bottom", fontsize=8)
    ax[1].set_xticks(range(len(ARMS)), ARMS)
    ax[1].set_title(f"Point {hi}, natural norm: token R on twin forecast\n(labels: off-target direction change)")
    for L, ls in ((12, "-"), (hi, "--")):
        pr = res["propagation"][str(L)]
        for arm in ARMS:
            ax[2].plot(pr["read_points"], [v["mean"] for v in pr[arm]["err_to_target"]], ls, c=cols[arm],
                       label=f"{arm} @{L}")
        ax[2].plot(pr["read_points"], pr["unedited_err_to_target"], ls, c="k", lw=0.8, label=f"unedited @{L}")
    ax[2].set_xlabel("read point")
    ax[2].set_ylabel(f"{VAR} error to target ({CFG['unit']}), probe refit per point")
    ax[2].set_title("Propagation through the encoder (own norm)")
    ax[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / CFG["fig"], dpi=130)
    print("wrote", FIG / CFG["fig"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "extract", "forward", "score"))
    ap.add_argument("--out", default=None, help="default artifacts/session3_<variable> (speed: artifacts/session3_speed)")
    ap.add_argument("--variable", choices=tuple(VARIABLES), default="speed")
    ap.add_argument("--n-carriers", type=int, default=128)
    ap.add_argument("--n-targets", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--groups", default=None, help="forward only: comma list of 1-based group numbers (box sharding)")
    ap.add_argument("--cost", default=None, help="JSON with dph_total etc. from `vastai show instances --raw`")
    a = ap.parse_args()
    configure(a.variable)
    a.out = a.out or str(ART)
    {"plan": plan, "extract": extract, "forward": forward, "score": score}[a.stage](a)

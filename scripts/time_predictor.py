"""Time as a 4th variable: does steering the within-clip TIME code change the PREDICTOR's forecast?

Speed set (constant velocity), edit points 12 and 22, context-only encoding of frames 1-8 (tubelets s = 0..3) as in
session 2/3; the predictor (mask token 0) forecasts tubelets 4..7 and per-step disk-position probes (fit on the
predictor's own unedited forecasts of probe clips, as scripts/session3_speed_predictor.py) read where the forecast
disk is.

Time spline (plan, CPU): full-clip timepool at the point, residual = step feature - clip mean, PCA-64 on knot-clip
(folds 0-2) residual rows, per-step centroids C_t and the count-weighted smoothing spline S(t) over t = 0..7
(continued linearly past the ends), exactly as run_time_manifold.py. Edits are per context step s, added to all 256
tokens of step s (not one vector for all tokens), in raw space (PCA lift), residual kept:
  sp+1, sp+2   S(s+k) - S(s)                      advance the time code by k steps along the spline
  ch+2         C_{s+2} - C_s                      raw-centroid chord (A.9)
  null-2       S(s-2) - S(s), rescaled per step to ||sp+2||      the same time path aimed backwards
  rand+2       a random PCA-64 direction per carrier, rescaled per step to ||sp+2||
  rev_sp       S(3-s) - S(s)                      reverse the context's time order (t -> 3 - t)
  rev_ch       C_{3-s} - C_s
References (real inputs, no edit): unedited; win+1 / win+2 (the same clip's context window shifted k tubelets later:
frames 2k+1 .. 2k+8, so its forecast is tubelets 4+k..7+k); rev_frames (context frames 8..1 reversed: the disk moves
backwards, a real time-reversed input).
Readouts (score, CPU): along-motion forecast advance in px and as a fraction of k tubelets of true displacement
(A = 1: the forecast moved k steps along the trajectory; 0: did not move), perpendicular shift, forecast speed and
direction change, and a context time probe (ridge on the context-only final-point per-step pools of probe clips,
target s) whose mean change should be +k if the pooled time readout moves.

Stages:  plan (CPU) -> ART/plan.npz;  native (GPU) -> ART/native_ctx_steps.npz;  forward (GPU) -> ART/forward/*.npz;
         score (CPU) -> results/p5_time_predictor.json
  python scripts/time_predictor.py plan|native|forward|score [--out ART]
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
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

POINTS = (12, 22)
ARMS = ("sp+1", "sp+2", "ch+2", "null-2", "rand+2", "rev_sp", "rev_ch")
ARM_K = {"sp+1": 1, "sp+2": 2, "ch+2": 2, "null-2": -2, "rand+2": 2, "rev_sp": 0, "rev_ch": 0}
REFS = ("win+1", "win+2", "rev_frames")
REF_K = {"win+1": 1, "win+2": 2, "rev_frames": 0}
D, FPS, PX_PER_M = 1024, 24.0, 32.0
STEP_S = 2 / FPS
LOCK = "/tmp/wm_gpu.lock"
ART = PROJECT_ROOT / "artifacts" / "time_predictor"


@contextmanager
def gpu_lock(path=LOCK):
    import os
    if os.environ.get("TP_NOLOCK") == "1":            # CPU run: the GPU lock is not needed
        yield 0.0
        return
    with open(path, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield time.time() - t0
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------- pure helpers (tests/test_time_predictor.py)

def step_deltas(S, C, k_steps=(1, 2), n_ctx=4):
    """Per-step PCA-space deltas [n_ctx, k] for the deterministic arms, from a spline S(t) (callable, [..] -> [.., k])
    and raw centroids C [8, k]. rand+2 is added per carrier by the caller."""
    s = np.arange(n_ctx, dtype=float)
    out = {"sp+1": S(s + 1) - S(s), "sp+2": S(s + 2) - S(s), "ch+2": C[2:2 + n_ctx] - C[:n_ctx]}
    back = S(s - 2) - S(s)
    out["null-2"] = back * (np.linalg.norm(out["sp+2"], axis=1) / np.maximum(np.linalg.norm(back, axis=1), 1e-12))[:, None]
    out["rev_sp"] = S(n_ctx - 1 - s) - S(s)
    out["rev_ch"] = C[n_ctx - 1 - np.arange(n_ctx)] - C[:n_ctx]
    return out


def along_across(dp, theta_deg):
    """Pixel displacement [..., 2] (col, row) -> (along the motion, across it); row grows downwards (y up in metres)."""
    th = np.radians(np.asarray(theta_deg, float))[..., None]
    u = np.concatenate([np.cos(th), -np.sin(th)], -1)
    n = np.concatenate([np.sin(th), np.cos(th)], -1)
    return (dp * u).sum(-1), (dp * n).sum(-1)


def step_px(speed):
    """True displacement per tubelet step in px."""
    return np.asarray(speed, float) * STEP_S * PX_PER_M


def heading_deg(pos):
    """Heading (deg, y up) of the forecast motion p_last - p_first from positions [..., S, 2] in px."""
    d = pos[..., -1, :] - pos[..., 0, :]
    return np.degrees(np.arctan2(-d[..., 1], d[..., 0])) % 360


# ---------------------------------------------------------------- plan (CPU)

def plan(args):
    from importlib.util import module_from_spec, spec_from_file_location
    from wm.p2_data import load_inputs
    spec = spec_from_file_location("tmr", Path(__file__).with_name("run_time_manifold.py"))
    tmr = module_from_spec(spec)
    spec.loader.exec_module(tmr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d0 = load_inputs("speed", 0)
    df, role = d0["df"], d0["role"]
    v, th = df["speed_mps"].to_numpy(float), df["theta_degrees"].to_numpy(float)
    m = tmr.meta("speed")
    pos_m = m["pos"]                                               # [N, 8, 2] metres (tubelet midpoints)
    rng = np.random.default_rng(args.seed)
    # carriers: test clips, speed >= 1 m/s, disk in frame (8 px margin) through tubelet 9 (win+2's last forecast step)
    tau = (2 * np.arange(10) + 0.5) / FPS
    u = np.stack([np.cos(np.radians(th)), np.sin(np.radians(th))], 1)
    p10 = np.stack([df["start_x"], df["start_y"]], 1)[:, None] + (v[:, None, None] * tau[None, :, None]) * u[:, None]
    px = np.stack([128 + PX_PER_M * p10[..., 0], 128 - PX_PER_M * p10[..., 1]], -1)
    inframe = ((px > 8) & (px < 248)).all((1, 2))
    pool = np.flatnonzero((role == "test") & (v >= 1.0) & inframe)
    carriers = np.sort(rng.choice(pool, size=min(args.n_carriers, len(pool)), replace=False))
    arrays = {"carrier_rows": carriers, "carrier_ids": df["id"].to_numpy()[carriers]}
    info = {"points": list(POINTS), "arms": list(ARMS), "refs": list(REFS), "n_carriers": int(len(carriers)),
            "carrier_pool": "speed-set test clips, speed >= 1 m/s, disk centre >= 8 px inside the frame through tubelet 9",
            "n_pool": int(len(pool)), "seed": args.seed, "per_point": {}}
    for L in POINTS:
        F = np.asarray(np.load(PROJECT_ROOT / "artifacts/activations/speed/vjepa2/timepool.npy", mmap_mode="r")[:, L],
                       np.float64)
        R = tmr.tm.remove_clip_mean(F)
        g = tmr.fit_time_geometry(R, m, role == "knot", 64)
        pca, S, C = g["pca"], g["spline"], g["cent"]["C"]
        dl = step_deltas(S, C)
        dz = np.stack([dl[a] if a != "rand+2" else np.zeros_like(dl["sp+2"]) for a in ARMS])          # [A, 4, 64]
        raw = np.einsum("ask,kd->asd", dz, pca.components).astype(np.float32)                         # [A, 4, D]
        rnd = rng.standard_normal((len(carriers), 4, dz.shape[-1]))
        rnd *= (np.linalg.norm(dl["sp+2"], axis=1)[None, :] / np.linalg.norm(rnd, axis=-1))[..., None]
        arrays[f"deltas_L{L}"] = raw
        arrays[f"rand_L{L}"] = np.einsum("csk,kd->csd", rnd, pca.components).astype(np.float32)       # [C, 4, D]
        info["per_point"][str(L)] = {
            "delta_norm_per_step": {a: np.linalg.norm(raw[i], axis=-1).round(3).tolist() for i, a in enumerate(ARMS)},
            "cos_sp2_vs_ch2_per_step": [float(raw[1, s] @ raw[2, s] / (np.linalg.norm(raw[1, s]) * np.linalg.norm(raw[2, s])))
                                        for s in range(4)],
            "centroid_step_norm": np.linalg.norm(np.diff(C, axis=0), axis=1).round(3).tolist(),
            "clip_residual_rms": float(np.sqrt((R[role == "knot"] ** 2).sum(-1).mean()))}
        arrays[f"C_L{L}"] = C.astype(np.float32)
        arrays[f"pca_L{L}"] = pca.components.astype(np.float32)
    np.savez(out / "plan.npz", **arrays)
    (out / "plan.json").write_text(json.dumps(info, indent=1))
    print("plan", out, len(carriers), "carriers")


# ---------------------------------------------------------------- GPU stages

def _model():
    import torch
    from wm.extract import load_model, pick_device, set_precision
    import os
    torch.set_num_threads(int(os.environ.get("TP_THREADS", "4")))
    set_precision()
    dev = pick_device()
    return load_model("vjepa2", dev), dev, torch


def finish(model, h, L, dtok=None):
    """Add per-token deltas dtok [B, N, D] to the residual stream h at point L, run blocks L+1..24 and the final LN."""
    enc = model.encoder
    if dtok is not None:
        h = h + dtok
    for layer in enc.layer[L:]:
        h = layer(h, None, None, False)[0]
    return enc.layernorm(h)


def native(args):
    """Context-only encoding of all speed clips: per-step pools (steps 0..3) at points 12, 22, 25 (final LN)."""
    from wm.extract import preprocess
    from wm.propagate import prefix
    from wm.p2_data import load_inputs
    role = load_inputs("speed", 0)["role"]
    df = load_table("speed")
    rng = np.random.default_rng(0)
    knot = np.flatnonzero(role == "knot")
    rows = np.sort(np.r_[np.flatnonzero(role == "probe"), rng.choice(knot, args.n_native_knot, replace=False)])
    si, sn = (int(x) for x in args.shard.split("/"))
    rows = rows[si::sn]
    df = df.iloc[rows]
    model, dev, torch = _model()
    frames = [decode(v) for v in df["video"]]
    res = {f"L{p}": np.zeros((len(df), 4, D), np.float16) for p in (*POINTS, 25)}
    with gpu_lock() as waited, torch.no_grad():
        t0 = time.time()
        for s in range(0, len(df), 8):
            pv = preprocess(frames[s:s + 8]).to(dev)[:, :8]
            saved, final, _ = prefix(model, pv, list(POINTS))
            for p in POINTS:
                res[f"L{p}"][s:s + len(pv)] = saved[p].reshape(len(pv), 4, 256, D).mean(2).cpu().numpy()
            res["L25"][s:s + len(pv)] = final.reshape(len(pv), 4, 256, D).mean(2).cpu().numpy()
        secs = time.time() - t0
    np.savez(Path(args.out) / f"native_ctx_steps_{si}of{sn}.npz", rows=rows, ids=df["id"].to_numpy(), gpu_seconds=secs,
             lock_wait_s=waited, device=str(dev), **res)
    print("native", secs)


def forward(args):
    from wm.extract import preprocess
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    out = Path(args.out)
    z = np.load(out / "plan.npz")
    rows = z["carrier_rows"]
    df = load_table("speed")
    (out / "forward").mkdir(exist_ok=True)
    model, dev, torch = _model()
    G = args.group
    si, sn = (int(x) for x in args.shard.split("/"))
    done = 0
    starts = list(range(0, len(rows), G))
    for g0 in (starts[::-1] if args.reverse else starts):
        path = out / "forward" / f"group_{g0 // G:03d}.npz"
        if path.exists() or (g0 // G) % sn != si:
            continue
        if args.max_groups is not None and done >= args.max_groups:
            break
        done += 1
        idx = np.arange(g0, min(g0 + G, len(rows)))
        frames = [decode(df["video"].iloc[rows[i]]) for i in idx]
        with gpu_lock() as waited, torch.no_grad():
            t0 = time.time()
            pv = preprocess(frames).to(dev)
            B = len(idx)
            saved, final, _ = prefix(model, pv[:, :8], list(POINTS))
            res = {"idx": idx, "unedited_pred": pool_steps(predict_future(model, final, 0)).cpu().numpy(),
                   "unedited_ctx": final.reshape(B, 4, 256, D).mean(2).cpu().numpy()}
            for r, sl in (("win+1", slice(2, 10)), ("win+2", slice(4, 12))):
                _, fr, _ = prefix(model, pv[:, sl], [])
                res[f"{r}_pred"] = pool_steps(predict_future(model, fr, 0)).cpu().numpy()
                res[f"{r}_ctx"] = fr.reshape(B, 4, 256, D).mean(2).cpu().numpy()
            _, fr, _ = prefix(model, pv[:, :8].flip(1).contiguous(), [])
            res["rev_frames_pred"] = pool_steps(predict_future(model, fr, 0)).cpu().numpy()
            res["rev_frames_ctx"] = fr.reshape(B, 4, 256, D).mean(2).cpu().numpy()
            for L in POINTS:
                dl = torch.as_tensor(z[f"deltas_L{L}"], device=dev)                     # [A, 4, D]
                rnd = torch.as_tensor(z[f"rand_L{L}"][idx], device=dev)                 # [B, 4, D]
                pred = np.zeros((B, len(ARMS), 4, D), np.float32)
                ctx = np.zeros((B, len(ARMS), 4, D), np.float32)
                for a, arm in enumerate(ARMS):
                    step_d = rnd if arm == "rand+2" else dl[a][None].expand(B, 4, D)
                    dtok = step_d[:, :, None, :].expand(B, 4, 256, D).reshape(B, 1024, D)
                    fe = finish(model, saved[L], L, dtok)
                    pred[:, a] = pool_steps(predict_future(model, fe, 0)).cpu().numpy()
                    ctx[:, a] = fe.reshape(B, 4, 256, D).mean(2).cpu().numpy()
                res[f"edit_pred_L{L}"], res[f"edit_ctx_L{L}"] = pred, ctx
            if dev.type == "cuda":
                torch.cuda.synchronize()
            res["device"] = np.array([str(dev)])
            res["gpu_seconds"] = np.array([time.time() - t0])
            res["lock_wait_s"] = np.array([waited])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        print(f"group {g0 // G} {res['gpu_seconds'][0]:.0f}s", flush=True)


# ---------------------------------------------------------------- score (CPU)

def score(args):
    from wm import timeman as tm
    from wm.p2_data import load_inputs
    from wm.provenance import provenance
    out = Path(args.out)
    z = np.load(out / "plan.npz")
    info = json.loads((out / "plan.json").read_text())
    files = sorted((out / "forward").glob("group_*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    fw = [np.load(f) for f in files]
    F = {k: np.concatenate([f[k] for f in fw]) for k in fw[0].files if k not in ("gpu_seconds", "lock_wait_s", "device")}
    idx = F["idx"]
    rows = z["carrier_rows"][idx]
    d0 = load_inputs("speed", 0)
    df, role = d0["df"], d0["role"]
    v, th = df["speed_mps"].to_numpy(float), df["theta_degrees"].to_numpy(float)
    probe = role == "probe"
    # position probes on the predictor's unedited native forecasts (session 3 artifacts), one per forecast step 4-7
    Zall = np.load(PROJECT_ROOT / "artifacts/session3_speed/native/pred_pooled_all.npy").astype(np.float64)
    cen = np.load(PROJECT_ROOT / "artifacts/session3_speed/centroids_speed.npy")
    pos_models, pos_info = [], {}
    for k, t in enumerate(range(4, 8)):
        ok = probe & np.isfinite(cen[:, t]).all(1)
        mdl = tm.ridge(Zall[ok, k], cen[ok, t])
        te = (role == "test") & np.isfinite(cen[:, t]).all(1)
        pos_models.append(mdl)
        pos_info[str(t)] = {"test_px_err_mean": float(np.linalg.norm(mdl.predict(Zall[te, k]) - cen[te, t], axis=1).mean())}
    read_pos = lambda P: np.stack([pos_models[k].predict(P[..., k, :].reshape(-1, D)).reshape(P.shape[:-2] + (2,))  # noqa: E731
                                   for k in range(4)], -2)
    parity = float(np.abs(F["unedited_pred"] - Zall[rows]).max() / np.abs(Zall[rows]).max())
    # context time probe: context-only final-point per-step pools of probe clips, target s = 0..3
    shards = [np.load(f) for f in sorted(out.glob("native_ctx_steps_*of*.npz"))]
    have = np.zeros(len(df), bool)
    nat = {f"L{p}": np.full((len(df), 4, D), np.nan) for p in (*POINTS, 25)}
    for sh in shards:
        have[sh["rows"]] = True
        for key in nat:
            nat[key][sh["rows"]] = sh[key]
    X25 = nat["L25"]
    pr_rows = np.flatnonzero(probe & have)
    folds = np.arange(len(pr_rows)) % 5
    oof = np.zeros((len(pr_rows), 4))
    for f in range(5):
        mdl = tm.ridge(X25[pr_rows[folds != f]].reshape(-1, D), np.tile(np.arange(4), (folds != f).sum()))
        oof[folds == f] = mdl.predict(X25[pr_rows[folds == f]].reshape(-1, D)).reshape(-1, 4)
    tp_err = float(np.abs(oof - np.arange(4)[None]).mean())
    tprobe = tm.ridge(X25[pr_rows].reshape(-1, D), np.tile(np.arange(4), len(pr_rows)))
    read_t = lambda X: tprobe.predict(X.reshape(-1, D)).reshape(X.shape[:-1]).mean(-1)                             # noqa: E731
    sp = step_px(v[rows])
    p0 = read_pos(F["unedited_pred"])
    h0 = heading_deg(p0)
    t0 = read_t(F["unedited_ctx"])
    speed_fc = lambda p: np.linalg.norm(p[..., -1, :] - p[..., 0, :], axis=-1) / 3 / PX_PER_M / STEP_S           # noqa: E731
    v0 = speed_fc(p0)

    def metrics(pred, ctx, k):
        p = read_pos(pred)
        al, ac = along_across(p - p0, th[rows][:, None])
        al, ac = al.mean(-1), np.abs(ac).mean(-1)
        m = {"along_px": tm.cluster_bootstrap(al), "across_abs_px": tm.cluster_bootstrap(ac),
             "forecast_speed_change_mps": tm.cluster_bootstrap(speed_fc(p) - v0),
             "heading_change_abs_deg": tm.cluster_bootstrap(np.abs((heading_deg(p) - h0 + 180) % 360 - 180)),
             "frac_heading_flipped_gt90": float((np.abs((heading_deg(p) - h0 + 180) % 360 - 180) > 90).mean()),
             "ctx_time_readout_change_steps": tm.cluster_bootstrap(read_t(ctx) - t0)}
        if k:
            m["advance_fraction"] = tm.cluster_bootstrap(al / (k * sp))
        return m, al

    res = {"design": info, "n_carriers_scored": int(len(idx)), "parity_unedited_vs_native_rel_maxabs": parity,
           "readouts": {"position_probes": pos_info, "ctx_time_probe_cv_mae_steps_probe_clips": tp_err,
                        "true_step_px_carriers_mean": float(sp.mean())},
           "references": {}, "arms": {}, "gaps": {}}
    for r in REFS:
        res["references"][r], _ = metrics(F[f"{r}_pred"], F[f"{r}_ctx"], REF_K[r])
    for L in POINTS:
        res["arms"][str(L)], als = {}, {}
        for a, arm in enumerate(ARMS):
            res["arms"][str(L)][arm], als[arm] = metrics(F[f"edit_pred_L{L}"][:, a], F[f"edit_ctx_L{L}"][:, a], ARM_K[arm])
        res["gaps"][str(L)] = {f"sp+2_minus_{o}_along_px": tm.cluster_bootstrap(als["sp+2"] - als[o])
                               for o in ("ch+2", "null-2", "rand+2")}
        res["gaps"][str(L)]["sp+2_minus_win+2_along_px"] = tm.cluster_bootstrap(
            als["sp+2"] - metrics(F["win+2_pred"], F["win+2_ctx"], 2)[1])
    # direction of the context-only time path vs the full-clip spline direction (does the edit live in ctx space?)
    res["ctx_space_check"] = {}
    for L in POINTS:
        Xc = nat[f"L{L}"]
        Rc = Xc - Xc.mean(1, keepdims=True)
        knot = (role == "knot") & have
        Cc = Rc[knot].mean(0)                                              # [4, D] ctx-only step centroids
        dc = np.diff(Cc, axis=0)
        comp = z[f"pca_L{L}"].astype(np.float64)
        Cfull = z[f"C_L{L}"].astype(np.float64) @ comp                     # [8, D]
        df_ = np.diff(Cfull[:4], axis=0)
        res["ctx_space_check"][str(L)] = {
            "cos_ctx_vs_full_step_differences": [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(dc, df_)],
            "ctx_step_share": float(1 - ((Rc[probe & have] - Cc[None]) ** 2).sum() / (Rc[probe & have] ** 2).sum())}
    res["gpu"] = {"devices": sorted({str(f["device"][0]) for f in fw if "device" in f.files}), "forward_gpu_seconds": float(sum(f["gpu_seconds"][0] for f in fw)),
                  "native_seconds": float(sum(float(sh["gpu_seconds"]) for sh in shards)),
                  "native_rows": int(have.sum())}
    res["keys"] = {
        "arms[point][arm]": "along_px: mean over forecast tubelets 4-7 of the forecast disk displacement along the clip's "
                            "motion, edited minus unedited (px, position probes); advance_fraction = along_px / (k * true "
                            "px per tubelet) (1 = advanced k steps; negative for null-2 means moved backwards as aimed); "
                            "across_abs_px: |perpendicular shift|; forecast_speed_change_mps; heading_change_abs_deg of "
                            "p7 - p4; ctx_time_readout_change_steps: context time probe (final point, per step, mean over "
                            "steps 0-3), edited minus unedited; CIs: clip bootstrap",
        "references[ref]": "real inputs: win+k = the clip's context window k tubelets later; rev_frames = context "
                           "frames reversed; same metrics vs the unedited forecast",
        "gaps[point]": "paired along_px differences (clip bootstrap)",
        "ctx_space_check": "cosine between the context-only step-centroid differences (steps 0-3) and the full-clip "
                           "ones the edits use; ctx_step_share = shared-curve variance share in context-only encodings"}
    res["provenance"] = provenance(seeds={"plan": info["seed"], "bootstrap": 0}, points=list(POINTS),
                                   forward_files=[f.name for f in files])
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "native", "forward", "score"])
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--result", default=str(PROJECT_ROOT / "results" / "p5_time_predictor.json"))
    ap.add_argument("--n-carriers", type=int, default=128)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="forward: process groups from the last one down")
    ap.add_argument("--n-native-knot", type=int, default=240)
    ap.add_argument("--max-groups", type=int, default=None, help="forward: stop after this many new groups (smoke)")
    ap.add_argument("--shard", type=str, default="0/1", help="forward: i/n, run groups with index %% n == i")
    args = ap.parse_args()
    {"plan": plan, "native": native, "forward": forward, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()

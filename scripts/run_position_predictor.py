"""Position through the predictor: does a start-position (x, y) edit made inside the encoder move where V-JEPA 2's
predictor forecasts the disk? The untested cell of the variables table (position is a plane in the encoder and was
steered there by scripts/run_position_sheet.py; nobody had pushed it through the predictor).

Design (speed set, split_v1 roles, same geometry as run_position_sheet.py at seed 0):
  grid       6 x 6 start-position cells over the knot rows' range; held-out interior 2 x 2 block held_block(6, 0)
             = cells [15, 16, 21, 22]. PCA-64 and every edit are built from knot clips (folds 0-2) outside the block.
  carriers   n test clips outside the held block (seed 0), each steered to 2 of the 4 held cells (target = cell centre).
  arms (additive at block L in {12, 22}, same delta on every token, residual kept)
             sheet          lift(TPS(target) - TPS(source position)), the thin-plate sheet of run_position_sheet.py
                            (smoothing chosen by its own held-out-block reconstruction rule, per layer)
             chord          lift(lin(target) - lin(source)), Delaunay piecewise-linear interpolant through the kept cell
                            centroids (the straight chord between neighbouring centroids; the sheet script's
                            chord_linear_interp, not endpoint-matched)
             zero           the unedited forecast (score time, no extra pass)
             twin           the carrier re-rendered with its start moved to the target cell centre (same speed and
                            direction): the physical ceiling, forecast from the twin's own context
  forward    context frames 1-8 encoded alone, delta added at block L, encoder finished, predictor (mask 0) -> pooled
             forecast tubelets 4-7 (as run_velocity_sheet_predictor / session3). Also the full 16-frame clip edited at
             block L and propagated to the final LN (meanpool at points L..25) for the encoder-side readout.
  judge      forecast: per-tubelet ridge probes (native predictor forecasts of probe clips, artifacts/session3_speed/
             native, same model/mask/pooling; never an edited forecast) -> disk pixel centroid; distance to the twin's
             rendered disk centroid (px, 256 px frame, 32 px/m). Encoder: start (x, y) ridge probe per point on stored
             full-clip meanpool of probe clips, read on the edited clip at points L..25 (exit = point 24 / 25).

  python scripts/run_position_predictor.py plan|forward|score [--out DIR] [--n-carriers 48]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, disk_pixels, frame_hash, load_table  # noqa: E402

RES = PROJECT_ROOT / "results"
ART = PROJECT_ROOT / "artifacts" / "session2" / "position_pred"
NATIVE = PROJECT_ROOT / "artifacts" / "session3_speed" / "native"
LAYERS = (12, 22)
ARMS = ("sheet", "chord")
D = 1024
PX_PER_M = 32.0
G_CELLS = 6


def sheet_mod():
    spec = importlib.util.spec_from_file_location("run_position_sheet", Path(__file__).with_name("run_position_sheet.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_clip_ci(v, n=1000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "se": None, "ci95": None, "n": 0}
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


# ================================================================ plan (CPU, Mac)

def plan(args):
    from session3_speed_predictor import tubelet_centroids
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.render_twin import load_meta, write_twin
    ps = sheet_mod()
    out = Path(args.out)
    (out / "twins").mkdir(parents=True, exist_ok=True)
    d0 = load_inputs("speed", LAYERS[0])
    df = d0["df"]
    pos = df[["start_x", "start_y"]].to_numpy(float)
    knot_all, probe, test = d0["role"] == "knot", d0["role"] == "probe", d0["role"] == "test"
    lo, hi = pos[knot_all].min(0).min(), pos[knot_all].max(0).max()
    cell, centres = ps.cells(pos, lo, hi, G_CELLS)
    held = ps.held_block(G_CELLS, args.seed)
    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero(test & ~np.isin(cell, held))
    carriers = np.sort(rng.choice(pool, size=min(args.n_carriers, len(pool)), replace=False))
    C, T = len(carriers), 2
    tcell = np.stack([rng.choice(held, size=T, replace=False) for _ in carriers])        # [C, T]
    tpos = centres[tcell]                                                                  # [C, T, 2]
    src = pos[carriers]
    scell = cell[carriers]
    arrays = {"carrier_rows": carriers, "carrier_ids": df["id"].to_numpy()[carriers], "target_cell": tcell,
              "target_pos": tpos, "source_cell": scell, "source_pos": src}
    info = {"layers": list(LAYERS), "arms": list(ARMS), "seed": args.seed, "n_carriers": C, "n_targets": T,
            "grid": {"G": G_CELLS, "range": [float(lo), float(hi)], "held_out_cells": held,
                     "held_out_centres": centres[held].tolist()},
            "carrier_pool": "test clips outside the held 2 x 2 block", "carrier_pool_size": int(len(pool)),
            "per_layer": {}}
    for L in LAYERS:
        t0 = time.time()
        d = load_inputs("speed", L)
        X = d["X"].astype(np.float64)
        tr = d["is_train"]
        sel_seeds, _ = ps.selection_blocks(G_CELLS, args.seed)
        smoothing, note = ps.choose_smoothing(X[tr], cell[tr], centres, G_CELLS, 64, sel_seeds, exclude=held)
        knot = knot_all & ~np.isin(cell, held)
        pca = mf.fit_pca(X[knot], 64)
        kept = np.unique(cell[knot])
        Ck = ps.cell_centroids(pca.project(X[knot]), cell[knot], kept)
        from scipy.interpolate import RBFInterpolator
        tps = RBFInterpolator(centres[kept], Ck, kernel="thin_plate_spline", smoothing=smoothing)
        lin, lin_count = ps.linear_interp(centres[kept], Ck)
        clo, chi = centres.min(0), centres.max(0)
        lin_at = lambda P: lin(np.clip(P, clo, chi))                                       # noqa: E731
        deltas = np.zeros((C, len(ARMS), T, D), np.float32)
        Ts = tps(src)
        Ls = lin_at(src)
        for j in range(T):
            deltas[:, 0, j] = pca.lift_delta(tps(tpos[:, j]) - Ts)
            deltas[:, 1, j] = pca.lift_delta(lin_at(tpos[:, j]) - Ls)
        arrays[f"deltas_L{L}"] = deltas
        # same-layer readout (the sheet script's probe model: xy ridge on probe folds at L)
        pm = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(X[probe], pos[probe])
        Xc = X[carriers]
        same = {}
        for i, a in enumerate(ARMS):
            P = pm.predict((Xc[:, None] + deltas[:, i]).reshape(-1, D)).reshape(C, T, 2)
            same[a] = {"err_to_target_m": float(np.linalg.norm(P - tpos, axis=-1).mean()),
                       "delta_norm_median": float(np.median(np.linalg.norm(deltas[:, i], axis=-1)))}
        P0 = pm.predict(Xc)
        same["zero"] = {"err_to_target_m": float(np.linalg.norm(P0[:, None] - tpos, axis=-1).mean()),
                        "err_to_source_m": float(np.linalg.norm(P0 - src, axis=-1).mean())}
        info["per_layer"][str(L)] = {"tps_smoothing": smoothing, "tps_smoothing_choice": note,
                                     "linear_interp_nan_fallbacks": lin_count["nan_fallbacks"],
                                     "same_layer_readout": same, "seconds": round(time.time() - t0, 1)}
        print(f"L{L}: smoothing {smoothing}; same-layer err to target (m) " +
              ", ".join(f"{a} {same[a]['err_to_target_m']:.3f}" for a in list(ARMS) + ["zero"]), flush=True)
    # twins: start moved to the target cell centre, same speed / direction / acceleration
    hashes, cen_tw, vis = [], np.zeros((C, T, 8, 2)), []
    for i, c in enumerate(carriers):
        meta = load_meta("speed", int(df["id"].iloc[c]))
        for j in range(T):
            m2 = {**meta, "start_position_xy_m": [float(tpos[i, j, 0]), float(tpos[i, j, 1])]}
            fr = write_twin(m2, float(meta["theta_degrees"]), out / "twins" / f"twin_{i:04d}_{j}.mp4")
            hashes.append(frame_hash(fr))
            cen_tw[i, j] = tubelet_centroids(fr)
            vis.append(float((disk_pixels(fr).reshape(16, -1).sum(1) > 0).mean()))
    np.save(out / "centroids_twins.npy", cen_tw)
    arrays["twin_frame_hash"] = np.array(hashes).reshape(C, T)
    full = np.load(PROJECT_ROOT / "artifacts" / "activations" / "speed" / "vjepa2" / "meanpool.npy", mmap_mode="r")
    arrays["carrier_stored_meanpool"] = np.asarray(full[carriers], np.float32)                # [C, 26, D]
    info["twins"] = {"dir": "twins/twin_{carrier:04d}_{target}.mp4", "visible_frac_mean": float(np.mean(vis)),
                     "visible_frac_min": float(np.min(vis)),
                     "def": "carrier metadata with start_position_xy_m replaced by the target cell centre"}
    np.savez(out / "plan.npz", **arrays)
    (out / "plan.json").write_text(json.dumps(info, indent=1, default=float))
    print("plan written", out, info["twins"], flush=True)


def load_plan(out):
    z = np.load(Path(out) / "plan.npz")
    return {k: z[k] for k in z.files}, json.loads((Path(out) / "plan.json").read_text())


# ================================================================ forward (GPU, box)

def forward(args):
    from session3_speed_predictor import _model, gpu_lock
    from wm.extract import preprocess
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix, suffix
    arrays, info = load_plan(args.out)
    fdir = Path(args.out) / "forward"
    fdir.mkdir(parents=True, exist_ok=True)
    df = load_table("speed")
    rows = arrays["carrier_rows"]
    C, A, T = arrays[f"deltas_L{LAYERS[0]}"].shape[:3]
    lim = C if args.limit is None else min(args.limit, C)
    groups = [list(range(s, min(s + args.group, lim))) for s in range(0, lim, args.group)]
    model, device, torch = _model()
    bs = args.batch_size
    log = []
    for g, idx in enumerate(groups):
        path = fdir / f"group_{g:03d}.npz"
        if path.exists():
            continue
        G = len(idx)
        frames = [decode(df["video"].iloc[rows[i]]) for i in idx]
        twins = [decode(Path(args.out) / "twins" / f"twin_{i:04d}_{j}.mp4") for i in idx for j in range(T)]
        assert [frame_hash(f) for f in twins] == [str(h) for h in arrays["twin_frame_hash"][idx].reshape(-1)]
        with gpu_lock() as waited:
            t0 = time.time()
            pv = preprocess(frames).to(device)
            full_saved, _, src_mp = prefix(model, pv, list(LAYERS))
            ctx_saved, cf, ctx_mp = prefix(model, pv[:, :8], list(LAYERS))
            src_pp = pool_steps(predict_future(model, cf, 0)).cpu().numpy()
            tw_pp, tw_mp = [], []
            for s in range(0, len(twins), bs):
                tpv = preprocess(twins[s:s + bs]).to(device)
                tw_mp.append(prefix(model, tpv, [])[2])
                tw_pp.append(pool_steps(predict_future(model, prefix(model, tpv[:, :8], [])[1], 0)).cpu().numpy())
            res = {"idx": np.array(idx), "src_pred_pooled": src_pp, "src_meanpool": src_mp, "src_ctx_meanpool": ctx_mp,
                   "twin_pred_pooled": np.concatenate(tw_pp).reshape(G, T, 4, D),
                   "twin_meanpool": np.concatenate(tw_mp).reshape(G, T, 26, D)}
            for L in LAYERS:
                dl = arrays[f"deltas_L{L}"][idx]                                               # [G, A, T, D]
                pp = np.zeros((G, A, T, 4, D), np.float32)
                cmp_ = np.zeros((G, A, T, 26 - L, D), np.float32)
                fmp = np.zeros((G, A, T, 26 - L, D), np.float32)
                jobs = [(a, i, j) for a in range(G) for i in range(A) for j in range(T)]
                for s in range(0, len(jobs), bs):
                    ch = jobs[s:s + bs]
                    ca = torch.as_tensor([c[0] for c in ch], device=device)
                    dd = torch.as_tensor(np.stack([dl[a, i, j] for a, i, j in ch]), device=device)
                    z, cm = edited_prediction(model, ctx_saved[L][ca], L, dd, 0)
                    p = pool_steps(z).cpu().numpy()
                    fm = suffix(model, full_saved[L][ca], L, dd)["meanpool"]
                    for q, (a, i, j) in enumerate(ch):
                        pp[a, i, j], cmp_[a, i, j], fmp[a, i, j] = p[q], cm[q], fm[q]
                res[f"edit_pred_pooled_L{L}"] = pp
                res[f"edit_ctx_meanpool_L{L}"] = cmp_
                res[f"edit_meanpool_L{L}"] = fmp
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


# ================================================================ score (CPU, Mac)

def score(args):
    from run_session2 import load_groups, sha256_file, write
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score as pscore
    out = Path(args.out)
    arrays, info = load_plan(out)
    F = load_groups(out / "forward")
    idx = F["idx"]
    rows = arrays["carrier_rows"][idx]
    tpos, src = arrays["target_pos"][idx], arrays["source_pos"][idx]
    C, T = tpos.shape[:2]
    df = load_table("speed")
    pos = df[["start_x", "start_y"]].to_numpy(float)
    d0 = load_inputs("speed", 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    assert test[rows].all() and not probe[rows].any()
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Zall = np.load(NATIVE / "pred_pooled_all.npy").astype(np.float64)
    ext = json.loads((NATIVE / "extract_info.json").read_text())
    assert ext["ids"] == [int(i) for i in df["id"]]
    cen = np.load(PROJECT_ROOT / "artifacts" / "session3_speed" / "centroids_speed.npy")          # [N, 8, 2] px
    cen_tw = np.load(out / "centroids_twins.npy")[idx][:, :, 4:8]                                  # [C, T, 4, 2]
    src_true = cen[rows][:, 4:8]                                                                   # [C, 4, 2]

    # forecast position probes: native forecasts of probe clips (never an edited forecast)
    pos_probes, pos_rows = {}, {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        m = probe & ok
        st = Standardizer().fit(Zall[m, k])
        cv = cv_select_alpha(st.transform(Zall[m, k]), cen[m, t], folds5[ok[probe]],
                             score_fn=lambda a, b: pscore(a, b, "linear"))
        W, b = fit_ridge(st.transform(Zall[m, k]), cen[m, t], cv["alpha"])
        pos_probes[k] = (st, W, b)
        e = np.linalg.norm(predict(st.transform(Zall[test & ok, k]), W, b) - cen[test & ok, t], axis=1)
        pos_rows[str(t)] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "test_px_mean": float(e.mean()),
                            "n_fit": int(m.sum())}

    def read_pos(Z):                                                                               # [..., 4, D] -> [..., 4, 2]
        return np.stack([predict(pos_probes[k][0].transform(Z[..., k, :].reshape(-1, D)), pos_probes[k][1],
                                 pos_probes[k][2]).reshape(Z.shape[:-2] + (2,)) for k in range(4)], -2)

    dist = lambda a, b: np.nanmean(np.linalg.norm(a - b, axis=-1), -1)                             # noqa: E731 mean over steps

    def proj(e, t):                                                                                # sum over steps & xy
        ok = np.isfinite(e).all(-1) & np.isfinite(t).all(-1)
        e, t = np.where(ok[..., None], e, 0), np.where(ok[..., None], t, 0)
        return (e * t).sum((-1, -2)) / np.maximum((t * t).sum((-1, -2)), 1e-9)

    Zs, Zt = F["src_pred_pooled"].astype(np.float64), F["twin_pred_pooled"].astype(np.float64)
    P0, Ptw = read_pos(Zs), read_pos(Zt)                                                           # [C,4,2], [C,T,4,2]
    parity = {"src_pred_vs_native_rel_maxabs": float(np.abs(Zs - Zall[rows]).max() / np.abs(Zall[rows]).max()),
              "src_meanpool_vs_stored_rel_maxabs_per_point_max": float(
                  (np.abs(F["src_meanpool"] - arrays["carrier_stored_meanpool"][idx]).max(axis=(0, 2))
                   / np.abs(arrays["carrier_stored_meanpool"][idx]).max(axis=(0, 2))).max())}
    true_shift = cen_tw - src_true[:, None]                                                        # [C, T, 4, 2]
    d_zero = dist(P0[:, None], cen_tw)                                                             # [C, T]
    ref = {"source_to_target_true_px": per_clip_ci(dist(src_true[:, None], cen_tw).mean(1)),
           "zero_edit_dist_to_target_px": per_clip_ci(d_zero.mean(1)),
           "zero_edit_dist_to_source_true_px": per_clip_ci(dist(P0, src_true)),
           "twin_forecast_dist_to_target_px": per_clip_ci(dist(Ptw, cen_tw).mean(1)),
           "twin_forecast_R_true_shift": per_clip_ci(proj(Ptw - P0[:, None], true_shift).mean(1)),
           "twin_forecast_frac_distance_closed": per_clip_ci((1 - dist(Ptw, cen_tw) / d_zero).mean(1))}

    # encoder-side start (x, y) probes per point, stored full-clip meanpool of probe clips
    full = np.load(PROJECT_ROOT / "artifacts" / "activations" / "speed" / "vjepa2" / "meanpool.npy", mmap_mode="r")
    enc_probe = {}
    for p in range(min(LAYERS), 26):
        Xp = np.asarray(full[:, p], np.float64)
        enc_probe[p] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(Xp[probe], pos[probe])
    enc_test_err = {str(p): float(np.linalg.norm(enc_probe[p].predict(np.asarray(full[test, p], np.float64))
                                                 - pos[test], axis=1).mean()) for p in (12, 22, 24, 25)}
    rd = lambda X, p: enc_probe[p].predict(X.reshape(-1, D).astype(np.float64)).reshape(X.shape[:-1] + (2,))  # noqa: E731
    dm = tpos - src[:, None]                                                                       # [C, T, 2] metres

    def enc_metrics(Pe, P0e):                                                                      # [C,T,2], [C,2]
        return {"dist_to_target_m": per_clip_ci(np.linalg.norm(Pe - tpos, axis=-1).mean(1)),
                "zero_dist_to_target_m": per_clip_ci(np.linalg.norm(P0e[:, None] - tpos, axis=-1).mean(1)),
                "R_true_shift": per_clip_ci((((Pe - P0e[:, None]) * dm).sum(-1)
                                             / np.maximum((dm * dm).sum(-1), 1e-9)).mean(1)),
                "frac_distance_closed": per_clip_ci((1 - np.linalg.norm(Pe - tpos, axis=-1)
                                                     / np.linalg.norm(P0e[:, None] - tpos, axis=-1)).mean(1))}

    src_mp = F["src_meanpool"].astype(np.float64)
    ctx_mp = F["src_ctx_meanpool"].astype(np.float64)
    tw_mp = F["twin_meanpool"].astype(np.float64)
    per_layer = {}
    for L in LAYERS:
        P = F[f"edit_pred_pooled_L{L}"].astype(np.float64)                                         # [C, A, T, 4, D]
        E = F[f"edit_meanpool_L{L}"].astype(np.float64)                                            # [C, A, T, 26-L, D]
        Ec = F[f"edit_ctx_meanpool_L{L}"].astype(np.float64)
        row = {}
        for i, arm in enumerate(ARMS):
            Pp = read_pos(P[:, i])                                                                 # [C, T, 4, 2]
            de = dist(Pp, cen_tw)
            enc = {}
            for p in sorted({L, 24, 25}):
                enc[str(p)] = enc_metrics(rd(E[:, i, :, p - L], p), rd(src_mp[:, p], p))
            enc_ctx = {str(p): enc_metrics(rd(Ec[:, i, :, p - L], p), rd(ctx_mp[:, p], p)) for p in (24, 25)}
            row[arm] = {"forecast": {
                "dist_to_target_px": per_clip_ci(de.mean(1)),
                "frac_distance_closed": per_clip_ci((1 - de / d_zero).mean(1)),
                "R_true_shift": per_clip_ci(proj(Pp - P0[:, None], true_shift).mean(1)),
                "R_twin_forecast_shift": per_clip_ci(proj(Pp - P0[:, None], Ptw - P0[:, None]).mean(1)),
                "shift_from_unedited_px": per_clip_ci(dist(Pp, P0[:, None]).mean(1)),
                "R_token_twin_forecast": per_clip_ci(np.mean(
                    ((P[:, i] - Zs[:, None]) * (Zt - Zs[:, None])).sum((-1, -2))
                    / np.maximum(((Zt - Zs[:, None]) ** 2).sum((-1, -2)), 1e-12), 1))},
                "encoder_full_clip": enc, "encoder_context_only": enc_ctx,
                "delta_norm_median": float(np.median(np.linalg.norm(arrays[f"deltas_L{L}"][idx][:, i], axis=-1)))}
        row["zero"] = {"forecast": {"dist_to_target_px": ref["zero_edit_dist_to_target_px"], "frac_distance_closed": 0.0,
                                    "R_true_shift": 0.0}}
        row["twin"] = {"forecast": {"dist_to_target_px": ref["twin_forecast_dist_to_target_px"],
                                    "frac_distance_closed": ref["twin_forecast_frac_distance_closed"],
                                    "R_true_shift": ref["twin_forecast_R_true_shift"]},
                       "encoder_full_clip": {str(p): enc_metrics(rd(tw_mp[:, :, p], p), rd(src_mp[:, p], p))
                                             for p in sorted({L, 24, 25})}}
        per_layer[str(L)] = row
        print(f"L{L}: forecast dist to target px " + ", ".join(
            f"{a} {row[a]['forecast']['dist_to_target_px']['mean']:.1f}" for a in list(ARMS) + ["zero", "twin"]), flush=True)
    timing = sorted((out / "forward").glob("timing_*.json"))
    gpu_name = json.loads(timing[-1].read_text()).get("gpu") if timing else None
    res = {"question": "does a start-position edit inside the encoder move where the predictor forecasts the disk?",
           "design": {k: info[k] for k in ("layers", "arms", "grid", "carrier_pool", "carrier_pool_size", "twins", "seed")},
           "n_carriers": int(C), "n_targets_per_carrier": int(T),
           "plan_per_layer": info["per_layer"],
           "readers": {"forecast_position_probes": pos_rows,
                       "forecast_fit_rows": f"{int(probe.sum())} probe clips (folds 3-4, all cells), native forecasts "
                                            f"{NATIVE.relative_to(PROJECT_ROOT)} (mask 0, frames 1-8), alpha 5-fold CV",
                       "encoder_start_xy_probe_test_err_m": enc_test_err,
                       "encoder_fit_rows": "probe clips, stored full-clip meanpool, StandardScaler + RidgeCV"},
           "parity": parity, "reference": ref, "per_layer": per_layer,
           "compute": {"gpu_seconds_forward": float(F["gpu_seconds"].sum()), "box": args.box, "gpu": gpu_name},
           "keys": {
               "per_layer.<L>.<arm>.forecast.dist_to_target_px": "mean over tubelets 4-7 of |forecast disk position "
                   "(per-step probe on the pooled forecast) - rendered twin disk centroid|, px in the 256 px frame; per "
                   "carrier mean over its 2 targets; per_clip_ci over carriers",
               "frac_distance_closed": "1 - dist(edit, target) / dist(unedited forecast, target), per carrier-target",
               "R_true_shift": "sum_steps <forecast_edit - forecast_unedited, twin_true - source_true> / "
                               "sum_steps |twin_true - source_true|^2 (1 = the forecast disk moved the whole true shift)",
               "R_twin_forecast_shift": "same against the predictor's own twin-forecast shift",
               "encoder_full_clip.<p>": "start (x, y) probe on the edited full 16-frame clip's meanpool at point p "
                                        "(p = L: edit point; 24 = block-24 output; 25 = final LN, what the predictor "
                                        "reads); metres; R_true_shift against (target centre - source start)",
               "encoder_context_only.<p>": "same probe on the edited frames-1-8 encoding (the predictor's actual input); "
                                           "the probe was fit on full clips, so read against its own zero baseline",
               "reference.source_to_target_true_px": "how far the target disk is from the source disk (true, px)"}}
    write(RES / "session2_position_predictor.json", res, stage="position_predictor_score",
          seeds={"plan": info["seed"], "bootstrap": 0, "cv_folds": 0}, plan_sha256=sha256_file(out / "plan.npz"),
          box=args.box, gpu_seconds_forward=res["compute"]["gpu_seconds_forward"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "forward", "score"))
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--n-carriers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="forward only the first N carriers (smoke)")
    ap.add_argument("--box", default="vast 53255819 (box 4, RTX 4060 Ti)")
    a = ap.parse_args()
    {"plan": plan, "forward": forward, "score": score}[a.stage](a)

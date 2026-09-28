"""Item 4 of the velocity-sheet follow-up: does the joint (direction, speed) sheet edit at point 22 reach the
predictor's forecast, and does the forecast disk move in the new direction at the new speed?

Design (speed set, split_v1 roles; one held-out block = scripts/run_velocity_sheet.py BLOCKS[0]: direction bins
[1, 2] x speed bins [2, 3], i.e. 4 held-out (theta, v) cells):
  geometry   PCA-64 of knot clips outside the held cells; TPS sheet over (cos, sin, lambda*(v-2.125)/1.875) fit on
             their knot cell centroids (lambda, smoothing chosen as run_velocity_sheet.choose_sheet with the block
             excluded); global ring (16 direction-bin centroids pooled over speeds) and global speed line (speed-value
             centroids pooled over directions), both from the same knot clips -- identical to run_velocity_sheet's
             joint arms.
  carriers   n test clips outside the held direction AND speed bins (seed 0); each steered to all 4 held cells
             (target theta, v = mean over the test clips in the cell, as in run_velocity_sheet).
  arms       sheet     P(theta*, v*) - P(theta_s, v_s) on the TPS, lifted to D (the joint 2-D edit)
             seq1d     global ring edit (theta_s -> theta*) + global speed-line edit (v_s -> v*): the additive
                       sequential 1-D edit (both orders share this endpoint)
             speed1d   speed-line edit only (v_s -> v*), the session-3 style 1-D speed edit
             ring1d    ring edit only (theta_s -> theta*)
             rawchord  raw-centroid polyline chords of the ring + the speed line (Goodfire's linear chord, joint)
  conditions own norm; sheet_norm (every arm rescaled to the sheet's per-clip norm)
  twins      each carrier re-rendered at the target cell (same start; theta and speed replaced): physical ceiling
  forward    context frames 1-8, delta added to every context token at point 22, encoder finished, predictor
             (mask 0) -> pooled forecast steps 4-7 (as session3_speed_predictor.forward).
  readouts   fit on the predictor's own forecasts of probe clips OUTSIDE the held cells (native forecasts from
             artifacts/session3_speed/native/pred_pooled_all.npy, same model / mask / pooling): ridge direction
             [sin, cos] and speed probes on the step-mean forecast; per-step disk position probes (tubelets 4-7)
             -> forecast displacement speed (|p7 - p4| / 3 steps, calibrated linearly on probe clips) and heading
             (world angle of p7 - p4, y up).

  python scripts/run_velocity_sheet_predictor.py plan|forward|score [--out DIR] [--n-carriers 48]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, disk_pixels, frame_hash, load_table  # noqa: E402

RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
ART = PROJECT_ROOT / "artifacts" / "p5_velocity_sheet_predictor"
NATIVE = PROJECT_ROOT / "artifacts" / "session3_speed" / "native"
POINT = 22
ARMS = ("sheet", "seq1d", "speed1d", "ring1d", "rawchord", "seq1d_interp", "speed1d_interp", "ring1d_interp")
SPLINE_OF_ARM = {"seq1d": "smooth", "speed1d": "smooth", "ring1d": "smooth", "seq1d_interp": "interp",
                 "speed1d_interp": "interp", "ring1d_interp": "interp", "sheet": "TPS", "rawchord": "raw polyline"}
CONDS = ("own", "sheet_norm")
D = 1024
FPS, FRAMES_PER_STEP, PX_PER_M = 24.0, 2, 32.0


# ================================================================ pure helpers (tests/test_velocity_sheet_predictor.py)

def heading_from_positions(pos):
    """Per-tubelet disk pixel positions [..., S, 2] (x = col, y = row, rows grow downward) -> world heading (deg,
    [0, 360), y up) of the first-to-last displacement."""
    pos = np.asarray(pos, float)
    d = pos[..., -1, :] - pos[..., 0, :]
    return np.degrees(np.arctan2(-d[..., 1], d[..., 0])) % 360.0


def circ_err(a, b):
    """Absolute circular difference in degrees, in [0, 180]."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def signed_circ(a, b):
    """Signed circular difference a - b in degrees, in (-180, 180]."""
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0


def rescale_to(delta, ref_norm, eps=1e-6):
    """delta [..., D] rescaled to ||delta|| = ref_norm [...]; a (near-)zero delta stays zero. Returns (delta', scale)."""
    delta = np.asarray(delta, float)
    n = np.linalg.norm(delta, axis=-1)
    s = np.where(n > eps, np.asarray(ref_norm, float) / np.maximum(n, eps), 0.0)
    return delta * s[..., None], s


def projection_fraction(edit_change, target_change):
    """Signed fraction of a target change achieved: sum(e * t) / sum(t * t) over the last axis (per row)."""
    e, t = np.asarray(edit_change, float), np.asarray(target_change, float)
    return np.sum(e * t, -1) / np.maximum(np.sum(t * t, -1), 1e-12)


def per_clip_ci(v, n=1000, seed=0):
    """Mean, SE and 95% clip-bootstrap CI of per-carrier values (NaNs dropped)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "se": None, "ci95": None, "n": 0}
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


# ================================================================ plan (CPU, Mac)

def plan(args):
    from run_velocity_sheet import BLOCKS, N_DIR, bins, cell_table, choose_sheet, fit_sheet, sheet_inputs
    from session3_speed_predictor import tubelet_centroids
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.render_twin import load_meta, validate, write_twin
    out = Path(args.out)
    (out / "twins").mkdir(parents=True, exist_ok=True)
    d = load_inputs("speed", POINT)
    df = d["df"]
    theta, speed = df["theta_degrees"].to_numpy(float), df["speed_mps"].to_numpy(float)
    dbin, sbin = bins(theta, speed)
    block = BLOCKS[0]
    held_d, held_s = [block[0], (block[0] + 1) % N_DIR], list(block[1])
    held = np.isin(dbin, held_d) & np.isin(sbin, held_s)
    knot, test = d["role"] == "knot", d["role"] == "test"
    kf = knot & ~held
    X = d["X"].astype(np.float64)
    pca = mf.fit_pca(X[kf], 64)
    Z = pca.project(X)
    lam, sm, sel = choose_sheet(Z, theta, speed, dbin, sbin, d["is_train"], block)
    sheet = fit_sheet(cell_table(Z, theta, speed, dbin, sbin, kf), lam, sm)
    labels_all = np.array([22.5 * b + 8.4375 for b in dbin[kf]])
    cent_ring = mf.centroids(Z[kf], labels_all)
    ring = mf.fit_curve(cent_ring, True, angle="labels", spline="smooth")
    ring_raw = mf.raw_knot_curve(ring, cent_ring)
    cent_v = mf.centroids(Z[kf], speed[kf])
    line = mf.fit_curve(cent_v, False, spline="smooth", extend="linear")
    line_raw = mf.raw_knot_curve(line, cent_v)
    ring_i = mf.fit_curve(cent_ring, True, angle="labels", spline="interp")       # the paper's interpolating spline
    line_i = mf.fit_curve(cent_v, False, spline="interp", extend="linear")
    # targets: the 4 held cells (test-clip means)
    cells = []
    for dd in held_d:
        for ss in held_s:
            c = (dbin == dd) & (sbin == ss)                    # labels only (all roles): the cell's mean (theta, v)
            assert c.sum() > 0
            cells.append((float(np.degrees(np.angle(np.exp(1j * np.radians(theta[c])).mean())) % 360),
                          float(speed[c].mean()), int(dd), int(ss), int((c & test).sum()), int(c.sum())))
    tth = np.array([c[0] for c in cells])
    tv = np.array([c[1] for c in cells])
    pool = np.flatnonzero(test & ~np.isin(dbin, held_d) & ~np.isin(sbin, held_s))
    rng = np.random.default_rng(args.seed)
    carriers = np.sort(rng.choice(pool, size=min(args.n_carriers, len(pool)), replace=False))
    C, T = len(carriers), len(cells)
    th_s, v_s, Z0 = theta[carriers], speed[carriers], Z[carriers]
    deltas = np.zeros((C, len(ARMS), T, D), np.float32)
    for j in range(T):
        Ps = sheet(sheet_inputs(th_s, v_s, lam))
        Pt = sheet(sheet_inputs(np.full(C, tth[j]), np.full(C, tv[j]), lam))
        dR = mf.manifold_coords(np.zeros_like(Z0), ring, ring.coord_of_value(th_s),
                                ring.coord_of_value(np.full(C, tth[j])), 2)[:, -1]
        dV = mf.manifold_coords(np.zeros_like(Z0), line, v_s, np.full(C, tv[j]), 2)[:, -1]
        rR = mf.piecewise_linear_point(ring_raw, np.full(C, tth[j])) - mf.piecewise_linear_point(ring_raw, th_s)
        rV = mf.piecewise_linear_point(line_raw, np.full(C, tv[j])) - mf.piecewise_linear_point(line_raw, v_s)
        dRi = mf.manifold_coords(np.zeros_like(Z0), ring_i, ring_i.coord_of_value(th_s),
                                 ring_i.coord_of_value(np.full(C, tth[j])), 2)[:, -1]
        dVi = mf.manifold_coords(np.zeros_like(Z0), line_i, v_s, np.full(C, tv[j]), 2)[:, -1]
        for i, dz in enumerate((Pt - Ps, dR + dV, dV, dR, rR + rV, dRi + dVi, dVi, dRi)):
            deltas[:, i, j] = pca.lift_delta(dz)
    # same-point sanity: ridge probes on stored point-22 meanpool (probe clips outside held cells)
    from wm.probes import Standardizer, fit_ridge, load_sweep, predict, targets
    pr = (d["role"] == "probe") & ~held
    alpha = load_sweep("speed", "speed")["layers"][POINT]["alpha"]
    st = Standardizer().fit(X[pr])
    Ws, bs = fit_ridge(st.transform(X[pr]), speed[pr], alpha)
    Yd = targets(df, "direction")[0]
    Wd, bd = fit_ridge(st.transform(X[pr]), Yd[pr], alpha)
    rs = lambda A: predict(st.transform(A.reshape(-1, D)), Ws, bs).reshape(A.shape[:-1])            # noqa: E731
    rdp = lambda A: predict(st.transform(A.reshape(-1, D)), Wd, bd).reshape(A.shape[:-1] + (2,))    # noqa: E731
    rd = lambda A: np.degrees(np.arctan2(rdp(A)[..., 0], rdp(A)[..., 1])) % 360                      # noqa: E731
    Xe = X[carriers][:, None, None] + deltas                                                         # [C, A, T, D]
    same = {a: {"dir_err_deg": float(circ_err(rd(Xe[:, i]), tth[None]).mean()),
                "speed_err_mps": float(np.abs(rs(Xe[:, i]) - tv[None]).mean()),
                "delta_norm_median": float(np.median(np.linalg.norm(deltas[:, i], axis=-1)))}
            for i, a in enumerate(ARMS)}
    same["unedited"] = {"dir_err_deg": float(circ_err(rd(X[carriers])[:, None], tth[None]).mean()),
                        "speed_err_mps": float(np.abs(rs(X[carriers])[:, None] - tv[None]).mean())}
    print(json.dumps(same, indent=1))
    # twins at the target cell (theta and speed replaced), MP4 through the clips' codec
    hashes, cen_tw, vis = [], np.zeros((C, T, 8, 2)), []
    for i, c in enumerate(carriers):
        meta = load_meta("speed", int(df["id"].iloc[c]))
        for j in range(T):
            fr = write_twin({**meta, "speed_mps": float(tv[j])}, float(tth[j]), out / "twins" / f"twin_{i:04d}_{j}.mp4")
            hashes.append(frame_hash(fr))
            cen_tw[i, j] = tubelet_centroids(fr)
            vis.append(float((disk_pixels(fr).reshape(16, -1).sum(1) > 0).mean()))
    np.save(out / "centroids_twins.npy", cen_tw)
    arrays = {"carrier_rows": carriers, "carrier_ids": df["id"].to_numpy()[carriers], "target_theta": tth,
              "target_speed": tv, "deltas_L22": deltas, "twin_frame_hash": np.array(hashes).reshape(C, T),
              "carrier_stored_meanpool": X[carriers].astype(np.float32)}
    val = validate("speed", n=10, seed=0)
    info = {"point": POINT, "arms": list(ARMS), "spline_of_arm": SPLINE_OF_ARM, "conditions": list(CONDS), "seed": args.seed,
            "block": {"held_dir_bins": held_d, "held_speed_bins": held_s},
            "target_cells": [{"theta_deg": c[0], "speed_mps": c[1], "dir_bin": c[2], "speed_bin": c[3],
                              "n_test_clips": c[4], "n_clips_all_roles": c[5],
                              "target_value_def": "mean label over all clips in the cell (labels only)"} for c in cells],
            "sheet_lambda": lam, "sheet_smoothing": sm, "selection_table": sel,
            "n_knot_clips": int(kf.sum()), "n_carriers": C, "carrier_pool_size": int(len(pool)),
            "carrier_pool": "test clips outside the held direction AND speed bins",
            "same_point_probe_check": same,
            "twins": {"dir": "twins/twin_{carrier:04d}_{target}.mp4", "visible_frac_mean": float(np.mean(vis)),
                      "visible_frac_min": float(np.min(vis))},
            "renderer_validation_speed": {k: val[k] for k in ("codec", "mask_twin_iou_mean", "verdict")}}
    np.savez(out / "plan.npz", **arrays)
    (out / "plan.json").write_text(json.dumps(info, indent=1, default=float))
    print("plan written", out, "twin visible frac", info["twins"])


def load_plan(out):
    z = np.load(Path(out) / "plan.npz")
    return {k: z[k] for k in z.files}, json.loads((Path(out) / "plan.json").read_text())


# ================================================================ forward (GPU, box)

def forward(args):
    from session3_speed_predictor import _model, gpu_lock
    from wm.extract import preprocess
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    arrays, info = load_plan(args.out)
    fdir = Path(args.out) / "forward"
    fdir.mkdir(parents=True, exist_ok=True)
    df = load_table("speed")
    rows = arrays["carrier_rows"]
    C, A, T = arrays["deltas_L22"].shape[:3]
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
        dl = arrays["deltas_L22"][idx].astype(np.float64)                                  # [G, A, T, D]
        ref = {"own": np.linalg.norm(dl, axis=-1),
               "sheet_norm": np.repeat(np.linalg.norm(dl[:, 0], axis=-1)[:, None], A, 1)}
        Dc = np.stack([rescale_to(dl, ref[c])[0] for c in CONDS], 1).astype(np.float32)     # [G, K, A, T, D]
        with gpu_lock() as waited:
            t0 = time.time()
            pv = preprocess(frames).to(device)
            ctx_saved, cf, ctx_mp = prefix(model, pv[:, :8], [POINT])
            src_pp = pool_steps(predict_future(model, cf, 0)).cpu().numpy()
            src_mp_full = prefix(model, pv, [])[2]
            tw_pp = []
            for s in range(0, len(twins), bs):
                tpv = preprocess(twins[s:s + bs]).to(device)
                tw_pp.append(pool_steps(predict_future(model, prefix(model, tpv[:, :8], [])[1], 0)).cpu().numpy())
            tw_pp = np.concatenate(tw_pp).reshape(G, T, 4, D)
            pp = np.zeros((G, len(CONDS), A, T, 4, D), np.float32)
            jobs = [(a, k, i, j) for a in range(G) for k in range(len(CONDS)) for i in range(A) for j in range(T)]
            for s in range(0, len(jobs), bs):
                ch = jobs[s:s + bs]
                ca = torch.as_tensor([c[0] for c in ch], device=device)
                dd = torch.as_tensor(np.stack([Dc[a, k, i, j] for a, k, i, j in ch]), device=device)
                z, _ = edited_prediction(model, ctx_saved[POINT][ca], POINT, dd, 0)
                p = pool_steps(z).cpu().numpy()
                for q, (a, k, i, j) in enumerate(ch):
                    pp[a, k, i, j] = p[q]
            torch.cuda.synchronize()
            secs = time.time() - t0
        res = {"idx": np.array(idx), "src_pred_pooled": src_pp, "twin_pred_pooled": tw_pp, "edit_pred_pooled": pp,
               "src_meanpool": np.asarray(src_mp_full), "gpu_seconds": np.array([secs])}
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
    from run_velocity_sheet import BLOCKS, N_DIR, bins
    from session3_speed_predictor import speed_from_positions
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score as pscore, targets
    out = Path(args.out)
    arrays, info = load_plan(out)
    F = load_groups(out / "forward")
    idx = F["idx"]
    rows = arrays["carrier_rows"][idx]
    tth, tv = arrays["target_theta"], arrays["target_speed"]
    C, T = len(idx), len(tth)
    df = load_table("speed")
    th = df["theta_degrees"].to_numpy(float)
    y = df["speed_mps"].to_numpy(float)
    dbin, sbin = bins(th, y)
    held = np.isin(dbin, info["block"]["held_dir_bins"]) & np.isin(sbin, info["block"]["held_speed_bins"])
    d0 = load_inputs("speed", 0)
    probe, test = (d0["role"] == "probe") & ~held, d0["role"] == "test"
    assert not probe[rows].any() and test[rows].all() and not held[rows].any()
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Zall = np.load(NATIVE / "pred_pooled_all.npy").astype(np.float64)
    ext = json.loads((NATIVE / "extract_info.json").read_text())
    assert ext["ids"] == [int(i) for i in df["id"]]
    cen = np.load(PROJECT_ROOT / "artifacts" / "session3_speed" / "centroids_speed.npy")
    cen_tw = np.load(out / "centroids_twins.npy")[idx]

    def fit(Xf, Yt, fl, kind):
        st = Standardizer().fit(Xf)
        cv = cv_select_alpha(st.transform(Xf), Yt, fl, score_fn=lambda a, b: pscore(a, b, kind))
        W, b = fit_ridge(st.transform(Xf), Yt, cv["alpha"])
        return (st, W, b), cv

    def apply(pr, Xf):
        st, W, b = pr
        sh = Xf.shape
        return predict(st.transform(Xf.reshape(-1, sh[-1])), W, b).reshape(sh[:-1] + (-1,))

    pos, pos_rows = {}, {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        pos[k], cv = fit(Zall[probe & ok, k], cen[probe & ok, t], folds5[ok[probe]], "linear")
        e = np.linalg.norm(apply(pos[k], Zall[test & ok, k]) - cen[test & ok, t], axis=1)
        pos_rows[str(t)] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "test_px_mean": float(e.mean())}
    read_pos = lambda Z: np.stack([apply(pos[k], Z[..., k, :]) for k in range(4)], -2)       # noqa: E731
    cal = np.polyfit(speed_from_positions(read_pos(Zall[probe])), y[probe], 1)
    read_vdisp = lambda Z: np.polyval(cal, speed_from_positions(read_pos(Z)))                # noqa: E731
    read_head = lambda Z: heading_from_positions(read_pos(Z))                                 # noqa: E731
    sp, cvs = fit(Zall.mean(1)[probe], y[probe, None], folds5, "linear")
    read_speed = lambda Z: apply(sp, Z.mean(-2))[..., 0]                                     # noqa: E731
    Yd = targets(df, "direction")[0]
    dp, cvd = fit(Zall.mean(1)[probe], Yd[probe], folds5, "circular")
    read_dir = lambda Z: np.degrees(np.arctan2(*np.moveaxis(apply(dp, Z.mean(-2)), -1, 0))) % 360   # noqa: E731
    readers = {
        "fit_rows": f"{int(probe.sum())} probe clips (folds 3-4) outside the 4 held cells; alpha by 5-fold CV; "
                    f"native forecasts from {NATIVE.relative_to(PROJECT_ROOT)} (mask 0, frames 1-8)",
        "position_probes": pos_rows, "disp_speed_calibration_a_b": cal.tolist(),
        "test_disp_speed_mae_mps": float(np.abs(read_vdisp(Zall[test]) - y[test]).mean()),
        "test_disp_heading_mae_deg": float(circ_err(read_head(Zall[test]), th[test]).mean()),
        "true_centroid_heading_mae_deg_test": float(np.nanmean(circ_err(heading_from_positions(cen[test][:, 4:8]), th[test]))),
        "speed_probe": {"alpha": cvs["alpha"], "cv_r2": cvs["cv_mean"],
                        "test_mae_mps": float(np.abs(read_speed(Zall[test]) - y[test]).mean())},
        "dir_probe": {"alpha": cvd["alpha"], "test_mae_deg": float(circ_err(read_dir(Zall[test]), th[test]).mean())},
        "heldout_cell_test_clips": {
            "n": int((test & held).sum()),
            "speed_probe_mae": float(np.abs(read_speed(Zall[test & held]) - y[test & held]).mean()),
            "dir_probe_mae_deg": float(circ_err(read_dir(Zall[test & held]), th[test & held]).mean()),
            "disp_speed_mae": float(np.abs(read_vdisp(Zall[test & held]) - y[test & held]).mean()),
            "disp_heading_mae_deg": float(circ_err(read_head(Zall[test & held]), th[test & held]).mean())}}
    Zs = F["src_pred_pooled"].astype(np.float64)
    Zt = F["twin_pred_pooled"].astype(np.float64)
    parity = {"src_pred_vs_native_rel_maxabs": float(np.abs(Zs - Zall[rows]).max() / np.abs(Zall[rows]).max()),
              "src_meanpool_L22_vs_stored_rel_maxabs": float(
                  np.abs(F["src_meanpool"][:, POINT] - arrays["carrier_stored_meanpool"][idx]).max()
                  / np.abs(arrays["carrier_stored_meanpool"][idx]).max())}
    th_s, v_s = th[rows], y[rows]
    s_dir0, s_sp0, s_vd0, s_hd0 = read_dir(Zs), read_speed(Zs), read_vdisp(Zs), read_head(Zs)
    true_dv = tv[None] - v_s[:, None]                                                   # [C, T]
    true_dth = signed_circ(tth[None], th_s[:, None])

    def metrics(Z):                                                                    # Z [C, T, 4, D]
        dr, spd, vd, hd = read_dir(Z), read_speed(Z), read_vdisp(Z), read_head(Z)
        m = {"dir_err_deg": per_clip_ci(circ_err(dr, tth[None]).mean(1)),
             "speed_err_mps": per_clip_ci(np.abs(spd - tv[None]).mean(1)),
             "disp_speed_err_mps": per_clip_ci(np.abs(vd - tv[None]).mean(1)),
             "disp_heading_err_deg": per_clip_ci(circ_err(hd, tth[None]).mean(1)),
             "disp_speed_mps": per_clip_ci(vd.mean(1)),
             "R_dir_probe": per_clip_ci(projection_fraction(signed_circ(dr, s_dir0[:, None]), true_dth)),
             "R_speed_probe": per_clip_ci(projection_fraction(spd - s_sp0[:, None], true_dv)),
             "R_disp_speed": per_clip_ci(projection_fraction(vd - s_vd0[:, None], true_dv)),
             "R_disp_heading": per_clip_ci(projection_fraction(signed_circ(hd, s_hd0[:, None]), true_dth)),
             "R_token_twin_forecast": per_clip_ci(np.mean(projection_fraction(
                 (Z - Zs[:, None]).reshape(C, T, -1), (Zt - Zs[:, None]).reshape(C, T, -1)), 1)),
             "px_to_twin_true": per_clip_ci(np.nanmean(np.linalg.norm(read_pos(Z) - cen_tw[:, :, 4:8], axis=-1), (1, 2)))}
        pc = {"dir": circ_err(dr, tth[None]).mean(1), "speed": np.abs(spd - tv[None]).mean(1),
              "disp_speed": np.abs(vd - tv[None]).mean(1), "disp_heading": circ_err(hd, tth[None]).mean(1)}
        return m, pc

    ref = {"unedited": metrics(np.repeat(Zs[:, None], T, 1))[0], "twin_forecast": metrics(Zt)[0]}
    ref["unedited_true"] = {"disp_speed_mps_source": per_clip_ci(v_s), "target_speed_mean": float(tv.mean()),
                            "target_theta": tth.tolist(), "target_speed": tv.tolist()}
    ref["twin_true_centroid_heading_err_deg"] = per_clip_ci(np.nanmean(circ_err(heading_from_positions(cen_tw[:, :, 4:8]), tth[None]), 1))
    P = F["edit_pred_pooled"].astype(np.float64)                                       # [C, K, A, T, 4, D]
    pred, paired = {}, {}
    for k, cond in enumerate(CONDS):
        pred[cond], pcs = {}, {}
        for i, arm in enumerate(ARMS):
            pred[cond][arm], pcs[arm] = metrics(P[:, k, i])
            pred[cond][arm]["delta_norm_median"] = float(np.median(np.linalg.norm(
                rescale_to(arrays["deltas_L22"][idx][:, i].astype(np.float64),
                           np.linalg.norm(arrays["deltas_L22"][idx][:, 0], axis=-1) if cond == "sheet_norm"
                           else np.linalg.norm(arrays["deltas_L22"][idx][:, i], axis=-1))[0], axis=-1)))
        paired[cond] = {f"sheet_minus_{b}_{r}_err": per_clip_ci(pcs["sheet"][r] - pcs[b][r])
                        for b in ARMS[1:] for r in ("dir", "speed", "disp_speed", "disp_heading")}
    gpu_s = float(F["gpu_seconds"].sum())
    res = {"design": {k: info[k] for k in ("point", "arms", "conditions", "block", "target_cells", "sheet_lambda",
                                           "sheet_smoothing", "n_knot_clips", "carrier_pool", "carrier_pool_size",
                                           "twins", "renderer_validation_speed")},
           "n_carriers": int(C), "same_point_probe_check": info["same_point_probe_check"], "readers": readers,
           "parity": parity, "reference": ref, "predictor": pred, "paired": paired,
           "compute": {"gpu_seconds_forward": gpu_s, "box": args.box,
                       "native_forecasts_reused_from": "artifacts/session3_speed/native (extract_info gpu_seconds "
                                                       f"{ext.get('gpu_seconds')}, not re-spent)"},
           "keys": {
               "predictor.<cond>.<arm>": "forecast of the edited context (frames 1-8, delta on every context token at "
                                         "point 22, mask 0), per carrier mean over the 4 held-out target cells; "
                                         "per_clip_ci = mean, SE, 1000-draw clip-bootstrap 95% CI, n carriers",
               "dir_err_deg / speed_err_mps": "|direction (speed) ridge probe on the step-mean forecast - target cell "
                                              "theta (speed)|",
               "disp_speed_err_mps / disp_heading_err_deg": "per-step disk position probes (tubelets 4-7) -> "
                   "|p7-p4|/3 steps calibrated to m/s on probe clips / world heading of p7-p4, vs the target cell",
               "disp_speed_mps": "forecast disk speed read from the per-step positions (m/s, calibrated)",
               "R_*": "signed fraction of the target change achieved relative to the unedited forecast "
                      "(sum_t edit_change * true_change / sum_t true_change^2); 1 = full, 0 = none",
               "R_token_twin_forecast": "<Z_edit - Z_src, Z_twin - Z_src> / ||Z_twin - Z_src||^2 over the 4 pooled "
                                        "forecast steps (probe-free), mean over targets",
               "reference.twin_forecast": "same readouts on the forecast of the carrier re-rendered at the target cell "
                                          "(physical ceiling)",
               "paired.<cond>.sheet_minus_<arm>_<readout>_err": "paired per-carrier difference of the error to target",
               "spline_of_arm": json.dumps(SPLINE_OF_ARM),
               "conditions": "own = each arm's own delta norm; sheet_norm = each arm rescaled to the sheet's per-clip norm"}}
    write(RES / "p5_velocity_sheet_predictor.json", res, stage="velocity_sheet_predictor_score",
          seeds={"plan": info["seed"], "bootstrap": 0, "cv_folds": 0}, plan_sha256=sha256_file(out / "plan.npz"))
    figure(res)
    return res


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"sheet": "#3b6fb6", "seq1d": "#d9822b", "speed1d": "#6aa84f", "ring1d": "#a64d79", "rawchord": "#999999",
            "seq1d_interp": "#f1c27d", "speed1d_interp": "#b6d7a8", "ring1d_interp": "#d5a6bd"}
    panels = (("dir_err_deg", "direction probe err (deg)"), ("speed_err_mps", "speed probe err (m/s)"),
              ("disp_heading_err_deg", "disk heading err, per-step positions (deg)"),
              ("disp_speed_err_mps", "disk speed err, per-step positions (m/s)"))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for a, (key, lab) in zip(ax, panels):
        for k, cond in enumerate(CONDS):
            for i, arm in enumerate(ARMS):
                r = res["predictor"][cond][arm][key]
                a.bar(k + i * 0.1 - 0.35, r["mean"], 0.1, color=cols[arm], label=arm if k == 0 else None,
                      yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], capsize=2)
        a.axhline(res["reference"]["unedited"][key]["mean"], ls="--", c="k", lw=1, label="unedited")
        a.axhline(res["reference"]["twin_forecast"][key]["mean"], ls=":", c="k", lw=1, label="twin forecast")
        a.set_xticks(range(len(CONDS)), CONDS)
        a.set_title(lab, fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("Point 22 edit -> predictor forecast, steer to 4 held-out (theta, v) cells (95% clip-bootstrap CI)",
                 fontsize=10)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_velocity_sheet_predictor.png", dpi=130)
    print("wrote", FIG / "fig_velocity_sheet_predictor.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "forward", "score"))
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--n-carriers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="forward only the first N carriers (smoke)")
    ap.add_argument("--box", default="vast box 4")
    a = ap.parse_args()
    {"plan": plan, "forward": forward, "score": score}[a.stage](a)

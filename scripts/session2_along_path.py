"""Predictor forecast read ALONG the steering path at point 22 (the Part 2 second-look note, B.1, gap 1).

Same 200 carriers x 4 held-out targets as session2_norm_matched.py. Paths are rebuilt exactly as run_session2.plan
builds them at point 22 (PCA-64 on knot clips at the kept values, interpolating and smoothing periodic splines, chord
between the polyline points, shift mode, residual kept), now with 9 waypoints t = 0, 0.125, ..., 1.0. Matched edit
size = the norm-matched run's natural condition: per (carrier, target, arm) one factor s = ||twin_meanpool_L22 -
src_meanpool_L22|| / ||endpoint delta||, applied to every waypoint (so t = 1.0 is the norm-matched natural edit).
random arm: one random unit direction in the same PCA-64 subspace per (carrier, target), a straight line whose
waypoint norms equal the scaled interpolating spline's.

  plan     CPU, local. -> artifacts/session2/along_path/plan.npz (PCA-64 waypoint deltas + components, scales)
  forward  box, GPU. Each waypoint delta added to every context token at point 22, blocks 23-24 + final LN, predictor
           (mask token 0), per-step pooled forecast. t = 0 is the unedited forecast (session-2 cache).
           -> artifacts/session2/along_path/group_*.npz
  score    CPU, local. Forecasts read with the session-2 predictor-native probes (stored alpha, no refit; checked by
           reproducing session2_predictor_norm_matched.json at t = 1.0), plus 5-NN distances of the edited point-22
           meanpool to real probe-fold clips (full space and the circular-chart ring plane).
           -> results/session2_predictor_along_path.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
AP = OUT / "along_path"
RES = PROJECT_ROOT / "results"
L = 22
KW = 9                                     # waypoints incl. t = 0
TS = np.linspace(0.0, 1.0, KW)
ARMS = ("spline", "spline_smooth", "chord", "random")


def build_paths():
    """Rebuild run_session2.plan's point-22 geometry. Returns dict with PCA, curves, carriers, waypoint dZ."""
    from run_session2 import load_groups, load_plan
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_meanpool", "twin_meanpool"})
    idx = F["idx"]
    rows, tg = arrays["carrier_rows"][idx], arrays["targets"][idx]
    d = load_inputs("direction", L)
    X, y = d["X"].astype(np.float64), d["y"]
    held = np.array(info["held_values"])
    knot = (d["role"] == "knot") & ~np.isin(y, held)
    pca = mf.fit_pca(X[knot], info["k"])
    cent = mf.centroids(pca.project(X[knot]), y[knot])
    choice = mf.choose_angle_source(cent["C"], cent["values"])
    curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=info["spline"])
    curve_s = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline="smooth")
    Z = pca.project(X[rows])
    src = y[rows]
    C, T = tg.shape
    dZ = np.zeros((C, T, len(ARMS), KW, Z.shape[1]))
    for j in range(T):
        Zsp = mf.manifold_coords(Z, curve, curve.coord_of_value(src), curve.coord_of_value(tg[:, j]), KW)
        Zss = mf.manifold_coords(Z, curve_s, curve_s.coord_of_value(src), curve_s.coord_of_value(tg[:, j]), KW)
        Zch = mf.linear_coords(Z, mf.piecewise_linear_point(curve, src), mf.piecewise_linear_point(curve, tg[:, j]), KW)
        for a, W in (("spline", Zsp), ("spline_smooth", Zss), ("chord", Zch)):
            dZ[:, j, ARMS.index(a)] = W - Z[:, None]
    # theta direction the interpolating spline traverses (decides the arc at exactly 180 deg, where no shorter arc
    # exists): sign of the intrinsic step times the orientation of theta -> intrinsic coordinate
    orient = np.sign(mf.wrap_pi(curve.coord_of_value(src + 1.0) - curve.coord_of_value(src)))
    arc_sign = np.stack([np.sign(curve.step(curve.coord_of_value(src), curve.coord_of_value(tg[:, j]))) * orient
                         for j in range(T)], 1)                                  # [C, T]
    return {"arc_sign": arc_sign, "arrays": arrays, "info": info, "F": F, "idx": idx, "rows": rows, "tg": tg, "d": d, "X": X, "y": y,
            "knot": knot, "pca": pca, "curve": curve, "curve_s": curve_s, "dZ": dZ, "held": held}


def plan():
    P = build_paths()
    arrays, info, F, pca, dZ = P["arrays"], P["info"], P["F"], P["pca"], P["dZ"]
    C, T = P["tg"].shape
    # parity: endpoints reproduce the stored session-2 deltas (plan.npz deltas_L22) for the three path arms
    stored = arrays[f"deltas_L{L}"][P["idx"]].astype(np.float64)                 # [C, 6, T, D]
    par = {}
    for a in ("spline", "spline_smooth", "chord"):
        ep = pca.lift_delta(dZ[:, :, ARMS.index(a), -1])                          # [C, T, D]
        ref = stored[:, info["arms"].index(a)]
        par[a] = float(np.abs(ep - ref).max() / np.abs(ref).max())
    assert max(par.values()) < 1e-5, par
    nat = np.linalg.norm(F["twin_meanpool"][:, :, L].astype(np.float64)
                         - F["src_meanpool"][:, None, L].astype(np.float64), axis=-1)   # [C, T]
    end = np.linalg.norm(dZ[:, :, :, -1], axis=-1)                               # [C, T, A] (components orthonormal)
    s = np.ones((C, T, len(ARMS)))
    for a in ("spline", "spline_smooth", "chord"):
        k = ARMS.index(a)
        s[:, :, k] = nat / np.maximum(end[:, :, k], 1e-12)
    dZs = dZ * s[..., None, None]
    rng = np.random.default_rng(7)
    r = rng.standard_normal((C, T, dZ.shape[-1]))
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    n_sp = np.linalg.norm(dZs[:, :, ARMS.index("spline")], axis=-1)              # [C, T, KW]
    dZs[:, :, ARMS.index("random")] = n_sp[..., None] * r[:, :, None]
    AP.mkdir(parents=True, exist_ok=True)
    np.savez(AP / "plan.npz", dZ=dZs, components=pca.components, scales=s, natural_twin_change_norm=nat,
             endpoint_norm_orig=end, idx=P["idx"], carrier_rows=P["rows"], targets=P["tg"], ts=TS,
             arms=np.array(ARMS), arc_sign=P["arc_sign"])
    (AP / "plan.json").write_text(json.dumps({"endpoint_parity_vs_plan_deltas_L22_rel_maxabs": par, "arms": ARMS,
                                              "ts": TS.tolist(), "random_seed": 7}, indent=1))
    print("parity", par, "scale medians", {a: float(np.median(s[:, :, i])) for i, a in enumerate(ARMS)})


def forward(batch_size=16):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    z = np.load(AP / "plan.npz")
    dZ, comp, rows = z["dZ"], z["components"], z["carrier_rows"]
    C, T, A, K, _ = dZ.shape
    ks = list(range(1, K))                                                        # t = 0 is the unedited forecast
    df = load_table("direction")
    t0 = time.time()
    G = 8
    for g in range(0, C, G):
        f = AP / f"group_{g:04d}.npz"
        if f.exists():
            continue
        cidx = list(range(g, min(g + G, C)))
        pv = preprocess([decode(df["video"].iloc[rows[i]]) for i in cidx]).to(device)
        saved, ctx_final, _ = prefix(model, pv[:, :8], [L])
        src_pp = pool_steps(predict_future(model, ctx_final, 0)).cpu().numpy()
        out = np.zeros((len(cidx), T, A, len(ks), 4, comp.shape[1]), np.float32)
        jobs = [(a, j, k, q) for a in range(len(cidx)) for j in range(T) for k in range(A) for q in range(len(ks))]
        for b in range(0, len(jobs), batch_size):
            ch = jobs[b:b + batch_size]
            ca = torch.as_tensor([c[0] for c in ch], device=device)
            dl = np.stack([dZ[cidx[a], j, k, ks[q]] @ comp for a, j, k, q in ch]).astype(np.float32)
            zz, _ = edited_prediction(model, saved[L][ca], L, torch.as_tensor(dl, device=device), 0)
            p = pool_steps(zz).cpu().numpy()
            for i, (a, j, k, q) in enumerate(ch):
                out[a, j, k, q] = p[i]
        np.savez(f, pred_pooled=out, src_pred_pooled=src_pp, cidx=np.array(cidx))
        print(f"{g + len(cidx)}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    info = {"seconds_this_invocation": time.time() - t0, "n_edits": int(C * T * A * len(ks)), "batch_size": batch_size,
            "arms": list(ARMS), "ts_edited": TS[1:].tolist(), "point": L, "mask_index": 0,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (AP / "forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def signed_wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def score():
    from run_session2 import angle_of, boot_ci, load_plan, wrap, write
    from wm import geometry_checks as gc
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets

    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    nm_json = json.loads((RES / "session2_predictor_norm_matched.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    d0 = load_inputs("direction", 0)
    probe = d0["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    cen = np.load(OUT / "centroids_direction.npy")

    def fit(X, Yt, alpha):
        st = Standardizer().fit(X)
        W, b = fit_ridge(st.transform(X), Yt, alpha)
        return st, W, b

    def apply(pr, X):
        st, W, b = pr
        sh = X.shape
        return predict(st.transform(X.reshape(-1, sh[-1])), W, b).reshape(sh[:-1] + (-1,))

    dir_nat = fit(Zall.mean(1)[probe], Y[probe], nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    pos_nat = {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        pos_nat[k] = fit(Zall[probe & ok, k], cen[probe & ok, t], nat_json["position"]["per_step"][str(t)]["native_alpha"])

    def read_pos(Z):
        return np.stack([apply(pos_nat[k], Z[..., k, :]) for k in range(4)], -2)

    z = np.load(AP / "plan.npz")
    idx, rows, tg = z["idx"], z["carrier_rows"], z["targets"]
    C, T, A, K, _ = z["dZ"].shape
    groups = sorted(AP.glob("group_*.npz"))
    pred = np.zeros((C, T, A, K, 4, Zall.shape[-1]), np.float32)
    src_pp = np.zeros((C, 4, Zall.shape[-1]), np.float32)
    for f in groups:
        g = np.load(f)
        pred[g["cidx"], :, :, 1:] = g["pred_pooled"]
        src_pp[g["cidx"]] = g["src_pred_pooled"]
    assert sum(len(np.load(f)["cidx"]) for f in groups) == C, "missing forward groups"
    pred[:, :, :, 0] = src_pp[:, None, None]                                     # t = 0: unedited forecast
    # parity: t = 1.0 of the path arms vs the norm-matched natural-norm forecasts (same edit, recomputed)
    nm = np.load(OUT / "norm_matched" / "pred_pooled.npz")
    assert (nm["idx"] == idx).all()
    nm_arms = nm_json["arms"]
    par = {a: float(np.abs(pred[:, :, ARMS.index(a), -1] - nm["pred_pooled"][:, 1, nm_arms.index(a)]).max()
                    / np.abs(nm["pred_pooled"][:, 1, nm_arms.index(a)]).max()) for a in ("spline", "spline_smooth", "chord")}
    par_src = float(np.abs(src_pp - nm["src_pred_pooled"]).max() / np.abs(nm["src_pred_pooled"]).max())

    src = y[rows]
    sw = signed_wrap(tg - src[:, None])
    shift = np.abs(sw)
    lt = shift < 179.99
    arc_agree = float((np.sign(sw[lt]) == z["arc_sign"][lt]).mean())
    shift_signed = z["arc_sign"] * shift                                          # [C, T] shorter arc (spline's at 180)
    ideal = (src[:, None, None] + TS[None, None, :] * shift_signed[..., None]) % 360.0   # [C, T, K]
    twin_true = np.load(OUT / "centroids_twins.npy")                             # [C, T, 4, 2]
    src_true = cen[rows][:, 4:8]
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                           # noqa: E731

    Pd = apply(dir_nat, pred.astype(np.float64).mean(-2))                        # [C, T, A, K, 2]
    ang = angle_of(Pd)
    rad = np.linalg.norm(Pd, axis=-1)
    pp = read_pos(pred.astype(np.float64))                                       # [C, T, A, K, 4, 2]
    err_ideal = wrap(ang - ideal[:, :, None])
    err_tgt = wrap(ang - tg[:, :, None, None])
    err_src = wrap(ang - src[:, None, None, None])
    a0 = ang[:, :, :, :1]
    progress = signed_wrap(ang - a0) * np.sign(shift_signed)[..., None, None] / np.maximum(shift, 1e-9)[..., None, None]
    px_twin = np.nanmean(dist(pp, twin_true[:, :, None, None]), -1)
    px_src = np.nanmean(dist(pp, src_true[:, None, None, None]), -1)
    jump = np.abs(signed_wrap(np.diff(ang, axis=-1))).max(-1)                   # [C, T, A] largest step change

    # geometry: edited point-22 meanpool = stored full-clip meanpool + delta; 5-NN to probe-fold clips
    d = load_inputs("direction", L)
    X = d["X"].astype(np.float64)
    held = np.array(load_plan(OUT)[1]["held_values"])
    knot = (d["role"] == "knot") & ~np.isin(d["y"], held)
    chart = gc.fit_circular_chart(X[knot], d["y"][knot])
    pinv = np.linalg.pinv(chart["A"]).T
    to_chart = lambda W: (W - chart["mu"]) @ pinv                                # noqa: E731
    ref_full, ref_ch = X[d["role"] == "probe"], to_chart(X[d["role"] == "probe"])
    Xt = X[d["role"] == "test"]
    floor = {"full": float(np.median(mf.knn_distance(Xt, ref_full, 5))),
             "chart2": float(np.median(mf.knn_distance(to_chart(Xt), ref_ch, 5)))}
    W = X[rows][:, None, None, None] + z["dZ"] @ z["components"]                 # [C, T, A, K, D]
    nn_full = mf.knn_distance(W.reshape(-1, W.shape[-1]), ref_full, 5).reshape(W.shape[:-1])
    Wc = to_chart(W.reshape(-1, W.shape[-1])).reshape(W.shape[:-1] + (2,))
    nn_ch = mf.knn_distance(Wc.reshape(-1, 2), ref_ch, 5).reshape(W.shape[:-1])
    ring_r = np.linalg.norm(Wc, axis=-1)
    del W

    per = {"err_to_ideal_intermediate": err_ideal, "err_to_target": err_tgt, "err_to_source": err_src,
           "forecast_radius": rad, "forecast_progress_frac": progress, "px_err_to_target_twin": px_twin,
           "px_err_to_source_future": px_src, "nn5_full": nn_full, "nn5_full_over_floor": nn_full / floor["full"],
           "nn5_ring_plane": nn_ch, "nn5_ring_plane_over_floor": nn_ch / floor["chart2"],
           "ring_plane_radius_chart_units": ring_r}
    sel_sets = {"all": np.ones_like(shift, bool), "shift_ge_90": shift >= 90, "shift_ge_135": shift >= 135}

    def carrier_mean(v, sel):                                                     # v [C, T] -> [C'] over selected T
        m = sel.sum(1) > 0
        return np.where(sel, v, 0).sum(1)[m] / sel.sum(1)[m]

    res = {"point": L, "n_carriers": int(C), "n_targets": int(T), "arms": list(ARMS), "ts": TS.tolist(),
           "forward": json.loads((AP / "forward_info.json").read_text()),
           "parity_t1_vs_norm_matched_natural_rel_maxabs": par,
           "spline_arc_equals_shorter_arc_frac_below_180": arc_agree, "n_exact_180": int((~lt).sum()), "parity_src_pred_vs_norm_matched_rel_maxabs": par_src,
           "real_clip_floor_5nn_median": floor, "shift_counts": {k: int(v.sum()) for k, v in sel_sets.items()},
           "applied_norms": {}, "by_t": {}, "summary": {}, "spline_minus_chord": {}, "definitions": {}}
    s, nat = z["scales"], z["natural_twin_change_norm"]
    wn = np.linalg.norm(z["dZ"], axis=-1)                                         # [C, T, A, K]
    for k, a in enumerate(ARMS):
        res["applied_norms"][a] = {
            "scale_median": float(np.median(s[:, :, k])),
            "endpoint_orig_norm_over_natural_median": float(np.median(z["endpoint_norm_orig"][:, :, k] / nat)),
            "waypoint_norm_over_natural_median_by_t": [float(np.median(wn[:, :, k, q] / nat)) for q in range(K)],
            "endpoint_norm_median": float(np.median(wn[:, :, k, -1]))}
    for sname, sel in sel_sets.items():
        bt, sm, sd = {}, {}, {}
        for k, a in enumerate(ARMS):
            bt[a] = {m: [boot_ci(carrier_mean(v[:, :, k, q], sel)) for q in range(K)] for m, v in per.items()}
            e = err_ideal[:, :, k, 1:].mean(-1)                                  # mean over t = 0.125..1
            sm[a] = {"mean_err_to_ideal_over_t": boot_ci(carrier_mean(e, sel)),
                     "min_forecast_radius_over_path": boot_ci(carrier_mean(rad[:, :, k].min(-1), sel)),
                     "max_step_jump_deg": boot_ci(carrier_mean(jump[:, :, k], sel)),
                     "mean_nn5_full_over_floor_over_t": boot_ci(carrier_mean(nn_full[:, :, k, 1:].mean(-1)
                                                                              / floor["full"], sel)),
                     "mean_nn5_ring_over_floor_over_t": boot_ci(carrier_mean(nn_ch[:, :, k, 1:].mean(-1)
                                                                              / floor["chart2"], sel))}
        for a in ("spline", "spline_smooth", "random"):
            ka, kc = ARMS.index(a), ARMS.index("chord")
            sd[f"{a}_minus_chord"] = {
                "mean_err_to_ideal_over_t": boot_ci(carrier_mean(err_ideal[:, :, ka, 1:].mean(-1)
                                                                 - err_ideal[:, :, kc, 1:].mean(-1), sel)),
                "min_forecast_radius": boot_ci(carrier_mean(rad[:, :, ka].min(-1) - rad[:, :, kc].min(-1), sel)),
                "err_to_ideal_by_t": [boot_ci(carrier_mean(err_ideal[:, :, ka, q] - err_ideal[:, :, kc, q], sel))
                                      for q in range(K)],
                "radius_by_t": [boot_ci(carrier_mean(rad[:, :, ka, q] - rad[:, :, kc, q], sel)) for q in range(K)],
                "px_twin_by_t": [boot_ci(carrier_mean(px_twin[:, :, ka, q] - px_twin[:, :, kc, q], sel))
                                 for q in range(K)]}
        res["by_t"][sname], res["summary"][sname], res["spline_minus_chord"][sname] = bt, sm, sd
    res["unedited_forecast_radius_all_probe_test_clips"] = float(np.median(np.linalg.norm(
        apply(dir_nat, Zall.mean(1)), axis=-1)))
    res["definitions"] = {
        "t": "fraction of the way along the path; waypoint k of 9 (t = 0 is the unedited clip, t = 1 the norm-matched "
             "natural-norm endpoint edit of session2_predictor_norm_matched.json)",
        "err_to_ideal_intermediate": "circular error (deg) of the native direction probe's forecast angle to "
                                     "src + t * (shorter-arc signed shift to target)",
        "forecast_radius": "norm of the native direction probe's (sin, cos) prediction of the step-mean pooled forecast",
        "forecast_progress_frac": "signed forecast-angle change from the unedited forecast along the shorter arc, "
                                  "divided by |shift| (1 = forecast moved the full shift)",
        "px_err_to_target_twin": "native per-step position probe vs the target twin's true disk centroids (px, 256 frame)",
        "nn5_*": "mean distance of the edited point-22 meanpool (stored full-clip meanpool + delta) to its 5 nearest "
                 "probe-fold clips; full = 1024-d, ring_plane = circular chart plane fit on knot clips (chart units); "
                 "_over_floor divides by the median of the same statistic for unedited test clips",
        "max_step_jump_deg": "largest |forecast-angle change| between consecutive waypoints",
        "CI": "bootstrap over carriers (targets averaged within carrier; shift subsets keep carriers with >= 1 target)",
        "random": "one random unit direction in the PCA-64 subspace per (carrier, target), straight line with the "
                  "scaled interpolating spline's waypoint norms"}
    write(RES / "session2_predictor_along_path.json", res, stage="along_path_score",
          seeds={"bootstrap": 0, "random_direction": 7}, cache=str(AP))
    for sname in ("all", "shift_ge_135"):
        print(sname)
        for a in ARMS:
            b = res["by_t"][sname][a]
            print(f"  {a:14s} " + " | ".join(
                f"t{TS[q]:.2f} e{b['err_to_ideal_intermediate'][q]['mean']:5.1f} r{b['forecast_radius'][q]['mean']:.2f}"
                f" px{b['px_err_to_target_twin'][q]['mean']:4.1f}" for q in (2, 4, 6, 8)),
                  "sum", round(res["summary"][sname][a]["mean_err_to_ideal_over_t"]["mean"], 1))
        print("  spline-chord", res["spline_minus_chord"][sname]["spline_minus_chord"]["mean_err_to_ideal_over_t"])
    print("parity", par, par_src)


# ================================================================ gap 2: reverse test (pullback through the predictor)
RV = AP / "reverse"
N_REV_CARRIERS, REV_STEPS, REV_LR, NORM_CAP = 20, 150, None, 1.2


def reverse_prep():
    """CPU, local. Pick 20 carriers x 2 targets at shift >= 135 deg from the gap-1 set; export the native direction
    probe as one affine map on the step-mean pooled forecast (P = pooled @ W + b)."""
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    X = Zall.mean(1)[probe]
    st = Standardizer().fit(X)
    W, b = fit_ridge(st.transform(X), targets(df, "direction")[0][probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    We, be = W / st.std[:, None], b - (st.mean / st.std) @ W
    z = np.load(AP / "plan.npz")
    rows, tg = z["carrier_rows"], z["targets"]
    src = y[rows]
    sh = np.abs(signed_wrap(tg - src[:, None]))
    ok = np.flatnonzero((sh >= 135).sum(1) >= 2)
    pick = np.sort(np.random.default_rng(0).choice(ok, N_REV_CARRIERS, replace=False))
    tj = np.stack([np.argsort(-np.where(sh[c] >= 135, sh[c], -1))[:2] for c in pick])      # 2 largest shifts
    ss = z["arc_sign"][pick[:, None], tj] * np.abs(signed_wrap(tg[pick[:, None], tj] - src[pick, None]))
    ideal = (src[pick, None, None] + TS[None, None, 1:] * ss[..., None]) % 360.0          # [N, 2, 8]
    nat = z["natural_twin_change_norm"][pick[:, None], tj]
    RV.mkdir(parents=True, exist_ok=True)
    np.savez(RV / "prep.npz", W=We, b=be, pick=pick, tj=tj, carrier_rows=rows[pick], ideal=ideal, nat=nat,
             components=z["components"], shift_signed=ss)
    print("picked", len(pick), "shifts", np.round(np.abs(ss), 1).tolist())


def reverse(steps=REV_STEPS, smoke=0):
    """Box, GPU. Per carrier (both targets batched: 2 x 8 waypoints), optimise PCA-64 waypoint coordinates with Adam
    so that the native probe's reading of the predictor forecast points at the ideal intermediate direction
    (loss 1 - cos(forecast angle - ideal)), projected after each step onto ||delta|| <= 1.2 x natural twin change.
    Init: zero edit + small noise (no path prior). Gradients through blocks 23-24, final LN and the predictor under
    bf16 autocast; the final waypoints are re-scored in fp32 with the gap-1 forward (edited_prediction)."""
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    pf = predict_future.__wrapped__                                            # the same forward, grad enabled
    z = np.load(RV / "prep.npz")
    Wt = torch.as_tensor(z["W"], dtype=torch.float32, device=device)
    bt = torch.as_tensor(z["b"], dtype=torch.float32, device=device)
    comp = torch.as_tensor(z["components"], dtype=torch.float32, device=device)
    N = len(z["pick"]) if not smoke else smoke
    ideal = torch.as_tensor(np.radians(z["ideal"]), dtype=torch.float32, device=device)   # [N, 2, 8]
    u = torch.stack([torch.sin(ideal), torch.cos(ideal)], -1)                             # [N, 2, 8, 2]
    cap = torch.as_tensor(NORM_CAP * z["nat"], dtype=torch.float32, device=device)        # [N, 2]
    df = load_table("direction")
    enc = model.encoder
    Wopt = np.zeros((N, 2, 8, comp.shape[0]))
    hist = np.zeros((N, steps, 2, 8))
    pred_fp32 = np.zeros((N, 2, 8, 4, comp.shape[1]), np.float32)
    lr = float(np.median(z["nat"])) / 20.0
    t0 = time.time()
    for n in range(N):
        pv = preprocess([decode(df["video"].iloc[int(z["carrier_rows"][n])])]).to(device)
        with torch.no_grad():
            saved, _, _ = prefix(model, pv[:, :8], [L])
        h0 = saved[L]                                                          # [1, 1024, D]
        g = torch.Generator(device="cpu").manual_seed(int(n))
        w = (1e-3 * float(z["nat"][n].mean()) * torch.randn(2, 8, comp.shape[0], generator=g)).to(device)
        w.requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = h0 + (w.reshape(16, -1) @ comp)[:, None, :]
                for layer in enc.layer[L:]:
                    h = layer(h, None, None, False)[0]
                zf = pf(model, enc.layernorm(h), 0)
            P = zf.float().mean(1) @ Wt + bt                                  # [16, 2]
            P = P.reshape(2, 8, 2)
            cosv = (P * u[n]).sum(-1) / P.norm(dim=-1).clamp_min(1e-6)
            loss = (1 - cosv).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                nr = w.norm(dim=-1)
                w.mul_(torch.clamp(cap[n][:, None] / nr.clamp_min(1e-12), max=1.0)[..., None])
            hist[n, it] = (1 - cosv).detach().cpu().numpy()
        Wopt[n] = w.detach().cpu().numpy()
        with torch.no_grad():                                                 # fp32 re-score, gap-1 forward
            d = (w.detach().reshape(16, -1) @ comp)
            zz, _ = edited_prediction(model, h0.expand(16, -1, -1), L, d, 0)
            pred_fp32[n] = pool_steps(zz).cpu().numpy().reshape(2, 8, 4, -1)
        print(f"{n + 1}/{N} carriers {time.time() - t0:.0f}s loss {hist[n, -1].mean():.3f} "
              f"(start {hist[n, 0].mean():.3f})", flush=True)
    info = {"seconds": time.time() - t0, "steps": steps, "lr": lr, "norm_cap_over_natural": NORM_CAP,
            "n_carriers": int(N), "autocast": "bf16 (optimisation only; final forecasts fp32)", "init": "zero + 1e-3 noise",
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if smoke:
        print(json.dumps(info, indent=1))
        return
    np.savez(RV / "reverse_out.npz", w=Wopt, loss_hist=hist, pred_pooled=pred_fp32)
    (RV / "reverse_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def reverse_score():
    """CPU, local. Does the predictor-optimised path lie on the ring? Per waypoint t: fp32 forecast read by the native
    probe (error to the ideal intermediate direction, radius); ring-plane radius and angle of the edited point-22
    meanpool; distance (PCA-64 = full space, since the edit lives in PCA-64) to the natural-norm spline and chord
    waypoints at the same t; share of the edit inside the ring plane."""
    from run_session2 import angle_of, boot_ci, load_plan, wrap, write
    from wm import geometry_checks as gc
    from wm.p2_data import load_inputs
    zr, zp, zo = np.load(RV / "prep.npz"), np.load(AP / "plan.npz"), np.load(RV / "reverse_out.npz")
    rinfo = json.loads((RV / "reverse_info.json").read_text())
    pick, tj, rows = zr["pick"], zr["tj"], zr["carrier_rows"]
    comp, nat, ideal = zr["components"], zr["nat"], zr["ideal"]                 # ideal [N, 2, 8]
    w = zo["w"]                                                                   # [N, 2, 8, 64]
    P = zo["pred_pooled"].astype(np.float64).mean(-2) @ zr["W"] + zr["b"]        # [N, 2, 8, 2]
    ang, rad = angle_of(P), np.linalg.norm(P, axis=-1)
    d = load_inputs("direction", L)
    X = d["X"].astype(np.float64)
    held = np.array(load_plan(OUT)[1]["held_values"])
    knot = (d["role"] == "knot") & ~np.isin(d["y"], held)
    chart = gc.fit_circular_chart(X[knot], d["y"][knot])
    pinv = np.linalg.pinv(chart["A"]).T
    to_chart = lambda V: (V - chart["mu"]) @ pinv                                # noqa: E731
    Qa = np.linalg.qr(chart["A"])[0]                                              # orthonormal ring-plane basis [D, 2]
    dZ = zp["dZ"][pick[:, None], tj][..., 1:, :]                                  # [N, 2, A, 8, 64] (t = 0.125..1)
    paths = {"optimised": w, "spline": dZ[:, :, ARMS.index("spline")], "chord": dZ[:, :, ARMS.index("chord")],
             "spline_smooth": dZ[:, :, ARMS.index("spline_smooth")]}
    x0 = X[rows][:, None, None]
    geo = {}
    for k, v in paths.items():
        delta = v @ comp
        c = to_chart(x0 + delta)
        geo[k] = {"ring_radius": np.linalg.norm(c, axis=-1),
                  "ring_angle_err_to_ideal": wrap(np.degrees(np.arctan2(c[..., 1], c[..., 0])) % 360.0 - ideal),
                  "edit_norm_over_natural": np.linalg.norm(v, axis=-1) / nat[..., None],
                  "ring_plane_share_of_edit": np.linalg.norm(delta @ Qa, axis=-1) / np.maximum(
                      np.linalg.norm(delta, axis=-1), 1e-12)}
    geo["optimised"]["dist_to_spline_over_natural"] = np.linalg.norm(w - paths["spline"], axis=-1) / nat[..., None]
    geo["optimised"]["dist_to_chord_over_natural"] = np.linalg.norm(w - paths["chord"], axis=-1) / nat[..., None]
    cosf = lambda a, b: (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)  # noqa: E731
    geo["optimised"]["cos_to_spline_edit"] = cosf(w, paths["spline"])
    geo["optimised"]["cos_to_chord_edit"] = cosf(w, paths["chord"])
    geo["optimised"]["ring_dist_to_spline_chart_units"] = np.linalg.norm(
        to_chart(x0 + w @ comp) - to_chart(x0 + paths["spline"] @ comp), axis=-1)
    geo["optimised"]["ring_dist_to_chord_chart_units"] = np.linalg.norm(
        to_chart(x0 + w @ comp) - to_chart(x0 + paths["chord"] @ comp), axis=-1)
    geo["optimised"]["forecast_err_to_ideal"] = wrap(ang - ideal)
    geo["optimised"]["forecast_radius"] = rad
    geo["optimised"]["final_loss_bf16"] = zo["loss_hist"][:, -1]
    ts = TS[1:]
    cm = lambda v: v.mean(1)                                                      # per carrier over its 2 targets  # noqa: E731
    out = {"n_carriers": int(len(pick)), "n_targets_per_carrier": 2, "ts": ts.tolist(),
           "shifts_deg": np.abs(zr["shift_signed"]).round(2).tolist(), "reverse": rinfo, "by_t": {}, "over_t": {},
           "paired_over_t": {}}
    for k, g in geo.items():
        out["by_t"][k] = {m: [boot_ci(cm(v[..., q])) for q in range(8)] for m, v in g.items()}
        out["over_t"][k] = {m: boot_ci(cm(v.mean(-1))) for m, v in g.items()}
    o = geo["optimised"]
    out["paired_over_t"] = {
        "dist_to_spline_minus_dist_to_chord": boot_ci(cm((o["dist_to_spline_over_natural"]
                                                         - o["dist_to_chord_over_natural"]).mean(-1))),
        "ring_radius_optimised_minus_chord": boot_ci(cm((o["ring_radius"] - geo["chord"]["ring_radius"]).mean(-1))),
        "ring_radius_optimised_minus_spline": boot_ci(cm((o["ring_radius"] - geo["spline"]["ring_radius"]).mean(-1))),
        "mid_ring_radius_optimised_minus_chord_t0.5": boot_ci(cm(o["ring_radius"][..., 3] - geo["chord"]["ring_radius"][..., 3]))}
    out["loss_curve_mean"] = zo["loss_hist"].mean((0, 2, 3))[::10].round(4).tolist()
    out["definitions"] = {
        "optimised": "PCA-64 waypoint edits at point 22 optimised through blocks 23-24 + predictor so the native "
                     "probe's forecast angle follows src + t * shift (loss 1 - cos), ||edit|| <= 1.2 x natural twin "
                     "change, zero init, Adam; forecasts re-scored in fp32",
        "spline / chord / spline_smooth": "the gap-1 natural-norm paths for the same carriers and targets",
        "ring_radius": "chart radius of the edited point-22 meanpool (1 = on the fitted ring; carriers' own ~1)",
        "ring_plane_share_of_edit": "||projection of the edit onto the ring plane|| / ||edit||",
        "dist_to_*_over_natural": "||w_opt(t) - w_path(t)|| / natural twin change norm (PCA-64 = full-space distance)",
        "CI": "bootstrap over 20 carriers (2 targets averaged within carrier)"}
    write(RES / "session2_reverse_path.json", out, stage="reverse_path_score", seeds={"bootstrap": 0, "init": "carrier index"},
          cache=str(RV))
    for k in ("optimised", "spline", "chord"):
        b = out["by_t"][k]
        print(k, "ring r", [round(b["ring_radius"][q]["mean"], 2) for q in range(8)],
              "share", [round(b["ring_plane_share_of_edit"][q]["mean"], 3) for q in (1, 3, 5, 7)])
    b = out["by_t"]["optimised"]
    print("fc err", [round(b["forecast_err_to_ideal"][q]["mean"], 1) for q in range(8)])
    print("fc rad", [round(b["forecast_radius"][q]["mean"], 2) for q in range(8)])
    print("d spline", [round(b["dist_to_spline_over_natural"][q]["mean"], 2) for q in range(8)])
    print("d chord", [round(b["dist_to_chord_over_natural"][q]["mean"], 2) for q in range(8)])
    print(out["paired_over_t"])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "reverse":
        reverse(steps=int(sys.argv[2]) if len(sys.argv) > 2 else REV_STEPS, smoke=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        {"plan": plan, "forward": forward, "score": score, "reverse_prep": reverse_prep,
         "reverse_score": reverse_score}[a]()

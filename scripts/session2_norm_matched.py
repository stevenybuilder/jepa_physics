"""Norm-matched predictor test at point 22 (external review fix: "spline best" may be a dose effect).

In session 2 the point-22 arms entered the predictor at different edit norms (interpolating spline 0.91x the natural
twin change, chord 0.65x, smoothing spline 0.62x, probe-QR 0.79x, radius-matched 0.58x), so forecast error to target
confounds edit direction with edit size. Here every arm's session-2 delta (plan.npz deltas_L22, the same 200 carriers
x 4 held-out targets) is rescaled per (carrier, target) to a common norm, in two conditions:

  chord     ||delta_arm|| := ||delta_chord||                       (the chord's own norm; chord is unchanged)
  natural   ||delta_arm|| := ||twin_meanpool_L22 - src_meanpool_L22||   (ratio 1.0 to the natural twin change, the
            same quantity as session2_propagation.json delta_over_natural_twin_change)

and added to every context token at point 22 exactly as run_session2.forward does (edited_prediction: context frames
1-8 encoded alone, blocks 23-24 + final LN, predictor with mask token 0, per-step pooled forecast).

  forward  box, GPU. -> artifacts/session2/norm_matched/pred_pooled.npz  (pooled [C, 2, A, T, 4, D] + applied scales)
  score    CPU, local. Forecasts read with the session-2 predictor-native probes (session2_predictor_native_readout.json:
           same rows, same alpha, no CV/refit; the fit is checked by reproducing that file's cached L22 numbers).
           -> results/session2_predictor_norm_matched.json
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
NM = OUT / "norm_matched"
RES = PROJECT_ROOT / "results"
L = 22
ARMS = ("probe_qr", "radius_matched", "spline", "spline_smooth", "chord", "random_matched")
CONDS = ("chord_norm", "natural_twin_norm")


def plan_scales(arrays, info, F):
    arms = info["arms"]
    ai = [arms.index(a) for a in ARMS]
    idx = F["idx"]
    dl = arrays[f"deltas_L{L}"][idx][:, ai].astype(np.float64)                     # [C, A, T, D]
    norm = np.linalg.norm(dl, axis=-1)                                             # [C, A, T]
    nat = np.linalg.norm(F["twin_meanpool"][:, :, L].astype(np.float64)
                         - F["src_meanpool"][:, None, L].astype(np.float64), axis=-1)   # [C, T]
    ref = np.stack([norm[:, ARMS.index("chord")], nat], 1)                         # [C, 2, T]
    s = ref[:, :, None] / np.maximum(norm[:, None], 1e-12)                         # [C, 2, A, T]
    return dl, norm, nat, s


def forward(batch_size=16):
    import torch
    from run_session2 import load_groups, load_plan
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_meanpool", "twin_meanpool", "src_pred_pooled",
                                           f"edit_pred_pooled_L{L}"})
    dl, norm, nat, s = plan_scales(arrays, info, F)
    D_ = (dl[:, None] * s[..., None]).astype(np.float32)                           # [C, 2, A, T, D]
    C, _, A, T, D = D_.shape
    rows = arrays["carrier_rows"][F["idx"]]
    df = load_table("direction")
    out = np.zeros((C, 2, A, T, 4, D), np.float32)
    src_pp = np.zeros((C, 4, D), np.float32)
    t0 = time.time()
    G = 8
    for g in range(0, C, G):
        cidx = list(range(g, min(g + G, C)))
        pv = preprocess([decode(df["video"].iloc[rows[i]]) for i in cidx]).to(device)
        saved, ctx_final, _ = prefix(model, pv[:, :8], [L])
        src_pp[cidx] = pool_steps(predict_future(model, ctx_final, 0)).cpu().numpy()
        jobs = [(a, c, k, t) for a in range(len(cidx)) for c in range(2) for k in range(A) for t in range(T)]
        for b in range(0, len(jobs), batch_size):
            ch = jobs[b:b + batch_size]
            ca = torch.as_tensor([j[0] for j in ch], device=device)
            d = torch.as_tensor(np.stack([D_[cidx[a], c, k, t] for a, c, k, t in ch]), device=device)
            z, _ = edited_prediction(model, saved[L][ca], L, d, 0)
            p = pool_steps(z).cpu().numpy()
            for q, (a, c, k, t) in enumerate(ch):
                out[cidx[a], c, k, t] = p[q]
        if g % 40 == 0:
            print(f"{g + len(cidx)}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    # parity: the chord at chord norm is the session-2 chord edit unchanged -> must match the cached forecast
    ref_ch = F[f"edit_pred_pooled_L{L}"][:, [a for a in info["arms"] if a in ARMS].index("chord")]
    par_chord = float(np.abs(out[:, 0, ARMS.index("chord")] - ref_ch).max() / np.abs(ref_ch).max())
    par_src = float(np.abs(src_pp - F["src_pred_pooled"]).max() / np.abs(F["src_pred_pooled"]).max())
    NM.mkdir(parents=True, exist_ok=True)
    np.savez(NM / "pred_pooled.npz", pred_pooled=out, src_pred_pooled=src_pp, scales=s.astype(np.float32),
             delta_norm_orig=norm.astype(np.float32), natural_twin_change_norm=nat.astype(np.float32),
             idx=F["idx"])
    info_out = {"seconds": secs, "n_edits": int(C * 2 * A * T), "batch_size": batch_size, "arms": list(ARMS),
                "conditions": list(CONDS), "point": L, "mask_index": 0,
                "parity_chord_at_chord_norm_vs_session2_cache_rel_maxabs": par_chord,
                "parity_src_pred_vs_session2_cache_rel_maxabs": par_src,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (NM / "forward_info.json").write_text(json.dumps(info_out, indent=1))
    print(json.dumps(info_out, indent=1))


def score():
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, shuffle_index, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets

    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    cen = np.load(OUT / "centroids_direction.npy")

    def fit(X, Yt, alpha):                                                         # same rows + stored alpha: no CV
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

    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", f"edit_pred_pooled_L{L}", "src_meanpool",
                                           "twin_meanpool"})
    z = np.load(NM / "pred_pooled.npz")
    finfo = json.loads((NM / "forward_info.json").read_text())
    assert (z["idx"] == F["idx"]).all()
    idx = F["idx"]
    rows, tg = arrays["carrier_rows"][idx], arrays["targets"][idx]
    C, T = tg.shape
    twin_true = np.load(OUT / "centroids_twins.npy")
    src_true = cen[rows][:, 4:8]
    p0 = cen[rows, 0]
    flipped_pos = 2 * p0[:, None, None] - twin_true
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                             # noqa: E731
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa: E731
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, b, ax: (np.nansum(a * b, ax) / np.maximum(np.nansum(b * b, ax), 1e-9))   # noqa: E731
    Ps = apply(dir_nat, F["src_pred_pooled"].astype(np.float64).mean(1))
    ps = read_pos(F["src_pred_pooled"].astype(np.float64))

    def per_carrier(Z):                                                            # Z [C, T, 4, D]
        Pe = apply(dir_nat, Z.mean(2))
        a = angle_of(Pe)
        pe = read_pos(Z)
        et, ef = wrap(a - tg).mean(1), wrap(a - (tg + 180.0)).mean(1)
        return {"dir_err_to_target": et, "dir_err_to_flipped180_target": ef, "dir_err_target_minus_flipped180": et - ef,
                "R_dir_real_change": proj(Pe - Ps[:, None], d_true, -1).mean(1),
                "px_err_to_twin_true": np.nanmean(dist(pe, twin_true), (1, 2)),
                "px_err_to_flipped180_twin": np.nanmean(dist(pe, flipped_pos), (1, 2))}

    # reproduction check: the session-2 cached L22 forecasts through these probes vs the native-readout file
    repro = {}
    for ip, arm in enumerate([a for a in info["arms"] if a in ARMS]):
        pc = per_carrier(F[f"edit_pred_pooled_L{L}"][:, ip].astype(np.float64))
        old = nat_json["per_layer"][str(L)][arm]
        repro[arm] = {k: [float(pc[k].mean()), old[k]["mean"]] for k in ("dir_err_to_target", "px_err_to_twin_true",
                                                                          "R_dir_real_change")}
    repro_maxabs = max(abs(a - b) for r in repro.values() for a, b in r.values())
    assert repro_maxabs < 1e-6, f"probe reconstruction does not reproduce the native readout ({repro_maxabs})"

    s, norm, nat = z["scales"].astype(np.float64), z["delta_norm_orig"].astype(np.float64), z["natural_twin_change_norm"]
    res = {"point": L, "n_carriers": int(C), "n_targets": int(T), "arms": list(ARMS), "conditions": {
        "chord_norm": "each arm's session-2 delta rescaled per (carrier, target) to ||delta_chord|| (chord unchanged)",
        "natural_twin_norm": "each arm's delta rescaled per (carrier, target) to ||twin_meanpool_L22 - src_meanpool_L22||"
                             " (full-clip meanpool, the session2_propagation delta_over_natural_twin_change quantity)"},
        "forward": finfo, "readout_reproduction_of_native_file_maxabs": repro_maxabs, "per_condition": {},
        "applied_norms": {}, "spline_minus_chord": {}}
    base = {}
    for ci, cond in enumerate(CONDS):
        rows_c, pcs = {}, {}
        applied = {}
        for k, arm in enumerate(ARMS):
            pc = per_carrier(z["pred_pooled"][:, ci, k].astype(np.float64))
            pcs[arm] = pc
            rows_c[arm] = {m: boot_ci(v) for m, v in pc.items()}
            new = norm[:, k] * s[:, ci, k]
            applied[arm] = {"scale_median": float(np.median(s[:, ci, k])),
                            "scale_p5_p95": [float(np.percentile(s[:, ci, k], 5)), float(np.percentile(s[:, ci, k], 95))],
                            "orig_norm_over_natural_median": float(np.median(norm[:, k] / nat)),
                            "applied_norm_over_natural_median": float(np.median(new / nat)),
                            "applied_norm_over_chord_median": float(np.median(new / norm[:, ARMS.index("chord")])),
                            "applied_norm_median": float(np.median(new))}
        res["per_condition"][cond] = rows_c
        res["applied_norms"][cond] = applied
        res["spline_minus_chord"][cond] = {m: boot_ci(pcs["spline"][m] - pcs["chord"][m]) for m in
                                           ("dir_err_to_target", "R_dir_real_change", "px_err_to_twin_true")}
        res["spline_minus_chord"][cond]["spline_smooth_minus_chord_dir_err"] = boot_ci(
            pcs["spline_smooth"]["dir_err_to_target"] - pcs["chord"]["dir_err_to_target"])
        base[cond] = pcs
    # session-2 unmatched numbers side by side (same probes)
    res["session2_unmatched"] = {arm: {k: v[0] for k, v in r.items()} for arm, r in repro.items()}
    res["unedited"] = {"dir_err_to_target": boot_ci(wrap(angle_of(Ps)[:, None] - tg).mean(1)),
                       "px_err_to_twin_true": boot_ci(np.nanmean(dist(ps[:, None], twin_true), (1, 2)))}
    res["definitions"] = {
        "dir_err_*": "circular error (deg) of the native direction probe's reading of the step-mean pooled forecast, "
                     "mean over the 4 targets per carrier; CI = bootstrap over 200 carriers",
        "dir_err_target_minus_flipped180": "paired per-carrier difference; flipped = target + 180 deg (for a single "
                                           "angle this equals 2*err - 180 by construction)",
        "R_dir_real_change": "<P_edit - P_src, u(target) - u(source)> / ||u(target) - u(source)||^2 in probe output space",
        "px_err_to_twin_true": "native per-step position probe vs the rendered twin's true disk centroids (256 px frame)",
        "readout": "probes = session2_predictor_native_readout.json (same 480 probe clips, stored alpha, no CV); "
                   "reconstruction reproduces that file's cached L22 numbers (readout_reproduction_of_native_file_maxabs)"}
    write(RES / "session2_predictor_norm_matched.json", res, stage="norm_matched_score",
          seeds={"bootstrap": 0}, cache=str(NM / "pred_pooled.npz"))
    for cond in CONDS:
        print(cond)
        for arm in ARMS:
            r = res["per_condition"][cond][arm]
            print(f"  {arm:15s} tgt {r['dir_err_to_target']['mean']:6.1f} flip {r['dir_err_to_flipped180_target']['mean']:6.1f}"
                  f" R {r['R_dir_real_change']['mean']:.3f} px {r['px_err_to_twin_true']['mean']:.1f}")
        print("  spline-chord", res["spline_minus_chord"][cond]["dir_err_to_target"])


if __name__ == "__main__":
    {"forward": forward, "score": score}[sys.argv[1]]()

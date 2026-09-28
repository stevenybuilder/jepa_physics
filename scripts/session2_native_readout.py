"""Predictor-native readouts for the session-2 predictor test (external review fix).

The session-2 predictor test read the predictor's forecast tokens with probes fit on REAL encoder tokens; the
unedited forecast then read its own clip's direction ~61 deg off, so the test had no power. Here the readouts are fit
on the predictor's OWN outputs:

  extract  box, GPU. All 1500 direction clips: frames 1-8 encoded alone, predictor (mask token 0, same convention as
           run_session2.forward) forecasts tubelets 4-7; the per-step pooled forecast [1500, 4, 1024] is saved to
           artifacts/session2/native/pred_pooled_all.npy (df row order). Unedited only.
  score    CPU, local. (1) direction probe (ridge on [sin, cos], alpha by 5-fold CV inside the 480 probe clips) and a
           per-step disk-position probe (pixel centroid, same targets as session2_extras) on the step-mean /
           per-step predictor forecasts of probe clips; scored on the 300 test clips' forecasts, next to the
           real-token probes applied to the same forecasts. (2) if the gate passes, the cached edited forecasts
           (forward/group_*.npz: edit_pred_pooled_L*) are re-read with these probes: error to the true target, to the
           45-deg-arc shuffled target, to the 180-deg-flipped target, and recovery against the real twin change.
-> results/session2_predictor_native_readout.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
NAT = OUT / "native"
RES = PROJECT_ROOT / "results"
D = 1024
GATE_PASS_DEG, GATE_STOP_DEG = 20.0, 40.0


def extract(batch_size=16):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    df = load_table("direction")
    NAT.mkdir(parents=True, exist_ok=True)
    out = np.zeros((len(df), 4, D), np.float32)
    t0 = time.time()
    for s in range(0, len(df), batch_size):
        frames = [decode(v) for v in df["video"].iloc[s:s + batch_size]]
        pv = preprocess(frames).to(device)
        _, ctx_final, _ = prefix(model, pv[:, :8], [])
        out[s:s + len(frames)] = pool_steps(predict_future(model, ctx_final, 0)).cpu().numpy()
        if s % 320 == 0:
            print(f"{s + len(frames)}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    np.save(NAT / "pred_pooled_all.npy", out)
    (NAT / "extract_info.json").write_text(json.dumps({
        "ids": df["id"].tolist(), "seconds": time.time() - t0, "batch_size": batch_size, "mask_index": 0,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}, indent=1))
    print("done", time.time() - t0)


def score_all():
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score, targets
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, shuffle_index, wrap, write

    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]                                               # [sin, cos]
    d0 = load_inputs("direction", 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    rng = np.random.default_rng(0)
    folds5 = rng.permutation(np.arange(probe.sum()) % 5)                          # 5-fold CV inside probe clips
    Zall = np.load(NAT / "pred_pooled_all.npy").astype(np.float64)               # [1500, 4, D]
    cen = np.load(OUT / "centroids_direction.npy")                               # [1500, 8, 2]
    tp = np.asarray(np.load(PROJECT_ROOT / "artifacts/activations/direction/vjepa2/timepool.npy",
                            mmap_mode="r")[:, 25, 4:8], np.float64)              # real future tokens, per step
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward")
    idx = F["idx"]
    rows, tg = arrays["carrier_rows"][idx], arrays["targets"][idx]
    C, T = tg.shape
    shuf = shuffle_index(T)
    pred_arms = [a for a in info["arms"] if a in ("probe_qr", "radius_matched", "spline", "spline_smooth", "chord",
                                                  "random_matched")]
    parity = float(np.abs(Zall[rows] - F["src_pred_pooled"]).max() / np.abs(F["src_pred_pooled"]).max())

    def fit(X, Yt, fl, kind):
        st = Standardizer().fit(X)
        cv = cv_select_alpha(st.transform(X), Yt, fl, score_fn=lambda a, b: score(a, b, kind))
        W, b = fit_ridge(st.transform(X), Yt, cv["alpha"])
        return (st, W, b), cv

    def apply(pr, X):
        st, W, b = pr
        sh = X.shape
        return predict(st.transform(X.reshape(-1, sh[-1])), W, b).reshape(sh[:-1] + (-1,))

    def dir_summary(P, theta):
        e = wrap(angle_of(P) - theta)
        return {"circ_mae_deg": float(e.mean()), "median_err_deg": float(np.median(e)),
                "acc15": float((e <= 15).mean()), "r2_sincos": score(np.c_[np.sin(np.radians(theta)),
                                                                            np.cos(np.radians(theta))], P,
                                                                      "circular")["r2"], "n": int(len(theta))}

    # ---------------- (1) direction readouts
    res = {"n_probe": int(probe.sum()), "n_test": int(test.sum()),
           "parity_native_cache_vs_session2_src_pred_pooled_rel_maxabs": parity}
    Xn = Zall.mean(1)
    dir_nat, cvn = fit(Xn[probe], Y[probe], folds5, "circular")
    Xc = Zall.reshape(len(Zall), -1)
    dir_cat, cvc = fit(Xc[probe], Y[probe], folds5, "circular")
    Xr = tp.mean(1)
    dir_real, cvr = fit(Xr[probe], Y[probe], d0["fold"][probe], "circular")     # session-2 future_probe replica
    res["direction"] = {
        "predictor_output_probe_stepmean": {"alpha": cvn["alpha"], "cv_r2": cvn["cv_mean"], "cv_mae": cvn["cv_mae_mean"],
                                            "test_on_predictor_output": dir_summary(apply(dir_nat, Xn[test]), y[test])},
        "predictor_output_probe_concat4steps": {"alpha": cvc["alpha"], "cv_r2": cvc["cv_mean"],
                                                "cv_mae": cvc["cv_mae_mean"],
                                                "test_on_predictor_output": dir_summary(apply(dir_cat, Xc[test]),
                                                                                        y[test])},
        "real_token_probe_stepmean": {"alpha": cvr["alpha"], "cv_r2": cvr["cv_mean"],
                                      "test_on_real_future": dir_summary(apply(dir_real, Xr[test]), y[test]),
                                      "test_on_predictor_output": dir_summary(apply(dir_real, Xn[test]), y[test])}}
    # ---------------- (1) per-step position readouts
    pos_nat, pos_real, prow = {}, {}, {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        m, mt = probe & ok, test & ok
        pos_nat[k], cv = fit(Zall[m, k], cen[m, t], folds5[ok[probe]], "linear")
        pos_real[k], cvr_ = fit(tp[m, k], cen[m, t], d0["fold"][probe][ok[probe]], "linear")
        en = np.linalg.norm(apply(pos_nat[k], Zall[mt, k]) - cen[mt, t], axis=1)
        er = np.linalg.norm(apply(pos_real[k], Zall[mt, k]) - cen[mt, t], axis=1)
        ereal = np.linalg.norm(apply(pos_real[k], tp[mt, k]) - cen[mt, t], axis=1)
        prow[str(t)] = {"native_alpha": cv["alpha"], "native_cv_r2": cv["cv_mean"],
                        "native_test_r2": score(cen[mt, t], apply(pos_nat[k], Zall[mt, k]), "linear")["r2"],
                        "native_px_mean": float(en.mean()), "native_px_median": float(np.median(en)),
                        "real_token_probe_on_predictor_px_mean": float(er.mean()),
                        "real_token_probe_on_real_future_px_mean": float(ereal.mean()), "n_test": int(mt.sum())}
    res["position"] = {"per_step": prow,
                       "native_px_mean_over_steps": float(np.mean([r["native_px_mean"] for r in prow.values()])),
                       "real_token_probe_on_predictor_px_mean_over_steps":
                           float(np.mean([r["real_token_probe_on_predictor_px_mean"] for r in prow.values()])),
                       "px_units": "256x256 frame; disk radius ~10.6 px; mean source-to-twin disk gap 58.1 px "
                                   "(session2_future_position.json)"}

    def read_pos(Z):                                                              # [..., 4, D] -> [..., 4, 2]
        return np.stack([apply(pos_nat[k], Z[..., k, :]) for k in range(4)], -2)

    # predictor forecast from the real twin context, read natively (what a perfect context edit could reach)
    twin_true = np.load(OUT / "centroids_twins.npy")                             # [C, T, 4, 2]
    src_true = cen[rows][:, 4:8]
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                           # noqa: E731
    Ps, Pt = apply(dir_nat, F["src_pred_pooled"].mean(1)), apply(dir_nat, F["twin_pred_pooled"].mean(2))
    ps, pt = read_pos(F["src_pred_pooled"]), read_pos(F["twin_pred_pooled"])
    res["twin_context_reference"] = {
        "dir_err_pred_src_vs_src_true": boot_ci(wrap(angle_of(Ps) - y[rows])),
        "dir_err_pred_twin_vs_target": boot_ci(wrap(angle_of(Pt) - tg).mean(1)),
        "dir_err_pred_twin_vs_src_true": boot_ci(wrap(angle_of(Pt) - y[rows][:, None]).mean(1)),
        "px_pred_src_vs_src_true": boot_ci(np.nanmean(dist(ps, src_true), 1)),
        "px_pred_twin_vs_twin_true": boot_ci(np.nanmean(dist(pt, twin_true), (1, 2))),
        "px_pred_src_vs_twin_true": boot_ci(np.nanmean(dist(ps[:, None], twin_true), (1, 2))),
        "note": "carriers (200 test clips) x 4 targets; the predictor's forecast from the rendered twin's own context"}

    gate = res["direction"]["predictor_output_probe_stepmean"]["test_on_predictor_output"]["circ_mae_deg"]
    px = res["position"]["native_px_mean_over_steps"]
    res["gate"] = {"unedited_test_dir_mae_deg": gate, "unedited_test_px_mean": px,
                   "rule": f"step 2 if MAE <= ~{GATE_PASS_DEG} deg or px well below the 58 px source-twin gap; "
                           f"stop if MAE > ~{GATE_STOP_DEG} deg",
                   "step2_ran": bool(gate <= GATE_PASS_DEG or px < 29.0) and not gate > GATE_STOP_DEG}
    print(json.dumps({"direction": {k: v.get("test_on_predictor_output") for k, v in res["direction"].items()},
                      "position_px": px, "gate": res["gate"]}, indent=1))

    # ---------------- (2) re-read cached edited forecasts
    if res["gate"]["step2_ran"]:
        u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)   # noqa: E731
        d_true = u(tg) - u(y[rows])[:, None]                                        # real direction change [C, T, 2]
        p0 = cen[rows, 0]                                                           # tubelet-0 centroid (shared start)
        flipped_pos = 2 * p0[:, None, None] - twin_true                             # twin reflected about the start
        d_pos = twin_true - src_true[:, None]                                       # real position change [C, T, 4, 2]
        proj = lambda a, b, ax: (np.nansum(a * b, ax) / np.maximum(np.nansum(b * b, ax), 1e-9))   # noqa: E731
        res["twin_context_reference"]["R_dir_ceiling"] = boot_ci(proj(Pt - Ps[:, None], d_true, -1).mean(1))
        res["twin_context_reference"]["R_px_ceiling"] = boot_ci(proj(pt - ps[:, None], d_pos, (-1, -2)).mean(1))
        old = json.loads((RES / "session2_predictor.json").read_text())["per_layer"]
        per = {}
        for L in info["layers"]:
            row = {}
            for ip, arm in enumerate(pred_arms):
                Z = F[f"edit_pred_pooled_L{L}"][:, ip].astype(np.float64)          # [C, T, 4, D]
                Pe = apply(dir_nat, Z.mean(2))
                a = angle_of(Pe)
                pe = read_pos(Z)
                row[arm] = {
                    "dir_err_to_target": boot_ci(wrap(a - tg).mean(1)),
                    "dir_err_to_shuffled45_target": boot_ci(wrap(a - tg[:, shuf]).mean(1)),
                    "dir_err_to_flipped180_target": boot_ci(wrap(a - (tg + 180.0)).mean(1)),
                    "dir_err_target_minus_shuffled45_paired": boot_ci((wrap(a - tg) - wrap(a - tg[:, shuf])).mean(1)),
                    "dir_err_to_source_true": boot_ci(wrap(a - y[rows][:, None]).mean(1)),
                    "dir_shift_from_unedited_deg": boot_ci(wrap(a - angle_of(Ps)[:, None]).mean(1)),
                    "R_dir_real_change": boot_ci(proj(Pe - Ps[:, None], d_true, -1).mean(1)),
                    "R_dir_real_change_shuffled45": boot_ci(proj(Pe - Ps[:, None], d_true[:, shuf], -1).mean(1)),
                    "px_err_to_twin_true": boot_ci(np.nanmean(dist(pe, twin_true), (1, 2))),
                    "px_err_to_shuffled45_twin": boot_ci(np.nanmean(dist(pe, twin_true[:, shuf]), (1, 2))),
                    "px_err_to_flipped180_twin": boot_ci(np.nanmean(dist(pe, flipped_pos), (1, 2))),
                    "px_err_twin_minus_shuffled45_paired": boot_ci(np.nanmean(dist(pe, twin_true)
                                                                              - dist(pe, twin_true[:, shuf]), (1, 2))),
                    "px_shift_from_unedited": boot_ci(np.nanmean(dist(pe, ps[:, None]), (1, 2))),
                    "R_px_real_change": boot_ci(proj(pe - ps[:, None], d_pos, (-1, -2)).mean(1)),
                    "R_token_realdiff_session2": old[str(L)][arm]["R_realdiff"]}
            row["zero"] = {"dir_err_to_target": boot_ci(wrap(angle_of(Ps)[:, None] - tg).mean(1)),
                           "dir_err_to_shuffled45_target": boot_ci(wrap(angle_of(Ps)[:, None] - tg[:, shuf]).mean(1)),
                           "dir_err_to_flipped180_target": boot_ci(wrap(angle_of(Ps)[:, None] - (tg + 180)).mean(1)),
                           "px_err_to_twin_true": boot_ci(np.nanmean(dist(ps[:, None], twin_true), (1, 2))),
                           "px_err_to_flipped180_twin": boot_ci(np.nanmean(dist(ps[:, None], flipped_pos), (1, 2)))}
            per[str(L)] = row
        res["per_layer"] = per
        res["definitions"] = {
            "shuffled45": "target (j+1) mod 4 of the same carrier; all 4 targets lie inside one 45-deg arc",
            "flipped180": "direction target + 180 deg; for position, the twin trajectory reflected about the source's "
                          "tubelet-0 centroid (ignores frame edges)",
            "R_dir_real_change": "<P_edit - P_src, u(target) - u(source)> / ||u(target) - u(source)||^2 in the native "
                                 "probe's [sin, cos] output space, u = unit vector of the true angle",
            "R_px_real_change": "<p_edit - p_src, twin_true - src_true> / ||twin_true - src_true||^2 over the 4 future "
                                "steps, native position probe, true pixel centroids",
            "R_token_realdiff_session2": "probe-free token R from session2_predictor.json, copied for reference"}
    write(RES / "session2_predictor_native_readout.json", res, stage="native_readout",
          seeds={"cv_folds": 0, "bootstrap": 0},
          readout="probes fit on the predictor's own unedited forecasts of the 480 probe clips (folds 3-4)",
          cache=str(NAT / "pred_pooled_all.npy"))
    return res


if __name__ == "__main__":
    {"extract": extract, "score": score_all}[sys.argv[1]]()

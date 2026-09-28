"""Session-2 follow-ups on the predictor-native readout (same carriers, targets, twins, probes as session 2).

A. Paper-site steering. Goodfire §5 steers the encoder output the world model consumes (v_t = LayerNorm f_enc(x_t)).
   For V-JEPA 2 that is point 25: the final LayerNorm of block 24 (wm.extract.encode: hs[24] = last_hidden_state),
   exactly the ctx_final tokens predictor_readout.predict_future receives. The six session-2 arms are rebuilt at point 25
   with run_session2.layer_deltas (same knot/probe/carrier rows, same held-out arc and targets), added to every
   context token of the post-LN encoding, and the predictor forecasts tubelets 4-7. Three conditions per arm:
   unmatched (the arm's own norm), chord_norm and natural_twin_norm (session2_norm_matched's two rescalings, with the
   natural twin change measured at point 25).
B. Point-12 interpolating spline on the labels angle. Session 2 built the point-12 spline on the unsupervised centroid-
   plane angle (choose_angle_source; scrambled knot order). Here only that arm is rebuilt with the labels angle
   (activation plane, monotone knot order) and pushed through the point-12 forward/predictor path; the stored chord
   at point 12 is re-run beside it as a bit-parity check against the session-2 cache.

  plan     CPU, local. -> <encout>/plan_encout.npz (+ parity of layer_deltas against plan.npz at points 12 and 22)
  forward  box, GPU. -> <encout>/pred_pooled.npz, <labels_out>/pred_pooled.npz, forward_info.json in each
  score    CPU, local. -> results/session2_encoder_output.json, results/session2_interp_labels_L12.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
RES = PROJECT_ROOT / "results"
ENC = 25                       # final LayerNorm = the predictor's input
LB = 12
ARMS = ("probe_qr", "radius_matched", "spline", "spline_smooth", "chord", "random_matched")
CONDS = ("unmatched", "chord_norm", "natural_twin_norm")


def rescale(dl, nat, chord_index):
    """Per (carrier, target) scales: [C, 2, A, T] for (chord norm, natural twin norm). dl [C, A, T, D], nat [C, T]."""
    norm = np.linalg.norm(dl, axis=-1)
    ref = np.stack([norm[:, chord_index], nat], 1)
    return ref[:, :, None] / np.maximum(norm[:, None], 1e-12)


def plan(args):
    from run_session2 import DATASET, layer_deltas, load_groups, load_plan
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_meanpool", "twin_meanpool"})
    idx = F["idx"]
    assert (idx == np.arange(len(idx))).all()
    carriers, tg = arrays["carrier_rows"], arrays["targets"]
    arms = info["arms"]
    assert tuple(arms) == ARMS
    d0 = load_inputs(DATASET, info["layers"][0])
    y = d0["y"]
    held = np.asarray(info["held_values"])
    kw = dict(holdout=info["holdout"]["design"], part1_mode=info["part1_basis"], k=info["k"], spline=info["spline"],
              waypoints=info["waypoints"])
    out = Path(args.encout)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    # parity: layer_deltas (current code, fresh INLP refit) reproduces session-2 plan.npz at points 12 and 22
    parity = {}
    for L in (LB, 22):
        dl, _ = layer_deltas(load_inputs(DATASET, L), L, y, carriers, tg, held, arms, out, **kw)
        ref = arrays[f"deltas_L{L}"]
        parity[str(L)] = float(np.abs(dl - ref).max() / np.abs(ref).max())
        print(f"parity L{L}: rel maxabs {parity[str(L)]:.2e}")
    assert max(parity.values()) < 1e-5, parity
    # A: every arm at point 25
    t0 = time.time()
    dA, mA = layer_deltas(load_inputs(DATASET, ENC), ENC, y, carriers, tg, held, arms, out, **kw)
    nat25 = np.linalg.norm(F["twin_meanpool"][:, :, ENC].astype(np.float64)
                           - F["src_meanpool"][:, None, ENC].astype(np.float64), axis=-1)
    sA = rescale(dA.astype(np.float64), nat25, arms.index("chord"))
    # B: labels-angle interpolating spline at point 12
    dB, mB = layer_deltas(load_inputs(DATASET, LB), LB, y, carriers, tg, held, arms, out, angle="labels", **kw)
    # natural centroid change (plan's reference: probe-clip raw centroids at source and target value)
    cc = {}
    for L, X in ((ENC, mA["X"]), (LB, mB["X"])):
        probe = d0["role"] == "probe"
        mu = {v: X[probe & np.isclose(y, v)].mean(0) for v in np.unique(y)}
        cc[L] = np.array([[np.linalg.norm(mu[tg[i, j]] - mu[y[c]]) for j in range(tg.shape[1])]
                          for i, c in enumerate(carriers)])
    np.savez(out / "plan_encout.npz", deltas_L25=dA, scales_L25=sA.astype(np.float32), nat_twin_L25=nat25,
             centroid_change_L25=cc[ENC], delta_L12_spline_labels=dB[:, arms.index("spline")],
             delta_L12_chord_labels=dB[:, arms.index("chord")], delta_L12_chord_stored=arrays["deltas_L12"][:, arms.index("chord")],
             delta_L12_spline_stored=arrays["deltas_L12"][:, arms.index("spline")], centroid_change_L12=cc[LB],
             carrier_rows=carriers, targets=tg, idx=idx)
    meta = {"parity_layer_deltas_vs_plan_rel_maxabs": parity, "seconds": time.time() - t0,
            "L25": {"K": mA["K"], "rank": int(mA["V"].shape[1]), "basis_source": mA["bsrc"],
                    "angle_choice": {k: mA["choice"][k] for k in ("angle", "plane")}},
            "L12_labels": {"angle_choice": {k: mB["choice"][k] for k in ("angle", "plane")}},
            "L12_session2_angle_choice": info["per_layer"][str(LB)]["angle_choice"]}
    (out / "plan_info.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def forward(args):
    import torch
    from run_session2 import load_groups
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    set_precision()
    torch.set_num_threads(args.threads)
    device = pick_device()
    model = load_model("vjepa2", device)
    enc, lab = Path(args.encout), Path(args.labels_out)
    lab.mkdir(parents=True, exist_ok=True)
    z = np.load(enc / "plan_encout.npz")
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", f"edit_pred_pooled_L{LB}"})
    assert (z["idx"] == F["idx"]).all()
    rows = z["carrier_rows"]
    dA, sA = z["deltas_L25"], z["scales_L25"]
    C, A, T, D = dA.shape
    DA = np.stack([dA, dA * sA[:, 0, :, :, None], dA * sA[:, 1, :, :, None]], 1).astype(np.float32)  # [C, 3, A, T, D]
    DB = np.stack([z["delta_L12_spline_labels"], z["delta_L12_chord_stored"]], 1).astype(np.float32)   # [C, 2, T, D]
    outA = np.zeros((C, 3, A, T, 4, D), np.float32)
    outB = np.zeros((C, 2, T, 4, D), np.float32)
    src_pp = np.zeros((C, 4, D), np.float32)
    df = load_table("direction")
    t0, tA, tB = time.time(), 0.0, 0.0
    bs, G = args.batch_size, 8
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    for g in range(0, C, G):
        cidx = list(range(g, min(g + G, C)))
        pv = preprocess([decode(df["video"].iloc[rows[i]]) for i in cidx]).to(device)
        saved, ctx_final, _ = prefix(model, pv[:, :8], [LB])
        src_pp[cidx] = pool_steps(predict_future(model, ctx_final, 0)).cpu().numpy()
        sync(); t1 = time.time()
        jobs = [(a, c, k, t) for a in range(len(cidx)) for c in range(3) for k in range(A) for t in range(T)]
        for b in range(0, len(jobs), bs):
            ch = jobs[b:b + bs]
            ca = torch.as_tensor([j[0] for j in ch], device=device)
            d = torch.as_tensor(np.stack([DA[cidx[a], c, k, t] for a, c, k, t in ch]), device=device)
            p = pool_steps(predict_future(model, ctx_final[ca] + d[:, None, :], 0)).cpu().numpy()
            for q, (a, c, k, t) in enumerate(ch):
                outA[cidx[a], c, k, t] = p[q]
        sync(); t2 = time.time()
        jobs = [(a, c, t) for a in range(len(cidx)) for c in range(2) for t in range(T)]
        for b in range(0, len(jobs), bs):
            ch = jobs[b:b + bs]
            ca = torch.as_tensor([j[0] for j in ch], device=device)
            d = torch.as_tensor(np.stack([DB[cidx[a], c, t] for a, c, t in ch]), device=device)
            ze, _ = edited_prediction(model, saved[LB][ca], LB, d, 0)
            p = pool_steps(ze).cpu().numpy()
            for q, (a, c, t) in enumerate(ch):
                outB[cidx[a], c, t] = p[q]
        sync(); tA += t2 - t1; tB += time.time() - t2
        if g % 40 == 0:
            print(f"{g + len(cidx)}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    par_src = float(np.abs(src_pp - F["src_pred_pooled"]).max() / np.abs(F["src_pred_pooled"]).max())
    ref = F[f"edit_pred_pooled_L{LB}"][:, ARMS.index("chord")]
    par_chord = float(np.abs(outB[:, 1] - ref).max() / np.abs(ref).max())
    common = {"commit": args.commit, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch": torch.__version__, "batch_size": bs, "mask_index": 0, "threads": args.threads,
              "parity_src_pred_vs_session2_cache_rel_maxabs": par_src}
    np.savez(enc / "pred_pooled.npz", pred_pooled=outA, src_pred_pooled=src_pp, idx=z["idx"])
    (enc / "forward_info.json").write_text(json.dumps({**common, "point": ENC, "arms": list(ARMS),
                                                       "conditions": list(CONDS), "n_edits": int(C * 3 * A * T),
                                                       "seconds_edits": tA, "seconds_total": time.time() - t0}, indent=1))
    np.savez(lab / "pred_pooled.npz", pred_pooled=outB, idx=z["idx"])
    (lab / "forward_info.json").write_text(json.dumps({**common, "point": LB, "arms": ["spline_labels", "chord_stored"],
                                                       "n_edits": int(C * 2 * T), "seconds_edits": tB,
                                                       "parity_chord_stored_vs_session2_cache_rel_maxabs": par_chord},
                                                      indent=1))
    print(json.dumps({"par_src": par_src, "par_chord_L12": par_chord, "sA": tA, "sB": tB}))


def score(args):
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    nm_json = json.loads((RES / "session2_predictor_norm_matched.json").read_text())
    prop = json.loads((RES / "session2_propagation.json").read_text())["delta_over_natural_twin_change"]
    plan_json = json.loads((RES / "session2_plan.json").read_text())["per_layer"]
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
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", "edit_pred_pooled_L22", "edit_pred_pooled_L12",
                                           "src_meanpool", "twin_meanpool"})
    enc, lab = Path(args.encout), Path(args.labels_out)
    zp = np.load(enc / "plan_encout.npz")
    zA, zB = np.load(enc / "pred_pooled.npz"), np.load(lab / "pred_pooled.npz")
    fA, fB = (json.loads((p / "forward_info.json").read_text()) for p in (enc, lab))
    pinfo = json.loads((enc / "plan_info.json").read_text())
    idx = F["idx"]
    assert (zA["idx"] == idx).all() and (zB["idx"] == idx).all()
    rows, tg = arrays["carrier_rows"][idx], arrays["targets"][idx]
    twin_true = np.load(OUT / "centroids_twins.npy")
    src_true = cen[rows][:, 4:8]
    flipped_pos = 2 * cen[rows, 0][:, None, None] - twin_true
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                             # noqa: E731
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa: E731
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, b, ax: (np.nansum(a * b, ax) / np.maximum(np.nansum(b * b, ax), 1e-9))   # noqa: E731
    Ps = apply(dir_nat, F["src_pred_pooled"].astype(np.float64).mean(1))
    assert src_true.shape[1] == 4

    def per_carrier(Z):                                                            # Z [C, T, 4, D]
        Pe = apply(dir_nat, Z.mean(2))
        a = angle_of(Pe)
        pe = read_pos(Z)
        return {"dir_err_to_target": wrap(a - tg).mean(1), "dir_err_to_flipped180_target": wrap(a - (tg + 180.0)).mean(1),
                "R_dir_real_change": proj(Pe - Ps[:, None], d_true, -1).mean(1),
                "px_err_to_twin_true": np.nanmean(dist(pe, twin_true), (1, 2)),
                "px_err_to_flipped180_twin": np.nanmean(dist(pe, flipped_pos), (1, 2))}

    KEYS = ("dir_err_to_target", "dir_err_to_flipped180_target", "R_dir_real_change", "px_err_to_twin_true",
            "px_err_to_flipped180_twin")
    # probe reconstruction reproduces the stored native-readout numbers (cached L22 and L12 forecasts)
    repro = 0.0
    for L in (22, LB):
        for ip, arm in enumerate(ARMS):
            pc = per_carrier(F[f"edit_pred_pooled_L{L}"][:, ip].astype(np.float64))
            repro = max(repro, max(abs(float(pc[k].mean()) - nat_json["per_layer"][str(L)][arm][k]["mean"]) for k in KEYS))
    assert repro < 1e-6, repro
    commit = {"forward_commit": fA["commit"]}

    # ------------------------------------------------ A
    nat25 = zp["nat_twin_L25"].astype(np.float64)
    dA = zp["deltas_L25"].astype(np.float64)
    nA = np.linalg.norm(dA, axis=-1)                                              # [C, A, T]
    sA = zp["scales_L25"].astype(np.float64)
    par_unm_vs_chordnorm = float(np.abs(zA["pred_pooled"][:, 0, ARMS.index("chord")]
                                        - zA["pred_pooled"][:, 1, ARMS.index("chord")]).max())
    pcs = {c: {a: per_carrier(zA["pred_pooled"][:, ci, k].astype(np.float64)) for k, a in enumerate(ARMS)}
           for ci, c in enumerate(CONDS)}
    A_res = {"point": ENC, "site": "encoder output = final LayerNorm (point 25), the predictor's input tokens; edit "
                                   "added to every post-LN context token, predictor forecasts tubelets 4-7",
             "n_carriers": int(len(idx)), "n_targets": int(tg.shape[1]), "arms": list(ARMS), "conditions": {
                 "unmatched": "each arm at its own norm (as the session-2 point-22 run)",
                 "chord_norm": "rescaled per (carrier, target) to ||delta_chord|| at point 25",
                 "natural_twin_norm": "rescaled to ||twin_meanpool_L25 - src_meanpool_L25||"},
             "plan": pinfo, "forward": fA, "readout_reproduction_of_native_file_maxabs": repro,
             "parity_chord_unmatched_vs_chord_norm_maxabs": par_unm_vs_chordnorm, "per_arm": {}, "spline_minus_chord": {}}
    st22 = nat_json["per_layer"]["22"]
    for k, arm in enumerate(ARMS):
        row = {}
        for ci, c in enumerate(CONDS):
            r = {m: boot_ci(v) for m, v in pcs[c][arm].items()}
            new = nA[:, k] * (1.0 if c == "unmatched" else sA[:, ci - 1, k])
            r["edit_norm_over_natural_twin_change_median"] = float(np.median(new / nat25))
            r["edit_norm_over_chord_median"] = float(np.median(new / nA[:, ARMS.index("chord")]))
            row[c] = r
        row["unmatched"]["edit_norm_over_centroid_change_median"] = float(np.median(nA[:, k] / zp["centroid_change_L25"]))
        row["point22_stored"] = {
            "unmatched": {m: st22[arm][m] for m in KEYS},
            "chord_norm": {m: nm_json["per_condition"]["chord_norm"][arm][m] for m in KEYS if m in nm_json["per_condition"]["chord_norm"][arm]},
            "natural_twin_norm": {m: nm_json["per_condition"]["natural_twin_norm"][arm][m] for m in KEYS
                                  if m in nm_json["per_condition"]["natural_twin_norm"][arm]},
            "edit_norm_over_natural_twin_change_median": prop["22"][arm]["median_ratio"]}
        A_res["per_arm"][arm] = row
    for c in CONDS:
        A_res["spline_minus_chord"][c] = {m: boot_ci(pcs[c]["spline"][m] - pcs[c]["chord"][m])
                                          for m in ("dir_err_to_target", "R_dir_real_change", "px_err_to_twin_true")}
    A_res["spline_minus_chord"]["point22_stored"] = {c: nm_json["spline_minus_chord"][c]
                                                     for c in ("chord_norm", "natural_twin_norm")}
    A_res["unedited"] = {"dir_err_to_target": boot_ci(wrap(angle_of(Ps)[:, None] - tg).mean(1)),
                         "px_err_to_twin_true": boot_ci(np.nanmean(dist(read_pos(F["src_pred_pooled"].astype(np.float64))[:, None],
                                                                         twin_true), (1, 2)))}
    A_res["definitions"] = {"metrics": "as session2_predictor_native_readout.json / session2_predictor_norm_matched.json: "
                                       "native probes (same 480 probe clips, stored alpha, no refit); CI = bootstrap "
                                       "over 200 carriers; flipped180 = target + 180 deg / twin reflected about the "
                                       "tubelet-0 centroid",
                            "edit_norm_over_natural_twin_change_median": "median ||delta|| / ||twin_meanpool_L - "
                                                                         "src_meanpool_L|| (session2_propagation quantity)"}
    write(RES / "session2_encoder_output.json", A_res, stage="encoder_output_score", seeds={"bootstrap": 0},
          **commit, cache=str(enc / "pred_pooled.npz"))

    # ------------------------------------------------ B
    natL12 = np.linalg.norm(F["twin_meanpool"][:, :, LB].astype(np.float64)
                            - F["src_meanpool"][:, None, LB].astype(np.float64), axis=-1)
    nb = {"spline_labels": np.linalg.norm(zp["delta_L12_spline_labels"].astype(np.float64), axis=-1),
          "chord_labels_curve": np.linalg.norm(zp["delta_L12_chord_labels"].astype(np.float64), axis=-1),
          "spline_stored": np.linalg.norm(zp["delta_L12_spline_stored"].astype(np.float64), axis=-1),
          "chord_stored": np.linalg.norm(zp["delta_L12_chord_stored"].astype(np.float64), axis=-1)}
    pcB = per_carrier(zB["pred_pooled"][:, 0].astype(np.float64))
    pcCh = per_carrier(zB["pred_pooled"][:, 1].astype(np.float64))
    pcS = per_carrier(F["edit_pred_pooled_L12"][:, ARMS.index("spline")].astype(np.float64))
    pcC = per_carrier(F["edit_pred_pooled_L12"][:, ARMS.index("chord")].astype(np.float64))
    ratios = lambda n: {"over_natural_twin_change_median": float(np.median(n / natL12)),               # noqa: E731
                        "over_centroid_change_median": float(np.median(n / zp["centroid_change_L12"])),
                        "over_stored_chord_median": float(np.median(n / nb["chord_stored"]))}
    st12 = nat_json["per_layer"][str(LB)]
    B_res = {"point": LB, "arm": "interpolating periodic spline endpoint, labels angle (activation plane, monotone knot "
                                 "order), shift mode, residual kept; everything else as session 2",
             "plan": pinfo, "forward": fB, "readout_reproduction_of_native_file_maxabs": repro,
             "spline_labels": {**{m: boot_ci(v) for m, v in pcB.items()}, "edit_norm": ratios(nb["spline_labels"])},
             "chord_rerun_parity": {"rel_maxabs_vs_session2_cache": fB["parity_chord_stored_vs_session2_cache_rel_maxabs"],
                                    "dir_err_to_target": boot_ci(pcCh["dir_err_to_target"])},
             "chord_on_labels_curve_edit_norm": ratios(nb["chord_labels_curve"]),
             "stored_point12": {arm: {**{m: st12[arm][m] for m in KEYS},
                                      "edit_norm_over_natural_twin_change_median": prop[str(LB)][arm]["median_ratio"],
                                      "edit_norm_over_centroid_change_median":
                                          plan_json[str(LB)]["delta_over_centroid_change_median"][arm]}
                                for arm in ARMS},
             "stored_spline_edit_norm_recomputed": ratios(nb["spline_stored"]),
             "spline_labels_minus_stored_chord": {m: boot_ci(pcB[m] - pcC[m]) for m in
                                                  ("dir_err_to_target", "R_dir_real_change", "px_err_to_twin_true")},
             "spline_labels_minus_stored_spline": {m: boot_ci(pcB[m] - pcS[m]) for m in
                                                   ("dir_err_to_target", "px_err_to_twin_true")},
             "definitions": A_res["definitions"]}
    write(RES / "session2_interp_labels_L12.json", B_res, stage="interp_labels_L12_score", seeds={"bootstrap": 0},
          **commit, cache=str(lab / "pred_pooled.npz"))
    for c in CONDS:
        print(c, {a: round(A_res["per_arm"][a][c]["dir_err_to_target"]["mean"], 1) for a in ARMS})
    print("B", {m: round(B_res["spline_labels"][m]["mean"], 2) for m in KEYS}, B_res["spline_labels"]["edit_norm"])


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("plan", "forward", "score"))
    p.add_argument("--encout", default=str(OUT / "encoder_output"))
    p.add_argument("--labels-out", default=str(OUT / "interp_labels_L12"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--commit", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse()
    {"plan": plan, "forward": forward, "score": score}[a.stage](a)

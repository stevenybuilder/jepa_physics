"""Direction edits at point 12 (and 22 as the reference) at the NATURAL twin norm: is the point-12 wash-out a dose effect?

Session 2 (REPORT §4.5) edited at each arm's own norm: at point 12 every direction arm leaves the predictor's forecast
within ~15 deg of the unedited 92.1 deg and the edit is erased within four blocks. The speed session
(session3_speed_predictor.py) found that at the natural twin norm a point-12 speed edit does reach the forecast.
This reruns the direction analogue with the session-2 machinery unchanged:

  carriers / targets  plan.npz (200 test clips x 4 held-out targets in the contiguous 45-deg arc, seed 0)
  arms   spline  session-2 interpolating periodic spline endpoint (the paper's), deltas_L{L}[:, 'spline']
         chord   session-2 chord endpoint (straight line between the polyline points, polyline through the raw kept
                 centroids), deltas_L{L}[:, 'chord']
         null    the same interpolating-spline edit, aimed at the far end of the ring: the angle 180 deg from the
                 held-out arc's centre (143.4375 deg); one delta per carrier (target independent), built here with
                 the same PCA-64 / centroid / curve recipe as run_session2.layer_deltas (checked by rebuilding the
                 stored spline and chord deltas: parity reported)
  norms  own      the delta as planned (session-2 dose; reproduces the stored numbers)
         natural  rescaled per (carrier, target) to ||twin_meanpool_L - src_meanpool_L|| (full-clip meanpool, the
                  same quantity as session2_predictor_norm_matched / session3 natural_norm)
  forward  context frames 1-8 encoded alone, delta added to every context token at point L, blocks L+1..24 + final
           LN, predictor with mask token 0 (edited_prediction, as run_session2 / norm_matched). Also keeps the
           context meanpool at points L..25 -> survival fraction <mp_edit - mp_src, delta> / ||delta||^2.
  score    session-2 predictor-native direction probe (480 probe clips, stored alpha, no CV); twin ceiling from the
           session-2 cached twin forecasts (twin_pred_pooled) through the same probe.

  plan     Mac CPU  -> artifacts/session2/natural_norm/deltas.npz
  forward  box GPU  -> <root>/pred.npz + forward_info.json
  score    Mac CPU  -> results/session2_direction_natural_norm.json + figures/fig_session2_direction_natural_norm.png
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
NN = OUT / "natural_norm"
RES = PROJECT_ROOT / "results"
POINTS = (12, 22)
ARMS = ("spline", "chord", "null")
CONDS = ("own", "natural")


def plan():
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from run_session2 import load_groups, load_plan
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_meanpool", "twin_meanpool"})
    idx = F["idx"]
    held = np.array(info["held_values"])
    centre = (held.mean()) % 360.0                       # arc does not wrap (303.75..343.125)
    far = (centre + 180.0) % 360.0
    carriers = arrays["carrier_rows"][idx]
    tg = arrays["targets"][idx]
    C, T = tg.shape
    out = {"idx": idx, "carrier_rows": carriers, "targets": tg}
    meta = {"far_value": far, "held_centre": centre, "parity": {}}
    for L in POINTS:
        d = load_inputs("direction", L)
        y = d["y"]
        X = d["X"].astype(np.float64)
        knot = (d["role"] == "knot") & ~np.isin(y, held)
        pca = mf.fit_pca(X[knot], info["k"])
        cent = mf.centroids(pca.project(X[knot]), y[knot])
        choice = info["per_layer"][str(L)]["angle_choice"]
        curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=info["spline"])
        Z = pca.project(X[carriers])
        src = y[carriers]
        ta = curve.coord_of_value(src)
        # parity: rebuild stored spline / chord deltas
        ai = [info["arms"].index(a) for a in ("spline", "chord")]
        stored = arrays[f"deltas_L{L}"][idx][:, ai].astype(np.float64)       # [C, 2, T, D]
        reb = np.zeros_like(stored)
        for j in range(T):
            tb = curve.coord_of_value(tg[:, j])
            reb[:, 0, j] = pca.lift_delta(mf.manifold_coords(Z, curve, ta, tb, 2)[:, -1] - Z)
            reb[:, 1, j] = pca.lift_delta(mf.linear_coords(Z, mf.piecewise_linear_point(curve, src),
                                                           mf.piecewise_linear_point(curve, tg[:, j]), 2)[:, -1] - Z)
        rel = np.linalg.norm(reb - stored, axis=-1) / np.maximum(np.linalg.norm(stored, axis=-1), 1e-12)
        meta["parity"][str(L)] = {"spline_rel_maxabs": float(rel[:, 0].max()), "chord_rel_maxabs": float(rel[:, 1].max()),
                                  "spline_rel_median": float(np.median(rel[:, 0])), "chord_rel_median": float(np.median(rel[:, 1]))}
        tf = curve.coord_of_value(np.full(C, far))
        dnull = pca.lift_delta(mf.manifold_coords(Z, curve, ta, tf, 2)[:, -1] - Z)       # [C, D]
        dl = np.stack([stored[:, 0], stored[:, 1], np.repeat(dnull[:, None], T, 1)], 1)  # [C, A, T, D]
        norm = np.linalg.norm(dl, axis=-1)                                                # [C, A, T]
        nat = np.linalg.norm(F["twin_meanpool"][:, :, L].astype(np.float64)
                             - F["src_meanpool"][:, None, L].astype(np.float64), axis=-1)  # [C, T]
        s = np.stack([np.ones_like(norm), nat[:, None] / np.maximum(norm, 1e-12)], 1)    # [C, 2, A, T]
        out[f"deltas_L{L}"] = (dl[:, None] * s[..., None]).astype(np.float32)             # [C, 2, A, T, D]
        out[f"own_norm_L{L}"] = norm.astype(np.float32)
        out[f"natural_norm_L{L}"] = nat.astype(np.float32)
        meta[str(L)] = {"own_over_natural_median": {a: float(np.median(norm[:, k] / nat)) for k, a in enumerate(ARMS)},
                        "natural_norm_median": float(np.median(nat)),
                        "null_src_to_far_deg_median": float(np.median(np.abs((src - far + 180) % 360 - 180)))}
        print(L, json.dumps(meta[str(L)]), json.dumps(meta["parity"][str(L)]), flush=True)
    NN.mkdir(parents=True, exist_ok=True)
    np.savez(NN / "deltas.npz", **out)
    (NN / "plan_meta.json").write_text(json.dumps(meta, indent=1))


def forward(root, n_carriers=200, batch_size=16):
    import torch
    torch.set_num_threads(4)
    from wm.extract import load_model, pick_device, set_precision
    from wm.extract import preprocess
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix
    root = Path(root)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    z = np.load(root / "deltas.npz")
    rows = z["carrier_rows"][:n_carriers]
    C = len(rows)
    df = load_table("direction")
    res = {"src_pred_pooled": np.zeros((C, 4, 1024), np.float32)}
    for L in POINTS:
        _, _, A, T, D = z[f"deltas_L{L}"].shape
        res[f"pred_L{L}"] = np.zeros((C, 2, A, T, 4, D), np.float32)
        res[f"surv_L{L}"] = np.zeros((C, 2, A, T, 26 - L), np.float32)   # points L..25
    t0 = time.time()
    G = 8
    for g in range(0, C, G):
        cidx = list(range(g, min(g + G, C)))
        pv = preprocess([decode(df["video"].iloc[rows[i]]) for i in cidx]).to(device)
        saved, ctx_final, ctx_pool = prefix(model, pv[:, :8], list(POINTS))
        res["src_pred_pooled"][cidx] = pool_steps(predict_future(model, ctx_final, 0)).cpu().numpy()
        for L in POINTS:
            DL = z[f"deltas_L{L}"]
            _, _, A, T, D = DL.shape
            mp0 = ctx_pool[:, L:]                                                     # [B, 26-L, D]
            jobs = [(a, c, k, t) for a in range(len(cidx)) for c in range(2) for k in range(A) for t in range(T)]
            for b in range(0, len(jobs), batch_size):
                ch = jobs[b:b + batch_size]
                ca = torch.as_tensor([j[0] for j in ch], device=device)
                dn = np.stack([DL[cidx[a], c, k, t] for a, c, k, t in ch])
                pz, mp = edited_prediction(model, saved[L][ca], L, torch.as_tensor(dn, device=device), 0)
                p = pool_steps(pz).cpu().numpy()
                dmp = mp.astype(np.float64) - mp0[[j[0] for j in ch]].astype(np.float64)
                sv = (dmp * dn[:, None].astype(np.float64)).sum(-1) / np.maximum((dn.astype(np.float64) ** 2).sum(-1), 1e-12)[:, None]
                for q, (a, c, k, t) in enumerate(ch):
                    res[f"pred_L{L}"][cidx[a], c, k, t] = p[q]
                    res[f"surv_L{L}"][cidx[a], c, k, t] = sv[q]
        print(f"{g + len(cidx)}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.savez(root / "pred.npz", idx=z["idx"][:C], **res)
    info = {"seconds": secs, "n_carriers": C, "batch_size": batch_size, "points": list(POINTS), "arms": list(ARMS),
            "conds": list(CONDS), "mask_index": 0, "torch_threads": 4,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def score():
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    alpha = nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"]
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe], alpha)
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", "twin_pred_pooled", "edit_pred_pooled_L12",
                                           "edit_pred_pooled_L22"})
    z = np.load(NN / "pred.npz")
    dz = np.load(NN / "deltas.npz")
    finfo = json.loads((NN / "forward_info.json").read_text())
    meta = json.loads((NN / "plan_meta.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all()
    rows, tg = arrays["carrier_rows"][F["idx"][:C]], arrays["targets"][F["idx"][:C]]
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    src_par = float(np.abs(z["src_pred_pooled"] - F["src_pred_pooled"][:C]).max() / np.abs(F["src_pred_pooled"]).max())
    Ps = readP(F["src_pred_pooled"][:C].astype(np.float64).mean(1))                 # [C, 2]

    def pc(Zp):                                                                      # [C, T, 4, D]
        Pe = readP(Zp.astype(np.float64).mean(2))
        return {"err": wrap(angle_of(Pe) - tg).mean(1), "R": proj(Pe - Ps[:, None], d_true).mean(1)}
    unedited = {"err_to_target": boot_ci(wrap(angle_of(Ps)[:, None] - tg).mean(1))}
    tw = pc(F["twin_pred_pooled"][:C])
    res = {"question": "does a direction edit at point 12 still wash out at the natural twin dose (was the session-2 "
                       "point-12 wash-out a matter of norm)?",
           "n_carriers": int(C), "n_targets": int(tg.shape[1]), "points": list(POINTS), "arms": list(ARMS),
           "conds": {"own": "session-2 delta as planned", "natural": "rescaled per (carrier, target) to "
                     "||twin_meanpool_L - src_meanpool_L|| (full-clip meanpool)"},
           "null_far_value_deg": meta["far_value"], "plan_parity_rebuild_vs_stored": meta["parity"],
           "parity_src_forecast_vs_session2_cache_rel_maxabs": src_par, "forward": finfo,
           "unedited": unedited, "twin_ceiling": {"err_to_target": boot_ci(tw["err"]), "R": boot_ci(tw["R"])},
           "table": {}, "survival": {}, "reproduction_of_session2_own_norm": {}, "paired": {}}
    armi = {a: info["arms"].index(a) for a in ("spline", "chord")}
    for L in POINTS:
        rL, sL, rep, pr = {}, {}, {}, {}
        own_n, nat_n = dz[f"own_norm_L{L}"][:C].astype(np.float64), dz[f"natural_norm_L{L}"][:C].astype(np.float64)
        pcs = {}
        for ci, cond in enumerate(CONDS):
            for k, arm in enumerate(ARMS):
                p = pc(z[f"pred_L{L}"][:, ci, k])
                pcs[(cond, arm)] = p
                applied = own_n[:, k] if cond == "own" else nat_n
                rL.setdefault(cond, {})[arm] = {"err_to_target": boot_ci(p["err"]), "R": boot_ci(p["R"]),
                                                "applied_norm_median": float(np.median(applied)),
                                                "applied_over_natural_median": float(np.median(applied / nat_n))}
                sv = z[f"surv_L{L}"][:, ci, k].astype(np.float64).mean(1)              # mean over targets [C, 26-L]
                pts = list(range(L, 26))
                want = [q for q in (13, 14, 15, 16, 24, 25) if q >= L] if L == 12 else [22, 23, 24, 25]
                sL.setdefault(cond, {})[arm] = {f"pt{q}": boot_ci(sv[:, pts.index(q)]) for q in want}
            for arm in ("spline", "chord"):
                pr.setdefault(cond, {})[f"{arm}_minus_null_err"] = boot_ci(pcs[(cond, arm)]["err"] - pcs[(cond, "null")]["err"])
            pr[cond]["spline_minus_chord_err"] = boot_ci(pcs[(cond, "spline")]["err"] - pcs[(cond, "chord")]["err"])
        for arm in ("spline", "chord"):
            pr[f"{arm}_natural_minus_own_err"] = boot_ci(pcs[("natural", arm)]["err"] - pcs[("own", arm)]["err"])
            old = pc(F[f"edit_pred_pooled_L{L}"][:C, armi[arm]])
            rep[arm] = {"stored_cache_err": float(old["err"].mean()), "rerun_own_err": float(pcs[("own", arm)]["err"].mean()),
                        "stored_cache_R": float(old["R"].mean()), "rerun_own_R": float(pcs[("own", arm)]["R"].mean()),
                        "native_readout_file_err": nat_json["per_layer"][str(L)][arm]["dir_err_to_target"]["mean"]}
        res["table"][str(L)], res["survival"][str(L)] = rL, sL
        res["reproduction_of_session2_own_norm"][str(L)], res["paired"][str(L)] = rep, pr
    res["definitions"] = {
        "err_to_target": "circular error (deg) of the predictor-native direction probe on the step-mean pooled forecast, "
                         "mean over 4 targets per carrier; CI = bootstrap over carriers (boot_ci, seed 0)",
        "R": "<P_edit - P_src, u(target) - u(source)> / ||u(target) - u(source)||^2 in probe output space "
             "(session-2 R_dir_real_change); twin ceiling uses the rendered twin's forecast",
        "survival_ptq": "context-only meanpool at point q: <mp_edit - mp_src, delta> / ||delta||^2 (1 at the edit point; "
                        "p5_repair_attribution ctx_survival quantity), mean over targets per carrier"}
    write(RES / "session2_direction_natural_norm.json", res, stage="natural_norm_score", seeds={"bootstrap": 0},
          cache=str(NN / "pred.npz"))
    for L in POINTS:
        for cond in CONDS:
            print(L, cond, {a: (round(res["table"][str(L)][cond][a]["err_to_target"]["mean"], 1),
                                round(res["table"][str(L)][cond][a]["R"]["mean"], 3)) for a in ARMS})
    print("twin", res["twin_ceiling"]["err_to_target"]["mean"], res["twin_ceiling"]["R"]["mean"],
          "unedited", unedited["err_to_target"]["mean"])
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"spline": "#3b6fb6", "chord": "#d9822b", "null": "#999999"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for pi, L in enumerate(POINTS):
        a = ax[pi]
        for k, arm in enumerate(ARMS):
            for ci, cond in enumerate(CONDS):
                r = res["table"][str(L)][cond][arm]["err_to_target"]
                x = k * 2.6 + ci
                a.bar(x, r["mean"], color=cols[arm], alpha=0.45 if cond == "own" else 1.0, width=0.9)
                a.errorbar(x, r["mean"], yerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], color="k", lw=1)
                a.text(x, 3, f"R {res['table'][str(L)][cond][arm]['R']['mean']:.2f}", ha="center", fontsize=7, rotation=90,
                       color="white" if cond == "natural" else "k")
        a.axhline(res["unedited"]["err_to_target"]["mean"], ls="--", color="k", lw=1, label="unedited")
        a.axhline(res["twin_ceiling"]["err_to_target"]["mean"], ls=":", color="g", lw=1.5, label="twin (ceiling)")
        a.set_xticks([k * 2.6 + c for k in range(3) for c in range(2)])
        a.set_xticklabels([f"{arm}\n{c}" for arm in ARMS for c in CONDS], fontsize=8)
        a.set_ylabel("forecast direction error to target (deg)")
        a.set_title(f"Point {L}: own norm (pale) vs natural twin norm (solid)")
        a.legend(fontsize=8)
    a = ax[2]
    for arm in ("spline", "chord"):
        for cond, ls in (("own", "--"), ("natural", "-")):
            s = res["survival"]["12"][cond][arm]
            ks = sorted(s, key=lambda q: int(q[2:]))
            a.plot([int(q[2:]) for q in ks], [s[q]["mean"] for q in ks], ls, color=cols[arm], marker="o", label=f"{arm} {cond}")
    a.set_xlabel("point (context-only encoder)")
    a.set_ylabel("survival  <Δmp, δ>/||δ||²")
    a.set_title("Point-12 edit survival downstream")
    a.axhline(0, color="k", lw=0.5)
    a.legend(fontsize=8)
    fig.suptitle(f"Session-2 direction edits at the natural twin norm ({res['n_carriers']} carriers x 4 held-out targets)")
    fig.tight_layout()
    out = PROJECT_ROOT / "figures" / "fig_session2_direction_natural_norm.png"
    fig.savefig(out, dpi=140)
    print(out)


# ---------------------------------------------------------------------------------------------------------------
# Follow-up (lead / interviewer): the point-12 edit applied ONLY to the object tokens, at the twin's per-token dose.
#   token set  obj = disk(source) | disk(twin), + 1-patch ring (3x3 spatial dilation per tubelet), context tubelets
#              0-3: run_token_patching.token_sets(mask_src, mask_twin)["obj"], per (carrier, target)
#   dose       each obj token i gets unit(arm delta) * n_i, n_i = ||h_twin[i] - h_src[i]|| at point 12 of the
#              context-only encoding (the twin's own per-token change on that token); background tokens unedited
#   arms       spline, chord, null (unit directions of the session-2 / far-end deltas above)
#   survival   pooled: <mp_edit - mp_src, mean_tok(edit)> / ||mean_tok(edit)||^2; obj: sum_i <dh_i, e_i> / sum_i ||e_i||^2
#              over the obj tokens, at points 16, 19, 24, 25
DISK_L = 12
SURV_PTS = (16, 19, 24, 25)


def disk_forward(root, n_carriers=200, batch_size=12):
    import torch
    torch.set_num_threads(4)
    from wm.data import disk_mask
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import _mean, prefix
    from run_token_patching import token_sets
    root = Path(root)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    enc = model.encoder
    z = np.load(root / "deltas.npz")
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    C = len(rows)
    dl = z[f"deltas_L{DISK_L}"][:C, 0].astype(np.float64)                          # own [C, A, T, D]
    A, T = dl.shape[1], dl.shape[2]
    unit = dl / np.maximum(np.linalg.norm(dl, axis=-1, keepdims=True), 1e-12)
    df = load_table("direction")
    pred = np.zeros((C, A, T, 4, 1024), np.float32)
    surv = np.zeros((C, A, T, 2, len(SURV_PTS)), np.float32)
    ntok = np.zeros((C, T), np.int32)
    tok_norm = np.zeros((C, T, 3), np.float32)                                     # mean / median n_i, pooled-equiv
    src_pp = np.zeros((C, 4, 1024), np.float32)
    t0 = time.time()

    @torch.no_grad()
    def run(h):
        keep = {}
        for li, layer in enumerate(enc.layer[DISK_L:]):
            h = layer(h, None, None, False)[0]
            if DISK_L + li + 1 in SURV_PTS:
                keep[DISK_L + li + 1] = h
        fin = enc.layernorm(h)
        keep[25] = fin
        return fin, keep

    for c in range(C):
        fs = decode(df["video"].iloc[rows[c]])
        ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
        ms = disk_mask(fs)[0]
        pv = preprocess([fs] + ft).to(device)
        saved, _, _ = prefix(model, pv[:, :8], [DISK_L])
        h = saved[DISK_L]                                                           # [1+T, 1024, D]
        fin0, keep0 = run(h[:1])
        src_pp[c] = pool_steps(predict_future(model, fin0, 0)).cpu().numpy()[0]
        edits = []
        for j in range(T):
            obj = torch.as_tensor(token_sets(ms, disk_mask(ft[j])[0])["obj"], device=device)
            n_i = (h[1 + j] - h[0]).norm(dim=-1) * obj                              # [1024]
            ntok[c, j] = int(obj.sum())
            tok_norm[c, j] = [float(n_i[obj].mean()), float(n_i[obj].median()), float(n_i.mean())]
            for k in range(A):
                u = torch.as_tensor(unit[c, k, j], dtype=h.dtype, device=device)
                edits.append((k, j, n_i[:, None] * u[None, :], obj))
        for b in range(0, len(edits), batch_size):
            ch = edits[b:b + batch_size]
            E = torch.stack([e[2] for e in ch])
            fin, keep = run(h[:1] + E)
            p = pool_steps(predict_future(model, fin, 0)).cpu().numpy()
            em = E.mean(1)                                                          # [B, D] meanpool shift
            for q, (k, j, e, obj) in enumerate(ch):
                pred[c, k, j] = p[q]
                for si, pt in enumerate(SURV_PTS):
                    dh = (keep[pt][q] - keep0[pt][0]).double()
                    surv[c, k, j, 0, si] = float(((_mean(dh[None])[0]) * em[q].double()).sum() / (em[q].double() ** 2).sum())
                    surv[c, k, j, 1, si] = float((dh[obj] * e[obj].double()).sum() / (e[obj].double() ** 2).sum())
        if c % 20 == 0:
            print(f"{c + 1}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.savez(root / "disk_pred.npz", idx=idx, pred=pred, surv=surv, ntok=ntok, tok_norm=tok_norm, src_pred_pooled=src_pp)
    info = {"seconds": secs, "n_carriers": C, "point": DISK_L, "arms": list(ARMS), "surv_points": list(SURV_PTS),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "disk_forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def disk_score():
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", "twin_pred_pooled"})
    z = np.load(NN / "disk_pred.npz")
    finfo = json.loads((NN / "disk_forward_info.json").read_text())
    pooled = json.loads((RES / "session2_direction_natural_norm.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all()
    rows, tg = arrays["carrier_rows"][F["idx"][:C]], arrays["targets"][F["idx"][:C]]
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    src_par = float(np.abs(z["src_pred_pooled"] - F["src_pred_pooled"][:C]).max() / np.abs(F["src_pred_pooled"]).max())
    Ps = readP(F["src_pred_pooled"][:C].astype(np.float64).mean(1))
    dz = np.load(NN / "deltas.npz")
    nat_pooled = dz[f"natural_norm_L{DISK_L}"][:C].astype(np.float64)
    res = {"question": "does the point-12 direction edit reach the forecast when applied only to the object tokens at the "
                       "twin's per-token dose (where the frame-wide pooled edit is repaired)?",
           "n_carriers": int(C), "n_targets": int(tg.shape[1]), "point": DISK_L, "arms": list(ARMS),
           "token_set": "obj = disk(source) | disk(twin) + 1-patch ring (3x3 dilation per tubelet), context tubelets 0-3 "
                        "(run_token_patching.token_sets(...)['obj']), per (carrier, target)",
           "dose": "each obj token i gets unit(arm delta) * ||h_twin[i] - h_src[i]|| at point 12 (context-only encoding); "
                   "background tokens unedited",
           "obj_token_fraction": {"mean": float(z["ntok"].mean() / 1024), "min": float(z["ntok"].min() / 1024),
                                  "max": float(z["ntok"].max() / 1024)},
           "twin_per_token_change_on_obj_tokens_median": float(np.median(z["tok_norm"][..., 0])),
           "equivalent_meanpool_shift_norm_over_pooled_natural_median": None,
           "parity_src_forecast_vs_session2_cache_rel_maxabs": src_par, "forward": finfo,
           "unedited_err_to_target": pooled["unedited"]["err_to_target"], "twin_ceiling": pooled["twin_ceiling"],
           "pooled_natural_norm_reference": {a: pooled["table"]["12"]["natural"][a] for a in ARMS},
           "table": {}, "survival": {}, "paired_vs_pooled_natural": {}}
    pz = np.load(NN / "pred.npz")
    for k, arm in enumerate(ARMS):
        Pe = readP(z["pred"][:, k].astype(np.float64).mean(2))
        err, R = wrap(angle_of(Pe) - tg).mean(1), proj(Pe - Ps[:, None], d_true).mean(1)
        res["table"][arm] = {"err_to_target": boot_ci(err), "R": boot_ci(R)}
        sv = z["surv"][:, k].astype(np.float64).mean(1)                            # [C, 2, P]
        res["survival"][arm] = {kind: {f"pt{pt}": boot_ci(sv[:, ki, si]) for si, pt in enumerate(SURV_PTS)}
                                for ki, kind in enumerate(("pooled", "obj_tokens"))}
        Pp = readP(pz[f"pred_L{DISK_L}"][:C, 1, k].astype(np.float64).mean(2))
        res["paired_vs_pooled_natural"][f"{arm}_disk_minus_pooled_err"] = boot_ci(err - wrap(angle_of(Pp) - tg).mean(1))
    # meanpool-equivalent size of the obj-only edit vs the pooled natural norm (how much smaller the frame-average is)
    eq = z["tok_norm"][..., 2].astype(np.float64)                                   # mean over all tokens of n_i*1[obj]
    res["equivalent_meanpool_shift_norm_over_pooled_natural_median"] = float(np.median(eq / nat_pooled))
    res["definitions"] = {"err_to_target": "as session2_direction_natural_norm.json", "R": "as session2_direction_natural_norm.json",
                          "survival.pooled": "<mp_edit - mp_src, mean_tok(e)> / ||mean_tok(e)||^2 at point q (context-only)",
                          "survival.obj_tokens": "sum_i <h_edit[i] - h_src[i], e_i> / sum_i ||e_i||^2 over obj tokens at point q"}
    write(RES / "session2_direction_natural_norm_disktokens.json", res, stage="natural_norm_disktokens_score",
          seeds={"bootstrap": 0}, cache=str(NN / "disk_pred.npz"))
    for arm in ARMS:
        print(arm, round(res["table"][arm]["err_to_target"]["mean"], 1), round(res["table"][arm]["R"]["mean"], 3),
              {kd: {p: round(v["mean"], 3) for p, v in d.items()} for kd, d in res["survival"][arm].items()})
    print("obj frac", res["obj_token_fraction"], "eq/pooled", res["equivalent_meanpool_shift_norm_over_pooled_natural_median"])


# ---------------------------------------------------------------------------------------------------------------
# Background-token control (lead): the point-12 chord direction on NON-object tokens, matched to the disk-token arm
#   bg_count   a random subset (seed 1000*c + j) of the non-obj context tokens, of the same size as obj, each given one
#              of the obj tokens' per-token twin-change magnitudes n_i (a random permutation of them)
#   bg_energy  all non-obj tokens, each at magnitude m = sqrt(sum_obj n_i^2 / |bg|) (same total edit energy)
#   disk12     the disk-token chord again (reproduction of session2_direction_natural_norm_disktokens.json)
#   disk16/19  the disk-token chord at points 16 / 19: chord delta planned at that point with the session-2 recipe
#              (PCA-64 / centroid / interpolating curve, choose_angle_source), unit direction x the twin's per-token
#              change at that point on the obj tokens
BG_ARMS = ("disk12", "bg_count", "bg_energy", "disk16", "disk19")


def bg_plan():
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from run_session2 import load_plan
    arrays, info = load_plan(OUT)
    z = np.load(NN / "deltas.npz")
    carriers, tg = z["carrier_rows"], z["targets"]
    held = np.array(info["held_values"])
    out, meta = {}, {}
    for L in (16, 19):
        d = load_inputs("direction", L)
        y, X = d["y"], d["X"].astype(np.float64)
        knot = (d["role"] == "knot") & ~np.isin(y, held)
        pca = mf.fit_pca(X[knot], info["k"])
        cent = mf.centroids(pca.project(X[knot]), y[knot])
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=info["spline"])
        Z = pca.project(X[carriers])
        src = y[carriers]
        dl = np.stack([pca.lift_delta(mf.linear_coords(Z, mf.piecewise_linear_point(curve, src),
                                                       mf.piecewise_linear_point(curve, tg[:, j]), 2)[:, -1] - Z)
                       for j in range(tg.shape[1])], 1)
        out[f"chord_L{L}"] = dl.astype(np.float32)
        meta[str(L)] = {"angle_choice": {"angle": choice["angle"], "plane": choice["plane"]},
                        "chord_norm_median": float(np.median(np.linalg.norm(dl, axis=-1)))}
        print(L, meta[str(L)], flush=True)
    np.savez(NN / "bg_deltas.npz", **out)
    (NN / "bg_plan_meta.json").write_text(json.dumps(meta, indent=1))


def bg_forward(root, n_carriers=200, batch_size=10):
    import torch
    torch.set_num_threads(4)
    from wm.data import disk_mask
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    from run_token_patching import token_sets
    root = Path(root)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    enc = model.encoder
    z = np.load(root / "deltas.npz")
    zb = np.load(root / "bg_deltas.npz")
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    C = len(rows)
    ch12 = z["deltas_L12"][:C, 0, ARMS.index("chord")].astype(np.float64)          # [C, T, D]
    dirs = {12: ch12, 16: zb["chord_L16"][:C].astype(np.float64), 19: zb["chord_L19"][:C].astype(np.float64)}
    dirs = {L: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12) for L, v in dirs.items()}
    T = ch12.shape[1]
    df = load_table("direction")
    pred = np.zeros((C, len(BG_ARMS), T, 4, 1024), np.float32)
    energy = np.zeros((C, len(BG_ARMS), T), np.float32)
    ntok = np.zeros((C, len(BG_ARMS), T), np.int32)
    src_pp = np.zeros((C, 4, 1024), np.float32)
    t0 = time.time()

    @torch.no_grad()
    def run(h, L):
        for layer in enc.layer[L:]:
            h = layer(h, None, None, False)[0]
        return pool_steps(predict_future(model, enc.layernorm(h), 0))

    for c in range(C):
        fs = decode(df["video"].iloc[rows[c]])
        ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
        ms = disk_mask(fs)[0]
        pv = preprocess([fs] + ft).to(device)
        saved, fin0, _ = prefix(model, pv[:, :8], [12, 16, 19])
        src_pp[c] = pool_steps(predict_future(model, fin0[:1], 0)).cpu().numpy()[0]
        jobs = []
        for j in range(T):
            obj_np = token_sets(ms, disk_mask(ft[j])[0])["obj"]
            obj = torch.as_tensor(obj_np, device=device)
            rng = np.random.default_rng(1000 * c + j)
            nobj = int(obj_np.sum())
            for ai, arm in enumerate(BG_ARMS):
                L = {"disk16": 16, "disk19": 19}.get(arm, 12)
                h = saved[L]
                u = torch.as_tensor(dirs[L][c, j], dtype=h.dtype, device=device)
                n_all = (h[1 + j] - h[0]).norm(dim=-1)
                mag = torch.zeros(1024, dtype=h.dtype, device=device)
                if arm.startswith("disk"):
                    mag[obj] = n_all[obj]
                elif arm == "bg_count":
                    bg = np.flatnonzero(~obj_np)
                    pick = torch.as_tensor(rng.choice(bg, size=nobj, replace=False), device=device)
                    vals = n_all[obj][torch.as_tensor(rng.permutation(nobj), device=device)]
                    mag[pick] = vals
                else:
                    e_obj = (n_all[obj] ** 2).sum()
                    mag[~obj] = torch.sqrt(e_obj / (~obj).sum())
                energy[c, ai, j] = float((mag ** 2).sum())
                ntok[c, ai, j] = int((mag > 0).sum())
                jobs.append((ai, j, L, mag[:, None] * u[None, :]))
        for L in (12, 16, 19):
            jl = [q for q in jobs if q[2] == L]
            for b in range(0, len(jl), batch_size):
                chk = jl[b:b + batch_size]
                p = run(saved[L][:1] + torch.stack([q[3] for q in chk]), L).cpu().numpy()
                for q, (ai, j, _, _) in enumerate(chk):
                    pred[c, ai, j] = p[q]
        if c % 20 == 0:
            print(f"{c + 1}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0
    np.savez(root / "bg_pred.npz", idx=idx, pred=pred, energy=energy, ntok=ntok, src_pred_pooled=src_pp)
    info = {"seconds": secs, "n_carriers": C, "arms": list(BG_ARMS),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "bg_forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def bg_score():
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled"})
    z = np.load(NN / "bg_pred.npz")
    finfo = json.loads((NN / "bg_forward_info.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all()
    rows, tg = arrays["carrier_rows"][F["idx"][:C]], arrays["targets"][F["idx"][:C]]
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    Ps = readP(F["src_pred_pooled"][:C].astype(np.float64).mean(1))
    dk = json.loads((RES / "session2_direction_natural_norm_disktokens.json").read_text())
    pooled = json.loads((RES / "session2_direction_natural_norm.json").read_text())
    dpred = np.load(NN / "disk_pred.npz")["pred"][:, ARMS.index("chord")]
    res = {"question": "is the disk-token point-12 chord effect about the object tokens or about a large per-token dose? "
                       "background-token controls matched on token count + per-token dose, and on total energy",
           "n_carriers": int(C), "n_targets": int(tg.shape[1]), "forward": finfo,
           "bg_plan": json.loads((NN / "bg_plan_meta.json").read_text()),
           "arms": {"disk12": "chord unit direction x twin per-token change n_i on obj tokens at point 12 (reproduction)",
                    "bg_count": "same number of tokens as obj, random non-obj subset (seed 1000*c+j), magnitudes = a random "
                                "permutation of the obj tokens' n_i",
                    "bg_energy": "all non-obj tokens, equal magnitude sqrt(sum_obj n_i^2 / |bg|): total energy matched",
                    "disk16": "disk-token chord edited at point 16 (chord planned at 16, per-token twin change at 16)",
                    "disk19": "disk-token chord edited at point 19"},
           "parity_src_forecast_vs_session2_cache_rel_maxabs": float(np.abs(z["src_pred_pooled"] - F["src_pred_pooled"][:C]).max()
                                                                      / np.abs(F["src_pred_pooled"]).max()),
           "references": {"unedited_err": pooled["unedited"]["err_to_target"], "twin_ceiling": pooled["twin_ceiling"],
                          "pooled_natural_chord_L12": pooled["table"]["12"]["natural"]["chord"],
                          "disk_token_chord_L12_previous_run": dk["table"]["chord"]},
           "table": {}, "paired_vs_disk12": {}}
    errs = {}
    for ai, arm in enumerate(BG_ARMS):
        Pe = readP(z["pred"][:, ai].astype(np.float64).mean(2))
        err, R = wrap(angle_of(Pe) - tg).mean(1), proj(Pe - Ps[:, None], d_true).mean(1)
        errs[arm] = err
        res["table"][arm] = {"err_to_target": boot_ci(err), "R": boot_ci(R),
                             "edit_energy_median": float(np.median(z["energy"][:, ai])),
                             "edited_tokens_median": float(np.median(z["ntok"][:, ai]))}
    Pd = readP(dpred.astype(np.float64).mean(2))
    res["disk12_reproduction_maxabs_err_deg"] = float(np.abs(wrap(angle_of(readP(z["pred"][:, 0].astype(np.float64).mean(2))) - angle_of(Pd))).max())
    for arm in BG_ARMS[1:]:
        res["paired_vs_disk12"][f"{arm}_minus_disk12_err"] = boot_ci(errs[arm] - errs["disk12"])
    write(RES / "session2_direction_natural_norm_bgcontrol.json", res, stage="natural_norm_bgcontrol_score",
          seeds={"bootstrap": 0, "bg_subset": "1000*c+j"}, cache=str(NN / "bg_pred.npz"))
    for arm in BG_ARMS:
        t = res["table"][arm]
        print(arm, t["err_to_target"], t["R"]["mean"], t["R"]["ci95"], t["edit_energy_median"], t["edited_tokens_median"])
    print("repro", res["disk12_reproduction_maxabs_err_deg"], res["paired_vs_disk12"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "forward":
        forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "score":
        score()
    elif cmd == "disk_forward":
        disk_forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "disk_score":
        disk_score()
    elif cmd == "bg_plan":
        bg_plan()
    elif cmd == "bg_forward":
        bg_forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "bg_score":
        bg_score()

"""Why are point-12 direction edits undone? Sublayer attribution of the repair (REPORT §4.5), plus a patching test.

Edits (recomputed with the run_part2 code path; run_part2 does not store deltas): the headline arc (direction,
contiguous held-out block seed 0, smoothing spline, PCA-64, K = 50, unsupervised angle rule, cubic extension, coord
aim; results/p2_steer_direction_direction_L{12,22}_contiguous_rawchord.json, which adds the raw-centroid chord to the
headline configuration). Target 320.625 deg (4th of the 8 held-out values, fixed before any read). Carriers = the
first 16 test clips run_part2.pick_clips draws for that target (the n_control_clips subset). Arms, each the K-th
(endpoint) waypoint minus the clip:
  spline   = run_part2 "manifold" arm (smoothing spline walk, residual kept)
  rawchord = run_part2 "linear_raw" arm (chord between the RAW kept centroids, A.9's baseline)
  random   = the spline delta's PCA-64 coordinates through a random orthonormal 64-d basis, rescaled to the spline
             delta's norm (rank- and norm-matched)
at point 12 and, as the positive control, at point 22 (same carriers, target, recipe; point-22 build).

Forward (box): each edit is added to every token at point L (wm.propagate.suffix). wm.repair.SublayerTap records
the residual after every attention and MLP sublayer of blocks L+1..24 and the sublayer outputs. Two forwards per
condition: the full 16-frame clip (probe readouts, as session2 readout_a) and the context-only frames 1-8 (encoder
-> final LN -> predictor, mask token 0, as session2 predictor test). Patching: the change a sublayer's output makes
relative to the clean run is projected off the edit direction u = delta / ||delta|| in blocks 13-16 (and 13-24),
attention only, MLP only, or both.

Readouts (score, CPU): direction probe per site fit on the 480 probe clips only (train folds 3-4): block outputs
(point b) = stored meanpool + step-1 sweep alpha (session2 readout_a exactly); post-attention sites = this run's
forward of the probe clips, alpha by 2-fold CV over folds 3/4. Predictor: session2_native_readout's step-mean probe
(ridge on the predictor's own unedited forecasts of the probe clips, alpha by 5-fold CV, rng 0).

  python scripts/run_repair_attribution.py plan
  python scripts/run_repair_attribution.py forward --root <out>   (box, GPU)
  python scripts/run_repair_attribution.py score                   (+ figure)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_repair"
RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
D = 1024
TARGET = 320.625
N_CAR = 16
POINTS = (12, 22)
FORECAST_POINTS = (2, 8, 12, 16, 19, 22)     # predictor obedience by edit point (spline, raw chord)
ARMS = ("spline", "rawchord", "random")
PATCHES = {"none": (), "attn_13_16": ("attn",), "mlp_13_16": ("mlp",), "both_13_16": ("attn", "mlp"),
           "attn_13_24": ("attn",), "mlp_13_24": ("mlp",), "both_13_24": ("attn", "mlp")}


def patch_sites(name):
    if name == "none":
        return ()
    hi = 16 if name.endswith("13_16") else 24
    return tuple((k, b) for b in range(13, hi + 1) for k in PATCHES[name])


def conditions():
    out = [(f"{a}{L}", L, "none") for L in POINTS for a in ARMS]
    out += [(f"{a}12", 12, p) for a in ARMS for p in PATCHES if p != "none"]
    return out


# ================================================================ plan (CPU)
def plan():
    from run_part2 import build, compose, pick_clips, subspace_arms
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.provenance import git_commit, sha256_file

    out, meta = {}, {"target": TARGET, "n_carriers": N_CAR, "points": {}}
    ids = None
    for L in FORECAST_POINTS:
        d = load_inputs("direction", L)
        m = build(d, 64, "unsupervised", "contiguous", 0, n_controls=0, spline="smooth")
        picks = pick_clips(d, m["held"], 48, 0)
        pick = picks[TARGET][:N_CAR]
        cid = d["df"]["id"].to_numpy()[pick]
        ids = cid if ids is None else ids
        assert (ids == cid).all()
        x, src = d["X"][pick].astype(float), d["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Za = subspace_arms(Z, src, TARGET, m, 50)
        dl = {a: compose(m["pca"], Za[k][:, -1:], resid)[:, 0] - x for a, k in (("spline", "manifold"),
                                                                                 ("rawchord", "linear_raw"))}
        c = Za["manifold"][:, -1] - Z                                                 # PCA-64 coordinates
        R = np.linalg.qr(np.random.default_rng(L).standard_normal((D, c.shape[1])))[0]
        dr = c @ R.T
        dl["random"] = dr * (np.linalg.norm(dl["spline"], axis=1) / np.linalg.norm(dr, axis=1))[:, None]
        # parity with the stored run_part2 rows (same ids, target, arms): probe error at the edit point, ||delta||
        refp = RES / f"p2_steer_direction_direction_L{L}_contiguous_rawchord.json"
        ref = json.loads(refp.read_text()) if refp.exists() else {"rows": []}
        rows = {(r["arm"], r["id"]): r for r in ref["rows"] if r["target"] == TARGET}
        probe = mf.ProbeReadout(d["X"][d["role"] == "probe"], d["y"][d["role"] == "probe"], True)
        par = {}
        for a, k in (("spline", "manifold"), ("rawchord", "linear_raw")) if rows else ():
            e = mf.value_error(probe.predict(x + dl[a]), TARGET, True)
            se = np.array([rows[(k, int(i))]["probe_err_to_target"] for i in cid])
            sn = np.array([rows[(k, int(i))]["delta_norm"] for i in cid])
            par[a] = {"probe_err_maxabs_diff_deg": float(np.abs(e - se).max()),
                      "delta_norm_max_rel_diff": float(np.abs(np.linalg.norm(dl[a], axis=1) - sn).max() / sn.max()),
                      "probe_err_to_target_mean": float(e.mean())}
        meta["points"][str(L)] = {"angle_source": m["curve"].coord_source, "spline": m["curve"].kind,
                                  "held_out_values": m["held"].tolist(), "parity_vs_run_part2_rows": par,
                                  "delta_norm_mean": {a: float(np.linalg.norm(v, axis=1).mean()) for a, v in dl.items()}}
        for a, v in dl.items():
            out[f"delta_{a}{L}"] = v.astype(np.float32)
        out[f"clean_mp_L{L}"] = d["X"][pick].astype(np.float32)
        if L == 12:
            out["src"] = src
            probe_ids = d["df"]["id"].to_numpy()[d["role"] == "probe"]
    out.update(carrier_ids=ids, probe_ids=probe_ids)
    meta["git"] = git_commit()
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", **out)
    meta["plan_sha256"] = sha256_file(ART / "plan.npz")
    (ART / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


# ================================================================ forward (GPU)
def forward(root, batch=4, probe_batch=8, smoke=False):
    import torch
    from wm.data import decode, load_table
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import _mean, prefix, suffix
    from wm.repair import SublayerTap

    set_precision()
    dev = pick_device()
    model = load_model("vjepa2", dev)
    enc = model.encoder
    P = dict(np.load(ART / "plan.npz"))
    vid = dict(zip(load_table("direction")["id"], load_table("direction")["video"]))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {}
    # ---- probe clips: residual meanpool after every sublayer of every block
    pids = P["probe_ids"][:16] if smoke else P["probe_ids"]
    acc = []
    for s in range(0, len(pids), probe_batch):
        pv = preprocess([decode(vid[i]) for i in pids[s:s + probe_batch]]).to(dev)
        with SublayerTap(enc.layer, range(1, 25)) as t:
            prefix(model, pv, [])
        acc.append(t.stacked("resid_mp").numpy())
    out["probe_resid_mp"] = np.concatenate(acc)                          # [n, 48, D] sites (attn,1),(mlp,1),..
    print(f"probe clips {len(pids)} {time.time() - t0:.0f}s", flush=True)
    conds = conditions()
    cids = P["carrier_ids"][:batch] if smoke else P["carrier_ids"]
    for mode in ("full", "ctx"):
        store = {}
        for s in range(0, len(cids), batch):
            rows = slice(s, s + batch)
            pv = preprocess([decode(vid[i]) for i in cids[rows]]).to(dev)
            if mode == "ctx":
                pv = pv[:, :8]
            saved, _, _ = prefix(model, pv, list(FORECAST_POINTS))
            for L in FORECAST_POINTS:
                store.setdefault(f"clean_mp_L{L}", []).append(_mean(saved[L]).float().cpu().numpy())
            with SublayerTap(enc.layer, range(13, 25), keep_tokens=True) as clean:
                r = suffix(model, saved[12], 12, None, return_final=True)
            _put(store, "clean", clean, r, mode, pool_steps, predict_future, model)
            for name, L, pname in conds:
                delta = torch.as_tensor(P[f"delta_{name}"][rows], device=dev)
                u = delta / delta.norm(dim=1, keepdim=True)
                with SublayerTap(enc.layer, range(L + 1, 25), clean=clean.tokens, u=u,
                                 patch=patch_sites(pname)) as t:
                    r = suffix(model, saved[L], L, delta, return_final=True)
                _put(store, f"{name}/{pname}", t, r, mode, pool_steps, predict_future, model)
            for L in sorted(set(FORECAST_POINTS) - set(POINTS)):        # forecast-by-edit-point extension, no tap
                for a in ("spline", "rawchord"):
                    delta = torch.as_tensor(P[f"delta_{a}{L}"][rows], device=dev)
                    r = suffix(model, saved[L], L, delta, return_final=True)
                    store.setdefault(f"fc/{a}{L}/final_mp", []).append(r["meanpool"][:, -1])
                    if mode == "ctx":
                        store.setdefault(f"fc/{a}{L}/pred_pooled", []).append(
                            pool_steps(predict_future(model, r["final"])).float().cpu().numpy())
            del clean
            torch.cuda.empty_cache() if dev.type == "cuda" else None
            print(f"{mode} carriers {s + batch}/{len(cids)} {time.time() - t0:.0f}s", flush=True)
        out.update({f"{mode}/{k}": np.concatenate(v) for k, v in store.items()})
    info = {"seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "forward_dtype": "float32 (tf32 off), as session 2", "batch": batch,
            "smoke": smoke, "conditions": conds}
    np.savez(root / "forward.npz", **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", info["seconds"])


def _put(store, key, tap, r, mode, pool_steps, predict_future, model):
    for q in ("resid_mp", "sub_mp", "resid_tok_norm_mean", "sub_tok_cos", "sub_tok_rms", "resid_tok_rms_change"):
        if q in tap.rec[tap.sites()[0]]:
            store.setdefault(f"{key}/{q}", []).append(tap.stacked(q).numpy())
    store.setdefault(f"{key}/final_mp", []).append(r["meanpool"][:, -1])
    if mode == "ctx":
        store.setdefault(f"{key}/pred_pooled", []).append(pool_steps(predict_future(model, r["final"])).float().cpu().numpy())


# ================================================================ score (CPU)
def site_names(blocks):
    return [f"{k}{b}" for b in blocks for k in ("attn", "mlp")]


def score(root):
    from run_session2 import angle_of, boot_ci, wrap, write
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, load_sweep, predict, score as pscore, targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    P = dict(np.load(ART / "plan.npz"))
    meta = json.loads((ART / "plan.json").read_text())
    finfo = json.loads((root / "forward_info.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    d0 = load_inputs("direction", 0)
    probe = d0["role"] == "probe"
    pid = np.flatnonzero(probe)
    assert (df["id"].to_numpy()[pid] == P["probe_ids"]).all()
    mp = np.load(PROJECT_ROOT / "artifacts/activations/direction/vjepa2/meanpool.npy", mmap_mode="r")
    sweep = load_sweep("direction", "direction")
    Yp, fold_p = Y[probe], d0["fold"][probe]
    circ = lambda a, b: pscore(a, b, "circular")                                           # noqa: E731
    readers, reader_info = {}, {}
    PR = F["probe_resid_mp"].astype(np.float64)                                            # [480, 48, D]

    def fit_reader(X, alpha):
        st = Standardizer().fit(X)
        W, b = fit_ridge(st.transform(X), Yp, alpha)
        return st, W, b

    for b in range(12, 26):
        readers[f"pt{b}"] = fit_reader(np.asarray(mp[pid, b], np.float64), sweep["layers"][b]["alpha"])
    parity_probe = {}
    for b in range(1, 25):
        Xa = PR[:, 2 * (b - 1)]
        st = Standardizer().fit(Xa)
        cv = cv_select_alpha(st.transform(Xa), Yp, fold_p, score_fn=circ)
        readers[f"attn{b}"] = fit_reader(Xa, cv["alpha"])
        readers[f"mlp{b}"] = readers[f"pt{b}"] if b >= 12 else None
        reader_info[f"attn{b}"] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "cv_mae_deg": cv["cv_mae_mean"]}
        Xm = np.asarray(mp[pid, b], np.float64)
        parity_probe[b] = float(np.abs(PR[:, 2 * b - 1] - Xm).max() / np.abs(Xm).max())
    for b in range(12, 26):
        reader_info[f"pt{b}"] = {"alpha": sweep["layers"][b]["alpha"], "source": "stored meanpool, step-1 sweep alpha"}

    def read(X, site):
        st, W, b = readers[site]
        return angle_of(predict(st.transform(np.asarray(X, np.float64)), W, b))

    # predictor-native probe (session2_native_readout, step-mean)
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Xn = Zall.mean(1)
    stn = Standardizer().fit(Xn[probe])
    cvn = cv_select_alpha(stn.transform(Xn[probe]), Yp, folds5, score_fn=circ)
    Wn, bn = fit_ridge(stn.transform(Xn[probe]), Yp, cvn["alpha"])
    pread = lambda Zp: angle_of(predict(stn.transform(Zp.mean(1).astype(np.float64)), Wn, bn))   # noqa: E731

    cid = P["carrier_ids"]
    rows = np.searchsorted(df["id"].to_numpy(), cid)
    src = y[rows]
    C = len(cid)
    parity = {"probe_clip_block_output_vs_stored_rel_maxabs": max(parity_probe.values()),
              "carrier_clean_final_ln_vs_stored_rel_maxabs": float(np.abs(F["full/clean/final_mp"] - mp[rows, 25]).max()
                                                                   / np.abs(mp[rows, 25]).max()),
              "carrier_clean_pt12_vs_plan_rel_maxabs": float(np.abs(F["full/clean_mp_L12"] - P["clean_mp_L12"]).max()
                                                             / np.abs(P["clean_mp_L12"]).max()),
              "ctx_clean_predictor_vs_native_cache_rel_maxabs": float(np.abs(F["ctx/clean/pred_pooled"] - Zall[rows]).max()
                                                                      / np.abs(Zall[rows]).max())}
    err = lambda a: wrap(a - TARGET)                                                        # noqa: E731
    B = lambda v: boot_ci(v)                                                                # noqa: E731

    def arm_block(mode, name, L, pname):
        k = f"{mode}/{name}/{pname}"
        blocks = list(range(L + 1, 25))
        sites = site_names(blocks)
        delta = P[f"delta_{name}"].astype(np.float64)
        dn2 = (delta ** 2).sum(1)
        cl = lambda q: F[f"{mode}/clean/{q}"][:, 2 * (L + 1 - 13):].astype(np.float64)    # noqa: E731
        R, S = F[f"{k}/resid_mp"].astype(np.float64), F[f"{k}/sub_mp"].astype(np.float64)
        dR, dS = R - cl("resid_mp"), S - cl("sub_mp")
        nclean = cl("resid_mp")
        rec = {"sites": ["edit_point"] + sites + ["final_ln"], "per_site": []}
        clean_L = F[f"{mode}/clean_mp_L{L}"].astype(np.float64)
        row0 = {"site": f"pt{L} (edited)", "survival": B(np.ones(C))}
        if mode == "full":
            row0["err_to_target"] = B(err(read(clean_L + delta, f"pt{L}")))
        rec["per_site"].append(row0)
        for j, s in enumerate(sites):
            row = {"site": s}
            if mode == "full":
                row["err_to_target"] = B(err(read(R[:, j], s)))
                row["err_to_true"] = B(wrap(read(R[:, j], s) - src))
            surv = (dR[:, j] * delta).sum(1) / dn2
            contrib = (dS[:, j] * delta).sum(1) / dn2
            nS = np.linalg.norm(dS[:, j], axis=1)
            nR = np.linalg.norm(dR[:, j], axis=1)
            row.update({
                "survival": B(surv),
                "sublayer_contribution_along_delta": B(contrib),
                "cos_sublayer_change_with_delta": B((dS[:, j] * delta).sum(1) / np.maximum(nS * np.sqrt(dn2), 1e-12)),
                "token_mean_cos_sublayer_change_with_delta": B(F[f"{k}/sub_tok_cos"][:, j]),
                "sublayer_change_norm_over_delta": B(nS / np.sqrt(dn2)),
                "sublayer_change_token_rms_over_delta": B(F[f"{k}/sub_tok_rms"][:, j] / np.sqrt(dn2)),
                "residual_change_norm_over_delta": B(nR / np.sqrt(dn2)),
                "residual_change_orthogonal_fraction": B(np.sqrt(np.maximum(nR ** 2 - (surv ** 2) * dn2, 0)) / np.maximum(nR, 1e-12)),
                "edit_share_of_clean_meanpool_norm": B(surv * np.sqrt(dn2) / np.linalg.norm(nclean[:, j], axis=1)),
                "edit_share_of_clean_token_norm": B(surv * np.sqrt(dn2) / F[f"{mode}/clean/resid_tok_norm_mean"][:, 2 * (L + 1 - 13) + j]),
                "clean_token_norm_mean": B(F[f"{mode}/clean/resid_tok_norm_mean"][:, 2 * (L + 1 - 13) + j])})
            rec["per_site"].append(row)
        fe = F[f"{k}/final_mp"].astype(np.float64)
        dfin = fe - F[f"{mode}/clean/final_mp"]
        row = {"site": "pt25 (final LN)", "survival": B((dfin * delta).sum(1) / dn2)}
        if mode == "full":
            row["err_to_target"] = B(err(read(fe, "pt25")))
        rec["per_site"].append(row)
        if mode == "ctx":
            pe = pread(F[f"{k}/pred_pooled"])
            rec["predictor_err_to_target"] = B(err(pe))
            rec["_pred_err"] = err(pe)
        rec["_pt16_err"] = err(read(R[:, sites.index("mlp16")], "pt16")) if (mode == "full" and "mlp16" in sites) else None
        rec["_pt25_err"] = err(read(fe, "pt25")) if mode == "full" else None
        rec["_surv"] = np.array([(dR[:, j] * delta).sum(1) / dn2 for j in range(len(sites))]).T
        rec["_contrib"] = np.array([(dS[:, j] * delta).sum(1) / dn2 for j in range(len(sites))]).T
        return rec

    res = {"question": "why are point-12 direction edits undone within four blocks while point-22 edits reach the "
                       "predictor? sublayer attribution (attention vs MLP) and a projection-removal patching test",
           "config": {"target_deg": TARGET, "n_carriers": C, "carrier_ids": cid.tolist(), "carrier_true_deg": src.tolist(),
                      "arms": {"spline": "run_part2 manifold arm endpoint (smoothing spline, headline arc)",
                               "rawchord": "run_part2 linear_raw arm endpoint (chord between raw kept centroids)",
                               "random": "spline delta's PCA-64 coords through a random orthonormal 64-d basis, "
                                         "rescaled to the spline delta's norm (seed = point)"},
                      "points": POINTS, "patches": {p: [f"{k}{b}" for k, b in patch_sites(p)] for p in PATCHES},
                      "plan": meta, "forward": finfo},
           "readers": reader_info,
           "predictor_probe": {"alpha": cvn["alpha"], "cv_r2": cvn["cv_mean"], "cv_mae_deg": cvn["cv_mae_mean"]},
           "parity": parity, "baseline_unedited": {}, "full": {}, "ctx": {}, "patching": {}}
    # unedited baseline per site
    base = {}
    for b in range(13, 25):
        for kind, j in (("attn", 2 * (b - 13)), ("mlp", 2 * (b - 13) + 1)):
            a = read(F["full/clean/resid_mp"][:, j], f"{kind}{b}")
            base[f"{kind}{b}"] = {"err_to_target": B(err(a)), "err_to_true": B(wrap(a - src))}
    base["pt12"] = {"err_to_target": B(err(read(F["full/clean_mp_L12"], "pt12")))}
    base["pt22"] = {"err_to_target": B(err(read(F["full/clean_mp_L22"], "pt22")))}
    base["pt25"] = {"err_to_target": B(err(read(F["full/clean/final_mp"], "pt25"))),
                    "err_to_true": B(wrap(read(F["full/clean/final_mp"], "pt25") - src))}
    p0 = pread(F["ctx/clean/pred_pooled"])
    base["predictor"] = {"err_to_target": B(err(p0)), "err_to_true": B(wrap(p0 - src))}
    res["baseline_unedited"] = base
    raw = {}
    for name, L, pname in conditions():
        for mode in ("full", "ctx"):
            rec = arm_block(mode, name, L, pname)
            raw[(mode, name, pname)] = rec
            res[mode].setdefault(name, {})[pname] = {k: v for k, v in rec.items() if not k.startswith("_")}
    # attribution summary: summed sublayer contributions over blocks 13-16 (point-12 arms) / 23-24 (point-22 arms)
    attrib = {}
    for name, L, _ in [c for c in conditions() if c[2] == "none"]:
        rec = raw[("full", name, "none")]
        sites = rec["sites"][1:-1]
        hi = 16 if L == 12 else 24
        idx_a = [i for i, s in enumerate(sites) if s.startswith("attn") and int(s[4:]) <= hi]
        idx_m = [i for i, s in enumerate(sites) if s.startswith("mlp") and int(s[3:]) <= hi]
        attrib[name] = {"blocks": f"{L + 1}-{hi}",
                        "sum_attn_contribution": B(rec["_contrib"][:, idx_a].sum(1)),
                        "sum_mlp_contribution": B(rec["_contrib"][:, idx_m].sum(1)),
                        "survival_after_block": B(rec["_surv"][:, idx_m[-1]]),
                        "sum_attn_contribution_all_later_blocks": B(rec["_contrib"][:, 0::2].sum(1)),
                        "sum_mlp_contribution_all_later_blocks": B(rec["_contrib"][:, 1::2].sum(1)),
                        "survival_final_block24": B(rec["_surv"][:, -1])}
    res["attribution_summary"] = attrib
    # patching: paired differences vs the unpatched edit
    for a in ARMS:
        name = f"{a}12"
        n0f, n0c = raw[("full", name, "none")], raw[("ctx", name, "none")]
        for p in PATCHES:
            rf, rc = raw[("full", name, p)], raw[("ctx", name, p)]
            i16 = rf["sites"][1:-1].index("mlp16")
            res["patching"].setdefault(name, {})[p] = {
                "survival_pt16": B(rf["_surv"][:, i16]), "survival_pt24": B(rf["_surv"][:, -1]),
                "probe_err_pt16": B(rf["_pt16_err"]), "probe_err_pt25": B(rf["_pt25_err"]),
                "predictor_err": B(rc["_pred_err"]),
                "ctx_survival_pt24": B(rc["_surv"][:, -1]),
                "paired_predictor_err_minus_unpatched": B(rc["_pred_err"] - n0c["_pred_err"]),
                "paired_pt25_err_minus_unpatched": B(rf["_pt25_err"] - n0f["_pt25_err"])}
    for L22 in ("spline22", "rawchord22", "random22"):
        rf = raw[("full", L22, "none")]
        res["patching"].setdefault("positive_control_point22", {})[L22] = {
            "survival_pt24": B(rf["_surv"][:, -1]), "probe_err_pt25": B(rf["_pt25_err"]),
            "predictor_err": B(raw[("ctx", L22, "none")]["_pred_err"])}
    # predictor obedience by edit point (spline, raw chord; no tap); points 12/22 reuse the unpatched conditions
    fr = {"points": list(FORECAST_POINTS), "target_deg": TARGET,
          "unedited": {"predictor_err_to_target": base["predictor"]["err_to_target"],
                       "probe_err_pt25_to_target": base["pt25"]["err_to_target"]},
          "readouts": "predictor = session2 native step-mean probe on the forecast from the edited context (frames "
                      "1-8); pt25 = session2 readout_a probe (point 25, probe clips) on the edited full clip",
          "per_point": {}}
    for L in FORECAST_POINTS:
        for a in ("spline", "rawchord"):
            key = f"{a}{L}/none" if L in POINTS else f"fc/{a}{L}"
            fe = F[f"full/{key}/final_mp"]
            pe = pread(F[f"ctx/{key}/pred_pooled"])
            fr["per_point"].setdefault(str(L), {})[a] = {
                "predictor_err_to_target": B(err(pe)), "predictor_err_to_true": B(wrap(pe - src)),
                "paired_predictor_err_minus_unedited": B(err(pe) - err(p0)),
                "probe_err_pt25_to_target": B(err(read(fe, "pt25"))),
                "delta_norm_mean": meta["points"][str(L)]["delta_norm_mean"][a],
                "angle_source": meta["points"][str(L)]["angle_source"]}
    res["forecast_response_by_edit_point"] = fr
    write(RES / "p5_repair_attribution_L12.json", res, stage="p5_repair_attribution",
          seeds={"arc": 0, "random_basis": list(POINTS), "native_probe_folds": 0},
          forward_npz_sha256=sha256_file(root / "forward.npz"), plan_npz_sha256=sha256_file(ART / "plan.npz"))
    figure(raw)


def figure(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"spline": "#c65d3b", "rawchord": "#4682b4", "random": "#8c8c8c"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a in ARMS:
        for L, ls in ((12, "-"), (22, "--")):
            rec = raw[("full", f"{a}{L}", "none")]
            s = rec["_surv"]
            xs = np.arange(len(s[0])) / 2 + L + 0.5
            m = np.r_[1.0, s.mean(0)]
            ax[0].plot(np.r_[L, xs], m, ls, color=col[a], marker="o", ms=3,
                       label=f"{a}, edit at point {L}")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("site (half-integers = residual after attention; integers = after MLP = block output)")
    ax[0].set_ylabel("survival fraction  <Δresid, Δ> / ||Δ||²")
    ax[0].set_title("How much of the edit survives, sublayer by sublayer (16 carriers)")
    ax[0].legend(fontsize=7)
    for p, ls in (("none", "-"), ("attn_13_16", "--"), ("mlp_13_16", ":"), ("both_13_16", "-.")):
        for a in ARMS:
            s = raw[("full", f"{a}12", p)]["_surv"].mean(0)
            xs = np.arange(len(s)) / 2 + 12.5
            ax[1].plot(np.r_[12, xs], np.r_[1.0, s], ls, color=col[a], lw=1.2, label=f"{a}, patch {p}")
    ax[1].axvspan(12.5, 16, color="0.9", zorder=0)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("site (point)")
    ax[1].set_ylabel("survival fraction")
    ax[1].set_title("Patched: sublayer change along Δ removed in blocks 13–16 (shaded)")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_repair_attribution_L12.png", dpi=150)
    print("wrote", FIG / "fig_repair_attribution_L12.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "forward", "score"))
    ap.add_argument("--root", default=str(ART / "forward"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.stage == "plan":
        plan()
    elif a.stage == "forward":
        forward(a.root, a.batch, smoke=a.smoke)
    else:
        score(a.root)

"""Twin-difference arm (Liu et al. 2608.15156, low-rank carriers from counterfactual-minus-factual hidden
differences): the strongest linear baseline not yet run on the session-2 carriers.

Fit set (disjoint from everything evaluated): knot clips (train folds 0-2; never carriers, never read probes), 2 per
source value (seed 0), each rendered (src/wm/render_twin.py, same renderer/codec as the session-2 twins) at every
held-out arc target except its own value. Per fit pair n: d_n = meanpool_L(twin at target) - meanpool_L(clip)
(full 16-frame clip, the space every session-2 delta lives in). For a held-out carrier c and target t:

  delta_full(c, t) = mean of d_n over the K_NN fit pairs with target t whose source angle is closest to the carrier's
  delta_r(c, t)    = V_r V_r^T delta_full, V_r = top-r right singular vectors of the (uncentred) difference matrix
                     [d_n] over all fit pairs, r in {1, 2, 4, 8}; 'full' = no rank reduction

then rescaled per (carrier, target) exactly as session2_norm_matched.py does (chord_norm: ||delta_chord||;
natural_twin_norm: ||twin_meanpool_L - src_meanpool_L|| of the carrier's own rendered twin) and added to every token at
point L: full clip -> blocks L+1..24 + LN (encoder-output probe at point 25, nearest-real R), context frames 1-8 ->
predictor, mask token 0 (predictor-native direction + position probes, as session2_norm_matched.score).

  prep     CPU, local. Fit clips, twins (MP4, rendered locally like run_session2.plan), comparison deltas and norms
           -> artifacts/session2/twin_diff/ship/
  forward  box, GPU. --layer L. Encodes the fit set once (cached), builds the deltas, runs the edits.
           -> artifacts/session2/twin_diff/L{L}/fwd.npz
  score    CPU, local. --layer L -> results/session2_twin_difference_L{L}.json, figures/fig_twin_difference.png
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

S2 = PROJECT_ROOT / "artifacts" / "session2"
TD = S2 / "twin_diff"
SHIP = TD / "ship"
RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
RANKS = (1, 2, 4, 8)
TWIN_ARMS = tuple(f"twin_r{r}" for r in RANKS) + ("twin_full",)
CMP_ARMS = ("spline", "chord", "probe_qr")
CONDS = ("chord_norm", "natural_twin_norm")
PER_VALUE = 2
K_NN = 4
D = 1024


# ---------------------------------------------------------------- pure functions (tested)

def circ_dist(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def svd_basis(diffs, r):
    """Top-r right singular vectors [r, D] of the uncentred difference matrix diffs [N, D]."""
    _, _, vt = np.linalg.svd(np.asarray(diffs, np.float64), full_matrices=False)
    return vt[:r]


def rank_reduce(delta, V):
    """Project delta [..., D] onto span(V) (V [r, D] orthonormal rows); V None = identity."""
    delta = np.asarray(delta, np.float64)
    return delta if V is None else (delta @ V.T) @ V


def knn_mean_difference(diffs, pair_src, pair_tgt, src, tgt, k=K_NN):
    """delta_full [C, T, D]: for each (carrier source src[c], target tgt[c, t]) the mean of the k fit differences with
    that target whose source angle is closest to src[c] (ties by index)."""
    C, T = tgt.shape
    out = np.zeros((C, T, diffs.shape[1]))
    used = np.zeros((C, T, k), int)
    for t_val in np.unique(tgt):
        cand = np.flatnonzero(np.isclose(pair_tgt, t_val))
        assert len(cand) >= k, (t_val, len(cand))
        for c, t in zip(*np.nonzero(np.isclose(tgt, t_val))):
            sel = cand[np.argsort(circ_dist(pair_src[cand], src[c]), kind="stable")[:k]]
            out[c, t] = diffs[sel].mean(0)
            used[c, t] = sel
    return out, used


def twin_deltas(diffs, pair_src, pair_tgt, src, tgt, ranks=RANKS, k=K_NN):
    """{arm: [C, T, D]} for twin_r{r} and twin_full, plus the SVD spectrum."""
    full, used = knn_mean_difference(diffs, pair_src, pair_tgt, src, tgt, k)
    s = np.linalg.svd(np.asarray(diffs, np.float64), compute_uv=False)
    out = {f"twin_r{r}": rank_reduce(full, svd_basis(diffs, r)) for r in ranks}
    out["twin_full"] = full
    return out, s, used


def rescale(delta, ref_norm):
    """Rescale delta [..., D] to norm ref_norm [...] (per row)."""
    n = np.linalg.norm(delta, axis=-1)
    return delta * (ref_norm / np.maximum(n, 1e-12))[..., None]


# ---------------------------------------------------------------- prep (local)

def _render(job):
    from wm.render_twin import load_meta, write_twin
    cid, th, path = job
    write_twin(load_meta("direction", cid), th, path)
    return path


def prep(args):
    from multiprocessing import Pool
    from run_session2 import load_groups, load_plan
    from wm.p2_data import load_inputs
    arrays, info = load_plan(S2)
    held = np.array(info["held_values"], float)
    d0 = load_inputs("direction", 0)
    df, y, role = d0["df"], d0["y"], d0["role"]
    rng = np.random.default_rng(0)
    fit_rows = []
    for v in np.unique(y):
        pool = np.flatnonzero((role == "knot") & np.isclose(y, v))
        fit_rows += sorted(rng.choice(pool, size=min(PER_VALUE, len(pool)), replace=False).tolist())
    fit_rows = np.array(fit_rows)
    assert not np.isin(fit_rows, arrays["carrier_rows"]).any() and (role[fit_rows] == "knot").all()
    pair_fit, pair_tgt = [], []
    for i, r in enumerate(fit_rows):
        for t in held:
            if not np.isclose(t, y[r]):
                pair_fit.append(i)
                pair_tgt.append(t)
    pair_fit, pair_tgt = np.array(pair_fit), np.array(pair_tgt)
    (SHIP / "twins").mkdir(parents=True, exist_ok=True)
    ids = df["id"].to_numpy()
    jobs = [(int(ids[fit_rows[i]]), float(t), str(SHIP / "twins" / f"fit_{n:04d}.mp4"))
            for n, (i, t) in enumerate(zip(pair_fit, pair_tgt))]
    t0 = time.time()
    with Pool(args.procs) as p:
        p.map(_render, jobs, chunksize=8)
    print(f"rendered {len(jobs)} fit twins in {time.time() - t0:.0f}s")
    F = load_groups(S2 / "forward", keys={"idx", "src_meanpool", "twin_meanpool"})
    idx = F["idx"]
    ai = [info["arms"].index(a) for a in CMP_ARMS]
    ship = {"fit_rows": fit_rows, "fit_ids": ids[fit_rows], "fit_src_angle": y[fit_rows], "pair_fit": pair_fit,
            "pair_tgt": pair_tgt, "carrier_rows": arrays["carrier_rows"][idx], "targets": arrays["targets"][idx],
            "carrier_src_angle": y[arrays["carrier_rows"][idx]], "idx": idx}
    for L in (12, 22):
        dl = arrays[f"deltas_L{L}"][idx][:, ai].astype(np.float32)                         # [C, 3, T, D]
        ship[f"cmp_deltas_L{L}"] = dl
        ship[f"chord_norm_L{L}"] = np.linalg.norm(dl[:, CMP_ARMS.index("chord")].astype(np.float64), axis=-1)
        ship[f"natural_norm_L{L}"] = np.linalg.norm(F["twin_meanpool"][:, :, L].astype(np.float64)
                                                    - F["src_meanpool"][:, None, L].astype(np.float64), axis=-1)
    np.savez(SHIP / "ship.npz", **ship)
    print("fit clips", len(fit_rows), "pairs", len(pair_fit), "->", SHIP)


# ---------------------------------------------------------------- forward (box)

def forward(args):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import prefix, suffix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    L = args.layer
    z = dict(np.load(SHIP / "ship.npz"))
    df = load_table("direction")
    bs = args.batch_size
    enc_path = TD / "fit_encodings.npz"
    t_all = time.time()
    if not enc_path.exists():
        t0 = time.time()
        fit_frames = [decode(df["video"].iloc[r]) for r in z["fit_rows"]]
        src_mp = np.concatenate([prefix(model, preprocess(fit_frames[s:s + bs]).to(device), [])[2]
                                 for s in range(0, len(fit_frames), bs)])
        tw = []
        n = len(z["pair_fit"])
        for s in range(0, n, bs):
            fr = [decode(SHIP / "twins" / f"fit_{k:04d}.mp4") for k in range(s, min(s + bs, n))]
            tw.append(prefix(model, preprocess(fr).to(device), [])[2])
            if s % 256 == 0:
                print(f"fit twins {s}/{n} {time.time() - t0:.0f}s", flush=True)
        np.savez(enc_path, fit_src_meanpool=src_mp, fit_twin_meanpool=np.concatenate(tw), seconds=time.time() - t0)
    E = np.load(enc_path)
    diffs = E["fit_twin_meanpool"][:, L].astype(np.float64) - E["fit_src_meanpool"][z["pair_fit"], L].astype(np.float64)
    dl, spec, used = twin_deltas(diffs, z["fit_src_angle"][z["pair_fit"]], z["pair_tgt"], z["carrier_src_angle"],
                                 z["targets"])
    ref = {"chord_norm": z[f"chord_norm_L{L}"], "natural_twin_norm": z[f"natural_norm_L{L}"]}
    conds = args.conds
    tw_d = np.stack([np.stack([rescale(dl[a], ref[c]) for a in TWIN_ARMS], 1) for c in conds], 1)   # [C,c,A,T,D]
    cmp_raw = z[f"cmp_deltas_L{L}"].astype(np.float64)                                            # [C,3,T,D]
    cmp_d = np.stack([rescale(cmp_raw, ref[c][:, None]) for c in conds], 1)                       # [C,c,3,T,D]
    C, nc, A, T, _ = tw_d.shape
    rows = z["carrier_rows"]
    enc25_tw = np.zeros((C, nc, A, T, D), np.float32)
    enc25_cmp = np.zeros((C, nc, 3, T, D), np.float32)
    pred_tw = np.zeros((C, nc, A, T, 4, D), np.float16)
    pred_cmp = np.zeros((C, nc, 3, T, 4, D), np.float16) if args.pred_compare else None
    pred_chord_raw = np.zeros((C, T, 4, D), np.float32)
    src_mp, src_pp = np.zeros((C, 26, D), np.float32), np.zeros((C, 4, D), np.float32)
    t0 = time.time()
    G = 8
    for g in range(0, C, G):
        ci = list(range(g, min(g + G, C)))
        pv = preprocess([decode(df["video"].iloc[rows[i]]) for i in ci]).to(device)
        saved, _, mp = prefix(model, pv, [L])
        src_mp[ci] = mp
        csaved, cfin, _ = prefix(model, pv[:, :8], [L])
        src_pp[ci] = pool_steps(predict_future(model, cfin, 0)).cpu().numpy()
        jobs = [("tw", a, c, k, t) for a in range(len(ci)) for c in range(nc) for k in range(A) for t in range(T)]
        jobs += [("cmp", a, c, k, t) for a in range(len(ci)) for c in range(nc) for k in range(3) for t in range(T)]
        jobs += [("raw", a, 0, 1, t) for a in range(len(ci)) for t in range(T)]
        for b in range(0, len(jobs), bs):
            ch = jobs[b:b + bs]
            ca = torch.as_tensor([j[1] for j in ch], device=device)
            dd = np.stack([tw_d[ci[a], c, k, t] if kind == "tw" else cmp_d[ci[a], c, k, t] if kind == "cmp"
                           else cmp_raw[ci[a], k, t] for kind, a, c, k, t in ch]).astype(np.float32)
            d = torch.as_tensor(dd, device=device)
            enc = [q for q, j in enumerate(ch) if j[0] != "raw"]
            if enc:
                r = suffix(model, saved[L][ca[enc]], L, d[enc])["meanpool"][:, -1]
                for q, e in enumerate(enc):
                    kind, a, c, k, t = ch[e]
                    (enc25_tw if kind == "tw" else enc25_cmp)[ci[a], c, k, t] = r[q]
            pj = [q for q, j in enumerate(ch) if j[0] != "cmp" or args.pred_compare]
            if pj:
                zz, _ = edited_prediction(model, csaved[L][ca[pj]], L, d[pj], 0)
                p = pool_steps(zz).cpu().numpy()
                for q, e in enumerate(pj):
                    kind, a, c, k, t = ch[e]
                    if kind == "tw":
                        pred_tw[ci[a], c, k, t] = p[q]
                    elif kind == "cmp":
                        pred_cmp[ci[a], c, k, t] = p[q]
                    else:
                        pred_chord_raw[ci[a], t] = p[q]
        print(f"L{L}: {ci[-1] + 1}/{C} carriers {time.time() - t0:.0f}s", flush=True)
    out = TD / f"L{L}"
    out.mkdir(parents=True, exist_ok=True)
    save = dict(idx=z["idx"], enc25_twin=enc25_tw, enc25_cmp=enc25_cmp, pred_twin=pred_tw,
                pred_chord_unscaled=pred_chord_raw, src_meanpool=src_mp, src_pred_pooled=src_pp,
                deltas_twin=tw_d.astype(np.float32), svd_spectrum=spec, knn_pairs=used,
                twin_norm_unscaled=np.stack([np.linalg.norm(dl[a], axis=-1) for a in TWIN_ARMS], 1))
    if pred_cmp is not None:
        save["pred_cmp"] = pred_cmp
    np.savez(out / "fwd.npz", **save)
    info = {"layer": L, "conds": list(conds), "twin_arms": list(TWIN_ARMS), "cmp_arms": list(CMP_ARMS),
            "pred_compare": bool(args.pred_compare), "k_nn": K_NN, "per_value": PER_VALUE,
            "n_fit_clips": int(len(z["fit_rows"])), "n_fit_pairs": int(len(z["pair_fit"])),
            "fit_encode_seconds": float(E["seconds"]), "edit_seconds": time.time() - t0,
            "total_seconds": time.time() - t_all, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    (out / "forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------- score (local)

def score(args):
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, load_sweep, predict, targets
    L = args.layer
    fdir = TD / f"L{L}"
    z = np.load(fdir / "fwd.npz")
    finfo = json.loads((fdir / "forward_info.json").read_text())
    conds = finfo["conds"]
    ship = np.load(SHIP / "ship.npz")
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    d0 = load_inputs("direction", 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    assert not np.isin(ship["fit_rows"], np.flatnonzero(probe | test)).any()
    Zall = np.load(S2 / "native" / "pred_pooled_all.npy").astype(np.float64)
    cen = np.load(S2 / "centroids_direction.npy")

    def fit(X, Yt, alpha):
        st = Standardizer().fit(X)
        W, b = fit_ridge(st.transform(X), Yt, alpha)
        return st, W, b

    def apply(pr, X):
        st, W, b = pr
        sh = X.shape
        return predict(st.transform(X.reshape(-1, sh[-1]).astype(np.float64)), W, b).reshape(sh[:-1] + (-1,))

    dir_nat = fit(Zall.mean(1)[probe], Y[probe], nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    pos_nat = {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        pos_nat[k] = fit(Zall[probe & ok, k], cen[probe & ok, t], nat_json["position"]["per_step"][str(t)]["native_alpha"])
    full = np.load(PROJECT_ROOT / "artifacts" / "activations" / "direction" / "vjepa2" / "meanpool.npy", mmap_mode="r")
    X25 = np.asarray(full[:, 25], np.float64)
    enc_probe = fit(X25[probe], Y[probe], load_sweep("direction", "direction")["layers"][25]["alpha"])

    F = load_groups(S2 / "forward", keys={"idx", "src_meanpool", "twin_meanpool", "src_pred_pooled"})
    assert (F["idx"] == z["idx"]).all()
    arrays, _ = load_plan(S2)
    rows, tg = arrays["carrier_rows"][z["idx"]], arrays["targets"][z["idx"]]
    C, T = tg.shape
    twin_true = np.load(S2 / "centroids_twins.npy")
    dist = lambda a, b: np.linalg.norm(a - b, axis=-1)                              # noqa: E731
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)   # noqa: E731
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, b, ax: (np.nansum(a * b, ax) / np.maximum(np.nansum(b * b, ax), 1e-9))   # noqa: E731
    src_pp = z["src_pred_pooled"].astype(np.float64)
    Ps = apply(dir_nat, src_pp.mean(1))
    src25 = z["src_meanpool"][:, 25].astype(np.float64)
    tmean = {v: X25[test & np.isclose(y, v)].mean(0) for v in np.unique(tg)}
    tm = np.stack([np.stack([tmean[v] for v in tg[:, j]]) for j in range(T)], 1)            # [C, T, D]
    tw25 = F["twin_meanpool"][:, :, 25].astype(np.float64)

    def pred_rows(Zp):                                                               # [C, T, 4, D]
        Zp = Zp.astype(np.float64)
        Pe = apply(dir_nat, Zp.mean(2))
        pe = np.stack([apply(pos_nat[k], Zp[..., k, :]) for k in range(4)], -2)
        return {"dir_err_to_target": wrap(angle_of(Pe) - tg).mean(1),
                "R_dir_real_change": proj(Pe - Ps[:, None], d_true, -1).mean(1),
                "px_err_to_twin_true": np.nanmean(dist(pe, twin_true), (1, 2))}

    def enc_rows(E25):                                                               # [C, T, D]
        E25 = E25.astype(np.float64)
        de, dt = E25 - src25[:, None], tw25 - src25[:, None]
        return {"enc_out_err_to_target": wrap(angle_of(apply(enc_probe, E25)) - tg).mean(1),
                "nearest_real_R_pt25": np.stack([mf.nearest_real_agreement(E25[:, j], src25, tm[:, j])
                                                 for j in range(T)], 1).mean(1),
                "R_act_twin_pt25": ((de * dt).sum(-1) / np.maximum((dt * dt).sum(-1), 1e-12)).mean(1)}

    # parity: unscaled session-2 chord through this box's predictor vs the session-2 cache (same readout)
    ref_chord = load_groups(S2 / "forward", keys={f"edit_pred_pooled_L{L}"})[f"edit_pred_pooled_L{L}"][:, 4]  # chord
    par = {"chord_pred_pooled_rel_maxabs_vs_session2_cache": float(np.abs(z["pred_chord_unscaled"] - ref_chord).max()
                                                                 / np.abs(ref_chord).max()),
           "chord_dir_err_diff_vs_session2_cache": float(pred_rows(z["pred_chord_unscaled"][:])["dir_err_to_target"].mean()
                                                      - pred_rows(ref_chord)["dir_err_to_target"].mean()),
           "src_meanpool_rel_maxabs_vs_session2": float(np.abs(z["src_meanpool"] - F["src_meanpool"]).max()
                                                        / np.abs(F["src_meanpool"]).max())}
    nm = json.loads((RES / "session2_predictor_norm_matched.json").read_text()) if L == 22 else None
    nm_pp = np.load(S2 / "norm_matched" / "pred_pooled.npz")["pred_pooled"] if nm is not None else None
    prop = json.loads((RES / "session2_propagation.json").read_text())
    res = {"point": L, "n_carriers": int(C), "n_targets": int(T), "forward": finfo, "parity": par,
           "fit_set": {"n_fit_clips": finfo["n_fit_clips"], "n_fit_pairs": finfo["n_fit_pairs"],
                       "role": "knot clips (train folds 0-2), 2 per source value, seed 0; disjoint from the 200 test "
                               "carriers and from the 480 probe clips that fit every readout",
                       "k_nn": K_NN, "svd_spectrum_top10": z["svd_spectrum"][:10].tolist(),
                       "svd_energy_top_r": {r: float((z["svd_spectrum"][:r] ** 2).sum() / (z["svd_spectrum"] ** 2).sum())
                                            for r in RANKS}},
           "unscaled_norm_over_natural_median": {a: float(np.median(z["twin_norm_unscaled"][:, i]
                                                                    / ship[f"natural_norm_L{L}"]))
                                                 for i, a in enumerate(TWIN_ARMS)},
           "per_condition": {}, "twin_minus_best_comparison": {}}
    unedited = {**{k: boot_ci(v) for k, v in {"dir_err_to_target": wrap(angle_of(Ps)[:, None] - tg).mean(1)}.items()},
                "enc_out_err_to_target": boot_ci(wrap(angle_of(apply(enc_probe, src25))[:, None] - tg).mean(1))}
    res["unedited"] = unedited
    for ci, cond in enumerate(conds):
        rows_c, pcs = {}, {}
        for k, a in enumerate(TWIN_ARMS):
            pcs[a] = {**pred_rows(z["pred_twin"][:, ci, k]), **enc_rows(z["enc25_twin"][:, ci, k])}
        for k, a in enumerate(CMP_ARMS):
            e = enc_rows(z["enc25_cmp"][:, ci, k])
            if "pred_cmp" in z.files:
                e.update(pred_rows(z["pred_cmp"][:, ci, k]))
            elif nm is not None:
                e.update(pred_rows(nm_pp[:, list(nm["conditions"]).index(cond), nm["arms"].index(a)]))
            pcs[a] = e
        for a, pc in pcs.items():
            rows_c[a] = {m: boot_ci(v) for m, v in pc.items()}
        res["per_condition"][cond] = rows_c
        # comparison-arm predictor readouts exist only with --pred-compare or the (point-22-only) norm-matched cache;
        # without them (point 12) select on the encoder-output error instead and say so
        sel = ("dir_err_to_target" if all("dir_err_to_target" in pcs[a] for a in CMP_ARMS)
               else "enc_out_err_to_target")
        best = min(CMP_ARMS, key=lambda a: pcs[a][sel].mean())
        bestr = min(TWIN_ARMS, key=lambda a: pcs[a][sel].mean())
        res["twin_minus_best_comparison"][cond] = {
            "best_comparison_arm": best, "best_twin_arm": bestr, "selection_metric": sel,
            **{m: boot_ci(pcs[bestr][m] - pcs[best][m]) for m in pcs[best] if m in pcs[bestr]}}
    if "pred_compare" not in z.files and nm is None:
        # no norm-matched predictor readouts for the comparison arms at this point: report instead (a) the session-2
        # cache's own-norm predictor readouts of the comparison arms and (b) twin (chord_norm) minus the unscaled chord
        # forwarded here, which IS norm-matched by construction (chord_norm rescales every twin edit to ||chord||)
        s2 = load_groups(S2 / "forward", keys={f"edit_pred_pooled_L{L}"})[f"edit_pred_pooled_L{L}"]
        s2_arms = json.loads((S2 / "plan.json").read_text())["arms"]
        own = {a: pred_rows(s2[:, s2_arms.index(a)]) for a in CMP_ARMS}
        raw = pred_rows(z["pred_chord_unscaled"])
        cn = ship[f"cmp_deltas_L{L}"].astype(np.float64)
        nrm = np.linalg.norm(cn, axis=-1)                                               # [C, 3, T]
        res["predictor_comparison_own_norm_session2_cache"] = {
            "note": "comparison arms at their own (unmatched) norm, predictor readouts from the session-2 forward cache "
                    "(edit_pred_pooled_L%d); the chord's own norm equals chord_norm, the spline's and probe-QR's do not" % L,
            "norm_over_chord_median": {a: float(np.median(nrm[:, k] / nrm[:, CMP_ARMS.index("chord")]))
                                       for k, a in enumerate(CMP_ARMS)},
            "arms": {a: {m: boot_ci(v) for m, v in own[a].items()} for a in CMP_ARMS},
            "raw_chord_forwarded_here": {m: boot_ci(v) for m, v in raw.items()}}
        if "chord_norm" in conds:
            ci = conds.index("chord_norm")
            res["twin_minus_raw_chord_chord_norm"] = {
                a: {m: boot_ci(v - raw[m]) for m, v in pred_rows(z["pred_twin"][:, ci, k]).items()}
                for k, a in enumerate(TWIN_ARMS)}
    res["session2_propagation_unmatched_pt25"] = {
        a: {"enc_out_err_to_target": prop["readout_a"][str(L)][a]["err_to_target"][-1]["mean"],
            "nearest_real_R_pt25": prop["readout_b"][str(L)][a]["nearest_real_agreement"][-1]["mean"]}
        for a in CMP_ARMS}
    res["definitions"] = {
        "dir_err_to_target": "predictor forecast, predictor-native direction probe (480 probe clips, stored alpha), "
                             "circular error (deg), mean over 4 targets; CI = bootstrap over carriers",
        "px_err_to_twin_true": "predictor-native per-step position probe vs the rendered twin's true centroids (px)",
        "R_dir_real_change": "as session2_predictor_norm_matched.json",
        "enc_out_err_to_target": "full edited clip propagated to the final LN (point 25), direction probe refit on the "
                                 "probe clips' stored point-25 meanpool (sweep alpha)",
        "nearest_real_R_pt25": "1 - ||x_edit - mu_target|| / ||x_src - mu_target|| at point 25, mu = mean of real test "
                               "clips at the target (session2_propagation readout_b)",
        "R_act_twin_pt25": "<x_edit - x_src, x_twin - x_src> / ||x_twin - x_src||^2 at point 25",
        "comparison arms": "session-2 deltas (plan.npz) at the same norm condition; encoder readouts forwarded here; "
                           "predictor readouts " + ("forwarded here" if "pred_compare" in z.files else
                                                    "from artifacts/session2/norm_matched/pred_pooled.npz"
                                                    if nm is not None else
                                                    "not available at matched norm at this point (forward run "
                                                    "without --pred-compare; norm-matched cache is point 22 only): "
                                                    "per_condition comparison arms carry encoder readouts only; see "
                                                    "predictor_comparison_own_norm_session2_cache and "
                                                    "twin_minus_raw_chord_chord_norm")}
    write(RES / f"session2_twin_difference_L{L}.json", res, stage="twin_difference_score", seeds={"fit": 0, "bootstrap": 0},
          cache=str(fdir / "fwd.npz"))
    for cond in conds:
        print(cond)
        for a, r in res["per_condition"][cond].items():
            print(f"  {a:10s} " + "  ".join(f"{m} {v['mean']:.3f}" for m, v in r.items()))
    figure()


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(RES.glob("session2_twin_difference_L*.json"))
    fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 3.8), squeeze=False)
    cols = {"spline": "#1f77b4", "chord": "#ff7f0e", "probe_qr": "#2ca02c"}
    for ax, f in zip(axes[0], files):
        r = json.loads(f.read_text())
        xs = list(RANKS) + [16]
        for cond, ls in zip(r["per_condition"], ("-", "--")):
            pc = r["per_condition"][cond]
            m = [pc[a]["dir_err_to_target"]["mean"] for a in TWIN_ARMS]
            lo = [pc[a]["dir_err_to_target"]["ci95"][0] for a in TWIN_ARMS]
            hi = [pc[a]["dir_err_to_target"]["ci95"][1] for a in TWIN_ARMS]
            ax.errorbar(xs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], color="k", ls=ls, marker="o", ms=4,
                        capsize=2, label=f"twin difference ({cond.replace('_', ' ')})")
            for a, c in cols.items():
                if "dir_err_to_target" in pc[a]:
                    ax.axhline(pc[a]["dir_err_to_target"]["mean"], color=c, ls=ls, lw=1,
                               label=f"{a.replace('_', '-')} ({cond.split('_')[0]})")
        ax.axhline(r["unedited"]["dir_err_to_target"]["mean"], color="grey", ls=":", lw=1, label="unedited")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in RANKS] + ["full"])
        ax.set_xlabel("rank r of the twin-difference edit")
        ax.set_ylabel("forecast direction error to target (deg)")
        ax.set_title(f"edit at point {r['point']}, 200 carriers x 4 targets")
        ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_twin_difference.png", dpi=150)
    print("wrote", FIG / "fig_twin_difference.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "forward", "score", "figure"])
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--pred-compare", action="store_true")
    a = ap.parse_args()
    {"prep": prep, "forward": forward, "score": score, "figure": lambda _: figure()}[a.stage](a)

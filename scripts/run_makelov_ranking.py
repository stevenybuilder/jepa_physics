"""Makelov-style ranking of INLP directions by their causal effect on the predictor's forecast (Makelov et al.,
"Towards principled evaluations of sparse autoencoders for interpretability and control": rank features by how much a
fixed-size edit along each changes the model's downstream output, then compare with the features' own order).

Directions: the first N columns of the point-22 INLP basis (artifacts/inlp/direction_direction_L22.npz, Q [1024, 176],
2 orthonormal columns per round, in step 1's train-standardised coordinates) mapped to raw activations as
normalise(sd * q) (wm.makelov.raw_directions). Controls: 16 random directions (orthonormal in the same standardised
coordinates, same mapping) and each carrier's own raw-chord edit direction.
Carriers and dose: the 16 carriers of the repair-attribution plan (artifacts/p5_repair/plan.npz: headline arc, target
320.625 deg, run_part2.pick_clips n_control_clips subset); every edit has the norm of that carrier's point-22
raw-chord edit (delta_rawchord22). Both signs +/- are run, since an INLP direction has no preferred sign.
Forward (GPU): context frames 1-8 encoded to point 22 once per carrier, the edit added to every token
(wm.propagate.suffix), blocks 23-24 + final LN, predictor with mask token 0 (wm.predictor_readout.predict_future),
forecast pooled per future step [4, D].
Readout (CPU): session2_native_readout's step-mean ridge probe on the predictor's own unedited forecasts of the 480
probe clips (the repair-attribution predictor reader, alpha by 5-fold CV, rng 0). Move of a direction on a carrier =
mean over the two signs of |wrap(angle(edited forecast) - angle(clean forecast))|; also the relative L2 change of the
pooled forecast (reader-free).

  python scripts/run_makelov_ranking.py plan [--n-inlp 32]
  python scripts/run_makelov_ranking.py forward --root <out>     (box, GPU; wrap in flock /tmp/wm_gpu.lock)
  python scripts/run_makelov_ranking.py score --root <out>
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

ART = PROJECT_ROOT / "artifacts" / "p5_makelov"
RES = PROJECT_ROOT / "results"
L = 22
N_RAND = 16
N_MAIN = 32                                  # directions in the headline ranking (brief: first 32)


def plan(n_inlp):
    from wm import makelov as mk
    from wm.probes import Standardizer, layer_matrix, load_activations, split_rows
    from wm.provenance import git_commit, sha256_file

    rp = dict(np.load(PROJECT_ROOT / "artifacts/p5_repair/plan.npz", allow_pickle=True))
    z = np.load(PROJECT_ROOT / f"artifacts/inlp/direction_direction_L{L}.npz")
    assert int(z["point"]) == L
    tr, te, _ = split_rows("direction")
    acts = load_activations("direction", "meanpool", "vjepa2")
    st = Standardizer().fit(layer_matrix(acts, L)[tr])
    dirs = mk.raw_directions(z["Q"], st.std, n_inlp)
    Qr = np.linalg.qr(np.random.default_rng(22).standard_normal((z["Q"].shape[0], N_RAND)))[0]
    rand = mk.raw_directions(Qr, st.std)
    dch = rp[f"delta_rawchord{L}"].astype(np.float64)
    norms = np.linalg.norm(dch, axis=1)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", carrier_ids=rp["carrier_ids"], norms=norms.astype(np.float32),
             dirs_inlp=dirs.astype(np.float32), dirs_rand=rand.astype(np.float32),
             dirs_chord=(dch / norms[:, None]).astype(np.float32), clean_mp=rp[f"clean_mp_L{L}"])
    meta = {"point": L, "n_inlp": n_inlp, "n_rand": N_RAND, "inlp_Q_cols": int(z["Q"].shape[1]),
            "inlp_alpha": float(z["alpha"]), "carrier_ids": rp["carrier_ids"].tolist(),
            "rawchord_norm_mean": float(norms.mean()), "git": git_commit(),
            "sd_rows": "step-1 train split (splits: split_rows('direction')), meanpool point 22",
            "cos_inlp_dir_vs_rawchord_mean_abs": float(np.abs(dirs @ (dch / norms[:, None]).T).mean())}
    meta["plan_sha256"] = sha256_file(ART / "plan.npz")
    (ART / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


def conditions(P):
    c = [("inlp", k, s) for k in range(len(P["dirs_inlp"])) for s in (1, -1)]
    c += [("rand", k, s) for k in range(len(P["dirs_rand"])) for s in (1, -1)]
    c += [("chord", 0, s) for s in (1, -1)]
    return c


def forward(root, batch=4, smoke=False):
    import torch
    from wm.data import decode, load_table
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix, suffix

    set_precision()
    dev = pick_device()
    model = load_model("vjepa2", dev)
    P = dict(np.load(ART / "plan.npz", allow_pickle=True))
    tab = load_table("direction")
    vid = dict(zip(tab["id"], tab["video"]))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    conds = conditions(P)
    if smoke:
        conds = conds[:4] + [c for c in conds if c[0] == "chord"]
    cids = P["carrier_ids"][:batch] if smoke else P["carrier_ids"]
    t0 = time.time()
    clean, edit = [], []
    for s in range(0, len(cids), batch):
        rows = slice(s, s + batch)
        pv = preprocess([decode(vid[i]) for i in cids[rows]]).to(dev)[:, :8]
        saved, _, pooled = prefix(model, pv, [L])
        r = suffix(model, saved[L], L, None, return_final=True)
        clean.append(pool_steps(predict_future(model, r["final"])).float().cpu().numpy())
        norms = torch.as_tensor(P["norms"][rows], device=dev)
        per = []
        for kind, k, sg in conds:
            if kind == "chord":
                u = torch.as_tensor(P["dirs_chord"][rows], device=dev)
            else:
                u = torch.as_tensor(P[f"dirs_{kind}"][k], device=dev)[None].expand(len(norms), -1)
            r = suffix(model, saved[L], L, sg * norms[:, None] * u, return_final=True)
            per.append(pool_steps(predict_future(model, r["final"])).float().cpu().numpy())
        edit.append(np.stack(per, 0))
        del saved
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"carriers {s + batch}/{len(cids)} {time.time() - t0:.0f}s", flush=True)
    np.savez(root / "forward.npz", clean_pred=np.concatenate(clean), edit_pred=np.concatenate(edit, 1),
             conds=np.array([f"{a}:{b}:{c}" for a, b, c in conds]))
    info = {"seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "batch": batch, "smoke": smoke, "n_conditions": len(conds),
            "n_carriers": int(len(cids)), "forward_dtype": "float32 (set_precision), as session 2"}
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", info["seconds"], flush=True)


def score(root):
    from run_session2 import angle_of
    from wm import makelov as mk
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score as pscore, targets
    from wm.provenance import git_commit, sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz", allow_pickle=True))
    P = dict(np.load(ART / "plan.npz", allow_pickle=True))
    meta = json.loads((ART / "plan.json").read_text())
    finfo = json.loads((root / "forward_info.json").read_text())
    df = load_table("direction")
    Y = targets(df, "direction")[0]
    d0 = load_inputs("direction", 0)
    probe = d0["role"] == "probe"
    Yp = Y[probe]
    circ = lambda a, b: pscore(a, b, "circular")                                            # noqa: E731
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Xn = Zall.mean(1)
    stn = Standardizer().fit(Xn[probe])
    cvn = cv_select_alpha(stn.transform(Xn[probe]), Yp, folds5, score_fn=circ)
    Wn, bn = fit_ridge(stn.transform(Xn[probe]), Yp, cvn["alpha"])
    pread = lambda Zp: angle_of(predict(stn.transform(Zp.mean(-2).reshape(-1, Zp.shape[-1]).astype(np.float64)),
                                        Wn, bn)).reshape(Zp.shape[:-2])                    # noqa: E731
    clean = F["clean_pred"].astype(np.float64)
    cid = P["carrier_ids"][:len(clean)]
    rows = np.searchsorted(df["id"].to_numpy(), cid)
    assert (df["id"].to_numpy()[rows] == cid).all()
    parity = float(np.abs(clean - Zall[rows]).max() / np.abs(Zall[rows]).max())
    conds = [c.split(":") for c in F["conds"]]
    E = F["edit_pred"].astype(np.float64)                                                  # [n_cond, C, 4, D]
    a0 = pread(clean)                                                                       # [C]
    ang = pread(E)                                                                          # [n_cond, C]
    dang = np.abs(mk.wrap_deg(ang - a0[None]))
    rel = np.linalg.norm(E - clean[None], axis=(2, 3)) / np.linalg.norm(clean, axis=(1, 2))[None]
    signed = mk.wrap_deg(ang - a0[None])

    def block(kind):
        ks = sorted({int(k) for kd, k, _ in conds if kd == kind})
        M, R, S = [], [], []
        for k in ks:
            ii = [i for i, (kd, kk, _) in enumerate(conds) if kd == kind and int(kk) == k]
            M.append(dang[ii].mean(0)); R.append(rel[ii].mean(0)); S.append(signed[ii])
        if not ks:
            return np.full((len(clean), 1), np.nan), np.full((len(clean), 1), np.nan), [np.full((2, len(clean)), np.nan)]
        return np.array(M).T, np.array(R).T, S                                              # [C, n]

    Mi, Ri, Si = block("inlp")
    Mi_all, Ri_all = Mi, Ri
    ext = None
    if Mi.shape[1] > N_MAIN:                    # headline = first N_MAIN directions (the brief); the rest reported apart
        ext = {"n_directions": int(Mi.shape[1]),
               "angle_move": {**mk.carrier_bootstrap_spearman(Mi, 2000, 0),
                              "perm_p_two_sided": mk.permutation_p(Mi.mean(0), 10000, 0)},
               "rel_l2_move": {**mk.carrier_bootstrap_spearman(Ri, 2000, 0),
                               "perm_p_two_sided": mk.permutation_p(Ri.mean(0), 10000, 0)},
               "by_round_angle_move": mk.carrier_bootstrap_spearman(
                   Mi.reshape(Mi.shape[0], -1, 2).mean(2), 2000, 0),
               "mean_move_deg_by_block_of_8": [float(Mi[:, j:j + 8].mean()) for j in range(0, Mi.shape[1], 8)]}
        Mi, Ri, Si = Mi[:, :N_MAIN], Ri[:, :N_MAIN], Si[:N_MAIN]
    Mr, Rr, _ = block("rand")
    Mc, Rc, Sc = block("chord")
    n = Mi.shape[1]
    mv, rv = Mi.mean(0), Ri.mean(0)
    order = np.argsort(-mv)
    rank_of = np.empty(n, int)
    rank_of[order] = np.arange(n)
    ang_boot = mk.carrier_bootstrap_spearman(Mi, n_boot=2000, seed=0)
    rel_boot = mk.carrier_bootstrap_spearman(Ri, n_boot=2000, seed=0)
    rounds = Mi.reshape(Mi.shape[0], n // 2, 2).mean(2)
    boot = lambda v: [float(x) for x in np.percentile(                                      # noqa: E731
        np.random.default_rng(0).choice(v, (2000, len(v))).mean(1), [2.5, 97.5])]

    def rowinfo(k):
        return {"inlp_index": int(k), "inlp_round": int(k // 2) + 1, "forecast_move_deg": float(mv[k]),
                "forecast_move_rank": int(rank_of[k]) + 1, "forecast_rel_l2_change": float(rv[k]),
                "signed_move_plus_deg": float(Si[k][0].mean()), "signed_move_minus_deg": float(Si[k][1].mean())}

    out = {
        "question": "Makelov-style: does the order in which INLP found the point-22 direction directions match how "
                    "much a fixed-norm edit along each moves the predictor's forecast readout?",
        "point": L, "n_carriers": int(len(cid)), "n_inlp_directions": int(n), "target_of_carriers_deg": 320.625,
        "dose": {"rule": "per carrier, ||edit|| = ||point-22 raw-chord edit|| (artifacts/p5_repair/plan.npz)",
                 "rawchord_norm_mean": meta["rawchord_norm_mean"]},
        "readout": {"reader": "session2 native step-mean ridge probe on the predictor's unedited forecasts of the 480 "
                              "probe clips (as run_repair_attribution score)", "alpha": cvn["alpha"],
                    "cv_mae_deg": cvn.get("cv_mae_mean"),
                    "move": "per carrier, mean over signs of |wrap(angle(edited forecast) - angle(clean forecast))|, deg",
                    "clean_forecast_angle_err_to_carrier_label_deg": float(np.mean(np.abs(mk.wrap_deg(
                        a0 - df["theta_degrees"].to_numpy(float)[rows]))))},
        "spearman_inlp_order_vs_move": {
            "angle_move": {**ang_boot, "perm_p_two_sided": mk.permutation_p(mv, 10000, 0),
                           "sign": "negative = earlier INLP directions move the forecast more"},
            "rel_l2_move": {**rel_boot, "perm_p_two_sided": mk.permutation_p(rv, 10000, 0)},
            "by_round_angle_move": {**mk.carrier_bootstrap_spearman(rounds, 2000, 0),
                                    "n_rounds": int(rounds.shape[1]),
                                    "note": "the two columns of a round are QR-ordered, so round order is the "
                                            "cleaner INLP order"}},
        "extension_all_planned_directions": ext,
        "top5_by_move": [rowinfo(k) for k in order[:5]],
        "bottom5_by_move": [rowinfo(k) for k in order[::-1][:5]],
        "per_direction": [rowinfo(k) for k in range(n)],
        "controls": {
            "inlp_mean_move_deg": {"mean": float(mv.mean()), "ci95_carriers": boot(Mi.mean(1))},
            "random_mean_move_deg": {"mean": float(Mr.mean()), "ci95_carriers": boot(Mr.mean(1)),
                                     "n_dirs": int(Mr.shape[1]),
                                     "rule": "orthonormal Gaussian in standardised coordinates, mapped as INLP"},
            "rawchord_move_deg": {"mean": float(Mc.mean()), "ci95_carriers": boot(Mc[:, 0]),
                                  "signed_plus_mean_deg": float(Sc[0][0].mean())},
            "frac_inlp_dirs_above_random_p95": float(np.mean(mv > np.percentile(Mr.mean(0), 95))),
            "inlp_mean_rel_l2": float(rv.mean()), "random_mean_rel_l2": float(Rr.mean()),
            "rawchord_rel_l2": float(Rc.mean())},
        "parity": {"clean_forecast_vs_native_cache_rel_maxabs": parity},
        "provenance": {"git": git_commit(), "plan": meta, "forward": finfo,
                       "forward_sha256": sha256_file(root / "forward.npz"),
                       "inlp_basis": f"artifacts/inlp/direction_direction_L{L}.npz"}}
    RES.mkdir(exist_ok=True)
    (RES / f"p5_makelov_ranking_L{L}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("spearman_inlp_order_vs_move", "controls", "parity")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("plan", "forward", "score"))
    p.add_argument("--root", default=str(ART / "forward"))
    p.add_argument("--n-inlp", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.cmd == "plan":
        plan(a.n_inlp)
    elif a.cmd == "forward":
        forward(a.root, a.batch, a.smoke)
    else:
        score(a.root)

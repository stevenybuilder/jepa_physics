"""Pre-registered, rank-matched coordinate competition with a true holdout (CPU only, stored mean-pooled activations).

Answers the obvious objection to run_motion_geometry.py exp4 (results/p5_motion_geometry.json
exp4_coordinate_search_and_exp5_shortcuts): there the 5-feature (cos, sin, cos2, sin2, v) frame beat 2-3 feature frames
on transfer R2 with no matched-rank control. Here every candidate is compared against baselines of the SAME rank.

PREREGISTRATION (written into the script and into results/p5_coordinate_competition.json["preregistration"] before
any number of this run was computed; the JSON also records the run timestamp):

Candidates (theta = direction, v = speed in m/s; rank = number of columns):
  cartesian        (v cos t, v sin t)                                     rank 2
  polar2           (cos t, sin t)                                         rank 2
  polar4           (cos t, sin t, cos 2t, sin 2t)                         rank 4
  polar6           (cos t, sin t, cos 2t, sin 2t, cos 3t, sin 3t)         rank 6
  cartesian_v      (v cos t, v sin t, v)                                  rank 3
  polar2_v         (cos t, sin t, v)                                      rank 3
  polar4_v         (cos t, sin t, cos 2t, sin 2t, v)                      rank 5
Baselines per rank k in {2, 3, 4, 5, 6}:
  pca_k            "no physics" ceiling: share of held-out variance inside the knot rows' top-k PCs (the best any
                   rank-k linear map could do; uses the held-out features themselves, so it is an upper bound)
  randfeat_k       matched-rank label null: k random smooth features of the same labels,
                   cos(W [cos t, sin t, z(v)] + b), W ~ N(0, 1), b ~ U(0, 2 pi); mean over 20 draws

Scoring rule (a), encoding, at EVERY stored read point 0..25, trained (vjepa2) and untrained (random) copy:
  features  Z = PCA-64 of the mean-pooled activation, PCA fit on the knot rows only
  split     splits/split_v1.json: knot = folds 0-2, probe = folds 3-4, test = fold -1 (NEVER used in (a))
  fit       OLS Z ~ a + Phi_c B on knot rows
  score     held-out R2 on probe rows: 1 - sum||Z - Zhat||^2 / sum||Z - mean_probe(Z)||^2 (clip level)
  primary set    speed set (constant velocity; 64 directions x 64 speeds; theta and v both vary)
  secondary      direction set, constant-velocity clips only (knot -> probe)
  transfer       fit on speed-set knot rows, score direction-set probe-fold velocity clips with 1 <= v <= 4 (each set
                 centred on its own mean, as exp4)
  PRIMARY WINNER per point = argmax over the 7 candidates of R2_c / R2_pca_k (share of the matched-rank ceiling).
  Also reported: raw-R2 winner (known to favour rank), margin R2_c - R2_pca_k, margin R2_c - R2_randfeat_k,
  and the two rank-matched head-to-heads that decide Cartesian vs polar without any rank confound:
  rank 2 polar2 - cartesian, rank 3 polar2_v - cartesian_v. CIs: 500-draw bootstrap over probe clips.

Scoring rule (b), steering, points 12 and 22, trained copy, the 16 held-out contiguous arcs of
scripts/run_bakeoff_unified_16arc.py (p2.build, seeds 0-15, PCA-64 on knot rows at kept values, same 48 picked
test-fold carriers per target, same readers): per candidate, B from OLS Z_knot ~ Phi_c(theta, v) (held arc excluded),
straight edit z + (Phi_c(tgt, v_clip) - Phi_c(src, v_clip)) B, residual kept, at its own norm and rescaled to the raw
chord's full-space norm; reader = linear probe fit on the probe fold (disjoint from the knot rows that build the edit);
nearest-real R on test clips as in the bake-off. Report endpoint error (deg) with a 1000-draw clip bootstrap CI and
the arc-level spread mean +/- 2 SD / sqrt(16) over per-arc means. chord_raw reported as reference.

  python scripts/run_coordinate_competition.py encode   --workers 34 --out <dir>
  python scripts/run_coordinate_competition.py steer    --workers 32 --out <dir>
  python scripts/run_coordinate_competition.py assemble --out <dir>
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
RESULT = ROOT / "results" / "p5_coordinate_competition.json"
POINTS = list(range(26))
STEER_POINTS = (12, 22)
N_BOOT, N_RAND, SEED = 500, 20, 0
PCA_K = 64

CANDIDATES = {
    "cartesian": 2, "polar2": 2, "polar4": 4, "polar6": 6, "cartesian_v": 3, "polar2_v": 3, "polar4_v": 5,
}


def fourier(th, K):
    t = np.radians(np.asarray(th, float))
    return np.column_stack([f(k * t) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def phi(name, th, v):
    v = np.asarray(v, float)
    if name == "cartesian":
        return v[:, None] * fourier(th, 1)
    if name == "cartesian_v":
        return np.column_stack([v[:, None] * fourier(th, 1), v])
    K = {"polar2": 1, "polar4": 2, "polar6": 3, "polar2_v": 1, "polar4_v": 2}[name]
    F = fourier(th, K)
    return np.column_stack([F, v]) if name.endswith("_v") else F


def rand_feats(th, v, k, rng, vmu, vsd):
    U = np.column_stack([fourier(th, 1), (np.asarray(v, float) - vmu) / vsd])
    W = rng.standard_normal((3, k))
    b = rng.uniform(0, 2 * np.pi, k)
    return W, b, U


def preregistration():
    return {
        "written": "2026-09-28 18:31 ET, before any number of this run was computed (script docstring = same text)",
        "candidates": {n: {"rank": k, "columns": {
            "cartesian": "v cos t, v sin t", "polar2": "cos t, sin t", "polar4": "cos t, sin t, cos 2t, sin 2t",
            "polar6": "cos t..sin 3t", "cartesian_v": "v cos t, v sin t, v", "polar2_v": "cos t, sin t, v",
            "polar4_v": "cos t, sin t, cos 2t, sin 2t, v"}[n]} for n, k in CANDIDATES.items()},
        "baselines": {"pca_k": "share of probe-fold variance (around the probe mean) inside the knot rows' top-k PCs, "
                               "k = 2..6; an upper bound for any rank-k linear map (uses held-out features)",
                      "randfeat_k": f"k random smooth features cos(W[cos t, sin t, z(v)] + b) of the same labels, "
                                    f"mean held-out R2 over {N_RAND} draws"},
        "features": f"PCA-{PCA_K} of meanpool at the read point, PCA fit on knot rows only",
        "split_roles": "splits/split_v1.json: knot = folds 0,1,2 (fit); probe = folds 3,4 (score); test = fold -1 "
                       "(never used in (a); in (b) only as the bake-off's steered carriers and nearest-real reference)",
        "sets": {"primary": "speed set, knot -> probe", "secondary": "direction set velocity clips, knot -> probe",
                 "transfer": "speed-set knot fit -> direction-set probe-fold velocity clips 1<=v<=4, own-mean centred"},
        "score_a": "held-out clip-level R2 = 1 - SSE / SST(probe mean) in PCA-64 space",
        "primary_winner": "per point, argmax over the 7 candidates of R2_c / R2_pca_rank(c) on the primary set",
        "also_reported": ["raw-R2 winner", "R2_c - R2_pca_k", "R2_c - R2_randfeat_k",
                          "rank-2 head-to-head polar2 - cartesian", "rank-3 head-to-head polar2_v - cartesian_v",
                          f"{N_BOOT}-draw probe-clip bootstrap CIs"],
        "points": POINTS, "models": ["vjepa2", "random"],
        "score_b": ("points 12 and 22, trained copy, 16 held-out contiguous arcs (seeds 0-15) of "
                    "run_bakeoff_unified_16arc.py; straight edit (Phi(tgt,v)-Phi(src,v)) B, B fit on knot rows at kept "
                    "values in the arc's PCA-64; own norm and at the raw chord's norm; probe-fold linear reader; "
                    "endpoint error deg with 1000-draw clip bootstrap CI and arc-level mean +/- 2 SD/sqrt(16)"),
        "seeds": {"bootstrap": SEED, "randfeat": SEED, "arcs": list(range(16))},
    }


# ------------------------------------------------------------------ (a) encoding
def _meta(dataset):
    from wm.data import load_table
    from wm.splits import load_split
    df = load_table(dataset)
    s = load_split(dataset)
    fold = {int(k): v for k, v in s["fold"].items()}
    return df, np.array([fold[int(i)] for i in df["id"]])


def _pca(Xk, k=PCA_K):
    mu = Xk.mean(0)
    _, _, Vt = np.linalg.svd(Xk - mu, full_matrices=False)
    return mu, Vt[:k].T


def ols(Phi_k, Z_k):
    Fm, Zm = Phi_k.mean(0), Z_k.mean(0)
    B, *_ = np.linalg.lstsq(Phi_k - Fm, Z_k - Zm, rcond=None)
    return Zm - Fm @ B, B


def r2_rows(Z, Zhat, mean=None):
    """Per-row SSE and SST terms (SST around the evaluation set's own mean unless given)."""
    mu = Z.mean(0) if mean is None else mean
    return ((Z - Zhat) ** 2).sum(1), ((Z - mu) ** 2).sum(1)


def score_block(Zk, thk, vk, Zp, thp, vp, boot_idx, rng, centre_own=False):
    """All candidates + baselines for one (fit, eval) pair. Returns dict of R2 and bootstrap arrays."""
    out, boots = {}, {}
    sst = ((Zp - Zp.mean(0)) ** 2).sum(1)
    SST_b = sst[boot_idx].sum(1)

    def fin(name, sse):
        out[name] = float(1 - sse.sum() / sst.sum())
        boots[name] = 1 - sse[boot_idx].sum(1) / SST_b

    for n in CANDIDATES:
        a, B = ols(phi(n, thk, vk), Zk)
        Ph = phi(n, thp, vp)
        Zh = a + Ph @ B
        if centre_own:
            Zh = Zh - Zh.mean(0) + Zp.mean(0)
        fin(n, ((Zp - Zh) ** 2).sum(1))
    # pca_k ceiling: probe variance in knot top-k PCs (Z columns are the knot PCs, ordered)
    Zc = Zp - Zp.mean(0)
    for k in sorted(set(CANDIDATES.values())):
        fin(f"pca_{k}", (Zc[:, k:] ** 2).sum(1))
    vmu, vsd = vk.mean(), vk.std() + 1e-9
    for k in sorted(set(CANDIDATES.values())):
        vals, bb = [], []
        for _ in range(N_RAND):
            W, b, Uk = rand_feats(thk, vk, k, rng, vmu, vsd)
            Up = np.column_stack([fourier(thp, 1), (vp - vmu) / vsd])
            a, B = ols(np.cos(Uk @ W + b), Zk)
            Zh = a + np.cos(Up @ W + b) @ B
            if centre_own:
                Zh = Zh - Zh.mean(0) + Zp.mean(0)
            sse = ((Zp - Zh) ** 2).sum(1)
            vals.append(1 - sse.sum() / sst.sum())
            bb.append(1 - sse[boot_idx].sum(1) / SST_b)
        out[f"randfeat_{k}"] = float(np.mean(vals))
        out[f"randfeat_{k}_max_of_{N_RAND}"] = float(np.max(vals))
        boots[f"randfeat_{k}"] = np.mean(bb, 0)
    return out, boots


def summarise(out, boots):
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
    res = {"r2": out, "per_candidate": {}}
    for n, k in CANDIDATES.items():
        res["per_candidate"][n] = {
            "rank": k, "r2": out[n], "r2_ci95": ci(boots[n]),
            "pca_k_r2": out[f"pca_{k}"], "share_of_pca_k": out[n] / out[f"pca_{k}"],
            "minus_pca_k": out[n] - out[f"pca_{k}"], "minus_pca_k_ci95": ci(boots[n] - boots[f"pca_{k}"]),
            "randfeat_k_r2": out[f"randfeat_{k}"], "minus_randfeat_k": out[n] - out[f"randfeat_{k}"],
            "minus_randfeat_k_ci95": ci(boots[n] - boots[f"randfeat_{k}"]),
            "minus_cartesian": out[n] - out["cartesian"], "minus_cartesian_ci95": ci(boots[n] - boots["cartesian"])}
    share = {n: out[n] / out[f"pca_{k}"] for n, k in CANDIDATES.items()}
    win = max(share, key=share.get)
    raw = max(CANDIDATES, key=lambda n: out[n])
    ws = boots[win] / boots[f"pca_{CANDIDATES[win]}"]
    runner = max((n for n in CANDIDATES if n != win), key=share.get)
    rs = boots[runner] / boots[f"pca_{CANDIDATES[runner]}"]
    res["winner_primary"] = win
    res["winner_primary_share_margin_over_runner_up"] = {"runner_up": runner, "diff": share[win] - share[runner],
                                                         "ci95": ci(ws - rs)}
    res["winner_raw_r2"] = raw
    res["h2h_rank2_polar2_minus_cartesian"] = {"diff": out["polar2"] - out["cartesian"],
                                               "ci95": ci(boots["polar2"] - boots["cartesian"])}
    res["h2h_rank3_polar2v_minus_cartesianv"] = {"diff": out["polar2_v"] - out["cartesian_v"],
                                                 "ci95": ci(boots["polar2_v"] - boots["cartesian_v"])}
    res["winner_rank2"] = "polar2" if out["polar2"] > out["cartesian"] else "cartesian"
    res["winner_rank3"] = "polar2_v" if out["polar2_v"] > out["cartesian_v"] else "cartesian_v"
    return res


def encode_job(args):
    p, model = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    res = {}
    dfs, fs = _meta("speed")
    dfd, fd = _meta("direction")
    Xs = np.asarray(np.load(ROOT / "artifacts/activations/speed" / model / "meanpool.npy", mmap_mode="r")[:, p], float)
    Xd = np.asarray(np.load(ROOT / "artifacts/activations/direction" / model / "meanpool.npy", mmap_mode="r")[:, p], float)
    ths, vs = dfs.theta_degrees.to_numpy(float), dfs.speed_mps.to_numpy(float)
    thd, vd = dfd.theta_degrees.to_numpy(float), dfd.speed_mps.to_numpy(float)
    velo = (dfd.motion == "velocity").to_numpy()
    ks, ps_ = np.isin(fs, (0, 1, 2)), np.isin(fs, (3, 4))
    kd, pd_ = np.isin(fd, (0, 1, 2)) & velo, np.isin(fd, (3, 4)) & velo
    rng = np.random.default_rng(SEED)
    # primary: speed set
    mu, V = _pca(Xs[ks])
    Zk, Zp = (Xs[ks] - mu) @ V, (Xs[ps_] - mu) @ V
    bi = rng.integers(0, ps_.sum(), (N_BOOT, ps_.sum()))
    o, b = score_block(Zk, ths[ks], vs[ks], Zp, ths[ps_], vs[ps_], bi, rng)
    res["primary_speedset"] = summarise(o, b)
    res["primary_speedset"]["n"] = {"knot": int(ks.sum()), "probe": int(ps_.sum())}
    # transfer: speed-set knot fit (same PCA) -> direction-set probe velocity clips 1..4 m/s, own-mean centred
    tm = pd_ & (vd >= 1) & (vd <= 4)
    Zt = (Xd[tm] - mu) @ V          # predictions are re-centred on this set's own mean (centre_own)
    bt = rng.integers(0, tm.sum(), (N_BOOT, tm.sum()))
    o, b = score_block(Zk, ths[ks], vs[ks], Zt, thd[tm], vd[tm], bt, rng, centre_own=True)
    res["transfer_speed_to_dirset"] = summarise(o, b)
    res["transfer_speed_to_dirset"]["n"] = {"knot": int(ks.sum()), "probe": int(tm.sum())}
    # secondary: direction set velocity clips
    mu2, V2 = _pca(Xd[kd])
    Zk2, Zp2 = (Xd[kd] - mu2) @ V2, (Xd[pd_] - mu2) @ V2
    b2 = rng.integers(0, pd_.sum(), (N_BOOT, pd_.sum()))
    o, b = score_block(Zk2, thd[kd], vd[kd], Zp2, thd[pd_], vd[pd_], b2, rng)
    res["secondary_dirset_velocity"] = summarise(o, b)
    res["secondary_dirset_velocity"]["n"] = {"knot": int(kd.sum()), "probe": int(pd_.sum())}
    res["seconds"] = round(time.time() - t0, 1)
    return p, model, res


# ------------------------------------------------------------------ (b) steering
def _bakeoff():
    s = importlib.util.spec_from_file_location("bu16", HERE.with_name("run_bakeoff_unified_16arc.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_D = {}


def steer_job(args):
    L, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    bu = _bakeoff()
    edp, p2 = bu.edp, bu.p2
    if L not in _D:
        d = load_inputs("direction", L, "direction")
        pr = d["role"] == "probe"
        test = np.flatnonzero(d["role"] == "test")
        probe = mf.ProbeReadout(d["X"][pr], d["y"][pr], True)
        near = mf.NearestRealReadout(d["X"][test], d["y"][test])
        pf_vals = np.unique(d["y"][pr]).astype(float)
        pf_C = np.array([d["X"][pr & np.isclose(d["y"], v)].astype(float).mean(0) for v in pf_vals])
        _D[L] = (d, probe, near, pf_vals, pf_C)
    d, probe, near, pf_vals, pf_C = _D[L]
    t0 = time.time()
    angle = "labels" if L == 22 else "unsupervised"
    m = p2.build(d, 64, angle, "contiguous", seed, 0, "smooth", aim="coord")
    pca, curve, raw_c, knot = m["pca"], m["curve"], m["raw_curve"], m["knot"]
    picks = p2.pick_clips(d, m["held"], 48, seed)
    v_all = d["df"].speed_mps.to_numpy(float)
    Zk = pca.project(d["X"][knot].astype(float))
    Bs = {n: ols(phi(n, d["y"][knot].astype(float), v_all[knot]), Zk)[1] for n in CANDIDATES}
    K = bu.K
    s = np.linspace(0.0, 1.0, K)
    acc, clips = {}, []
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick].astype(float)
        vv = v_all[pick]
        Z, resid = pca.project(x), pca.complement(x)
        arc = p2.traversed_arc(curve, src, tgt)
        shift_signed = arc * (((tgt - src) * arc) % 360.0)
        tv = np.full(len(src), tgt)
        Zp = {"chord_raw": mf.linear_coords(Z, mf.piecewise_linear_point(raw_c, src),
                                            mf.piecewise_linear_point(raw_c, tv), K)}
        for n, B in Bs.items():
            dF = (phi(n, tv, vv) - phi(n, src, vv)) @ B
            Zp[n] = Z[:, None] + s[None, :, None] * dF[:, None]
        W = {a: p2.compose(pca, zp, resid) for a, zp in Zp.items()}
        nl = np.linalg.norm(W["chord_raw"] - x[:, None], axis=-1, keepdims=True)
        for a in list(W):
            if a == "chord_raw":
                continue
            da = W[a] - x[:, None]
            na = np.linalg.norm(da, axis=-1, keepdims=True)
            W[f"{a}@chord_norm"] = x[:, None] + da * np.where(na > 0, nl / np.where(na > 0, na, 1.0), 0.0)
        for a, w in W.items():
            r = edp.path_metrics(w, x, src, tgt, s, shift_signed, probe, near, pf_vals, pf_C)
            e = acc.setdefault(a, {"err_end": [], "R_end_test": [], "delta_norm_end": []})
            e["err_end"].append(r["err_end"]); e["R_end_test"].append(r["R_end_test"])
            e["delta_norm_end"].append(np.linalg.norm(w[:, -1] - x, axis=-1))
        clips.append(pick)
    return L, seed, {"clip": np.concatenate(clips).tolist(),
                     "arms": {a: {q: np.concatenate(v).tolist() for q, v in e.items()} for a, e in acc.items()},
                     "held": [float(h) for h in m["held"]], "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ assemble
def steer_summary(raw, keep=None):
    """keep: optional boolean over manifest rows (post-hoc carrier subset); None = all carriers (pre-registered)."""
    from wm import motiongeom as mg
    out = {}
    for L in STEER_POINTS:
        arcs = [raw[f"{L}_{s}"] for s in range(16) if f"{L}_{s}" in raw]
        sel = [np.ones(len(a["clip"]), bool) if keep is None else keep[np.asarray(a["clip"])] for a in arcs]
        clip = np.concatenate([np.asarray(a["clip"])[m] for a, m in zip(arcs, sel)])
        arm_names = list(arcs[0]["arms"])
        res = {"n_arcs": len(arcs), "n_rows": int(len(clip)), "n_clips": int(len(np.unique(clip))), "arms": {}}
        g = lambda r, a, q, m: np.asarray(r["arms"][a][q], float)[m]
        E = {a: np.concatenate([g(r, a, "err_end", m) for r, m in zip(arcs, sel)]) for a in arm_names}
        for a in arm_names:
            per_arc = np.array([g(r, a, "err_end", m).mean() for r, m in zip(arcs, sel)])
            Rv = np.concatenate([g(r, a, "R_end_test", m) for r, m in zip(arcs, sel)])
            nv = np.concatenate([g(r, a, "delta_norm_end", m) for r, m in zip(arcs, sel)])
            res["arms"][a] = {"err_end_deg": mg.clip_boot(E[a], clip),
                              "arc_level": {"mean": float(per_arc.mean()),
                                            "pm_2sd_over_sqrt_n": float(2 * per_arc.std(ddof=1) / np.sqrt(len(per_arc))),
                                            "per_arc": [float(x) for x in per_arc]},
                              "R_end_test": mg.clip_boot(Rv, clip), "delta_norm_end_mean": float(nv.mean())}
        for ref in ("chord_raw", "polar4"):
            res[f"minus_{ref}"] = {a: mg.clip_boot(E[a] - E[ref], clip) for a in arm_names if a != ref}
        res["cartesian_minus_polar2"] = {"own_norm": mg.clip_boot(E["cartesian"] - E["polar2"], clip),
                                         "chord_norm": mg.clip_boot(E["cartesian@chord_norm"] - E["polar2@chord_norm"], clip)}
        own = {a: res["arms"][a]["err_end_deg"]["mean"] for a in CANDIDATES}
        cn = {a: res["arms"][f"{a}@chord_norm"]["err_end_deg"]["mean"] for a in CANDIDATES}
        res["best_own_norm"], res["best_chord_norm"] = min(own, key=own.get), min(cn, key=cn.get)
        out[str(L)] = res
    return out


def fig(enc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY, ACC, GREY = "#1b2a49", "#d9822b", "#9aa3b2"
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    pts = POINTS
    for ax, model, title in ((axs[0], "vjepa2", "Trained V-JEPA 2"), (axs[1], "random", "Untrained copy")):
        P = [enc[model][str(p)]["primary_speedset"] for p in pts]
        c4 = [q["per_candidate"]["polar4_v"]["r2"] for q in P]
        pc = [q["per_candidate"]["polar4_v"]["pca_k_r2"] for q in P]
        cart = [q["per_candidate"]["cartesian"]["r2"] for q in P]
        pol2 = [q["per_candidate"]["polar2"]["r2"] for q in P]
        cartv = [q["per_candidate"]["cartesian_v"]["r2"] for q in P]
        pol2v = [q["per_candidate"]["polar2_v"]["r2"] for q in P]
        ax.plot(pts, pc, color=GREY, lw=1.5, ls="--")
        ax.plot(pts, c4, color=NAVY, lw=2)
        ax.plot(pts, pol2v, color=NAVY, lw=1.2, ls=":")
        ax.plot(pts, cartv, color=ACC, lw=1.2, ls=":")
        ax.plot(pts, pol2, color=NAVY, lw=1.2, alpha=0.55)
        ax.plot(pts, cart, color=ACC, lw=2)
        labs = [(pc, "top-5 PCs (ceiling)", GREY), (c4, "polar 4-D + v", NAVY), (pol2v, "polar 2-D + v", NAVY),
                (cartv, "Cartesian + v", ACC), (pol2, "polar 2-D", NAVY), (cart, "Cartesian (vx, vy)", ACC)]
        labs.sort(key=lambda t: -t[0][-1])
        gap = 0.045 * (max(pc) - min(min(cart), 0))
        ys = []
        for y, lab, col in labs:            # stagger end labels so they never overlap
            yy = y[-1] if not ys else min(y[-1], ys[-1] - gap)
            ys.append(yy)
            ax.annotate(lab, (pts[-1], y[-1]), xytext=(pts[-1] + 0.6, yy), textcoords="data", color=col, fontsize=8,
                        va="center")
        ax.axvspan(7.5, 9.5, color=GREY, alpha=0.15, lw=0)
        ax.text(8.5, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.5, "PEZ", ha="center", va="top", fontsize=8, color=GREY)
        ax.set_title(title + ": held-out R² (speed set, knot→probe)", fontsize=10, color=NAVY, loc="left")
        ax.set_xlabel("read point (0 = patch embedding)")
        ax.set_xlim(0, 30.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # winner strip
    ax = axs[2]
    order = list(CANDIDATES)
    for row, model, lab in ((1, "vjepa2", "trained"), (0, "random", "untrained")):
        for p in pts:
            q = enc[model][str(p)]["primary_speedset"]
            w = q["winner_primary"]
            col = ACC if w.startswith("cartesian") else NAVY
            if q["per_candidate"][w]["r2"] <= 0.01:       # nothing explained (patch embedding): no winner
                col = GREY
            ax.scatter(p, row + 0.18, color=col, s=36, marker="s")
            h2 = q["winner_rank2"]
            c2 = ACC if h2 == "cartesian" else NAVY
            if q["per_candidate"][w]["r2"] <= 0.01:
                c2 = GREY
            ax.scatter(p, row - 0.18, color=c2, s=36, marker="o")
        ax.text(-0.8, row, lab, ha="right", va="center", fontsize=9, color=NAVY)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlim(-6, 26)
    ax.axvspan(7.5, 9.5, color=GREY, alpha=0.15, lw=0)
    ax.set_xlabel("read point")
    ax.set_title("Winner per point\n\u25a0 best share of matched-rank PCA ceiling\n\u25cf rank-2 head-to-head (polar 2-D vs Cartesian)\nnavy polar, orange Cartesian, grey nothing explained",
                 fontsize=9, color=NAVY, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def provenance(workers):
    def sh(c):
        try:
            return subprocess.check_output(c, shell=True, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": sh("git rev-parse HEAD") or "8ee1deb (Mac HEAD; box copy is not a git checkout)",
            "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
            "host": os.uname().nodename, "cpus": os.cpu_count(), "workers": workers,
            "activations": "artifacts/activations/{speed,direction}/{vjepa2,random}/meanpool.npy (sha256 = Mac copies)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "steer", "assemble", "prereg"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--points", type=int, nargs="*", default=POINTS)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    if a.cmd == "prereg":
        (out / "prereg.json").write_text(json.dumps({"preregistration": preregistration(),
                                                     "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest()}, indent=1))
        return
    if a.cmd == "encode":
        jobs = [(p, m) for m in ("vjepa2", "random") for p in a.points]
        res = {}
        with Pool(a.workers) as pool:
            for p, m, r in pool.imap_unordered(encode_job, jobs):
                res.setdefault(m, {})[str(p)] = r
                print(time.strftime("%H:%M:%S"), "encode", m, p, r["seconds"], r["primary_speedset"]["winner_primary"], flush=True)
                (out / "encode.json").write_text(json.dumps(res))
        (out / "encode_meta.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                          **provenance(a.workers)}))
    if a.cmd == "steer":
        jobs = [(L, s) for L in STEER_POINTS for s in range(16)]
        res = {}
        with Pool(a.workers) as pool:
            for L, s, r in pool.imap_unordered(steer_job, jobs):
                res[f"{L}_{s}"] = r
                print(time.strftime("%H:%M:%S"), "steer", L, s, r["seconds"], flush=True)
                (out / "steer_raw.json").write_text(json.dumps(res))
        (out / "steer_meta.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                         **provenance(a.workers)}))
    if a.cmd == "assemble":
        pre = json.loads((out / "prereg.json").read_text())
        enc = json.loads((out / "encode.json").read_text())
        em = json.loads((out / "encode_meta.json").read_text())
        st = json.loads((out / "steer_raw.json").read_text()) if (out / "steer_raw.json").exists() else None
        sm = json.loads((out / "steer_meta.json").read_text()) if (out / "steer_meta.json").exists() else None
        wins = {m: {p: enc[m][p]["primary_speedset"]["winner_primary"] for p in sorted(enc[m], key=int)} for m in enc}
        J = {"preregistration": pre["preregistration"],
             "preregistration_script_sha256_at_writing": pre["script_sha256"],
             "run": {"encode_finished": em["finished"], "steer_finished": sm and sm["finished"]},
             "provenance": {**em, "assembled_on_mac_script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
                            "seeds": pre["preregistration"]["seeds"], "n_boot": N_BOOT, "n_randfeat": N_RAND,
                            "n_clips_per_role": {m: enc[m]["12"]["primary_speedset"]["n"] for m in enc} if "12" in enc.get("vjepa2", {}) else None,
                            "n_clips_transfer": enc["vjepa2"]["12"]["transfer_speed_to_dirset"]["n"],
                            "n_clips_secondary": enc["vjepa2"]["12"]["secondary_dirset_velocity"]["n"]},
             "existing_exp4_selection_note": (
                 "run_motion_geometry.py exp4 (not exp3) ran the earlier search: 6 candidates (polar, log_polar, cartesian, "
                 "fourier_polar, fourier_logpolar, fourier2_polar) registered 17:10 ET before its numbers; ranks 2-5; "
                 "selection = transfer encoding R2 of an OLS map fit on ALL speed-set train clips (folds 0-4, no knot/probe "
                 "split) scored on direction-set velocity cell centroids (16 dir x speed 1-4); a true set holdout but no "
                 "rank matching and no baseline of equal rank"),
             "winner_by_point": wins,
             "encoding": enc,
             "steering": steer_summary(st) if st else None}
        if st:
            from wm.data import load_table
            dfd = load_table("direction")
            vel = ((dfd.motion == "velocity") & (dfd.speed_mps > 0)).to_numpy()
            J["steering_posthoc_velocity_carriers"] = {
                "note": ("POST-HOC, not pre-registered: 48% of the bake-off's test-fold carriers are accelerating clips "
                         "whose speed_mps (initial speed) is 0, so every Cartesian-frame edit v(cos, sin) is exactly zero "
                         "on them (error ~93 deg). This block restricts the same rows to constant-velocity carriers "
                         "(v > 0); the edit maps B are unchanged (still fit on all knot rows, incl. v = 0 accelerating clips)"),
                **steer_summary(st, vel)}
        RESULT.write_text(json.dumps(J, indent=1, default=float))
        fig(enc, ROOT / "figures" / "fig_coordinate_competition.png")
        print("wrote", RESULT)


if __name__ == "__main__":
    main()

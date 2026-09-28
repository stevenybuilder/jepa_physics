"""Shape of the two-disk relative-speed code |v1 - v2| (E4 follow-up, probing + encoder-level edits; no predictor).

Inputs: the stored meanpool features of results/p5_relational_motion.json
(artifacts/activations/stimuli_relational/{vjepa2,random}/meanpool.npy, [637, 26, 1024], point 0 = embedding,
point p = block p) and splits/split_relational.json.

Split roles (the relational split has train/test + 5 stratified folds over train; no knot/probe roles exist, so
they are assigned here from the stored folds, never from test):
  knot  = train clips in folds 0, 1, 2   (fit: centroids, line, v1/v2/|v_rel| readers for Q1/Q2)
  probe = train clips in folds 3, 4      (fit: the Q3 readers that score the edit; disjoint from knot)
  test  = the 128 held-out clips         (evaluation only; nothing is ever fit on them)

Q1 line: class-mean trajectory over the 7 |v1 - v2| levels (0..6 m/s) in standardised activation space (knot):
  residual fraction of a straight fit (centroid ~ a + b * level), raw and split-half cross-validated
  (<r_A, r_B> / <c_A, c_B>, unbiased under independent noise), PC1 share, ordering (Spearman of PC1 projection
  vs level, knot and probe-clip centroids on the knot PC1), spacing (CV of consecutive PC1 steps; late/early span
  ratio). Reference manifolds: |v1| (the single-disk speed) and signed v1, same recipe.
Q2 separate axis: ridge readers (alpha by 5-fold CV on knot) for v1, v2, |v_rel|; principal angle between w_|vrel|
  and span(w_v1, w_v2); held-out R^2 of |v_rel| from (a) the activation reader vs (b) the decoded v1, v2 only
  (b_abs = |v1^ - v2^|; b_flex = a quadratic + |diff| regression on the decoded pair, trained on 5-fold cross-fitted
  knot decodes); paired clip-bootstrap of (a) - (b) on test and on probe clips.
Q3 leakage: straight edit along the knot |v_rel| centroid line (slope b, units of 1 m/s of |v_rel|) applied to
  test clips; readers for v1, v2, |v_rel| refit on probe clips score the change per +1 m/s; also the change of
  the composite |v1^ - v2^|. Positive control: edit along the knot v1 centroid line. Null: matched-norm random.
CIs: 1000-draw clip bootstrap for evaluation-only quantities; 200-draw refit bootstrap (knot and probe clips
  resampled with replacement, readers/lines refit at the selected alpha) for angles, line metrics and edits.

  python scripts/run_relational_shape.py --make-labels            # Mac: writes artifacts/relational_shape/labels.npz
  python scripts/run_relational_shape.py --root DIR --out JSON    # box: DIR holds labels.npz, split, features
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

POINTS = [1, 12, 16, 19, 22]
ALPHAS = np.logspace(-2, 5, 15)
N_BOOT = 1000
N_REFIT = 200
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- ridge (dual form, n << d)
def standardise(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd


def ridge_fit(X, y, alpha):
    """w, b for y ~ X w + b (X already standardised on some rows; centred here on the fit rows)."""
    mx, my = X.mean(0), y.mean()
    Xc = X - mx
    K = Xc @ Xc.T
    a = np.linalg.solve(K + alpha * np.eye(len(y)), y - my)
    w = Xc.T @ a
    return w, my - mx @ w


def ridge_cv_alpha(X, y, folds):
    """alpha by fold-mean R^2 over the given folds (eigendecomposition per fold, all alphas at once)."""
    scores = np.zeros(len(ALPHAS))
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        mx, my = X[tr].mean(0), y[tr].mean()
        Xt, Xv = X[tr] - mx, X[va] - mx
        lam, U = np.linalg.eigh(Xt @ Xt.T)
        Uy = U.T @ (y[tr] - my)
        KvU = (Xv @ Xt.T) @ U
        for i, al in enumerate(ALPHAS):
            p = KvU @ (Uy / (lam + al)) + my
            scores[i] += r2(y[va], p)
    return float(ALPHAS[int(np.argmax(scores))])


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def boot_r2(y, p, idx):
    return np.array([r2(y[i], p[i]) for i in idx])


# ---------------------------------------------------------------- Q1 line metrics
def centroids(X, lev, levels):
    return np.stack([X[lev == l].mean(0) for l in levels])


def line_metrics(C, levels):
    """C [K, d] class means at numeric levels. Straight fit C ~ a + b * level."""
    L = np.asarray(levels, float)
    Cc = C - C.mean(0)
    Lc = L - L.mean()
    b = (Lc @ Cc) / (Lc @ Lc)
    R = Cc - np.outer(Lc, b)
    tot = float((Cc ** 2).sum())
    u, s, vt = np.linalg.svd(Cc, full_matrices=False)
    proj = Cc @ vt[0]
    if np.corrcoef(proj, L)[0, 1] < 0:
        proj = -proj
    steps = np.diff(proj)
    half = len(L) // 2
    return {"resid_frac": float((R ** 2).sum() / tot), "pc1_share": float(s[0] ** 2 / (s ** 2).sum()),
            "spearman_pc1": float(spearmanr(proj, L)[0]), "n_inversions": int((steps <= 0).sum()),
            "step_cv": float(steps.std() / abs(steps.mean())),
            "late_over_early_span": float((proj[-1] - proj[half]) / (proj[half] - proj[0])),
            "pc1_proj": proj.tolist()}, b, vt[0] * np.sign(np.corrcoef(Cc @ vt[0], L)[0, 1])


def cv_resid_frac(X, lev, levels, rng, n_split=50):
    """split-half cross-validated residual fraction: <r_A, r_B> / <C_A, C_B> (independent noise cancels)."""
    L = np.asarray(levels, float)
    Lc = L - L.mean()
    out = []
    for _ in range(n_split):
        A = np.zeros(len(lev), bool)
        for l in levels:
            ix = np.flatnonzero(lev == l)
            A[rng.permutation(ix)[: len(ix) // 2]] = True
        CA, CB = centroids(X[A], lev[A], levels), centroids(X[~A], lev[~A], levels)
        CA, CB = CA - CA.mean(0), CB - CB.mean(0)
        RA = CA - np.outer(Lc, (Lc @ CA) / (Lc @ Lc))
        RB = CB - np.outer(Lc, (Lc @ CB) / (Lc @ Lc))
        out.append(float((RA * RB).sum() / (CA * CB).sum()))
    return float(np.mean(out)), float(np.std(out))


def q1(X, lab, knot, probe, rng):
    res = {}
    specs = {"abs_v_rel": np.round(lab["abs_v_rel"]).astype(int),
             "abs_v1": np.round(np.abs(lab["v1"])).astype(int),
             "v1": np.round(lab["v1"]).astype(int)}
    for name, lev_all in specs.items():
        levels = sorted(set(lev_all[knot]) & set(lev_all[probe]))
        levels = [l for l in levels if (lev_all[knot] == l).sum() >= 4 and (lev_all[probe] == l).sum() >= 2]
        lev_k = lev_all[knot]
        m = np.isin(lev_k, levels)
        Xk, lk = X[knot][m], lev_k[m]
        C = centroids(Xk, lk, levels)
        met, b, pc1 = line_metrics(C, levels)
        cvm, cvs = cv_resid_frac(Xk, lk, levels, rng)
        # probe-clip centroids on the knot PC1 (held-out ordering)
        lp = lev_all[probe]
        mp = np.isin(lp, levels)
        Cp = centroids(X[probe][mp], lp[mp], levels)
        pp = (Cp - C.mean(0)) @ pc1
        # shuffle null for the raw residual fraction (levels permuted over clips)
        null = []
        for _ in range(200):
            lperm = rng.permutation(lk)
            null.append(line_metrics(centroids(Xk, lperm, levels), levels)[0]["resid_frac"])
        # refit bootstrap over knot clips (stratified within level)
        bs = {"resid_frac": [], "spearman_pc1": [], "step_cv": [], "late_over_early_span": [], "cv_resid": []}
        for _ in range(N_REFIT):
            ix = np.concatenate([rng.choice(np.flatnonzero(lk == l), (lk == l).sum()) for l in levels])
            mm, _, _ = line_metrics(centroids(Xk[ix], lk[ix], levels), levels)
            for k in ("resid_frac", "spearman_pc1", "step_cv", "late_over_early_span"):
                bs[k].append(mm[k])
        res[name] = {**met, "levels": [int(l) for l in levels],
                     "n_per_level_knot": [int((lk == l).sum()) for l in levels],
                     "cv_resid_frac": cvm, "cv_resid_frac_sd_over_splits": cvs,
                     "resid_frac_shuffle_null_mean": float(np.mean(null)), "resid_frac_shuffle_null_ci": ci(null),
                     "probe_spearman_on_knot_pc1": float(spearmanr(pp, levels)[0]),
                     "probe_pc1_proj": pp.tolist(),
                     **{k + "_ci": ci(v) for k, v in bs.items() if v}}
        if name == "abs_v_rel":
            res["_line_slope_abs_v_rel"] = b
        if name == "v1":
            res["_line_slope_v1"] = b
    return res


# ---------------------------------------------------------------- Q2 / Q3
def principal_angle(w, B):
    Q, _ = np.linalg.qr(B)
    c = np.linalg.norm(Q.T @ w) / np.linalg.norm(w)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def cos_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def pair_feats(p1, p2):
    d = p1 - p2
    return np.column_stack([p1, p2, np.abs(d), p1 ** 2, p2 ** 2, p1 * p2])


def lstsq_fit(F, y):
    A = np.column_stack([F, np.ones(len(F))])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def lstsq_pred(F, c):
    return np.column_stack([F, np.ones(len(F))]) @ c


def q2_q3(X, lab, knot, probe, test, kfold, pfold, slope_a, slope_v1, rng):
    T = {"v1": lab["v1"], "v2": lab["v2"], "abs_v_rel": lab["abs_v_rel"]}
    Xk, Xp, Xt = X[knot], X[probe], X[test]
    out = {"alpha_knot": {}, "alpha_probe": {}}
    W, B, OOF = {}, {}, {}
    for t, y in T.items():
        al = ridge_cv_alpha(Xk, y[knot], kfold)
        out["alpha_knot"][t] = al
        W[t], B[t] = ridge_fit(Xk, y[knot], al)
        oof = np.zeros(len(knot))
        for f in np.unique(kfold):
            w, b = ridge_fit(Xk[kfold != f], y[knot][kfold != f], al)
            oof[kfold == f] = Xk[kfold == f] @ w + b
        OOF[t] = oof
    ang = principal_angle(W["abs_v_rel"], np.column_stack([W["v1"], W["v2"]]))
    d = X.shape[1]
    rnd_ang = [principal_angle(rng.standard_normal(d), np.column_stack([W["v1"], W["v2"]])) for _ in range(200)]
    # (a) vs (b) on held-out clips
    ev = {}
    cflex = lstsq_fit(pair_feats(OOF["v1"], OOF["v2"]), T["abs_v_rel"][knot])
    for split_name, rows, Xe in (("test", test, Xt), ("probe", probe, Xp)):
        y = T["abs_v_rel"][rows]
        pa = Xe @ W["abs_v_rel"] + B["abs_v_rel"]
        p1, p2 = Xe @ W["v1"] + B["v1"], Xe @ W["v2"] + B["v2"]
        pb_abs = np.abs(p1 - p2)
        pb_flex = lstsq_pred(pair_feats(p1, p2), cflex)
        idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
        ba, bb, bf = boot_r2(y, pa, idx), boot_r2(y, pb_abs, idx), boot_r2(y, pb_flex, idx)
        ev[split_name] = {"n": int(len(rows)),
                          "r2_v1": r2(T["v1"][rows], p1), "r2_v2": r2(T["v2"][rows], p2),
                          "a_direct_r2": r2(y, pa), "a_direct_r2_ci": ci(ba),
                          "b_abs_diff_decoded_r2": r2(y, pb_abs), "b_abs_diff_decoded_r2_ci": ci(bb),
                          "b_flex_decoded_r2": r2(y, pb_flex), "b_flex_decoded_r2_ci": ci(bf),
                          "a_minus_b_abs": r2(y, pa) - r2(y, pb_abs), "a_minus_b_abs_ci": ci(ba - bb),
                          "a_minus_b_flex": r2(y, pa) - r2(y, pb_flex), "a_minus_b_flex_ci": ci(ba - bf)}
    # Q3 readers on probe clips
    Wp, Bp = {}, {}
    for t, y in T.items():
        al = ridge_cv_alpha(Xp, y[probe], pfold)
        out["alpha_probe"][t] = al
        Wp[t], Bp[t] = ridge_fit(Xp, y[probe], al)

    def edit_effect(Wd, Bd, direction, delta=1.0):
        Xe = Xt + delta * direction
        base = {t: Xt @ Wd[t] + Bd[t] for t in T}
        new = {t: Xe @ Wd[t] + Bd[t] for t in T}
        comp0 = np.abs(base["v1"] - base["v2"])
        comp1 = np.abs(new["v1"] - new["v2"])
        return {"d_v1": float((new["v1"] - base["v1"]).mean()), "d_v2": float((new["v2"] - base["v2"]).mean()),
                "d_abs_v_rel": float((new["abs_v_rel"] - base["abs_v_rel"]).mean()),
                "d_abs_diff_decoded": float((comp1 - comp0).mean())}

    rdir = rng.standard_normal(d)
    rdir *= np.linalg.norm(slope_a) / np.linalg.norm(rdir)
    q3 = {"units": "mean change of the probe-clip reader outputs (m/s) on the 128 test clips per +1 m/s step "
                   "along the knot centroid line",
          "abs_v_rel_line": edit_effect(Wp, Bp, slope_a), "v1_line_positive_control": edit_effect(Wp, Bp, slope_v1),
          "random_matched_norm": edit_effect(Wp, Bp, rdir),
          "line_norm_abs_v_rel": float(np.linalg.norm(slope_a)), "line_norm_v1": float(np.linalg.norm(slope_v1)),
          "angle_line_abs_v_rel_to_span_probe_v1_v2_deg": principal_angle(slope_a, np.column_stack([Wp["v1"], Wp["v2"]]))}
    # refit bootstrap: knot clips (readers, line) and probe clips (Q3 readers) resampled with replacement
    lev = np.round(lab["abs_v_rel"][knot]).astype(int)
    levels = sorted(set(lev))
    bs = {"angle": [], "angle_v1_v2": [], "q3_d_v1": [], "q3_d_v2": [], "q3_d_abs": [], "q3_d_comp": [],
          "q3_leak_frac": []}
    for _ in range(N_REFIT):
        ik = rng.integers(0, len(knot), len(knot))
        ip = rng.integers(0, len(probe), len(probe))
        Wb = {t: ridge_fit(Xk[ik], T[t][knot][ik], out["alpha_knot"][t])[0] for t in T}
        bs["angle"].append(principal_angle(Wb["abs_v_rel"], np.column_stack([Wb["v1"], Wb["v2"]])))
        bs["angle_v1_v2"].append(cos_deg(Wb["v1"], Wb["v2"]))
        lk = lev[ik]
        okl = [l for l in levels if (lk == l).sum() > 0]
        C = centroids(Xk[ik], lk, okl)
        Lc = np.asarray(okl, float) - np.mean(okl)
        sl = (Lc @ (C - C.mean(0))) / (Lc @ Lc)
        Wpb, Bpb = {}, {}
        for t in T:
            Wpb[t], Bpb[t] = ridge_fit(Xp[ip], T[t][probe][ip], out["alpha_probe"][t])
        e = edit_effect(Wpb, Bpb, sl)
        bs["q3_d_v1"].append(e["d_v1"]); bs["q3_d_v2"].append(e["d_v2"])
        bs["q3_d_abs"].append(e["d_abs_v_rel"]); bs["q3_d_comp"].append(e["d_abs_diff_decoded"])
        bs["q3_leak_frac"].append(max(abs(e["d_v1"]), abs(e["d_v2"])) / abs(e["d_abs_v_rel"]))
    e0 = q3["abs_v_rel_line"]
    q3["leak_frac_max_abs_dv_over_d_abs_v_rel"] = max(abs(e0["d_v1"]), abs(e0["d_v2"])) / abs(e0["d_abs_v_rel"])
    q3["refit_ci"] = {"d_v1": ci(bs["q3_d_v1"]), "d_v2": ci(bs["q3_d_v2"]), "d_abs_v_rel": ci(bs["q3_d_abs"]),
                      "d_abs_diff_decoded": ci(bs["q3_d_comp"]), "leak_frac": ci(bs["q3_leak_frac"])}
    out["q2"] = {"angle_w_abs_v_rel_to_span_w_v1_w_v2_deg": ang, "angle_ci": ci(bs["angle"]),
                 "angle_random_direction_mean_deg": float(np.mean(rnd_ang)),
                 "angle_random_direction_ci": ci(rnd_ang),
                 "angle_w_v1_w_v2_deg": cos_deg(W["v1"], W["v2"]), "angle_w_v1_w_v2_ci": ci(bs["angle_v1_v2"]),
                 "angle_w_abs_v_rel_to_w_v1_deg": cos_deg(W["abs_v_rel"], W["v1"]),
                 "angle_w_abs_v_rel_to_w_v2_deg": cos_deg(W["abs_v_rel"], W["v2"]),
                 "angle_centroid_line_to_w_abs_v_rel_deg": cos_deg(slope_a, W["abs_v_rel"]),
                 "heldout": ev}
    out["q3"] = q3
    return out


# ---------------------------------------------------------------- driver
def make_labels():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from run_relational_motion import table  # noqa: E402
    df = table()
    out = PROJECT_ROOT / "artifacts" / "relational_shape"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "labels.npz", id=df["id"].to_numpy(), v1=df["v1"].to_numpy(float), v2=df["v2"].to_numpy(float),
             abs_v_rel=df["abs_v_rel"].to_numpy(float), cell=df["cell"].astype(str).to_numpy())
    print("wrote", out / "labels.npz")


def roles(lab, split):
    pos = {int(i): r for r, i in enumerate(lab["id"])}
    tr = np.array([pos[i] for i in split["train_ids"]])
    te = np.array([pos[i] for i in split["test_ids"]])
    fold = np.array([split["fold"][str(i)] for i in split["train_ids"]])
    knot, probe = tr[fold <= 2], tr[fold >= 3]
    assert not set(knot) & set(probe) and not (set(knot) | set(probe)) & set(te)
    # 5 folds inside knot / probe for alpha selection (the stored fold ids are reused modulo the role)
    kf = np.arange(len(knot)) % 5
    pf = np.arange(len(probe)) % 5
    rng = np.random.default_rng(0)
    return knot, probe, te, rng.permutation(kf), rng.permutation(pf)


def run(root, out_path, points, sources):
    t0 = time.time()
    root = Path(root)
    L = np.load(root / "labels.npz", allow_pickle=True)
    lab = {k: L[k] for k in L.files}
    split = json.loads((root / "split_relational.json").read_text())["relational"]
    knot, probe, test, kf, pf = roles(lab, split)
    res = {"experiment": "E4 follow-up: shape of the |v1 - v2| code (probing + encoder-level line edits)",
           "doc": __doc__, "points": points,
           "roles": {"knot": "train folds 0-2", "probe": "train folds 3-4", "test": "held-out 128 (eval only)",
                     "n": {"knot": int(len(knot)), "probe": int(len(probe)), "test": int(len(test))}},
           "sources": {}, "provenance": {}}
    for src in sources:
        f = root / src / "meanpool.npy"
        ids = json.loads((root / src / "ids.json").read_text())
        assert ids == [int(i) for i in lab["id"]]
        res["provenance"][f"{src}_meanpool_sha256"] = sha(f)
        acts = np.load(f, mmap_mode="r")
        res["sources"][src] = {}
        for p in points:
            rng = np.random.default_rng(1000 * p + (0 if src == "vjepa2" else 1))
            X = standardise(np.asarray(acts[:, p], dtype=np.float64), knot)
            r1 = q1(X, lab, knot, probe, rng)
            sa, sv = r1.pop("_line_slope_abs_v_rel"), r1.pop("_line_slope_v1")
            r23 = q2_q3(X, lab, knot, probe, test, kf, pf, sa, sv, rng)
            res["sources"][src][str(p)] = {"q1_line": r1, **r23}
            h = r23["q2"]["heldout"]["test"]
            print(f"{src} point {p:2d} ({time.time() - t0:.0f}s): resid {r1['abs_v_rel']['resid_frac']:.3f} "
                  f"cv {r1['abs_v_rel']['cv_resid_frac']:.3f} angle {r23['q2']['angle_w_abs_v_rel_to_span_w_v1_w_v2_deg']:.1f} "
                  f"a {h['a_direct_r2']:.3f} b_abs {h['b_abs_diff_decoded_r2']:.3f} b_flex {h['b_flex_decoded_r2']:.3f} "
                  f"leak {r23['q3']['leak_frac_max_abs_dv_over_d_abs_v_rel']:.3f}", flush=True)
    res["provenance"].update({"labels_sha256": sha(root / "labels.npz"),
                              "split_sha256": sha(root / "split_relational.json"),
                              "seeds": "roles/fold permutation seed 0; per (source, point) rng seed 1000*point + (0 vjepa2 | 1 random)",
                              "n_boot_eval": N_BOOT, "n_refit_boot": N_REFIT, "alphas": ALPHAS.tolist(),
                              "host": os.uname().nodename, "wall_seconds": time.time() - t0})
    Path(out_path).write_text(json.dumps(res, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-labels", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p5_relational_shape.json"))
    ap.add_argument("--points", default=",".join(map(str, POINTS)))
    ap.add_argument("--sources", default="vjepa2,random")
    ap.add_argument("--refit", type=int, default=N_REFIT)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    N_REFIT, N_BOOT = a.refit, a.boot
    if a.make_labels:
        make_labels()
    else:
        run(a.root, a.out, [int(x) for x in a.points.split(",")], a.sources.split(","))

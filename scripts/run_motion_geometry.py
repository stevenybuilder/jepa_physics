"""Representational geometry of motion as a whole: subspace atlas, interference/superposition as steering leakage,
within- vs between-subspace steering (incl. Fourier-linear direction edits), a registered model-native coordinate
search, and shortcut rejection. CPU only, stored pooled activations. Each experiment writes
results/p5_motion_geometry_parts/<exp>.json; `merge` assembles results/p5_motion_geometry.json.

  python scripts/run_motion_geometry.py exp1|exp2|exp3|exp4|merge
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.splits import load_split  # noqa: E402
from wm import motiongeom as mg  # noqa: E402

PARTS = PROJECT_ROOT / "results" / "p5_motion_geometry_parts"
POINTS = [1, 4, 8, 12, 16, 19, 22]
ACT = PROJECT_ROOT / "artifacts" / "activations"
ARC = np.arange(303.75, 343.2, 5.625)              # the contiguous held-out 45 deg arc of §4.2 (8 values)
ARC_NEIGH = (298.125, 348.75)
BLOCK_IDX = np.arange(24, 32)                       # interior block of 8 speed / acceleration values (sorted index)
HELD_STEP = 5
SVAR = {"speed": "speed_mps", "acceleration": "acceleration_mps2"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def meta(dataset):
    df = load_table(dataset)
    s = load_split(dataset)
    fold = {int(k): v for k, v in s["fold"].items()}
    f = np.array([fold[int(i)] for i in df["id"]])
    ids = json.loads((ACT / dataset / "vjepa2" / "ids.json").read_text())
    assert ids == [int(i) for i in df["id"]]
    return df, f


def load_mp(dataset, model, p):
    return np.asarray(np.load(ACT / dataset / model / "meanpool.npy", mmap_mode="r")[:, p], dtype=np.float64)


def load_tp(dataset, model, p):
    return np.asarray(np.load(ACT / dataset / model / "timepool.npy", mmap_mode="r")[:, p], dtype=np.float32)


def design(df, svar):
    """[fourier2(theta) (4), s, x0, y0] -> column groups."""
    F = np.column_stack([mg.fourier_features(df.theta_degrees, 2), df[svar].to_numpy(float),
                         df.start_x.to_numpy(float), df.start_y.to_numpy(float)])
    return F, {"dir": [0, 1], "dir2": [2, 3], "s": [4], "pos": [5, 6]}


def time_basis(R_clip, k_max=3, frac=0.9):
    """R_clip [n, 8, D] clip-mean-removed step features -> (Q [D, k], k, var share per PC of the step centroids)."""
    M = R_clip.mean(0)
    M = M - M.mean(0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    k = int(min(k_max, np.searchsorted(np.cumsum(share), frac) + 1))
    return Vt[:k].T, k, share


def ridge_fit(X, Y):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))).fit(X, Y)


def mlp_fit(X, Y, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(128, random_state=seed),
                         MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=seed)
                         ).fit(X, Y)


def angle_of(P):
    return np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360.0


def write(name, obj):
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    log("wrote", name)


def jf(x):
    return [float(v) for v in np.asarray(x).ravel()]


# ================================================================ exp1: subspace atlas ============================
def atlas_bases(Xs, dfs, Xa, dfa, Rt):
    _, Bs = mg.fit_linear_encoding(Xs, design(dfs, "speed_mps")[0])
    _, Ba = mg.fit_linear_encoding(Xa, design(dfa, "acceleration_mps2")[0])
    Q = {"dir": mg.orth(Bs[[0, 1]].T), "dir2": mg.orth(Bs[[2, 3]].T), "speed": mg.orth(Bs[[4]].T),
         "acc": mg.orth(Ba[[4]].T), "pos": mg.orth(Bs[[5, 6]].T)}
    if Rt is not None:
        Q["time"] = time_basis(Rt)[0]
    return Q, Bs, Ba


def pair_table(Q):
    names = list(Q)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ang = mg.principal_angles_deg(Q[a], Q[b])
            out[f"{a}|{b}"] = {"angles_deg": jf(ang), "min_angle_deg": float(ang.min()), "overlap": mg.overlap(Q[a], Q[b])}
    return out


def exp1():
    dfs, fs = meta("speed")
    dfa, fa = meta("acceleration")
    dfd, fd = meta("direction")
    trs, tra, trd = fs >= 0, fa >= 0, fd >= 0
    rng = np.random.default_rng(0)
    out = {"method": (
        "Per point, train clips only (folds 0-4). Joint OLS encoding model on the speed set: meanpool ~ 1 + [cos t, "
        "sin t, cos 2t, sin 2t, v, x0, y0]; S_dir = span of the (cos, sin) coefficient columns, S_dir2 = (cos 2t, "
        "sin 2t), S_speed = v column, S_pos = (x0, y0). S_acc = the a column of the same model on the acceleration "
        "set. S_time = top-k PCs (k = smallest giving >= 0.90 of variance, cap 3) of the 8 step centroids of the "
        "clip-mean-removed timepool features (speed set; random-init has no speed-set timepool, so its S_time and a "
        "V-JEPA S_time_dirset twin come from the direction set). Nulls: (perm) refit with design rows permuted / step "
        "labels permuted within clip, 20 draws; (sigma) random subspaces of matching dims drawn from N(0, Sigma_train), "
        "200 draws, which is the right null for 'more overlap than data-shaped random directions'. CIs: 100 clip "
        "bootstrap refits of the train clips. Strength ||P_S (h - mean)|| on test clips."), "points": {}}
    for model in ("vjepa2", "random"):
        for p in POINTS:
            log("exp1", model, p)
            Xs, Xa = load_mp("speed", model, p), load_mp("acceleration", model, p)
            if model == "vjepa2":
                T = load_tp("speed", model, p)[trs]
                Rt = (T - T.mean(1, keepdims=True)).astype(np.float64)
                Td = load_tp("direction", model, p)[trd]
            else:
                Rt = None
                Td = load_tp("direction", model, p)[trd]
            Rd = (Td - Td.mean(1, keepdims=True)).astype(np.float64)
            Q, Bs, Ba = atlas_bases(Xs[trs], dfs[trs], Xa[tra], dfa[tra], Rt)
            Qtd, kd, shd = time_basis(Rd)
            Q["time_dirset"] = Qtd
            if Rt is not None:
                _, kt, sht = time_basis(Rt)
            res = {"dims": {k: int(v.shape[1]) for k, v in Q.items()}}
            if Rt is not None:
                res["time_pc_share"] = jf(sht[:4])
            res["time_dirset_pc_share"] = jf(shd[:4])
            # variance captured
            Xtr = Xs[trs]
            mu = Xtr.mean(0)
            vc = {k: mg.captured_variance(Xtr, Q[k]) for k in ("dir", "dir2", "speed", "pos")}
            vc["acc_in_accset"] = mg.captured_variance(Xa[tra], Q["acc"])
            vc["dir_in_accset"] = mg.captured_variance(Xa[tra], Q["dir"])
            if Rt is not None:
                Rf = Rt.reshape(-1, Rt.shape[-1])
                vc["time_of_within_clip"] = mg.captured_variance(Rf, Q["time"], mean=0)
                vc["time_of_total_steps"] = float(((Rf @ Q["time"]) ** 2).sum() /
                                                  ((T.reshape(-1, T.shape[-1]).astype(np.float64) - T.reshape(-1, T.shape[-1]).mean(0)) ** 2).sum())
            Rdf = Rd.reshape(-1, Rd.shape[-1])
            vc["time_dirset_of_within_clip"] = mg.captured_variance(Rdf, Qtd, mean=0)
            # signal R2 of the joint encoding model (held-out test clips), per group
            Fs_all, _ = design(dfs, "speed_mps")
            a0, _ = mg.fit_linear_encoding(Xtr, Fs_all[trs])
            te = fs < 0
            pred = a0 + Fs_all[te] @ Bs
            vc["joint_model_test_R2"] = float(1 - ((Xs[te] - pred) ** 2).sum() / ((Xs[te] - Xs[te].mean(0)) ** 2).sum())
            res["variance_captured"] = vc
            names = ["dir", "dir2", "speed", "acc", "pos"] + (["time"] if Rt is not None else []) + ["time_dirset"]
            res["pairs"] = pair_table({k: Q[k] for k in names})
            res["cos_speed_acc_axes"] = float(abs(Q["speed"][:, 0] @ Q["acc"][:, 0]))
            # dir subspace from acceleration set vs speed set (consistency)
            res["dir_speedset_vs_accset_angles_deg"] = jf(mg.principal_angles_deg(Q["dir"], mg.orth(Ba[[0, 1]].T)))
            six = [Q[k] for k in names if k != "time_dirset"] if Rt is not None else [Q[k] for k in names]
            res["union"] = mg.union_dimension(six)
            res["union"]["members"] = [k for k in names if k != "time_dirset"] if Rt is not None else names
            # sigma null: random data-shaped subspaces
            C = np.cov(Xtr.T)
            w, V = np.linalg.eigh(C)
            L = V * np.sqrt(np.clip(w, 0, None))
            dims = [Q[k].shape[1] for k in res["union"]["members"]]
            null_ov, null_pr = {}, []
            for d in range(200):
                Rq = [mg.orth(L @ rng.normal(size=(L.shape[1], k))) for k in dims]
                null_pr.append(mg.union_dimension(Rq)["participation_ratio"])
                for i in range(len(dims)):
                    for j in range(i + 1, len(dims)):
                        null_ov.setdefault(f"{dims[i]}x{dims[j]}", []).append(mg.overlap(Rq[i], Rq[j]))
            res["null_sigma"] = {"union_participation_ratio": {"mean": float(np.mean(null_pr)), "p5": float(np.percentile(null_pr, 5)), "p95": float(np.percentile(null_pr, 95))},
                                 "overlap_by_dims": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99))} for k, v in null_ov.items()}}
            # permutation null (design rows permuted)
            perm_ov = {}
            for d in range(20):
                ps, pa = rng.permutation(trs.sum()), rng.permutation(tra.sum())
                Qp, _, _ = atlas_bases(Xs[trs], dfs[trs].iloc[ps], Xa[tra], dfa[tra].iloc[pa],
                                       None if Rt is None else Rt[:, :, :][np.arange(len(Rt))[:, None], np.argsort(rng.random((len(Rt), 8)), 1)])
                for k, v in pair_table(Qp).items():
                    perm_ov.setdefault(k, []).append(v["overlap"])
            res["null_perm"] = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95))} for k, v in perm_ov.items()}
            # bootstrap CIs on pair overlaps / angles
            boot = {}
            ns, na = trs.sum(), tra.sum()
            for b in range(100):
                i_s, i_a = rng.integers(0, ns, ns), rng.integers(0, na, na)
                Qb, _, _ = atlas_bases(Xs[trs][i_s], dfs[trs].iloc[i_s], Xa[tra][i_a], dfa[tra].iloc[i_a],
                                       None if Rt is None else Rt[i_s])
                for k, v in pair_table(Qb).items():
                    boot.setdefault(k, []).append((v["overlap"], v["min_angle_deg"]))
            for k, v in boot.items():
                v = np.array(v)
                res["pairs"][k]["overlap_ci95"] = jf(np.percentile(v[:, 0], [2.5, 97.5]))
                res["pairs"][k]["min_angle_ci95"] = jf(np.percentile(v[:, 1], [2.5, 97.5]))
            # strength on test clips
            from scipy.stats import spearmanr
            Xte, dte = Xs[te], dfs[te]
            st = {}
            for k in ("dir", "dir2", "speed", "pos"):
                s = mg.projector_strength(Xte, Q[k], mu)
                st[k] = {"q10_50_90": jf(np.percentile(s, [10, 50, 90])),
                         "spearman_vs_speed": float(spearmanr(s, dte.speed_mps).correlation),
                         "spearman_vs_start_radius": float(spearmanr(s, np.hypot(dte.start_x, dte.start_y)).correlation)}
            st["speed"]["signed_proj_spearman_vs_speed"] = float(spearmanr((Xte - mu) @ Q["speed"][:, 0], dte.speed_mps).correlation)
            res["strength_test"] = st
            out["points"].setdefault(model, {})[str(p)] = res
            write("exp1", out)
    write("exp1", out)


# ================================================================ exp3: within vs between + Fourier-linear =======
def knn_excess(Q, R, k=5):
    """Mean distance of Q rows to their k nearest R rows / median of the same for R rows (leave-one-out)."""
    def kd(A, self_rows):
        d2 = (A ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None] - 2 * A @ R.T
        d = np.sqrt(np.maximum(d2, 0))
        d.sort(1)
        return d[:, 1:k + 1].mean(1) if self_rows else d[:, :k].mean(1)
    return kd(Q, False) / np.median(kd(R, True))


_ED = None


def _endpoint_diag():
    """scripts/run_endpoint_diagnosis.py (cv_lambda, causalab_curve), loaded once."""
    global _ED
    if _ED is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_endpoint_diagnosis", Path(__file__).with_name("run_endpoint_diagnosis.py"))
        _ED = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ED)
    return _ED


def exp3_fourier(p, out):
    """Direction set: Fourier-linear (K = 1, 2, 3) edits vs smoothing spline vs raw chord, held-out 45 deg arc."""
    from wm.manifold import centroids, fit_curve, fit_pca
    df, f = meta("direction")
    X = load_mp("direction", "vjepa2", p)
    th = df.theta_degrees.to_numpy(float)
    held = np.isin(np.round(th, 4), np.round(ARC, 4))
    knot = np.isin(f, (0, 1, 2)) & ~held
    probe = np.isin(f, (3, 4))
    test = (f < 0) & ~held
    pr = ridge_fit(X[probe], mg.fourier_features(th[probe], 1))
    ml = mlp_fit(X[probe], mg.fourier_features(th[probe], 1))
    # transferred speed probe (speed set, all train) for leakage
    dfs, fs = meta("speed")
    Xs = load_mp("speed", "vjepa2", p)
    sp = ridge_fit(Xs[fs >= 0], dfs.speed_mps[fs >= 0].to_numpy(float))
    rng = np.random.default_rng(p)
    ci = np.where(test)[0]
    rows_c = np.repeat(ci, 3)
    tgt = rng.choice(ARC, len(rows_c))
    src = th[rows_c]
    Xc = X[rows_c]
    arms = {}
    for K in (1, 2, 3):
        _, B = mg.fit_linear_encoding(X[knot], mg.fourier_features(th[knot], K))
        arms[f"fourier{K}_{2 * K}d"] = (mg.fourier_features(tgt, K) - mg.fourier_features(src, K)) @ B
    pca = fit_pca(X[knot], 64)
    cent = centroids(pca.project(X[knot]), th[knot])
    # interpolating periodic cubic through the raw kept centroids, value-ordered (labels) coordinates (Goodfire A.3)
    curve = fit_curve(cent, True, angle="labels", spline="interp")
    arms["spline_interp_pca64"] = pca.lift_delta(curve(np.radians(tgt)) - curve(np.radians(src)))
    # causalab Reinsch smoother, lambda by leave-block-out CV on kept knots only (run_endpoint_diagnosis.cv_lambda)
    ed = _endpoint_diag()
    coords = np.radians(cent["values"])
    cvl = ed.cv_lambda(coords, cent["C"], cent["values"])
    fr, _ = ed.causalab_curve(coords, cent["C"], cvl["lambda_cv"])
    arms["spline_reinsch_cv_pca64"] = pca.lift_delta(fr(np.radians(tgt)) - fr(np.radians(src)))
    # raw chord in full space: target = interpolation between raw centroids of the arc neighbours; source = raw centroid
    cfull = {v: X[knot & np.isclose(th, v)].mean(0) for v in np.unique(th[knot])}
    a, b = ARC_NEIGH
    w = ((tgt - a) / (b - a))[:, None]
    ctgt = (1 - w) * cfull[a] + w * cfull[b]
    arms["raw_chord"] = ctgt - np.array([cfull[v] for v in src])
    # matched-norm Fourier-4 (rescaled to the spline's norm per row)
    n_sp = np.linalg.norm(arms["spline_reinsch_cv_pca64"], axis=1, keepdims=True)
    arms["fourier2_4d_normmatched_to_reinsch"] = arms["fourier2_4d"] * n_sp / np.linalg.norm(arms["fourier2_4d"], axis=1, keepdims=True)
    real_t = {v: X[probe & np.isclose(th, v)].mean(0) for v in ARC}
    RT = np.array([real_t[v] for v in tgt])
    Rref = pca.project(X[probe])
    base_speed = sp.predict(Xc)
    res = {}
    for name, D in arms.items():
        Xe = Xc + D
        e_pr = mg.circ_err_deg(angle_of(pr.predict(Xe)), tgt)
        e_ml = mg.circ_err_deg(angle_of(ml.predict(Xe)), tgt)
        R = 1 - np.linalg.norm(Xe - RT, axis=1) / np.linalg.norm(Xc - RT, axis=1)
        kn = knn_excess(pca.project(Xe), Rref)
        dv = np.abs(sp.predict(Xe) - base_speed)
        res[name] = {"probe_err_deg": mg.clip_boot(e_pr, rows_c), "mlp_err_deg": mg.clip_boot(e_ml, rows_c),
                     "nearest_real_R": mg.clip_boot(R, rows_c), "knn5_excess_pca64": mg.clip_boot(kn, rows_c),
                     "speed_leak_mps": mg.clip_boot(dv, rows_c), "edit_norm": float(np.linalg.norm(D, axis=1).mean()),
                     "_e": (e_pr, e_ml, R)}
    diffs = {}
    for a1, a2 in [("fourier2_4d", "spline_interp_pca64"), ("fourier2_4d", "spline_reinsch_cv_pca64"),
                   ("fourier1_2d", "spline_interp_pca64"), ("fourier2_4d", "raw_chord"), ("fourier1_2d", "fourier2_4d"),
                   ("fourier3_6d", "fourier2_4d"), ("fourier2_4d_normmatched_to_reinsch", "spline_reinsch_cv_pca64"),
                   ("spline_interp_pca64", "raw_chord"), ("spline_reinsch_cv_pca64", "raw_chord")]:
        diffs[f"{a1}-{a2}"] = {m: mg.clip_boot(res[a1]["_e"][i] - res[a2]["_e"][i], rows_c)
                               for i, m in enumerate(("probe_err_deg", "mlp_err_deg", "nearest_real_R"))}
    for v in res.values():
        v.pop("_e")
    unedited = mg.circ_err_deg(angle_of(pr.predict(Xc)), tgt)
    out[str(p)] = {"arms": res, "differences": diffs, "unedited_probe_err_deg": float(unedited.mean()),
                   "reinsch_lambda_cv": cvl,
                   "n_rows": int(len(rows_c)), "n_clips": int(len(ci)),
                   "probe_test_mae_deg_kept": float(mg.circ_err_deg(angle_of(pr.predict(X[test])), th[test]).mean())}


def exp3_joint(p, out):
    """Speed set: edits to held-out (direction, speed) cells: within S_dir, between S_dir (+) S_speed, full chord."""
    from wm.manifold import fit_pca
    from wm.vsheet import idw_cell_estimate
    df, f = meta("speed")
    X = load_mp("speed", "vjepa2", p)
    th, v = df.theta_degrees.to_numpy(float), df.speed_mps.to_numpy(float)
    vals = np.unique(v)
    hv = vals[BLOCK_IDX]
    hth = np.isin(np.round(th, 4), np.round(ARC, 4))
    hsp = np.isin(v, hv)
    fit = np.isin(f, (0, 1, 2)) & ~hth & ~hsp
    probe = np.isin(f, (3, 4))
    test = (f < 0) & ~hth & ~hsp
    prd = ridge_fit(X[probe], mg.fourier_features(th[probe], 1))
    prs = ridge_fit(X[probe], v[probe])
    mld = mlp_fit(X[probe], mg.fourier_features(th[probe], 1))
    mls = mlp_fit(X[probe], v[probe])
    _, B = mg.fit_linear_encoding(X[fit], np.column_stack([mg.fourier_features(th[fit], 2), v[fit]]))
    rng = np.random.default_rng(100 + p)
    ci = np.where(test)[0]
    rows_c = np.repeat(ci, 2)
    tt, tv = rng.choice(ARC, len(rows_c)), rng.choice(hv, len(rows_c))
    st, sv = th[rows_c], v[rows_c]
    Xc = X[rows_c]
    f1 = mg.fourier_features(tt, 1) - mg.fourier_features(st, 1)
    f2 = mg.fourier_features(tt, 2)[:, 2:] - mg.fourier_features(st, 2)[:, 2:]
    dv = (tv - sv)[:, None]
    arms = {"a_within_Sdir": f1 @ B[[0, 1]], "a2_within_Sdir+Sdir2": f1 @ B[[0, 1]] + f2 @ B[[2, 3]],
            "b_between_Sdir+Sspeed": f1 @ B[[0, 1]] + dv @ B[[4]],
            "b2_Sdir+Sdir2+Sspeed": f1 @ B[[0, 1]] + f2 @ B[[2, 3]] + dv @ B[[4]]}
    # full-space chord between cell centroids (16 dir bins x 8 speed bins), IDW estimate of the unseen target cell
    dbin = np.round(th / 22.5).astype(int) % 16
    sidx = np.searchsorted(vals, v)
    sbin = sidx // 8
    cells = [(d, s) for d in range(16) for s in range(8) if (fit & (dbin == d) & (sbin == s)).sum() >= 3]
    cd, cs = np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
    C = np.array([X[fit & (dbin == d) & (sbin == s)].mean(0) for d, s in cells])
    cmap = {c: i for i, c in enumerate(cells)}
    tsi = np.searchsorted(vals, tv)
    ct = np.array([idw_cell_estimate(cd, cs, C, a / 22.5 % 16, (b - 3.5) / 8, k=4) for a, b in zip(tt, tsi)])
    csrc = np.array([C[cmap[(dbin[i], sbin[i])]] if (dbin[i], sbin[i]) in cmap else
                     idw_cell_estimate(cd, cs, C, dbin[i], sbin[i], k=4) for i in rows_c])
    arms["c_full_space_cell_chord"] = ct - csrc
    # real target: mean of the 5 nearest real probe/test clips (not the carrier) in (theta/45 deg, v/0.5) at held values
    real = np.where((probe | (f < 0)) & hth & hsp)[0]
    RT = np.empty_like(Xc)
    for j, (a, b, c) in enumerate(zip(tt, tv, rows_c)):
        dd = np.hypot(mg.circ_err_deg(th[real], a) / 45.0, (v[real] - b) / 0.5)
        nn = real[np.argsort(dd)[:5]]
        RT[j] = X[nn].mean(0)
    pca = fit_pca(X[fit], 64)
    Rref = pca.project(X[probe | (f < 0)])
    res = {}
    for name, D in arms.items():
        Xe = Xc + D
        m = {"dir_probe_err_deg": mg.circ_err_deg(angle_of(prd.predict(Xe)), tt),
             "dir_mlp_err_deg": mg.circ_err_deg(angle_of(mld.predict(Xe)), tt),
             "speed_probe_err_mps": np.abs(prs.predict(Xe) - tv), "speed_mlp_err_mps": np.abs(mls.predict(Xe) - tv),
             "nearest_real_R": 1 - np.linalg.norm(Xe - RT, axis=1) / np.linalg.norm(Xc - RT, axis=1),
             "knn5_excess_pca64": knn_excess(pca.project(Xe), Rref)}
        res[name] = {k: mg.clip_boot(x, rows_c) for k, x in m.items()}
        res[name]["edit_norm"] = float(np.linalg.norm(D, axis=1).mean())
        res[name]["_m"] = m
    diffs = {}
    for a1, a2 in [("b_between_Sdir+Sspeed", "c_full_space_cell_chord"), ("b2_Sdir+Sdir2+Sspeed", "c_full_space_cell_chord"),
                   ("b2_Sdir+Sdir2+Sspeed", "b_between_Sdir+Sspeed"), ("a2_within_Sdir+Sdir2", "a_within_Sdir")]:
        diffs[f"{a1}-{a2}"] = {k: mg.clip_boot(res[a1]["_m"][k] - res[a2]["_m"][k], rows_c) for k in res[a1]["_m"]}
    for r in res.values():
        r.pop("_m")
    out[str(p)] = {"arms": res, "differences": diffs, "n_rows": int(len(rows_c)), "held_speeds": jf(hv),
                   "unedited_speed_err_mps": float(np.abs(prs.predict(Xc) - tv).mean()),
                   "unedited_dir_err_deg": float(mg.circ_err_deg(angle_of(prd.predict(Xc)), tt).mean())}


def exp3(parts=("fourier", "joint")):
    out = {"method": (
        "fourier: direction set, held-out contiguous arc 303.75-343.125 deg (8 values); knot folds 0-2 at kept values "
        "fit X ~ a + B phi_K(theta) (K = 1: 2-D ring, K = 2: 4-D ring + second harmonic, K = 3: 6-D), two spline arms "
        "in PCA-64 on value-ordered (labels) knot coordinates at every point: spline_interp = the interpolating periodic "
        "cubic through the raw kept centroids (Goodfire A.3, scipy; matches causalab to 1e-13), spline_reinsch_cv = "
        "causalab's Reinsch smoother with lambda chosen by leave-block-out CV on kept knots only (functions reused from "
        "scripts/run_endpoint_diagnosis.py); the FITPACK smoother is NOT used; and the raw-centroid chord (neighbours 298.125 / "
        "348.75); additive edits h + [f(target) - f(source)], oracle source; each kept-value test clip gets 3 random "
        "arc targets. Readouts on disjoint probe folds 3-4: ridge (cos, sin), MLP (PCA-128, 256 hidden); nearest-real "
        "R vs the probe-fold centroid at the target; kNN-5 excess distance in PCA-64 (on-manifold proxy; NOT the A.7 "
        "Eq. 10 energy, which needs the spline behaviour and is not recomputed here); speed leakage from a speed-set "
        "ridge probe. joint: speed set, held cells = arc x speed block (sorted speed idx 24-31); fitting data exclude "
        "the arc AND the block; edits (a) within S_dir (source speed kept), (a2) S_dir + S_dir2, (b) block-diagonal "
        "S_dir (+) S_speed, (b2) + S_dir2, (c) full-space cell chord (16 dir x 8 speed cells, IDW k = 4 estimate of "
        "the unseen target cell). Real target = mean of the 5 nearest real probe/test clips in the held cells. CIs: "
        "clip bootstrap."), "fourier_dirset": {}, "joint_speedset": {}}
    old = json.loads((PARTS / "exp3.json").read_text()) if (PARTS / "exp3.json").exists() else {}
    if "joint" not in parts and "joint_speedset" in old:
        out["joint_speedset"] = old["joint_speedset"]
    for p in (POINTS if "fourier" in parts else []):
        log("exp3 fourier", p)
        exp3_fourier(p, out["fourier_dirset"])
        write("exp3", out)
    for p in ((12, 19, 22) if "joint" in parts else ()):
        log("exp3 joint", p)
        exp3_joint(p, out["joint_speedset"])
        write("exp3", out)


# ================================================================ exp4 + exp5: coordinate search, shortcuts ======
REGISTRATION = {
    "registered_at": "2026-09-28T17:10-04:00, before any exp4 number was computed",
    "candidates": ["polar", "log_polar", "cartesian", "fourier_polar", "fourier_logpolar", "fourier2_polar"],
    "skipped": {"intrinsic_sheet_spline": "needs a nonparametric 2-D chart of the sheet; the velocity-sheet engineer owns "
                "that fit (p5_velocity_sheet.json), not re-implemented here"},
    "points": POINTS,
    "primary_selection_metric": ("transfer encoding R2: the linear-in-C encoding map X ~ a + phi_C A fit on all speed-set "
                                 "train clips predicts the direction-set constant-velocity clips' (16 dir bin x speed) "
                                 "cell centroids (speeds 1-4 m/s, the overlap; never fit on), each set centred on its own "
                                 "mean; higher is better"),
    "test": ("per point, winner vs each other candidate: paired bootstrap over direction-set clips (300 draws), one-sided "
             "p = share of draws with winner <= other; Holm over all 5 x 7 = 35 comparisons"),
    "secondary_reported_not_selected": ["within-set decode-through-C error (theta MAE, v MAE, 5-fold grouped CV)",
                                        "within-set encoding R2 on held-out cell centroids",
                                        "straight-in-C steering endpoint error and nearest-real R to held-out cells",
                                        "decode transfer speed set -> direction set and back"],
}


def cell_centroids(X, th, v, vbins):
    d = np.round(th / 22.5).astype(int) % 16
    s = np.digitize(v, vbins)
    keys = sorted({(a, b) for a, b in zip(d, s)})
    C, T, V, N = [], [], [], []
    for a, b in keys:
        m = (d == a) & (s == b)
        if m.sum() < 2:
            continue
        C.append(X[m].mean(0)); T.append(mg.circ_mean(th[m]) if hasattr(mg, "circ_mean") else np.degrees(np.angle(np.exp(1j * np.radians(th[m])).mean())) % 360)
        V.append(v[m].mean()); N.append(m.sum())
    return np.array(C), np.array(T), np.array(V), np.array(N)


def enc_r2(A, a0, Cc, T, V, name, center=True):
    Phi = mg.COORDS[name](T, V)
    pred = a0 + Phi @ A
    if center:
        pred, Cc = pred - pred.mean(0), Cc - Cc.mean(0)
    return float(1 - ((Cc - pred) ** 2).sum() / ((Cc - Cc.mean(0)) ** 2).sum())


def coord_block(Xs, dfs, fs, Xd, dfd, fd, p, with_steer=True, seed=0):
    """All candidate coordinate systems at one point. Returns (per-C dict, bootstrap matrix for the primary metric)."""
    from sklearn.model_selection import GroupKFold  # noqa: F401
    th, v = dfs.theta_degrees.to_numpy(float), dfs.speed_mps.to_numpy(float)
    tr, te = fs >= 0, fs < 0
    dvel = (dfd.motion == "velocity").to_numpy() & (dfd.speed_mps.between(1, 4)).to_numpy()
    thd, vd = dfd.theta_degrees.to_numpy(float), dfd.speed_mps.to_numpy(float)
    vbins = np.quantile(v[tr], np.linspace(0, 1, 9)[1:-1])
    # direction-set transfer centroids: bins by exact speed (1..4)
    rng = np.random.default_rng(seed)
    idx_d = np.where(dvel)[0]
    n_boot = 300 if with_steer else 0
    bcent = [cell_centroids(Xd[bi], thd[bi], vd[bi], np.array([1.5, 2.5, 3.5]))
             for bi in (rng.choice(idx_d, len(idx_d)) for _ in range(n_boot))]
    res, boot = {}, {}
    # held-cell design for steering
    vals = np.unique(v)
    hv = vals[BLOCK_IDX]
    hth = np.isin(np.round(th, 4), np.round(ARC, 4)); hsp = np.isin(v, hv)
    fit = np.isin(fs, (0, 1, 2)) & ~hth & ~hsp
    probe = np.isin(fs, (3, 4))
    carriers = np.where(te & ~hth & ~hsp)[0]
    SV_cv = {k: mg.SVDRidge(Xs[tr & (fs != k)]) for k in range(5)}
    SV_tr = mg.SVDRidge(Xs[tr])
    SV_rev = mg.SVDRidge(Xd[dvel & (fd >= 0)])
    if with_steer:
        SV_pr = mg.SVDRidge(Xs[probe])
        _pd = mg.SVDRidge.__new__(mg.SVDRidge); _pd.__dict__.update(SV_pr.__dict__); _pd.fit(mg.fourier_features(th[probe], 1)); prd = _pd
        _ps = mg.SVDRidge.__new__(mg.SVDRidge); _ps.__dict__.update(SV_pr.__dict__); _ps.fit(v[probe]); prs = _ps
        r2 = np.random.default_rng(7)
        rows_c = np.repeat(carriers, 2)
        tt, tv = r2.choice(ARC, len(rows_c)), r2.choice(hv, len(rows_c))
        real = np.where((probe | te) & hth & hsp)[0]
        RT = np.array([Xs[real[np.argsort(np.hypot(mg.circ_err_deg(th[real], a) / 45.0, (v[real] - b) / 0.5))[:5]]].mean(0)
                       for a, b in zip(tt, tv)])
    # within-set held-out centroids for encoding R2 (fit folds 0-2, evaluate folds 3-4 + test)
    ev = ~np.isin(fs, (0, 1, 2))
    Ce, Te, Ve, _ = cell_centroids(Xs[ev], th[ev], v[ev], vbins)
    Cd, Td, Vd, _ = cell_centroids(Xd[dvel], thd[dvel], vd[dvel], np.array([1.5, 2.5, 3.5]))
    for name, fmap in mg.COORDS.items():
        r = {}
        Phi = fmap(th, v)
        # (i) decode-through-C, 5-fold CV over train folds
        oof = np.zeros_like(Phi)
        for k in range(5):
            m = fs == k
            oof[m] = SV_cv[k].fit(Phi[tr & ~m]).predict(Xs[m]).reshape(oof[m].shape)
        t_hat, v_hat = mg.coords_to_state(name, oof[tr])
        r["decode_cv"] = {"theta_mae_deg": float(mg.circ_err_deg(t_hat, th[tr]).mean()),
                          "v_mae_mps": float(np.nan_to_num(np.abs(v_hat - v[tr]), nan=9.0).mean()),
                          "coord_r2_mean": float(np.mean([1 - ((Phi[tr][:, j] - oof[tr][:, j]) ** 2).sum() / ((Phi[tr][:, j] - Phi[tr][:, j].mean()) ** 2).sum() for j in range(Phi.shape[1])]))}
        # decode transfer speed set -> direction set (constant velocity 1-4 m/s) and back
        pm = SV_tr.fit(Phi[tr])
        t2, v2 = mg.coords_to_state(name, pm.predict(Xd[dvel]))
        pmr = SV_rev.fit(fmap(thd[dvel & (fd >= 0)], vd[dvel & (fd >= 0)]))
        sb = te & (v >= 1) & (v <= 4)
        t3, v3 = mg.coords_to_state(name, pmr.predict(Xs[sb]))
        r["decode_transfer"] = {"speed_to_dir": {"theta_mae_deg": float(mg.circ_err_deg(t2, thd[dvel]).mean()),
                                                 "v_mae_mps": float(np.nan_to_num(np.abs(v2 - vd[dvel]), nan=9.0).mean())},
                                "dir_to_speed": {"theta_mae_deg": float(mg.circ_err_deg(t3, th[sb]).mean()),
                                                 "v_mae_mps": float(np.nan_to_num(np.abs(v3 - v[sb]), nan=9.0).mean())}}
        # (ii) encoding linearity, within set (fit folds 0-2 -> other rows' cell centroids)
        a0, A = mg.fit_linear_encoding(Xs[np.isin(fs, (0, 1, 2))], Phi[np.isin(fs, (0, 1, 2))])
        r["encoding_r2_within_heldout_cells"] = enc_r2(A, a0, Ce, Te, Ve, name, center=False)
        # (iv) PRIMARY: transfer encoding R2 (fit all speed-set train clips)
        a1, A1 = mg.fit_linear_encoding(Xs[tr], Phi[tr])
        r["encoding_r2_transfer_to_dirset"] = enc_r2(A1, a1, Cd, Td, Vd, name)
        if n_boot:
            boot[name] = np.array([enc_r2(A1, a1, Cb, Tb, Vb, name) for Cb, Tb, Vb, _ in bcent])
            r["encoding_r2_transfer_ci95"] = jf(np.percentile(boot[name], [2.5, 97.5]))
        # (iii) steering straight in C to held cells
        if with_steer:
            _, Af = mg.fit_linear_encoding(Xs[fit], Phi[fit])
            D = (fmap(tt, tv) - fmap(th[rows_c], v[rows_c])) @ Af
            Xe = Xs[rows_c] + D
            e_t = mg.circ_err_deg(angle_of(prd.predict(Xe)), tt)
            e_v = np.abs(prs.predict(Xe) - tv)
            R = 1 - np.linalg.norm(Xe - RT, axis=1) / np.linalg.norm(Xs[rows_c] - RT, axis=1)
            r["steer_straight_in_C"] = {"theta_err_deg": mg.clip_boot(e_t, rows_c), "v_err_mps": mg.clip_boot(e_v, rows_c),
                                        "nearest_real_R": mg.clip_boot(R, rows_c), "edit_norm": float(np.linalg.norm(D, axis=1).mean())}
        res[name] = r
    return res, boot


def pixel_feats(dataset, train_mask, diff_only=False, pca=None):
    g = np.load(PROJECT_ROOT / "artifacts" / "pixels" / f"{dataset}.npy").astype(np.float32)
    g = g.reshape(len(g), 16, -1)
    F = np.diff(g, axis=1).reshape(len(g), -1) if diff_only else np.concatenate([g.reshape(len(g), -1), np.diff(g, axis=1).reshape(len(g), -1)], 1)
    if pca is None:
        from sklearn.decomposition import PCA
        pca = PCA(256, random_state=0).fit(F[train_mask])
    return pca.transform(F), pca


def exp4(points=POINTS, name="exp4", final=True):
    dfs, fs = meta("speed")
    dfd, fd = meta("direction")
    out = {"registration": REGISTRATION, "points": {}, "random_init": {}, "exp5": {}}
    names = list(mg.COORDS)
    allp, keys = [], []
    for p in points:
        log("exp4", p)
        Xs, Xd = load_mp("speed", "vjepa2", p), load_mp("direction", "vjepa2", p)
        res, boot = coord_block(Xs, dfs, fs, Xd, dfd, fd, p)
        prim = {n: res[n]["encoding_r2_transfer_to_dirset"] for n in names}
        win = max(prim, key=prim.get)
        comp = {}
        for n in names:
            if n == win:
                continue
            pv = float(((boot[win] - boot[n]) <= 0).mean())
            comp[n] = {"diff": prim[win] - prim[n], "diff_ci95": jf(np.percentile(boot[win] - boot[n], [2.5, 97.5])), "p_one_sided": pv}
            allp.append(max(pv, 1.0 / 301)); keys.append((p, n))
        out["points"][str(p)] = {"per_coordinate": res, "winner": win, "vs_winner": comp}
        # random-init comparison (no steering)
        Rs, Rd = load_mp("speed", "random", p), load_mp("direction", "random", p)
        rr, _ = coord_block(Rs, dfs, fs, Rd, dfd, fd, p, with_steer=False)
        out["random_init"][str(p)] = {n: {"encoding_r2_transfer_to_dirset": rr[n]["encoding_r2_transfer_to_dirset"],
                                          "encoding_r2_within_heldout_cells": rr[n]["encoding_r2_within_heldout_cells"],
                                          "decode_cv": rr[n]["decode_cv"]} for n in names}
        out["random_init"][str(p)]["winner"] = max(names, key=lambda n: rr[n]["encoding_r2_transfer_to_dirset"])
        # exp5: position residualisation H_res = (I - P_pos) H (S_pos from the joint model on speed-set train clips)
        _, B = mg.fit_linear_encoding(Xs[fs >= 0], design(dfs, "speed_mps")[0][fs >= 0])
        Qp = mg.orth(B[[5, 6]].T)
        rp, _ = coord_block(Xs - (Xs @ Qp) @ Qp.T, dfs, fs, Xd - (Xd @ Qp) @ Qp.T, dfd, fd, p, with_steer=False)
        out["exp5"].setdefault("position_residualised", {})[str(p)] = {
            n: {"encoding_r2_transfer_to_dirset": rp[n]["encoding_r2_transfer_to_dirset"], "decode_cv": rp[n]["decode_cv"],
                "decode_transfer": rp[n]["decode_transfer"]} for n in names}
        out["exp5"]["position_residualised"][str(p)]["winner"] = max(names, key=lambda n: rp[n]["encoding_r2_transfer_to_dirset"])
        write(name, out)
    if not final:
        return
    exp4_final(out)


def exp4_final(out=None):
    """Combine exp4 point parts (exp4.json, exp4b.json), apply Holm over every (point x coordinate) comparison, add
    the pixel and position-only baselines, write exp4.json."""
    dfs, fs = meta("speed")
    dfd, fd = meta("direction")
    if out is None:
        out = json.loads((PARTS / "exp4.json").read_text())
        if (PARTS / "exp4b.json").exists():
            b = json.loads((PARTS / "exp4b.json").read_text())
            for k in ("points", "random_init"):
                out[k].update(b[k])
            out["exp5"].setdefault("position_residualised", {}).update(b["exp5"].get("position_residualised", {}))
    allp, keys = [], []
    for p in POINTS:
        if str(p) not in out["points"]:
            continue
        for n, c in out["points"][str(p)]["vs_winner"].items():
            allp.append(max(c["p_one_sided"], 1.0 / 301)); keys.append((p, n))
    POINTS_DONE = [p for p in POINTS if str(p) in out["points"]]
    adj = mg.holm(allp)
    for (p, n), a in zip(keys, adj):
        out["points"][str(p)]["vs_winner"][n]["p_holm"] = float(a)
    out["winner_by_point"] = {str(p): out["points"][str(p)]["winner"] for p in POINTS_DONE}
    out["holm_family_size"] = len(allp)
    # the empirical bootstrap p is floored at 1/301 (300 draws), so Holm x35 cannot go below 0.116; add a normal-
    # approximation p from the paired-bootstrap SE (SE = CI width / 3.92) and Holm it over the same family.
    from scipy.stats import norm
    zp = []
    for p, n in keys:
        c = out["points"][str(p)]["vs_winner"][n]
        se = max((c["diff_ci95"][1] - c["diff_ci95"][0]) / 3.92, 1e-12)
        c["z"] = float(c["diff"] / se)
        zp.append(float(norm.sf(c["z"])))
    for (p, n), a, q in zip(keys, mg.holm(zp), zp):
        out["points"][str(p)]["vs_winner"][n]["p_normal_one_sided"] = q
        out["points"][str(p)]["vs_winner"][n]["p_holm_normal"] = float(a)
    out["holm_note"] = ("p_holm uses the empirical bootstrap p (floor 1/301, so min Holm-adjusted value 35/301 = 0.116); "
                        "p_holm_normal uses z = diff / bootstrap SE; both over the 35-comparison family")
    # exp5: pixel baselines and position-only probes (point-free)
    log("exp5 pixels")
    tr = fs >= 0
    th, v = dfs.theta_degrees.to_numpy(float), dfs.speed_mps.to_numpy(float)
    dvel = (dfd.motion == "velocity").to_numpy() & (dfd.speed_mps.between(1, 4)).to_numpy()
    pix = {}
    for kind, diff_only in (("pixels_plus_diff", False), ("pixel_diff_only", True)):
        Fs, pca = pixel_feats("speed", tr, diff_only)
        Fd, _ = pixel_feats("direction", tr, diff_only, pca)
        pix[kind] = {}
        for n, fmap in mg.COORDS.items():
            Phi = fmap(th, v)
            oof = np.zeros_like(Phi)
            for k in range(5):
                m = fs == k
                oof[m] = ridge_fit(Fs[tr & ~m], Phi[tr & ~m]).predict(Fs[m]).reshape(oof[m].shape)
            t_hat, v_hat = mg.coords_to_state(n, oof[tr])
            t2, v2 = mg.coords_to_state(n, ridge_fit(Fs[tr], Phi[tr]).predict(Fd[dvel]))
            pix[kind][n] = {"decode_cv": {"theta_mae_deg": float(mg.circ_err_deg(t_hat, th[tr]).mean()),
                                          "v_mae_mps": float(np.nan_to_num(np.abs(v_hat - v[tr]), nan=9.0).mean())},
                            "decode_transfer_speed_to_dir": {"theta_mae_deg": float(mg.circ_err_deg(t2, dfd.theta_degrees.to_numpy(float)[dvel]).mean()),
                                                             "v_mae_mps": float(np.nan_to_num(np.abs(v2 - dfd.speed_mps.to_numpy(float)[dvel]), nan=9.0).mean())}}
    out["exp5"]["pixels"] = pix
    P0 = np.column_stack([dfs.start_x, dfs.start_y, dfs.start_x ** 2, dfs.start_y ** 2, dfs.start_x * dfs.start_y])
    pos_only = {}
    for n, fmap in mg.COORDS.items():
        Phi = fmap(th, v)
        oof = np.zeros_like(Phi)
        for k in range(5):
            m = fs == k
            oof[m] = ridge_fit(P0[tr & ~m], Phi[tr & ~m]).predict(P0[m]).reshape(oof[m].shape)
        t_hat, v_hat = mg.coords_to_state(n, oof[tr])
        pos_only[n] = {"theta_mae_deg": float(mg.circ_err_deg(t_hat, th[tr]).mean()),
                       "v_mae_mps": float(np.nan_to_num(np.abs(v_hat - v[tr]), nan=9.0).mean())}
    out["exp5"]["start_position_only_quadratic"] = pos_only
    out["exp5"]["chance"] = {"theta_mae_deg": 90.0, "v_mae_mps_predict_mean": float(np.abs(v[tr] - v[tr].mean()).mean())}
    write("exp4", out)


# ================================================================ exp2: interference as superposition ============
def step_fit(H, df, f, svar, fit_clip):
    """Joint per-step OLS on fit clips, steps != HELD_STEP: rows ~ [fourier2, s, x0, y0, onehot(step 1..7 minus 5)].
    Returns dict of coefficient blocks, bases and the time centroid vectors m_t (m_0 = 0, m_5 = chord estimate)."""
    steps = [t for t in range(8) if t != HELD_STEP]
    ci = np.where(fit_clip)[0]
    Hr = H[ci][:, steps].reshape(-1, H.shape[-1]).astype(np.float64)
    F, _ = design(df.iloc[ci], svar)
    Fr = np.repeat(F, len(steps), 0)
    st = np.tile(steps, len(ci))
    oh = np.column_stack([(st == t).astype(float) for t in steps if t != 0])
    _, B = mg.fit_linear_encoding(Hr, np.column_stack([Fr, oh]))
    m = {0: np.zeros(H.shape[-1])}
    for j, t in enumerate([t for t in steps if t != 0]):
        m[t] = B[7 + j]
    m[HELD_STEP] = 0.5 * (m[HELD_STEP - 1] + m[HELD_STEP + 1])
    M = np.array([m[t] for t in steps])
    Qt, kt, _ = time_basis(M[None])
    return {"B": B, "m": m, "Q": {"dir": mg.orth(B[[0, 1]].T), "dir2": mg.orth(B[[2, 3]].T), "s": mg.orth(B[[4]].T),
                                  "pos": mg.orth(B[[5, 6]].T), "time": Qt}, "k_time": kt}


def exp2_set(dataset, p, other_probe, other_axis_Q):
    svar = SVAR[dataset]
    df, f = meta(dataset)
    H = load_tp(dataset, "vjepa2", p)
    th, sv = df.theta_degrees.to_numpy(float), df[svar].to_numpy(float)
    vals = np.unique(sv)
    hv = vals[BLOCK_IDX]
    hth = np.isin(np.round(th, 4), np.round(ARC, 4)); hs = np.isin(sv, hv)
    fitc = np.isin(f, (0, 1, 2)) & ~hth & ~hs
    SF = step_fit(H, df, f, svar, fitc)
    B, Q = SF["B"], SF["Q"]
    # readouts on probe clips, all steps
    pc = np.where(np.isin(f, (3, 4)))[0]
    Hp = H[pc].reshape(-1, H.shape[-1]).astype(np.float64)
    rep = lambda a: np.repeat(np.asarray(a, float), 8, 0)
    tp = np.tile(np.arange(8), len(pc))
    RD = {"dir": ridge_fit(Hp, rep(mg.fourier_features(th[pc], 1))), "s": ridge_fit(Hp, rep(sv[pc])),
          "time": ridge_fit(Hp, tp.astype(float)),
          "pos": ridge_fit(Hp, rep(np.column_stack([df.start_x.to_numpy()[pc], df.start_y.to_numpy()[pc]]))),
          "other_scalar": other_probe}
    MLP = {"dir": mlp_fit(Hp, rep(mg.fourier_features(th[pc], 1))), "s": mlp_fit(Hp, rep(sv[pc]))}
    # carriers
    rng = np.random.default_rng(10 + p)
    cc = np.where((f < 0) & ~hth & ~hs)[0]
    steps = np.array([1, 3, 6])
    rc = np.repeat(cc, len(steps)); rt = np.tile(steps, len(cc))
    Xc = H[rc, rt].astype(np.float64)
    tt, ts = rng.choice(ARC, len(rc)), rng.choice(hv, len(rc))
    other = rng.choice(cc, len(rc))
    pt = np.column_stack([df.start_x.to_numpy()[other], df.start_y.to_numpy()[other]])
    ps = np.column_stack([df.start_x.to_numpy()[rc], df.start_y.to_numpy()[rc]])
    f1 = mg.fourier_features(tt, 1) - mg.fourier_features(th[rc], 1)
    f2 = mg.fourier_features(tt, 2)[:, 2:] - mg.fourier_features(th[rc], 2)[:, 2:]
    dt = np.array([SF["m"][HELD_STEP] - SF["m"][t] for t in rt])
    edits = {"dir": f1 @ B[[0, 1]], "dir2": f2 @ B[[2, 3]], "s": (ts - sv[rc])[:, None] @ B[[4]],
             "time": (dt @ Q["time"]) @ Q["time"].T, "pos": (pt - ps) @ B[[5, 6]]}

    def read(Xe):
        return {"dir": angle_of(RD["dir"].predict(Xe)), "s": RD["s"].predict(Xe), "time": RD["time"].predict(Xe),
                "pos": RD["pos"].predict(Xe), "other_scalar": RD["other_scalar"].predict(Xe),
                "dir_mlp": angle_of(MLP["dir"].predict(Xe)), "s_mlp": MLP["s"].predict(Xe)}

    def change(r0, r1):
        return {"dir": mg.circ_err_deg(r1["dir"], r0["dir"]), "s": np.abs(r1["s"] - r0["s"]),
                "time": np.abs(r1["time"] - r0["time"]), "pos": np.linalg.norm(r1["pos"] - r0["pos"], axis=1),
                "other_scalar": np.abs(r1["other_scalar"] - r0["other_scalar"]),
                "dir_mlp": mg.circ_err_deg(r1["dir_mlp"], r0["dir_mlp"]), "s_mlp": np.abs(r1["s_mlp"] - r0["s_mlp"])}
    r0 = read(Xc)
    truth_err = {"dir": mg.circ_err_deg(r0["dir"], th[rc]), "s": r0["s"] - sv[rc], "time": r0["time"] - rt,
                 "pos": np.linalg.norm(r0["pos"] - ps, axis=1), "other_scalar": r0["other_scalar"] - r0["other_scalar"].mean(),
                 "dir_mlp": mg.circ_err_deg(r0["dir_mlp"], th[rc]), "s_mlp": r0["s_mlp"] - sv[rc]}
    spread = {k: float(np.sqrt(np.mean(np.asarray(v) ** 2))) for k, v in truth_err.items()}
    out = {"k_time": SF["k_time"], "natural_spread_rms": spread, "edit_norm_natural": {}, "natural": {}, "matched": {}}
    ref = np.linalg.norm(edits["dir"], axis=1, keepdims=True)
    for e, D in edits.items():
        out["edit_norm_natural"][e] = float(np.linalg.norm(D, axis=1).mean())
        for mode, DD in (("natural", D), ("matched", D * ref / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12))):
            ch = change(r0, read(Xc + DD))
            out[mode][e] = {g: mg.clip_boot(ch[g] / spread[g], rc, n_boot=300) for g in ch}
    # on-target accuracy (natural edits)
    r_dir = read(Xc + edits["dir"])
    out["on_target_natural"] = {"dir_err_deg": mg.clip_boot(mg.circ_err_deg(r_dir["dir"], tt), rc, 300),
                                "s_err": mg.clip_boot(np.abs(read(Xc + edits["s"])["s"] - ts), rc, 300),
                                "time_err_steps": mg.clip_boot(np.abs(read(Xc + edits["time"])["time"] - HELD_STEP), rc, 300)}
    # overlaps for the leakage-prediction test
    SG = {"dir": Q["dir"], "s": Q["s"], "time": Q["time"], "pos": Q["pos"], "other_scalar": other_axis_Q}

    def wdir(model):
        lr = model[-1]; sc = model[0].scale_
        return mg.orth((np.atleast_2d(lr.coef_) / sc).T)
    DG = {g: wdir(RD[g]) for g in ("dir", "s", "time", "pos", "other_scalar")}
    out["overlap_subspace"] = {e: {g: mg.overlap(Q[e], SG[g]) for g in SG} for e in Q}
    out["overlap_decoder"] = {e: {g: mg.overlap(Q[e], DG[g]) for g in DG} for e in Q}
    return out, SF, RD, MLP, (H, df, f, th, sv, rc, rt, Xc, tt, fitc, hv)


def exp2():
    from scipy.stats import spearmanr
    res = {"method": (
        "Per-step timepool features (8 steps per clip), points 12 and 22, speed set and acceleration set separately. "
        "Joint OLS per-step encoding on fit clips (folds 0-2, held-out arc 303.75-343.125, held interior block of 8 "
        "speed/acc values and held step 5 excluded): rows ~ [cos t, sin t, cos 2t, sin 2t, s, x0, y0, one-hot step]. "
        "S_time = top-k PCs of the step coefficient vectors. Edits (additive, in-subspace, to held-out targets): dir "
        "(ring plane), dir2 (harmonic plane, same theta targets), s (speed or acceleration axis), time (to held step 5, "
        "chord estimate m5 = (m4 + m6) / 2, projected on S_time), pos (to another test clip's start). 'matched' rescales "
        "every edit to the direction edit's norm per row. Readouts: ridge probes on probe clips (folds 3-4, all steps) "
        "for direction, s, step, start position, plus the OTHER set's scalar probe (acceleration on the speed set, "
        "speed on the acceleration set) and MLPs for direction and s. Leakage = mean |readout change| / natural spread "
        "(RMS error of that readout on the unedited carriers; for the other-set scalar, its SD). Carriers: test clips "
        "outside the held values, steps 1, 3, 6. Overlap predictors: subspace overlap (mean cos^2 of principal angles) "
        "between the edit subspace and the readout variable's encoding subspace, and between the edit subspace and the "
        "readout probe's weight vectors (decoder)."), "sets": {}}
    for p in (12, 22):
        log("exp2", p)
        # cross probes: fit each set's scalar probe on its probe clips (all steps) for use on the other set
        cross, axes = {}, {}
        for ds in ("speed", "acceleration"):
            df, f = meta(ds)
            H = load_tp(ds, "vjepa2", p)
            pc = np.where(np.isin(f, (3, 4)))[0]
            cross[ds] = ridge_fit(H[pc].reshape(-1, H.shape[-1]), np.repeat(df[SVAR[ds]].to_numpy(float)[pc], 8))
            fitc = np.isin(f, (0, 1, 2))
            axes[ds] = step_fit(H, df, f, SVAR[ds], fitc)["Q"]["s"]
            del H
        for ds, other in (("speed", "acceleration"), ("acceleration", "speed")):
            log("exp2", p, ds)
            o, SF, RD, MLP, ctx = exp2_set(ds, p, cross[other], axes[other])
            res["sets"].setdefault(ds, {})[str(p)] = o
            if ds == "speed":
                res.setdefault("residualised", {})[str(p)] = exp2_residualise(SF, RD, MLP, ctx)
            write("exp2", res)
    # leakage vs overlap across off-diagonal pairs
    xs_sub, xs_dec, ys = [], [], []
    gmap = {"dir": "dir", "dir2": "dir", "s": "s", "time": "time", "pos": "pos"}
    for ds in res["sets"]:
        for p, o in res["sets"][ds].items():
            for e in o["matched"]:
                for g in ("dir", "s", "time", "pos", "other_scalar"):
                    if gmap[e] == g:
                        continue
                    ys.append(o["matched"][e][g]["mean"])
                    xs_sub.append(o["overlap_subspace"][e][g]); xs_dec.append(o["overlap_decoder"][e][g])
    r1, r2 = spearmanr(xs_sub, ys), spearmanr(xs_dec, ys)
    res["leakage_vs_overlap"] = {"n_pairs": len(ys), "spearman_subspace_overlap": {"rho": float(r1.correlation), "p": float(r1.pvalue)},
                                 "spearman_decoder_overlap": {"rho": float(r2.correlation), "p": float(r2.pvalue)}}
    write("exp2", res)


def exp2_residualise(SF, RD, MLP, ctx):
    H, df, f, th, sv, rc, rt, Xc, tt, fitc, hv = ctx
    B, Q = SF["B"], SF["Q"]
    f1 = mg.fourier_features(tt, 1) - mg.fourier_features(th[rc], 1)
    f2 = mg.fourier_features(tt, 2)[:, 2:] - mg.fourier_features(th[rc], 2)[:, 2:]
    # raw chord: per-step-pooled centroids of fit clips at each theta (steps != 5)
    steps = [t for t in range(8) if t != HELD_STEP]
    cfull = {}
    for v in np.unique(th[fitc]):
        ci = np.where(fitc & np.isclose(th, v))[0]
        cfull[v] = H[ci][:, steps].reshape(-1, H.shape[-1]).astype(np.float64).mean(0)
    a, b = ARC_NEIGH
    w = ((tt - a) / (b - a))[:, None]
    base = {"fourier_ring_2d": f1 @ B[[0, 1]], "fourier_4d": f1 @ B[[0, 1]] + f2 @ B[[2, 3]],
            "raw_chord": (1 - w) * cfull[a] + w * cfull[b] - np.array([cfull[v] for v in th[rc]])}
    lr, sc = RD["s"][-1], RD["s"][0].scale_
    wprobe = mg.orth((lr.coef_ / sc)[:, None])
    nuis = {"none": None, "S_speed": Q["s"], "S_speed+S_pos+S_time": mg.orth(np.hstack([Q["s"], Q["pos"], Q["time"]])),
            "orthogonal_to_speed_probe_weight (reader; circular for the probe readout)": wprobe,
            "oblique_speed (along encoder b_s, zeroes the speed probe; circular for the probe readout)": ("obl", B[[4]].T, (lr.coef_ / sc)[None, :]),
            "oblique_speed+pos (encoders b_s, B_pos; zeroes speed and position probes)": ("obl", np.hstack([B[[4]].T, B[[5, 6]].T]),
                np.vstack([(lr.coef_ / sc)[None, :], RD["pos"][-1].coef_ / RD["pos"][0].scale_]))}
    # real target: probe/test rows (not the carrier clip) at the target theta, same step, speed octile
    vals = np.unique(sv)
    oct_ = np.searchsorted(vals, sv) // 8
    pool = np.where(~fitc & ~np.isin(sv, hv))[0]
    RT = np.empty_like(Xc)
    for j, (t, c, st) in enumerate(zip(tt, rc, rt)):
        m = pool[np.isclose(th[pool], t) & (oct_[pool] == oct_[c]) & (pool != c)]
        if len(m) == 0:
            m = pool[np.isclose(th[pool], t) & (pool != c)]
        RT[j] = H[m, st].astype(np.float64).mean(0)
    s0, sm0 = RD["s"].predict(Xc), MLP["s"].predict(Xc)
    spread_s = float(np.sqrt(np.mean((s0 - sv[rc]) ** 2))); spread_sm = float(np.sqrt(np.mean((sm0 - sv[rc]) ** 2)))
    out = {"speed_natural_spread_rms_probe": spread_s, "speed_natural_spread_rms_mlp": spread_sm}
    for bn, D in base.items():
        per = {}
        for nn, Qn in nuis.items():
            if Qn is None:
                DD = D
            elif isinstance(Qn, tuple):
                DD = mg.oblique_residualise(D, Qn[1], Qn[2])
            else:
                DD = mg.residualise(D, Qn)
            Xe = Xc + DD
            m = {"dir_probe_err_deg": mg.circ_err_deg(angle_of(RD["dir"].predict(Xe)), tt),
                 "dir_mlp_err_deg": mg.circ_err_deg(angle_of(MLP["dir"].predict(Xe)), tt),
                 "speed_leak_probe_over_spread": np.abs(RD["s"].predict(Xe) - s0) / spread_s,
                 "speed_leak_mlp_over_spread": np.abs(MLP["s"].predict(Xe) - sm0) / spread_sm,
                 "nearest_real_R": 1 - np.linalg.norm(Xe - RT, axis=1) / np.linalg.norm(Xc - RT, axis=1)}
            if Qn is None or isinstance(Qn, tuple):
                sh = 0.0 if Qn is None else float((np.linalg.norm(D - mg.oblique_residualise(D, Qn[1], Qn[2], renorm=False), axis=1) / np.linalg.norm(D, axis=1)).mean())
            else:
                sh = float((np.linalg.norm((D @ Qn) @ Qn.T, axis=1) / np.linalg.norm(D, axis=1)).mean())
            per[nn] = {"_m": m, "share_of_edit_removed": sh}
        o = {}
        for nn in per:
            o[nn] = {k: mg.clip_boot(v, rc, 300) for k, v in per[nn]["_m"].items()}
            o[nn]["share_of_edit_norm_removed_before_renorm"] = per[nn]["share_of_edit_removed"]
            if nn != "none":
                o[nn]["minus_none"] = {k: mg.clip_boot(per[nn]["_m"][k] - per["none"]["_m"][k], rc, 300) for k in per[nn]["_m"]}
        out[bn] = o
    return out


def exp1w():
    """Principal angles in the noise-whitened metric: PCA-128 of pooled speed+acceleration train meanpool, whiten by the
    pooled residual covariance of the two joint encoding models (shrinkage 0.1 toward the mean eigenvalue). In this
    metric a direction's length is its signal-to-noise, and random subspaces are near-orthogonal (null overlap ~ k/128),
    so overlap here is the interference that matters for a readout."""
    dfs, fs = meta("speed"); dfa, fa = meta("acceleration")
    trs, tra = fs >= 0, fa >= 0
    out = {"method": exp1w.__doc__, "points": {}}
    rng = np.random.default_rng(1)
    for model in ("vjepa2", "random"):
        for p in POINTS:
            log("exp1w", model, p)
            Xs, Xa = load_mp("speed", model, p)[trs], load_mp("acceleration", model, p)[tra]
            Fs, Fa = design(dfs[trs], "speed_mps")[0], design(dfa[tra], "acceleration_mps2")[0]
            from wm.manifold import fit_pca
            pca = fit_pca(np.vstack([Xs, Xa]), 128)
            Zs, Za = pca.project(Xs), pca.project(Xa)
            a_s, Bs = mg.fit_linear_encoding(Zs, Fs); a_a, Ba = mg.fit_linear_encoding(Za, Fa)
            E = np.vstack([Zs - a_s - Fs @ Bs, Za - a_a - Fa @ Ba])
            C = np.cov(E.T)
            C = 0.9 * C + 0.1 * np.trace(C) / len(C) * np.eye(len(C))
            w, V = np.linalg.eigh(C)
            W = V @ np.diag(w ** -0.5) @ V.T
            G = {"dir": Bs[[0, 1]], "dir2": Bs[[2, 3]], "speed": Bs[[4]], "acc": Ba[[4]], "pos": Bs[[5, 6]]}
            if model == "vjepa2":
                T = load_tp("speed", model, p)[trs].astype(np.float64)
                R = T - T.mean(1, keepdims=True)
                M = R.mean(0); M = M - M.mean(0)
                _, sv, Vt = np.linalg.svd(M @ pca.components.T, full_matrices=False)
                G["time"] = Vt[:2]
            Q = {k: mg.orth((g @ W).T) for k, g in G.items()}
            res = {"pairs": pair_table(Q), "union": mg.union_dimension(list(Q.values())),
                   "snr_norm": {k: jf(np.linalg.norm(g @ W, axis=1)) for k, g in G.items()}}
            res["union"]["members"] = list(Q)
            nulls = []
            for _ in range(200):
                a1, a2 = mg.orth(rng.normal(size=(128, 2))), mg.orth(rng.normal(size=(128, 2)))
                nulls.append(mg.overlap(a1, a2))
            res["null_isotropic_2x2_overlap"] = {"mean": float(np.mean(nulls)), "p95": float(np.percentile(nulls, 95))}
            # bootstrap CIs (refit on resampled train clips)
            boot = {}
            for b in range(50):
                i_s, i_a = rng.integers(0, len(Xs), len(Xs)), rng.integers(0, len(Xa), len(Xa))
                _, Bsb = mg.fit_linear_encoding(Zs[i_s], Fs[i_s]); _, Bab = mg.fit_linear_encoding(Za[i_a], Fa[i_a])
                Gb = {"dir": Bsb[[0, 1]], "dir2": Bsb[[2, 3]], "speed": Bsb[[4]], "acc": Bab[[4]], "pos": Bsb[[5, 6]]}
                if "time" in G:
                    Gb["time"] = G["time"]
                for k, v in pair_table({k: mg.orth((g @ W).T) for k, g in Gb.items()}).items():
                    boot.setdefault(k, []).append(v["overlap"])
            for k, v in boot.items():
                res["pairs"][k]["overlap_ci95"] = jf(np.percentile(v, [2.5, 97.5]))
            out["points"].setdefault(model, {})[str(p)] = res
        write("exp1w", out)


if __name__ == "__main__":
    {"exp1": exp1, "exp3": exp3, "exp4": exp4, "exp2": exp2, "exp1w": exp1w, "exp3fourier": lambda: exp3(("fourier",)), "exp4b": lambda: exp4([16, 19, 22], "exp4b", False), "exp4final": exp4_final}[sys.argv[1]]()

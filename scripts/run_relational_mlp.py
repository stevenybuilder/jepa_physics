"""Relational structure beyond composition (non-linear targets): relative speed |v1 - v2| and closing rate
-(d(T) - d(0)) / T of the inter-disk centre distance d(t) = |p1(t) - p2(t)| (T = 15/24 s; positive = approaching),
both non-linear in the single-disk variables.

Per source (V-JEPA 2 / random-init meanpool, 26 points; pixels) and target, on the stored split (509 train / 128 test):
  lin_acts   ridge on the activations (wm.probes recipe: standardise on train, alpha by fold-mean R^2 on the 5 folds)
  mlp_acts   MLP 2 x 256 ReLU (sklearn MLPRegressor, adam, L2 alpha in {1e-3, 1e-1, 10} by 5-fold CV on the train
             folds, max_iter 1000, seed 0) on the standardised activations
  lin_null / mlp_null   the same probes on the concatenated single-disk readouts of the SAME point: ridge predictions
             of v1, v2, x1, x2, y1, y2 (mid-clip, frame 7.5) -- out-of-fold for the train clips, refit-on-train for the
             test clips, so the null never sees test labels (the compositional null)
  mlp_null_v the MLP on the (v1, v2) readouts only
  mlp_oracle the MLP on the true (v1, v2, x1, x2, y1, y2): the ceiling of the null family
Score: test R^2 (128 held-out clips) with 1000-draw clip-bootstrap CI; paired CI of mlp_acts - mlp_null.
Relational structure beyond composition = mlp_acts beats mlp_null on held-out clips (CI excludes 0).

  python scripts/run_relational_mlp.py [--workers 32]  -> artifacts/relational/mlp.json (merged into
  results/p5_relational_motion.json key "beyond_composition")
"""
import argparse
import importlib.util
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT  # noqa: E402
from wm.metrics import r2  # noqa: E402
from wm.probes import Standardizer, fit_ridge, predict, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("rrm", Path(__file__).with_name("run_relational_motion.py"))
rrm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrm)

CACHE = PROJECT_ROOT / "artifacts" / "relational" / "mlp.json"
MLP_ALPHAS = [1e-3, 1e-1, 10.0]
SINGLE = ["v1", "v2", "x1", "x2", "y1", "y2"]
TARGETS = ["rel_speed", "closing_rate"]


def design():
    df = rrm.table()
    rows = []
    for i in df["id"]:
        m = json.loads((rrm.ROOT / "videos" / f"scene_{i:04d}" / "metadata.json").read_text())
        T = 15 / 24
        p = {}
        for d in ("disk1", "disk2"):
            x0, y = m[d]["start_xy_m"]
            p[d] = (x0, y, m[d]["vx_mps"])
        d0 = np.hypot(p["disk1"][0] - p["disk2"][0], p["disk1"][1] - p["disk2"][1])
        dT = np.hypot(p["disk1"][0] + p["disk1"][2] * T - p["disk2"][0] - p["disk2"][2] * T, p["disk1"][1] - p["disk2"][1])
        tm = 7.5 / 24
        rows.append({"x1": p["disk1"][0] + p["disk1"][2] * tm, "x2": p["disk2"][0] + p["disk2"][2] * tm,
                     "y1": p["disk1"][1], "y2": p["disk2"][1], "closing_rate": -(dT - d0) / T})
    for k in rows[0]:
        df[k] = [r[k] for r in rows]
    df["rel_speed"] = df["abs_v_rel"]
    return df


def mlp_fit_predict(Xtr, ytr, Xte, alpha):
    from sklearn.neural_network import MLPRegressor
    mu, sd = ytr.mean(), ytr.std() + 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MLPRegressor(hidden_layer_sizes=(256, 256), alpha=alpha, max_iter=1000, random_state=0,
                         learning_rate_init=1e-3).fit(Xtr, (ytr - mu) / sd)
    return m.predict(Xte) * sd + mu


def mlp_cv(Xtr, ytr, Xte, folds):
    """alpha by fold-mean R^2 over the train folds, refit on all train, predict test."""
    best = None
    for a in MLP_ALPHAS:
        s = []
        for k in np.unique(folds):
            v = folds == k
            s.append(r2(ytr[v], mlp_fit_predict(Xtr[~v], ytr[~v], Xtr[v], a)))
        if best is None or np.mean(s) > best[0]:
            best = (float(np.mean(s)), a)
    return mlp_fit_predict(Xtr, ytr, Xte, best[1]), {"alpha": best[1], "cv_mean": best[0]}


def lin_cv(Xtr, Ytr, Xte, folds):
    cvs = rrm.cv_multi(Xtr, Ytr, folds)
    oof = np.stack([c["oof"] for c in cvs], 1)
    te = np.stack([predict(Xte, *fit_ridge(Xtr, Ytr[:, j], c["alpha"]))[:, 0] for j, c in enumerate(cvs)], 1)
    return oof, te, cvs


def task(args):
    src, p, Xtr, Xte, D, folds, idx = args
    S_tr, S_te, T_tr, T_te = D
    out = {"point": p, "targets": {}}
    oof_s, te_s, _ = lin_cv(Xtr, S_tr, Xte, folds)                    # single-disk readouts (compositional null)
    st = Standardizer().fit(oof_s)
    N_tr, N_te = st.transform(oof_s), st.transform(te_s)
    so = Standardizer().fit(S_tr)
    O_tr, O_te = so.transform(S_tr), so.transform(S_te)
    out["single_readout_test_r2"] = {v: r2(S_te[:, j], te_s[:, j]) for j, v in enumerate(SINGLE)}
    _, lin_te, lin_cvs = lin_cv(Xtr, T_tr, Xte, folds)
    _, nlin_te, _ = lin_cv(N_tr, T_tr, N_te, folds)
    for j, t in enumerate(TARGETS):
        y, yt = T_tr[:, j], T_te[:, j]
        preds = {"lin_acts": lin_te[:, j], "lin_null": nlin_te[:, j]}
        info = {}
        preds["mlp_acts"], info["mlp_acts"] = mlp_cv(Xtr, y, Xte, folds)
        preds["mlp_null"], info["mlp_null"] = mlp_cv(N_tr, y, N_te, folds)
        preds["mlp_null_v"], info["mlp_null_v"] = mlp_cv(N_tr[:, :2], y, N_te[:, :2], folds)
        preds["mlp_oracle"], info["mlp_oracle"] = mlp_cv(O_tr, y, O_te, folds)
        rec = {}
        for k, pr in preds.items():
            rec[k] = {"test_r2": r2(yt, pr), "test_r2_ci": rrm.ci(rrm.boot_r2(yt, pr, idx)), **info.get(k, {})}
        d = rrm.boot_r2(yt, preds["mlp_acts"], idx) - rrm.boot_r2(yt, preds["mlp_null"], idx)
        rec["mlp_acts_minus_mlp_null"] = {"value": rec["mlp_acts"]["test_r2"] - rec["mlp_null"]["test_r2"],
                                          "ci": rrm.ci(d)}
        dl = rrm.boot_r2(yt, preds["lin_acts"], idx) - rrm.boot_r2(yt, preds["lin_null"], idx)
        rec["lin_acts_minus_lin_null"] = {"value": rec["lin_acts"]["test_r2"] - rec["lin_null"]["test_r2"],
                                          "ci": rrm.ci(dl)}
        out["targets"][t] = rec
    print(src, p, {t: (round(out["targets"][t]["mlp_acts"]["test_r2"], 3), round(out["targets"][t]["mlp_null"]["test_r2"], 3))
                   for t in TARGETS}, flush=True)
    return src, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    df = design()
    split = json.loads(rrm.SPLIT.read_text())["relational"]
    pos = {int(i): r for r, i in enumerate(df["id"])}
    tr = np.array([pos[i] for i in split["train_ids"]])
    te = np.array([pos[i] for i in split["test_ids"]])
    folds = np.array([split["fold"][str(i)] for i in split["train_ids"]])
    S, T = df[SINGLE].to_numpy(float), df[TARGETS].to_numpy(float)
    D = (S[tr], S[te], T[tr], T[te])
    idx = rrm.boot_idx(len(te))
    jobs = []
    for src in ("vjepa2", "random"):
        acts = np.load(rrm.ACT / src / "meanpool.npy")
        for p in range(acts.shape[1]):
            Xtr, Xte = rrm.std_pair(acts[:, p].astype(np.float64), tr, te)
            jobs.append((src, p, Xtr, Xte, D, folds, idx))
    if rrm.PIX.exists():
        from wm.pixels import features
        Xtr, Xte = features("pixels", np.load(rrm.PIX), None, tr, te)
        jobs.append(("pixels", 0, Xtr, Xte, D, folds, idx))
    with Pool(a.workers) as pool:
        res = pool.map(task, jobs, chunksize=1)
    out = {"targets": {"rel_speed": "|v1 - v2| (m/s)",
                       "closing_rate": "-(d(T) - d(0)) / T, d = centre distance incl. vertical offset (m/s)"},
           "method": __doc__.split("\n\n")[1], "sources": {}}
    for src, r in res:
        out["sources"].setdefault(src, []).append(r)
    for src in out["sources"]:
        out["sources"][src].sort(key=lambda r: r["point"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    path = PROJECT_ROOT / "results" / "p5_relational_motion.json"
    if path.exists():
        res_all = json.loads(path.read_text())
        res_all.pop("provenance", None)
        res_all["beyond_composition"] = out
        res_all["keys"]["beyond_composition"] = ("sources.{vjepa2,random,pixels}[p].targets.{rel_speed,closing_rate}: "
                                                 "lin_acts, lin_null, mlp_acts, mlp_null, mlp_null_v, mlp_oracle "
                                                 "(test_r2 + CI, alpha, cv_mean) and mlp_acts_minus_mlp_null / "
                                                 "lin_acts_minus_lin_null (paired CI)")
        write_json(path, res_all, rrm.SPLIT)


if __name__ == "__main__":
    main()

"""E4 relational motion: does the encoder carry relative velocity v_rel = v1 - v2 between two disks beyond each disk's
absolute motion? Probing only (CPU), on artifacts/stimuli/relational (scripts/render_relational_stimuli.py).

Feature sources: V-JEPA 2 meanpool (26 points), random-init ViT-L meanpool (26 points), pixels (wm.pixels: 16 x 32 x 32
grey frames + 15 frame differences, PCA-256 fit on the training rows). Probe = wm.probes recipe: train-standardised
features, closed-form ridge, alpha from wm.probes.ALPHAS by fold-mean R^2 over the 5 folds of
splits/split_relational.json (stratified by (c, v_rel) cell), refit on all train, test once.

Analyses (every point):
  main       targets v_rel, c, v1, v2, |v_rel|: fold-mean CV R^2, pooled out-of-fold (OOF) R^2 with a 1000-draw clip
             bootstrap 95% CI, test R^2.
  composite  v_rel from the separate v1, v2 probes (OOF v1_hat - v2_hat) and |v1_hat - v2_hat| for |v_rel|; paired
             bootstrap CI of direct - composite. For a linear ridge probe at a shared alpha the v_rel probe IS
             W_v1 - W_v2 exactly, so the linear contrast can differ only through alpha choice; |v_rel| is the
             non-linear contrast.
  galilean   v_rel probe fit on c < 0, tested on c > 0 and vice versa (c = 0 in neither); c probe fit on v_rel < 0,
             tested on v_rel > 0 and vice versa. Alpha by 5-fold CV (stratified by cell, seed 0) inside the fit half;
             all 637 clips used. Reported: transfer R^2 per direction, their mean, in-half CV R^2 for reference.
  mirror     v_rel probe fit on c = 0 only (91 clips, where v1 = v_rel/2 = -v2), tested on c != 0 (546 clips);
             plus the least-squares fit pred ~ a v1 + b v2 + k on the test clips (relational readout: a = 1, b = -1;
             disk-1 shortcut: a = 2, b = 0; disk-2 shortcut: a = 0, b = -2).

  python scripts/run_relational_motion.py  -> results/p5_relational_motion.json, figures/fig_relational_motion.png
"""
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.metrics import r2  # noqa: E402
from wm.pixels import _decode_one, features  # noqa: E402
from wm.probes import (ALPHAS, LAST_BLOCK, N_POINTS, Standardizer, availability, fit_ridge, layer_matrix,  # noqa: E402
                       predict, ridge_path, write_json)

ROOT = PROJECT_ROOT / "artifacts" / "stimuli" / "relational"
ACT = PROJECT_ROOT / "artifacts" / "activations" / "stimuli_relational"
SPLIT = PROJECT_ROOT / "splits" / "split_relational.json"
PIX = PROJECT_ROOT / "artifacts" / "pixels" / "relational.npy"
TARGETS = ["v_rel", "c", "v1", "v2", "abs_v_rel", "v_top_minus_bottom"]   # composite code indexes 0 and 4
N_BOOT = 1000


def table():
    df = load_table("speed", root=ROOT)
    metas = [json.loads((ROOT / "videos" / f"scene_{i:04d}" / "metadata.json").read_text()) for i in df["id"]]
    df["v1"] = [m["v1_mps"] for m in metas]
    df["v2"] = [m["v2_mps"] for m in metas]
    df["v_rel"] = df["v1"] - df["v2"]
    df["c"] = (df["v1"] + df["v2"]) / 2
    df["abs_v_rel"] = df["v_rel"].abs()
    df["cell"] = [m["cell"] for m in metas]
    top1 = np.array([m["disk1_on_top"] for m in metas])
    df["v_top_minus_bottom"] = np.where(top1, 1, -1) * df["v_rel"]    # relative velocity with identity = row, not colour
    return df


def identity_blind_ceiling(df):
    """R^2 of the best predictor that knows only the identity-symmetric summary (c, |v_rel|): E[t | c, |v_rel|]
    per target, computed on the design (what a colour-blind / permutation-invariant code can reach)."""
    out = {}
    key = df["c"].astype(str) + "_" + df["abs_v_rel"].astype(str)
    for t in TARGETS:
        pred = df.groupby(key)[t].transform("mean").to_numpy()
        out[t] = r2(df[t].to_numpy(float), pred)
    return out


def pixels(df):
    if not PIX.exists():
        with Pool(8) as p:
            out = p.map(_decode_one, df["video"].tolist(), chunksize=16)
        PIX.parent.mkdir(parents=True, exist_ok=True)
        np.save(PIX, np.stack([o[0] for o in out]))
    return np.load(PIX)


def cv_multi(X, Y, folds, alphas=ALPHAS):
    """wm.probes.cv_select_alpha applied column by column (scalar R^2 score), sharing one eigh per fold across the
    columns. Returns per column: alpha, fold-mean R^2, fold SD, and OOF predictions at that alpha."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    ks = np.unique(folds)
    paths = {k: ridge_path(X[folds != k], Y[folds != k], alphas) for k in ks}
    out = []
    for j in range(Y.shape[1]):
        best = None
        for i, a in enumerate(alphas):
            oof = np.zeros(len(Y))
            fs = []
            for k in ks:
                v = folds == k
                W, b = paths[k][i]
                oof[v] = predict(X[v], W[:, j:j + 1], b[j:j + 1])[:, 0]
                fs.append(r2(Y[v, j], oof[v]))
            m = float(np.mean(fs))
            if best is None or m > best[0]:
                best = (m, a, fs, oof)
        out.append({"alpha": float(best[1]), "cv_mean": best[0], "cv_sd": float(np.std(best[2], ddof=1)),
                    "oof": best[3]})
    return out


def boot_idx(n, seed=0):
    return np.random.default_rng(seed).integers(0, n, (N_BOOT, n))


def boot_r2(y, p, idx):
    """R^2 over bootstrap rows, vectorised: [N_BOOT]."""
    yb, pb = y[idx], p[idx]
    ss_res = ((yb - pb) ** 2).sum(1)
    ss_tot = ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1)
    return 1 - ss_res / ss_tot


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def strat_folds(cells, seed=0):
    f = np.full(len(cells), -1)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(cells)), cells)):
        f[v] = k
    return f


def fit_transfer(Xtr, Xte, ytr, yte, folds_tr):
    """alpha by in-half CV, refit on the half, predict the other half."""
    cv = cv_multi(Xtr, ytr, folds_tr)[0]
    W, b = fit_ridge(Xtr, ytr, cv["alpha"])
    return predict(Xte, W, b)[:, 0], cv


def std_pair(X, a, b):
    st = Standardizer().fit(X[a])
    return st.transform(X[a]), st.transform(X[b])


def analyse(get_X, df, tr, te, folds, idx_tr, idx_te_by_mask, subsets, n_points):
    """All analyses for one feature source; get_X(point, rows_fit, rows_eval) -> standardised (X_fit, X_eval)."""
    Y = df[TARGETS].to_numpy(float)
    rows = []
    for p in range(n_points):
        t0 = time.time()
        Xtr, Xte = get_X(p, tr, te)
        cvs = cv_multi(Xtr, Y[tr], folds)
        row = {"point": p, "targets": {}}
        oof = {}
        for j, t in enumerate(TARGETS):
            cv = cvs[j]
            W, b = fit_ridge(Xtr, Y[tr, j], cv["alpha"])
            pte = predict(Xte, W, b)[:, 0]
            bo = boot_r2(Y[tr, j], cv["oof"], idx_tr)
            oof[t] = cv["oof"]
            row["targets"][t] = {"alpha": cv["alpha"], "cv_mean": cv["cv_mean"], "cv_sd": cv["cv_sd"],
                                 "oof_r2": r2(Y[tr, j], cv["oof"]), "oof_r2_ci": ci(bo),
                                 "test_r2": r2(Y[te, j], pte)}
        # composite from separate v1, v2 probes
        comp = oof["v1"] - oof["v2"]
        y_vr, y_abs = Y[tr, 0], Y[tr, 4]
        d_lin = boot_r2(y_vr, oof["v_rel"], idx_tr) - boot_r2(y_vr, comp, idx_tr)
        d_abs = boot_r2(y_abs, oof["abs_v_rel"], idx_tr) - boot_r2(y_abs, np.abs(comp), idx_tr)
        row["composite"] = {"v_rel_from_v1_v2_oof_r2": r2(y_vr, comp), "v_rel_from_v1_v2_oof_r2_ci": ci(
                                boot_r2(y_vr, comp, idx_tr)),
                            "direct_minus_composite_v_rel": float(r2(y_vr, oof["v_rel"]) - r2(y_vr, comp)),
                            "direct_minus_composite_v_rel_ci": ci(d_lin),
                            "abs_v_rel_from_v1_v2_oof_r2": r2(y_abs, np.abs(comp)),
                            "abs_v_rel_from_v1_v2_oof_r2_ci": ci(boot_r2(y_abs, np.abs(comp), idx_tr)),
                            "direct_minus_composite_abs_v_rel": float(r2(y_abs, oof["abs_v_rel"]) - r2(y_abs, np.abs(comp))),
                            "direct_minus_composite_abs_v_rel_ci": ci(d_abs)}
        # galilean + mirror
        gal = {}
        for name, (target, fit_mask, eval_mask) in subsets.items():
            a, e = np.flatnonzero(fit_mask), np.flatnonzero(eval_mask)
            Xa, Xe = get_X(p, a, e)
            j = TARGETS.index(target)
            pred, cv = fit_transfer(Xa, Xe, Y[a, j], Y[e, j], strat_folds(df["cell"].to_numpy()[a]))
            rec = {"target": target, "n_fit": len(a), "n_eval": len(e), "alpha": cv["alpha"],
                   "in_half_cv_r2": cv["cv_mean"], "transfer_r2": r2(Y[e, j], pred),
                   "transfer_r2_ci": ci(boot_r2(Y[e, j], pred, idx_te_by_mask[name]))}
            v1, v2 = df["v1"].to_numpy(), df["v2"].to_numpy()
            inr = ((v1[e] >= v1[a].min()) & (v1[e] <= v1[a].max()) & (v2[e] >= v2[a].min()) & (v2[e] <= v2[a].max()))
            rec["in_range"] = {"rule": "eval clips whose v1 AND v2 lie inside the fit half's v1, v2 ranges",
                               "n_eval": int(inr.sum()), "transfer_r2": r2(Y[e[inr], j], pred[inr]),
                               "transfer_r2_ci": ci(boot_r2(Y[e[inr], j], pred[inr],
                                                            boot_idx(int(inr.sum()))))}
            if name == "mirror_c0_to_cnonzero":
                A = np.stack([df["v1"].to_numpy()[e], df["v2"].to_numpy()[e], np.ones(len(e))], 1)
                coef = np.linalg.lstsq(A, pred, rcond=None)[0]
                rec["pred_on_v1_v2"] = {"a_v1": float(coef[0]), "b_v2": float(coef[1]), "k": float(coef[2])}
                rec["r2_vs_2v1"] = r2(2 * df["v1"].to_numpy()[e], pred)
                rec["r2_vs_minus_2v2"] = r2(-2 * df["v2"].to_numpy()[e], pred)
            gal[name] = rec
        for tgt, (n1, n2) in {"v_rel": ("vrel_fit_cneg_eval_cpos", "vrel_fit_cpos_eval_cneg"),
                              "c": ("c_fit_vrelneg_eval_vrelpos", "c_fit_vrelpos_eval_vrelneg")}.items():
            gal[f"{tgt}_transfer_mean_r2"] = float((gal[n1]["transfer_r2"] + gal[n2]["transfer_r2"]) / 2)
            gal[f"{tgt}_transfer_in_range_mean_r2"] = float((gal[n1]["in_range"]["transfer_r2"] +
                                                             gal[n2]["in_range"]["transfer_r2"]) / 2)
            gal[f"{tgt}_in_half_cv_mean_r2"] = float((gal[n1]["in_half_cv_r2"] + gal[n2]["in_half_cv_r2"]) / 2)
        row["galilean"] = gal
        rows.append(row)
        print(f"point {p:2d} ({time.time() - t0:.1f}s): " + " ".join(
            f"{t}={row['targets'][t]['cv_mean']:.3f}" for t in TARGETS) +
              f" | vrel xfer {gal['v_rel_transfer_mean_r2']:.3f} c xfer {gal['c_transfer_mean_r2']:.3f}"
              f" mirror {gal['mirror_c0_to_cnonzero']['transfer_r2']:.3f}", flush=True)
    return rows, None


def profile(rows):
    """Availability (wm.probes.availability, fold-mean CV R^2 over points 0..24) per target."""
    out = {}
    for t in TARGETS:
        curve = [r["targets"][t]["cv_mean"] for r in rows[:LAST_BLOCK + 1]]
        out[t] = availability(curve)
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"v_rel": "#c0392b", "c": "#2471a3", "v1": "#d68910", "v2": "#7d3c98", "abs_v_rel": "#555555"}
    lab = {"v_rel": "v_rel = v1 - v2", "c": "c = (v1 + v2)/2", "v1": "v1 (orange)", "v2": "v2 (blue)",
           "abs_v_rel": "|v_rel|"}
    cols["v_top_minus_bottom"] = "#17a589"
    lab["v_top_minus_bottom"] = "v_top - v_bottom"
    bc = res.get("beyond_composition")
    fig, ax = plt.subplots(1, 4 if bc else 3, figsize=(21 if bc else 16, 4.6))
    pts = np.arange(N_POINTS)
    for t in TARGETS:
        for src, ls, a in (("vjepa2", "-", 1.0), ("random", ":", 0.8)):
            rows = res["sources"][src]["points"]
            y = [r["targets"][t]["oof_r2"] for r in rows]
            lo = [r["targets"][t]["oof_r2_ci"][0] for r in rows]
            hi = [r["targets"][t]["oof_r2_ci"][1] for r in rows]
            ax[0].plot(pts, y, ls, color=cols[t], alpha=a, label=f"{lab[t]} ({'V-JEPA 2' if src == 'vjepa2' else 'random-init'})")
            ax[0].fill_between(pts, lo, hi, color=cols[t], alpha=0.12 if src == "vjepa2" else 0.06)
        px = res["sources"]["pixels"]["points"][0]["targets"][t]["oof_r2"]
        ax[0].axhline(px, color=cols[t], lw=0.8, ls="--")
    ax[0].set(xlabel="layer point (0 = embedding, 25 = post-LN)", ylabel="pooled out-of-fold R$^2$ (95% CI)",
              title="Probe R$^2$ per point (solid V-JEPA 2, dotted random-init, dashed pixels)", ylim=(-0.1, 1.02))
    ax[0].legend(fontsize=6, ncol=2, loc="lower right")
    for k, (key, ttl) in enumerate((("v_rel", "Galilean transfer: v_rel probe fit on c<0, tested on c>0 (and reverse)"),
                                    ("c", "c probe fit on v_rel<0, tested on v_rel>0 (and reverse)"))):
        a = ax[1 + k]
        for src, ls in (("vjepa2", "-"), ("random", ":")):
            rows = res["sources"][src]["points"]
            a.plot(pts, [r["galilean"][f"{key}_transfer_mean_r2"] for r in rows], ls, color=cols[key],
                   label=f"{src} transfer")
            a.plot(pts, [r["galilean"][f"{key}_transfer_in_range_mean_r2"] for r in rows], ls, color="black",
                   label=f"{src} transfer, eval within fit speed range")
            a.plot(pts, [r["galilean"][f"{key}_in_half_cv_mean_r2"] for r in rows], ls, color="grey",
                   label=f"{src} in-half CV")
            if key == "v_rel":
                a.plot(pts, [r["galilean"]["mirror_c0_to_cnonzero"]["transfer_r2"] for r in rows], ls, color="#1e8449",
                       label=f"{src} mirror (fit c=0, test c!=0)")
        g = res["sources"]["pixels"]["points"][0]["galilean"]
        a.axhline(g[f"{key}_transfer_mean_r2"], color=cols[key], ls="--", lw=0.8, label="pixels transfer")
        if key == "v_rel":
            a.axhline(g["mirror_c0_to_cnonzero"]["transfer_r2"], color="#1e8449", ls="--", lw=0.8, label="pixels mirror")
        a.set(xlabel="layer point", ylabel="R$^2$", title=ttl, ylim=(-0.5, 1.02))
        a.title.set_fontsize(8)
        a.legend(fontsize=6)
    if bc:
        a = ax[3]
        for src, ls in (("vjepa2", "-"), ("random", ":")):
            rows = bc["sources"][src]
            for t, col in (("rel_speed", "#555555"), ("closing_rate", "#1e8449")):
                pp = [r["point"] for r in rows]
                a.plot(pp, [r["targets"][t]["mlp_acts"]["test_r2"] for r in rows], ls, color=col, lw=1.6,
                       label=f"{src} MLP on activations: {t}")
                a.plot(pp, [r["targets"][t]["mlp_null"]["test_r2"] for r in rows], ls, color=col, lw=0.8, alpha=0.6,
                       marker="x", ms=3, label=f"{src} MLP on decoded single-disk readouts: {t}")
        a.set(xlabel="layer point", ylabel="held-out test R$^2$", ylim=(-0.5, 1.02),
              title="Beyond composition: |v1 - v2| and closing rate")
        a.title.set_fontsize(8)
        a.legend(fontsize=5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    df = table()
    split = json.loads(SPLIT.read_text())["relational"]
    pos = {int(i): r for r, i in enumerate(df["id"])}
    tr = np.array([pos[i] for i in split["train_ids"]])
    te = np.array([pos[i] for i in split["test_ids"]])
    folds = np.array([split["fold"][str(i)] for i in split["train_ids"]])
    c, vr = df["c"].to_numpy(), df["v_rel"].to_numpy()
    subsets = {"vrel_fit_cneg_eval_cpos": ("v_rel", c < 0, c > 0), "vrel_fit_cpos_eval_cneg": ("v_rel", c > 0, c < 0),
               "c_fit_vrelneg_eval_vrelpos": ("c", vr < 0, vr > 0), "c_fit_vrelpos_eval_vrelneg": ("c", vr > 0, vr < 0),
               "mirror_c0_to_cnonzero": ("v_rel", c == 0, c != 0)}
    idx_tr = boot_idx(len(tr))
    idx_sub = {k: boot_idx(int(m[2].sum())) for k, m in subsets.items()}

    res = {"sources": {}}
    hashes = {}
    for src in ("vjepa2", "random"):
        folder = ACT / src
        acts = np.load(folder / "meanpool.npy", mmap_mode="r")
        assert json.loads((folder / "ids.json").read_text()) == df["id"].tolist()
        hashes[src] = {"index_json_sha256": hashlib.sha256((folder / "index.json").read_bytes()).hexdigest(),
                       "meanpool_sha256": hashlib.sha256((folder / "meanpool.npy").read_bytes()).hexdigest()}
        cache = {}

        def get_X(p, a, b, acts=acts, cache=cache):
            if p not in cache:
                cache.clear()
                cache[p] = layer_matrix(acts, p)
            return std_pair(cache[p], a, b)
        print(f"== {src}", flush=True)
        rows, _ = analyse(get_X, df, tr, te, folds, idx_tr, idx_sub, subsets, N_POINTS)
        res["sources"][src] = {"points": rows, "profile": profile(rows)}
    gray = pixels(df)
    print("== pixels", flush=True)
    get_px = lambda p, a, b: features("pixels", gray, None, a, b)   # noqa: E731
    rows, _ = analyse(get_px, df, tr, te, folds, idx_tr, idx_sub, subsets, 1)
    res["sources"]["pixels"] = {"points": rows}

    # V-JEPA 2 minus baselines, per point, per target (paired bootstrap over the same OOF rows)
    Y = df[TARGETS].to_numpy(float)
    best = {}
    for t in TARGETS:
        vj = [r["targets"][t]["oof_r2"] for r in res["sources"]["vjepa2"]["points"]]
        rn = [r["targets"][t]["oof_r2"] for r in res["sources"]["random"]["points"]]
        px = res["sources"]["pixels"]["points"][0]["targets"][t]["oof_r2"]
        pk = int(np.argmax(vj[:LAST_BLOCK + 1]))
        best[t] = {"vjepa2_peak_point": pk, "vjepa2_peak_oof_r2": vj[pk],
                   "vjepa2_peak_oof_r2_ci": res["sources"]["vjepa2"]["points"][pk]["targets"][t]["oof_r2_ci"],
                   "random_best_oof_r2": float(max(rn)), "random_best_point": int(np.argmax(rn)), "pixels_oof_r2": px,
                   "vjepa2_minus_random_best": float(vj[pk] - max(rn)), "vjepa2_minus_pixels": float(vj[pk] - px)}
    summary = {}
    for src in ("vjepa2", "random"):
        rows = res["sources"][src]["points"]
        g = [r["galilean"] for r in rows]
        pk = int(np.argmax([x["v_rel_transfer_mean_r2"] for x in g[:LAST_BLOCK + 1]]))
        summary[src] = {"v_rel_transfer_best_point": pk, **{k: g[pk][k] for k in (
            "v_rel_transfer_mean_r2", "v_rel_transfer_in_range_mean_r2", "v_rel_in_half_cv_mean_r2", "c_transfer_mean_r2",
            "c_transfer_in_range_mean_r2", "c_in_half_cv_mean_r2")},
            "mirror_at_that_point": g[pk]["mirror_c0_to_cnonzero"]}
    summary["pixels"] = {k: res["sources"]["pixels"]["points"][0]["galilean"][k] for k in (
        "v_rel_transfer_mean_r2", "v_rel_transfer_in_range_mean_r2", "v_rel_in_half_cv_mean_r2", "c_transfer_mean_r2",
        "c_transfer_in_range_mean_r2", "c_in_half_cv_mean_r2", "mirror_c0_to_cnonzero")}

    out = {
        "experiment": "E4 relational motion (probing only)",
        "keys": {
            "sources.{vjepa2,random}.points[p].targets[t]": "t in v_rel, c, v1, v2, abs_v_rel: alpha, cv_mean/cv_sd "
                "(fold-mean R^2, 5 stratified-by-cell folds of the 509 train clips), oof_r2 + oof_r2_ci (pooled OOF R^2, "
                "1000-draw clip bootstrap 95% CI), test_r2 (refit on train, 128 held-out clips)",
            "sources.*.points[p].composite": "v_rel and |v_rel| from the separate v1, v2 probes' OOF predictions; "
                "direct_minus_composite_* with paired bootstrap CI (same bootstrap rows)",
            "sources.*.points[p].galilean": "transfer tests over all 637 clips (alpha by 5-fold CV inside the fit half): "
                "vrel_fit_cneg_eval_cpos / vrel_fit_cpos_eval_cneg / c_fit_vrelneg_eval_vrelpos / "
                "c_fit_vrelpos_eval_vrelneg / mirror_c0_to_cnonzero; transfer_r2 + clip-bootstrap CI over eval clips; "
                "*_transfer_mean_r2 = mean of the two directions; *_in_half_cv_mean_r2 = the same fits' in-half CV",
            "sources.*.profile": "wm.probes.availability of the fold-mean CV curve over points 0..24 per target",
            "sources.pixels.points[0]": "the same analyses on the wm.pixels feature set (single row, point = -1 role)",
            "headline.best_by_target": "V-JEPA 2 peak point (0..24) per target vs best random-init point and pixels",
            "headline.identity_blind_ceiling_r2": "R^2 per target of E[t | c, |v_rel|] on the design: the ceiling of a "
                "code that keeps common motion and closing speed but not which disk is which",
            "targets.v_top_minus_bottom": "v_rel with identity by vertical row (top disk minus bottom disk) instead of "
                "colour; disk 1 is on top in a random half of the clips",
            "headline.transfer": "Galilean/mirror at the V-JEPA 2 / random best v_rel-transfer point, and pixels"},
        "design": json.loads((PROJECT_ROOT / "results" / "p5_relational_stimuli_validation.json").read_text()),
        "notes": ["Galilean halves differ in the range of absolute velocities (c < 0: v1, v2 in [-6, 2.5]; c > 0: [-2.5, 6]), "
                  "so a transfer failure mixes relational non-invariance with absolute-speed extrapolation; the c probe "
                  "across v_rel halves faces the same kind of range shift and is the matched comparison",
                  "linear identity: with the same alpha a ridge v_rel probe equals W_v1 - W_v2 exactly, so "
                  "'v_rel beyond v1, v2' is only testable non-linearly (|v_rel|) or by transfer",
                  "diskpool/timepool are extracted (V-JEPA 2) but not probed here; diskmask is the red-channel rule so "
                  "it covers disk 1 (orange) only",
                  "bootstrap resamples clips (7 per cell share labels), so CIs are optimistic relative to a cell bootstrap"],
        "headline": {"best_by_target": best, "transfer": summary,
                     "identity_blind_ceiling_r2": identity_blind_ceiling(df)},
        **res,
        "activation_hashes": hashes,
        "wall_seconds": time.time() - t_start,
    }
    bcache = PROJECT_ROOT / "artifacts" / "relational" / "binding.json"
    if bcache.exists():
        out["binding"] = json.loads(bcache.read_text())
    write_json(PROJECT_ROOT / "results" / "p5_relational_motion.json", out, SPLIT)
    figure(out, PROJECT_ROOT / "figures" / "fig_relational_motion.png")
    print(json.dumps(out["headline"], indent=1))


if __name__ == "__main__":
    main()

"""Frozen-probe linearity-and-extrapolation test on time-rescaled clips (ideas brief E3, first run 2026-09-28).

NOT a clock / physical-units test. At the fixed 24 fps, x(lam t) is exactly the clip of speed lam v (acceleration
lam^2 a): lam = 2 at v = 1..3 is bit-identical to paper_layout at 2..6 (168 of 392 clips; activations equal to 3e-10),
so the design cannot separate a clock from an odometer. What it measures: when the true speed doubles (acceleration
x4), by how much does the readout of a probe fit on the supplied set change, relative to the intercept-adjusted linear
expectation (a lam v + b) / (a v + b) calibrated at lam = 1 inside the training range, and relative to an explicit
displacement readout. Probes are fit on the SUPPLIED sets' train split (splits/split_v1.json; train-standardised,
alpha by the 5 train folds, refit on all train rows) and read, frozen, on the rescaled paper_layout clips
(scripts/render_clock_stimuli.py): lam = 0.5, 1 (paper_layout itself), 2. Speed probe <- supplied speed set,
direction probe <- supplied direction set, acceleration probe <- supplied acceleration set.

Readouts: V-JEPA 2 meanpool at every point (detail at 9, 12, 19, 22), and two baselines read the same way:
  pixels      16 x 32 x 32 grayscale frames + 15 frame differences, PCA-256 fit on the supplied train rows (wm.pixels)
  trajectory  disk centroid per frame + 15 differences (wm.pixels), i.e. an explicit displacement readout.
Per readout and lam: OLS slope / intercept / R^2 of predicted vs effective label (lam v, lam^2 a), paired ratios
pred(lam = 2) / pred(lam = 1) and pred(lam = 1) / pred(lam = 0.5) on the same path vs the intercept-adjusted linear
expectation, direction circular MAE; 200-draw cluster bootstrap over the 56 straight-line paths (direction x start;
clips shared across lam are never counted as independent); the pooled OLS keeps each distinct clip once.
Clip subsets: all 392, and 'in_frame' (disk fully inside the frame at every frame, at every lam compared).

  python scripts/run_clock_test.py [--all-points]   -> results/p5_clock_test.json, figures/fig_clock_test.png
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.pixels import _decode_one, load_pixels, pixel_matrix, trajectory_matrix  # noqa: E402
from wm.probes import (ALPHAS, Standardizer, cv_select_alpha, fit_ridge, layer_matrix, load_activations,  # noqa: E402
                       predict, split_rows, targets)
from wm.provenance import git_commit, sha256_file  # noqa: E402
from wm.splits import SPLIT_PATH  # noqa: E402

STIM = PROJECT_ROOT / "artifacts" / "stimuli"
ACT = PROJECT_ROOT / "artifacts" / "activations"
PIX = PROJECT_ROOT / "artifacts" / "pixels"
DETAIL = (9, 12, 19, 22)
LAMS = (0.5, 1.0, 2.0)
FAMILIES = {"velocity": ({0.5: "clock_lam0.5", 1.0: "paper_layout", 2.0: "clock_lam2"}, "speed"),
            "acceleration": ({0.5: "clock_acc_lam0.5", 1.0: "clock_acc_lam1", 2.0: "clock_acc_lam2"}, "acceleration")}
N_BOOT, SEED = 200, 0
POINTS = DETAIL if "--all-points" not in sys.argv else tuple(range(26))   # default: the four detail points


def set_table(name):
    df = load_table("direction", root=STIM / name)
    meta = [json.loads((STIM / name / "videos" / f"scene_{i:04d}" / "metadata.json").read_text()) for i in df["id"]]
    if name == "paper_layout":        # lam = 1 reference: effective = nominal, in-frame flags recomputed
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from render_clock_stimuli import clock_meta
        meta = [clock_meta(m, 1.0, name) for m in meta]
    df["in_frame"] = [m["in_frame_all_frames"] for m in meta]
    df["source_id"] = [m.get("source_id", m["id"]) for m in meta]
    assert (df["source_id"] == df["id"]).all()
    return df


def set_pixels(name):
    f_g, f_c = PIX / f"stimuli_{name}.npy", PIX / f"stimuli_{name}_centroid.npy"
    if not f_c.exists():
        vids = load_table("direction", root=STIM / name)["video"].tolist()
        with Pool(8) as pool:
            out = pool.map(_decode_one, vids, chunksize=8)
        np.save(f_g, np.stack([o[0] for o in out]))
        np.save(f_c, np.stack([o[1] for o in out]))
    return np.load(f_g), np.load(f_c)


class Probe:
    """wm.probes recipe on one supplied dataset / variable / feature matrix builder."""

    def __init__(self, Xtr, Ytr, folds, score_fn):
        self.st = Standardizer().fit(Xtr)
        Z = self.st.transform(Xtr)
        cv = cv_select_alpha(Z, Ytr, folds, ALPHAS, score_fn)
        self.alpha, self.cv_r2 = cv["alpha"], cv["cv_mean"]
        self.W, self.b = fit_ridge(Z, Ytr, self.alpha)

    def __call__(self, X):
        return predict(self.st.transform(X), self.W, self.b)


def ols(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    (m, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - np.sum((y - (m * x + c)) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(m), float(c), float(r2)


def circ_err(theta, P):
    return np.abs((np.degrees(np.arctan2(P[:, 0], P[:, 1])) - theta + 180.0) % 360.0 - 180.0)


def q(a, p):
    return float(np.percentile(a, p))


def path_of(i):
    """Straight-line path (direction index, start index) of layout id i = (dir * 7 + speed) * 7 + start: 56 paths.
    At a fixed 24 fps, x(lam t) is the speed-lam v clip, so clips on one path are shared across lam (e.g. lam = 2 at
    v = 1..3 is bit-identical to lam = 1 at 2..6); the bootstrap resamples whole paths."""
    i = np.asarray(i)
    return (i // 49) * 7 + i % 7


def summarise(pred, eff, theta, dirP, mask, path_draws, train_hi):
    """pred / eff / dirP: dict lam -> [392] (dirP [392, 2]); mask: bool [392] clips kept; path_draws: list of
    resampled path-index arrays (cluster bootstrap over the 56 paths); train_hi: top of the probe's training range."""
    ids = np.flatnonzero(mask)
    paths = path_of(ids)
    members = {p_: np.flatnonzero(paths == p_) for p_ in np.unique(paths)}
    draws = [np.concatenate([members[p_] for p_ in d if p_ in members] or [np.array([], int)]) for d in path_draws]
    out = {"n_clips": int(len(ids)), "n_paths": len(members)}
    for lam in LAMS:
        m, c, r2 = ols(eff[lam][ids], pred[lam][ids])
        bs = [ols(eff[lam][ids][b], pred[lam][ids][b])[0] for b in draws]
        de = circ_err(theta[ids], dirP[lam][ids])
        dbs = [de[b].mean() for b in draws]
        out[f"lam{lam:g}"] = {"slope": m, "slope_ci": [q(bs, 2.5), q(bs, 97.5)], "intercept": c, "r2": r2,
                              "mae_vs_eff": float(np.mean(np.abs(pred[lam][ids] - eff[lam][ids]))),
                              "pred_mean": float(pred[lam][ids].mean()),
                              "direction_mae_deg": float(de.mean()), "direction_mae_ci": [q(dbs, 2.5), q(dbs, 97.5)]}
    # pooled over lam, each distinct clip once: key (path, effective label); duplicates across lam dropped
    rows, seen = [], set()
    for lam in LAMS:
        for j, i in enumerate(ids):
            k = (int(paths[j]), round(float(eff[lam][i]), 6))
            if k not in seen:
                seen.add(k)
                rows.append((eff[lam][i], pred[lam][i], paths[j]))
    X, Y, Pp = (np.array(v) for v in zip(*rows))
    pm = {p_: np.flatnonzero(Pp == p_) for p_ in np.unique(Pp)}
    pdraws = [np.concatenate([pm[p_] for p_ in d if p_ in pm]) for d in path_draws]
    m, c, r2 = ols(X, Y)
    bs = [ols(X[b], Y[b])[0] for b in pdraws]
    out["pooled"] = {"slope": m, "slope_ci": [q(bs, 2.5), q(bs, 97.5)], "intercept": c, "r2": r2,
                     "n_distinct_clips": int(len(X)), "n_duplicates_dropped": int(3 * len(ids) - len(X))}
    # intercept-adjusted linear null: calibrate pred = a eff + b on lam = 1 clips inside the training range; a linear
    # readout then predicts ratio (a eff_hi + b) / (a eff_lo + b), not 2 (speed) or 4 (acceleration)
    cal = eff[1.0][ids] <= train_hi

    def calib(sel):
        s_ = sel[cal[sel]]
        return ols(eff[1.0][ids][s_], pred[1.0][ids][s_])[:2]

    a_, b_ = calib(np.arange(len(ids)))
    out["calibration_lam1_in_range"] = {"a": a_, "b": b_, "n": int(cal.sum())}
    for hi, lo in ((2.0, 1.0), (1.0, 0.5)):
        r = pred[hi][ids] / pred[lo][ids]
        med = [np.median(r[b]) for b in draws]

        def exp_ratio(sel, a, b):
            return (a * eff[hi][ids][sel] + b) / (a * eff[lo][ids][sel] + b)

        e = exp_ratio(np.arange(len(ids)), a_, b_)
        oe = r / e
        oe_b, ex_b = [], []
        for bd in draws:
            ab = calib(bd)
            eb = exp_ratio(bd, *ab)
            ex_b.append(np.median(eb))
            oe_b.append(np.median(r[bd] / eb))
        out[f"ratio_{hi:g}_over_{lo:g}"] = {"median": float(np.median(r)), "iqr": [q(r, 25), q(r, 75)],
                                            "median_ci": [q(med, 2.5), q(med, 97.5)],
                                            "expected_linear_median": float(np.median(e)),
                                            "expected_linear_ci": [q(ex_b, 2.5), q(ex_b, 97.5)],
                                            "obs_over_expected_median": float(np.median(oe)),
                                            "obs_over_expected_iqr": [q(oe, 25), q(oe, 75)],
                                            "obs_over_expected_ci": [q(oe_b, 2.5), q(oe_b, 97.5)],
                                            "n_nonpositive_denominator": int((pred[lo][ids] <= 0).sum())}
    return out


def main():
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, 56, 56) for _ in range(N_BOOT)]      # cluster bootstrap over the 56 straight-line paths

    sup = {d: load_table(d) for d in ("speed", "direction", "acceleration")}
    rows = {d: split_rows(d, sup[d]) for d in sup}
    sup_acts = {d: np.asarray(load_activations(d)) for d in sup}
    sup_pix = {d: load_pixels(d) for d in sup}
    tabs = {s: set_table(s) for fam in FAMILIES.values() for s in fam[0].values()}
    acts = {s: np.asarray(load_activations(f"stimuli_{s}", table=tabs[s])) for s in tabs}
    pix = {s: set_pixels(s) for s in tabs}

    cache = {}

    def fit(d, variable, feat, key):
        if (d, variable, key) in cache:
            return cache[(d, variable, key)]
        tr, _, folds = rows[d]
        Y, _, score_fn = targets(sup[d], variable)
        cache[(d, variable, key)] = Probe(feat(d)[tr], Y[tr], folds, score_fn)
        return cache[(d, variable, key)]

    readouts = {f"vjepa2_p{p}": (lambda s, p=p: layer_matrix(sup_acts[s] if s in sup else acts[s], p)) for p in POINTS}
    pca = {}

    def pix_feat(s):
        X = pixel_matrix((sup_pix[s] if s in sup else pix[s])[0])
        return X

    def pixels_for(d):
        if d not in pca:
            tr = rows[d][0]
            pca[d] = PCA(n_components=256, svd_solver="randomized", random_state=0).fit(pix_feat(d)[tr])
        return lambda s: pca[d].transform(pix_feat(s))

    traj = lambda s: trajectory_matrix((sup_pix[s] if s in sup else pix[s])[1])            # noqa: E731

    res = {"test": "frozen-probe linearity and extrapolation under time rescaling (NOT a clock / physical-units test)",
           "design": __doc__.strip().split("\n\n")[1], "families": {}}
    for fam, (sets, mag_var) in FAMILIES.items():
        mag_ds = mag_var
        cart_var = {"speed": "vxvy", "acceleration": "axay"}[mag_var]
        tabl = {lam: tabs[sets[lam]] for lam in LAMS}
        theta = tabl[1.0]["theta_degrees"].to_numpy(float)
        col = "speed_mps" if fam == "velocity" else "acceleration_mps2"
        eff = {lam: tabl[lam][col].to_numpy(float) for lam in LAMS}
        for lam in LAMS:                                  # same trajectories, effective label = lam^k * nominal
            k = 1 if fam == "velocity" else 2
            assert np.allclose(eff[lam], lam ** k * eff[1.0]) and (tabl[lam]["theta_degrees"] == tabl[1.0]["theta_degrees"]).all()
        in_frame = np.all([tabl[lam]["in_frame"].to_numpy() for lam in LAMS], axis=0)
        in_frame_12 = tabl[1.0]["in_frame"].to_numpy() & tabl[2.0]["in_frame"].to_numpy()
        train_hi = float(sup[mag_ds]["magnitude"].max())
        fres = {"sets": {f"lam{lam:g}": sets[lam] for lam in LAMS},
                "n_in_frame": {f"lam{lam:g}": int(tabl[lam]["in_frame"].sum()) for lam in LAMS},
                "n_in_frame_all_lams": int(in_frame.sum()), "probe_train_range": [float(sup[mag_ds]["magnitude"].min()), train_hi],
                "readouts": {}}
        for rname, feat in list(readouts.items()) + [("pixels", None), ("trajectory", traj)]:
            fm = pixels_for(mag_ds) if rname == "pixels" else feat
            fd = pixels_for("direction") if rname == "pixels" else feat
            dprobe = fit("direction", "direction", fd, rname)
            dirP = {lam: dprobe(fd(sets[lam])) for lam in LAMS}
            entry = {"probe": {"direction_alpha": dprobe.alpha, "direction_cv_r2": dprobe.cv_r2}}
            detail = rname in ("pixels", "trajectory") or int(rname.split("p")[-1]) in DETAIL
            for tag, var in (("scalar", mag_var), ("cartesian_norm", cart_var)):
                mprobe = fit(mag_ds, var, fm, rname)
                P = {lam: mprobe(fm(sets[lam])) for lam in LAMS}
                pred = {lam: (P[lam][:, 0] if tag == "scalar" else np.linalg.norm(P[lam], axis=1)) for lam in LAMS}
                entry["probe"].update({f"{var}_alpha": mprobe.alpha, f"{var}_cv_r2": mprobe.cv_r2})
                e = {"all": summarise(pred, eff, theta, dirP, np.ones(len(theta), bool), boot, train_hi),
                     "in_frame": summarise(pred, eff, theta, dirP, in_frame, boot, train_hi)}
                # effective labels inside the probe's training range at both lam = 1 and lam = 2, in frame at both
                sub = summarise(pred, eff, theta, dirP, in_frame_12 & (eff[2.0] <= train_hi), boot, train_hi)
                e["in_train_range_lam1_2"] = {"n_clips": sub["n_clips"], "n_paths": sub["n_paths"],
                                              "ratio_2_over_1": sub["ratio_2_over_1"],
                                              "lam1": {q_: sub["lam1"][q_] for q_ in ("slope", "slope_ci", "r2")},
                                              "lam2": {q_: sub["lam2"][q_] for q_ in ("slope", "slope_ci", "r2")}}
                if detail:
                    e["pred_by_source"] = {f"lam{lam:g}": pred[lam].round(4).tolist() for lam in LAMS}
                entry[tag] = e
                print(fam, rname, tag, "cv_r2 %.3f" % mprobe.cv_r2, "pooled slope %.3f" % e["in_frame"]["pooled"]["slope"],
                      "ratio2/1 %.3f" % e["in_frame"]["ratio_2_over_1"]["median"],
                      "dirMAE", [round(e["in_frame"][f"lam{lam:g}"]["direction_mae_deg"], 1) for lam in LAMS], flush=True)
            fres["readouts"][rname] = entry
        fres["effective"] = {f"lam{lam:g}": eff[lam].tolist() for lam in LAMS}
        fres["theta"] = theta.tolist()
        fres["in_frame_mask"] = in_frame.astype(int).tolist()
        res["families"][fam] = fres

    idx = {}
    for s in tabs:
        f = ACT / f"stimuli_{s}" / "vjepa2" / "index.json"
        ix = json.loads(f.read_text())
        idx[s] = {"index_sha256": sha256_file(f), "meanpool_sha256": sha256_file(f.parent / "meanpool.npy"),
                  "n_ids": len(ix["ids"]), "device": ix.get("device"), "gpu": ix.get("gpu")}
    dup = {}
    for fam, (sets, _) in FAMILIES.items():
        k1 = {(int(path_of(i)), round(float(x), 6)): i for i, x in
              enumerate(tabs[sets[1.0]]["speed_mps" if fam == "velocity" else "acceleration_mps2"])}
        dup[fam] = {}
        for lam in (0.5, 2.0):
            col = tabs[sets[lam]]["speed_mps" if fam == "velocity" else "acceleration_mps2"]
            pairs = [(i, k1[(int(path_of(i)), round(float(x), 6))]) for i, x in enumerate(col)
                     if (int(path_of(i)), round(float(x), 6)) in k1]
            A = np.stack([acts[sets[lam]][i, 12] for i, _ in pairs]) if pairs else None
            B = np.stack([acts[sets[1.0]][j, 12] for _, j in pairs]) if pairs else None
            dup[fam][f"lam{lam:g}"] = {"n_clips_identical_to_a_lam1_clip": len(pairs),
                                       "max_rel_meanpool_diff_point12": float(np.abs(A - B).max() / np.abs(B).max())
                                       if pairs else None}
    res["duplicates_across_lam"] = dup
    res["provenance"] = {"script_sha256": {n: sha256_file(Path(__file__).resolve().parent / n)
                                           for n in ("run_clock_test.py", "render_clock_stimuli.py")},
                         "scripts_untracked_at_commit": True,
                         "overwrites": "replaces the 14:29 ET (velocity-only) and 14:45 ET versions of this file, which "
                                       "used a per-clip bootstrap, counted cross-lam duplicate clips as independent, "
                                       "and compared ratios with 2 / 4 instead of the intercept-adjusted expectation",
"split_file": str(SPLIT_PATH.relative_to(PROJECT_ROOT)), "split_sha256": sha256_file(SPLIT_PATH),
                         **git_commit(), "activation_index": idx, "n_boot": N_BOOT, "seed": SEED,
                         "detail_points": list(DETAIL), "points_run": list(POINTS)}
    out = PROJECT_ROOT / "results" / ("p5_clock_test_linearity.json")
    out.write_text(json.dumps(res, indent=1))
    print("->", out)
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [f"vjepa2_p{p}" for p in DETAIL] + ["pixels", "trajectory"]
    colors = {0.5: "#1f77b4", 1.0: "#2ca02c", 2.0: "#d62728"}
    rows = [("velocity", "scalar", "speed probe"), ("velocity", "cartesian_norm", "|(vx, vy) probe|"),
            ("acceleration", "scalar", "acceleration probe")]
    rows = [r for r in rows if r[0] in res["families"]]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(2.9 * len(names), 2.9 * len(rows)), squeeze=False)
    for (fam_name, tag, label), axrow in zip(rows, axes):
        fam = res["families"][fam_name]
        mask = np.array(fam["in_frame_mask"], bool)
        unit = "m/s" if fam_name == "velocity" else "m/s²"
        for ax, n in zip(axrow, names):
            e = fam["readouts"][n][tag]
            jit = np.random.default_rng(0).uniform(-0.04, 0.04, int(mask.sum()))
            for lam in LAMS:
                x = np.array(fam["effective"][f"lam{lam:g}"])[mask]
                y = np.array(e["pred_by_source"][f"lam{lam:g}"])[mask]
                ax.scatter(x * (1 + jit), y, s=3, alpha=0.3, color=colors[lam], label=f"λ={lam:g}")
            lim = np.array(fam["effective"]["lam2"])[mask].max()
            ax.plot([0, lim], [0, lim], "k--", lw=0.7)
            lo, hi = fam["probe_train_range"]
            ax.axvspan(lo, hi, color="0.92", zorder=-1)
            p, r = e["in_frame"]["pooled"], e["in_frame"]["ratio_2_over_1"]
            ax.set_title(f"{n.replace('vjepa2_p', 'V-JEPA point ')}: slope {p['slope']:.2f}\n"
                         f"ratio λ2/λ1 {r['median']:.2f} vs linear {r['expected_linear_median']:.2f}", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlabel(f"effective label ({unit})", fontsize=7)
            ins = ax.inset_axes([0.64, 0.07, 0.32, 0.24])
            ins.bar(range(3), [e["in_frame"][f"lam{lam:g}"]["direction_mae_deg"] for lam in LAMS],
                    color=[colors[lam] for lam in LAMS])
            ins.set_xticks([])
            ins.tick_params(labelsize=5)
            ins.set_title("direction MAE (°)", fontsize=5)
        axrow[0].set_ylabel(f"{label} readout ({unit})", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="upper left", markerscale=3)
    fig.suptitle("Frozen-probe linearity/extrapolation (time-rescaled clips = speed x λ, accel x λ²; not a clock test): probes fit on supplied sets (training range shaded), "
                 "(in-frame clips); dashed = identity", fontsize=9)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "fig_clock_test.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

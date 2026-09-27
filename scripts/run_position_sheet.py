"""A 2-D sheet variable: start position (start_x, start_y) on the speed set (constant speed; starts span the data's
own range, [-1.2, 1.2]^2 here). Lowest-priority Part 2 extra.

Geometry (train rows): G x G grid of position bins over the knot rows' range, PCA-k, cell centroids, participation
ratio, Procrustes R^2 of the centroids onto the true (x, y) cell centres, and the unsupervised 2-D coordinate (top-2
principal axes of the centroids) vs the labels (Procrustes R^2, reflection allowed).

Steering (same roles as run_part2: knot folds 0-2 build, probe folds 3-4 read, test is steered): one interior 2 x 2
block of cells is held out. Arms, all additive in PCA-k with the clip's residual kept:
  tps          thin-plate spline (Goodfire A.3, TPS case) through the kept cell centroids, walked along the straight
               line in (x, y) from the clip's position to the target cell centre. Smoothing: --tps-smoothing auto
               (default) picks it by held-out-block reconstruction on train rows (like run_part2 --spline smooth)
  chord        straight line between the TPS values at source and target (endpoint-matched to tps)
  chord_linear_interp  straight line between piecewise-linear (Delaunay) interpolants at source and target (not
               endpoint-matched; nearest-neighbour fallback outside the hull, counted)
  chord_dose_matched, projected, reflected   as in run_part2
Controls: endpoint-matched random bends (>= 20 draws). Readouts: a ridge (x, y) probe on probe folds (endpoint and
along the path), nearest-real agreement with real test clips in the target cell, off-manifold energy against a
reference TPS through probe-fold centroids (no arm was built from it) and the mean distance to the 5 nearest
probe-fold clips. Gaps: paired bootstrap over clips.

Writes results/p2_sheet_speed_L{layer}.json and figures/fig4_sheet_speed_L{layer}.png.

  python scripts/run_position_sheet.py --layer 12
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

_spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)


def cells(pos, lo, hi, G):
    """Cell index (row-major, x fastest) and the G x G cell centres."""
    edges = np.linspace(lo, hi, G + 1)
    ij = np.clip(np.searchsorted(edges[1:-1], pos, side="right"), 0, G - 1)            # [n, 2]
    ctr = (edges[:-1] + edges[1:]) / 2
    centres = np.array([[ctr[i], ctr[j]] for j in range(G) for i in range(G)])        # cell c = j*G + i
    return ij[:, 1] * G + ij[:, 0], centres


def cell_centroids(Z, cell, ids):
    return np.array([Z[cell == c].mean(0) for c in ids])


def geometry(X, cell, centres, k):
    pca = mf.fit_pca(X, k)
    ids = np.unique(cell)
    C = cell_centroids(pca.project(X), cell, ids)
    T = centres[ids]
    Cc = C - C.mean(0)
    top2 = Cc @ np.linalg.svd(Cc, full_matrices=False)[2][:2].T
    return {"n_cells": int(len(ids)), "min_count": int(np.bincount(cell)[ids].min()),
            "participation_ratio_centroids": gc.participation_ratio(C),
            "procrustes_r2_centroids_to_xy": gc.procrustes_r2(C, T)["r2"],
            "unsupervised_top2_vs_xy_procrustes_r2": gc.procrustes_r2(top2, T)["r2"],
            "top2_frac_of_centroid_variance": float((top2 ** 2).sum() / (Cc ** 2).sum())}, pca, C, ids, top2


SMOOTHING_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def linear_interp(pts, vals):
    """Delaunay-linear interpolant with a nearest-neighbour fallback where it is NaN (outside the hull). Returns
    (f, counter) where counter["nan_fallbacks"] counts the rows that fell back."""
    lin, near = LinearNDInterpolator(pts, vals), NearestNDInterpolator(pts, vals)
    counter = {"nan_fallbacks": 0}

    def f(P):
        P = np.atleast_2d(P)
        out = lin(P)
        bad = np.isnan(out).any(-1)
        if bad.any():
            out[bad] = near(P[bad])
            counter["nan_fallbacks"] += int(bad.sum())
        return out
    return f, counter


def selection_blocks(G, eval_seed, n=4, max_seed=500):
    """Blocks for choosing the smoothing: distinct 2 x 2 blocks that do not overlap the evaluation block (so no
    train-row centroid of an evaluated cell ever enters the choice). Returns (seeds, blocks)."""
    ev = set(held_block(G, eval_seed))
    seeds, blocks = [], []
    for sd in range(max_seed):
        b = held_block(G, sd)
        if sd == eval_seed or ev & set(b) or b in blocks:
            continue
        seeds.append(sd)
        blocks.append(b)
        if len(seeds) == n:
            break
    return seeds, blocks


def choose_smoothing(X, cell, centres, G, k, seeds, exclude=()):
    """Held-out 2 x 2 block reconstruction on train rows: TPS at each smoothing in SMOOTHING_GRID and the Delaunay
    chord rebuild the block's centroids from the other cells; mean error over the `seeds` blocks (selection_blocks:
    never the evaluation block, never overlapping it). A development decision made on train only. Rows of the
    `exclude` cells (the evaluation block) are dropped before anything is fitted, so they are never anchors either.
    Returns (best smoothing, table)."""
    keep_ex = ~np.isin(cell, list(exclude))
    X, cell = X[keep_ex], cell[keep_ex]
    table = {str(sm): [] for sm in SMOOTHING_GRID}
    table["linear_interp"] = []
    for seed in seeds:
        held = held_block(G, seed)
        keep = ~np.isin(cell, held)
        pca = mf.fit_pca(X[keep], k)
        ids = np.unique(cell[keep])
        C = cell_centroids(pca.project(X[keep]), cell[keep], ids)
        truth = cell_centroids(pca.project(X), cell, held)
        for sm in SMOOTHING_GRID:
            f = RBFInterpolator(centres[ids], C, kernel="thin_plate_spline", smoothing=sm)
            table[str(sm)].append(float(np.linalg.norm(f(centres[held]) - truth, axis=1).mean()))
        li, _ = linear_interp(centres[ids], C)
        table["linear_interp"].append(float(np.linalg.norm(li(centres[held]) - truth, axis=1).mean()))
    mean = {kk: float(np.mean(v)) for kk, v in table.items()}
    best = min(SMOOTHING_GRID, key=lambda sm: mean[str(sm)])
    return best, {"mean_error": mean, "seeds": list(seeds), "blocks": [held_block(G, sd) for sd in seeds],
                  "best_tps_smoothing": best,
                  "tps_beats_linear_interp": mean[str(best)] < mean["linear_interp"]}


def sheet_verdict(cat, metrics=("err_path", "excess_to_ref", "excess_to_nearest_real"), seed=0):
    """negative when the TPS path is not better than the endpoint-matched chord on any path metric (paired bootstrap
    over clips, 95% CI): "TPS path indistinguishable from chord" if no CI excludes 0, else "chord better"."""
    better, worse = [], []
    for q in metrics:
        b = p2.paired_bootstrap(cat["tps"][q] - cat["chord"][q], cat["tps"]["_id"], seed=seed)
        if b["ci95"][1] < 0:
            better.append(q)
        elif b["ci95"][0] > 0:
            worse.append(q)
    if not better:
        text = ("negative: TPS path indistinguishable from chord" if not worse
                else f"negative: chord better than the TPS path on {worse}")
    else:
        text = f"TPS path better than chord on {better}" + (f", worse on {worse}" if worse else "")
    return {"text": text, "tps_better_on": better, "tps_worse_on": worse, "rule": "95% CI of tps - chord (clips)"}


def held_block(G, seed):
    """One interior 2 x 2 block of cells (never touching the grid edge), chosen by seed."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(1, G - 2, size=2)
    return sorted(int((j + b) * G + (i + a)) for a in (0, 1) for b in (0, 1))


def run(args):
    d = load_inputs("speed", args.layer, "speed", args.act_dir, args.table, args.split)
    pos = d["df"][["start_x", "start_y"]].to_numpy(float)
    knot_all, probe, test = d["role"] == "knot", d["role"] == "probe", d["role"] == "test"
    lo, hi = pos[knot_all].min(0).min(), pos[knot_all].max(0).max()
    cell, centres = cells(pos, lo, hi, args.grid)
    tr = d["is_train"]
    geo, *_ = geometry(d["X"][tr], cell[tr], centres, args.k)

    smooth_note = None
    if args.tps_smoothing == "auto":
        sel_seeds, _ = selection_blocks(args.grid, args.seed)
        smoothing, smooth_note = choose_smoothing(d["X"][tr], cell[tr], centres, args.grid, args.k, sel_seeds,
                                                  exclude=held_block(args.grid, args.seed))
        smooth_note["evaluation_block_excluded"] = held_block(args.grid, args.seed)
    else:
        smoothing = float(args.tps_smoothing)
    held = held_block(args.grid, args.seed)
    knot = knot_all & ~np.isin(cell, held)
    pca = mf.fit_pca(d["X"][knot], args.k)
    kept = np.unique(cell[knot])
    Ck = cell_centroids(pca.project(d["X"][knot]), cell[knot], kept)
    tps = RBFInterpolator(centres[kept], Ck, kernel="thin_plate_spline", smoothing=smoothing)
    lin, lin_count = linear_interp(centres[kept], Ck)
    clo, chi = centres.min(0), centres.max(0)
    lin_at = lambda P: lin(np.clip(P, clo, chi))
    # reference manifold nobody built from: TPS through probe-fold centroids at all cells, on a dense grid
    pids = np.unique(cell[probe])
    ref = RBFInterpolator(centres[pids], cell_centroids(pca.project(d["X"][probe]), cell[probe], pids),
                          kernel="thin_plate_spline", smoothing=smoothing)
    g = np.linspace(clo, chi, 60)
    ref_pts = ref(np.array([[a, b] for a in g[:, 0] for b in g[:, 1]]))
    X_ref = d["X"][probe]
    probe_model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15))).fit(d["X"][probe], pos[probe])
    rng = np.random.default_rng(args.seed)
    coefs = [rng.standard_normal((3, pca.components.shape[0])) / np.arange(1, 4)[:, None] for _ in range(args.n_controls)]

    arms = ("tps", "chord", "chord_linear_interp", "chord_dose_matched", "projected", "reflected")
    acc = {a: {q: [] for q in ("err_end", "err_path", "nearest_real_R", "excess_to_ref", "excess_to_nearest_real",
                               "delta_norm", "_id")} for a in arms}
    ctrl = [{q: [] for q in ("err_path", "excess_to_ref")} for _ in coefs]   # endpoints are matched
    ctrl_ref = {q: [] for q in ("err_path", "excess_to_ref")}
    s = np.linspace(0, 1, args.K)
    pool = np.flatnonzero(test & ~np.isin(cell, held))
    tpos_all = {c: centres[c] for c in held}
    for c, tgt in tpos_all.items():
        pick = rng.choice(pool, size=min(args.n_clips, len(pool)), replace=False)
        x = d["X"][pick].astype(float)
        src = pos[pick]
        Z, resid = pca.project(x), pca.complement(x)
        path_xy = src[:, None] + s[None, :, None] * (tgt - src)[:, None]                  # [n, K, 2]
        T = tps(path_xy.reshape(-1, 2)).reshape(len(pick), args.K, -1)
        Zs = Z[:, None] + T - T[:, :1]
        Zl = mf.linear_coords(Z, T[:, 0], T[:, -1], args.K)                    # endpoint-matched to tps
        Zli = mf.linear_coords(Z, lin_at(src), lin_at(tgt[None])[0], args.K)
        dl, ds = Zl - Z[:, None], Zs - Z[:, None]
        nl, ns = np.linalg.norm(dl, axis=-1, keepdims=True), np.linalg.norm(ds, axis=-1, keepdims=True)
        Zd = Z[:, None] + dl * np.where(nl > 0, ns / np.where(nl > 0, nl, 1.0), 0.0)
        proj, refl = mf.chord_coords(Zs)
        real_t = d["X"][test & (cell == c)].mean(0)

        def score(Zk, xs=x, rs=resid):
            W = pca.lift(Zk) + rs[:, None]
            pr = probe_model.predict(W.reshape(-1, W.shape[-1])).reshape(len(xs), args.K, 2)
            err = np.linalg.norm(pr - tgt, axis=-1)
            dc = mf.min_distance(pca.project(W), ref_pts)
            dn = mf.knn_distance(W, X_ref, 5)
            return {"err_end": err[:, -1], "err_path": err.mean(1),
                    "nearest_real_R": mf.nearest_real_agreement(W[:, -1], xs, real_t),
                    "excess_to_ref": dc.mean(1) - dc[:, 0], "excess_to_nearest_real": dn.mean(1) - dn[:, 0],
                    "delta_norm": np.linalg.norm(W[:, -1] - xs, axis=-1)}

        for a, Zk in zip(arms, (Zs, Zl, Zli, Zd, proj, refl)):
            sc = score(Zk)
            for q in acc[a]:
                acc[a][q].append(d["df"]["id"].to_numpy()[pick] if q == "_id" else sc[q])
        sub = slice(0, args.n_control_clips)
        sc = score(Zs[sub], x[sub], resid[sub])
        for q in ctrl_ref:
            ctrl_ref[q].append(sc[q])
        for i, cf in enumerate(coefs):
            sc = score(mf.endpoint_matched_random(Zs[sub], cf), x[sub], resid[sub])
            for q in ctrl[i]:
                ctrl[i][q].append(sc[q])

    cat = {a: {q: np.concatenate(v) for q, v in qs.items()} for a, qs in acc.items()}
    metrics = [q for q in acc["tps"] if q != "_id"]
    out = {"dataset": "speed", "variable": "start_position", "layer": args.layer,
           **layer_role("speed", args.layer, "speed"),
           "provenance": provenance(args.split, seeds={"seed": args.seed}, layer=args.layer, pool="meanpool",
                                    k=args.k, grid=args.grid, K=args.K),
           "grid": {"G": args.grid, "range": [float(lo), float(hi)], "held_out_cells": held,
                    "held_out_centres": centres[held].tolist()},
           "geometry_train": geo,
           "tps_smoothing": smoothing, "tps_smoothing_choice": smooth_note,
           "linear_interp_nan_fallbacks": lin_count["nan_fallbacks"],
           "arms": {"tps": "thin-plate spline through kept cell centroids, walked along the straight (x, y) line",
                    "chord": "straight line between the TPS values at source and target (endpoint-matched)",
                    "chord_linear_interp": "straight line between Delaunay-linear interpolants at source and target "
                                           "(not endpoint-matched; nearest-neighbour fallback outside the hull)",
                    "chord_dose_matched": "chord delta rescaled per clip and waypoint to the tps arm's ||delta||",
                    "projected": "tps path's own chord with its arc-length spacing",
                    "reflected": "2 * projected - tps"},
           "summary": {a: {q: float(cat[a][q].mean()) for q in metrics} for a in arms},
           "gaps": {f"tps_minus_{a}": {q: p2.paired_bootstrap(cat["tps"][q] - cat[a][q], cat["tps"]["_id"],
                                                               seed=args.seed) for q in metrics}
                    for a in arms if a != "tps"},
           "controls": {"random_endpoint_matched": {
               q: p2.band([np.concatenate(c[q]).mean() for c in ctrl], np.concatenate(ctrl_ref[q]).mean(), False)
               for q in ctrl_ref}},
           "verdict": sheet_verdict(cat),
           "notes": ["err = distance (m) from the (x, y) probe readout to the target cell centre",
                     "energy reference: TPS through probe-fold cell centroids (all cells), dense 60 x 60 grid",
                     "gaps: paired bootstrap over clips (each clip's targets together)"]}
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    tag = f"speed_L{args.layer}"
    (Path(args.results_dir) / f"p2_sheet_{tag}.json").write_text(json.dumps(out, indent=1))
    plot(d, cell, centres, tr, args, out, Path(args.figures_dir) / f"fig4_sheet_{tag}.png")
    return out


def plot(d, cell, centres, tr, args, out, path):
    _, _, C, ids, top2 = geometry(d["X"][tr], cell[tr], centres, args.k)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    G = args.grid
    lut = {c: p for c, p in zip(ids, top2)}
    for j in range(G):
        row = [lut[j * G + i] for i in range(G) if j * G + i in lut]
        col = [lut[i * G + j] for i in range(G) if i * G + j in lut]
        if len(row) > 1:
            axes[0].plot(*np.array(row).T, color="C0", lw=0.8)
        if len(col) > 1:
            axes[0].plot(*np.array(col).T, color="C1", lw=0.8)
    held = out["grid"]["held_out_cells"]
    axes[0].scatter(*np.array([lut[c] for c in held if c in lut]).T, color="k", zorder=3, label="held-out block")
    axes[0].set(title=f"Cell centroids, top-2 axes (Procrustes R2 vs xy "
                      f"{out['geometry_train']['unsupervised_top2_vs_xy_procrustes_r2']:.2f})",
                xlabel="centroid axis 1", ylabel="centroid axis 2")
    axes[0].legend(fontsize=7)
    arms = list(out["summary"])
    axes[1].bar(range(len(arms)), [out["summary"][a]["err_end"] for a in arms], color="0.6", label="endpoint")
    axes[1].bar(range(len(arms)), [out["summary"][a]["err_path"] for a in arms], fill=False, label="path mean")
    axes[1].set_xticks(range(len(arms)), arms, rotation=30, fontsize=7)
    axes[1].set(ylabel="probe error to target cell centre (m)", title="Held-out 2 x 2 block, steered test clips")
    axes[1].legend(fontsize=7)
    fig.text(0.01, 0.005, "verdict: " + out["verdict"]["text"], fontsize=6, color="0.25", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--grid", type=int, default=6)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--n-clips", type=int, default=48)
    p.add_argument("--n-controls", type=int, default=20)
    p.add_argument("--n-control-clips", type=int, default=16)
    p.add_argument("--tps-smoothing", default="auto",
                   help="'auto' (held-out-block reconstruction on train rows) or a number; 0 = interpolating (A.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

"""Local intrinsic dimension and density where the model reads (the Part 2 second-look note, section D, "To make the full
claim"). CPU only, stored meanpool activations, direction set; roles as run_part2 (knot folds 0-2 fit every space and
the spline, probe folds 3-4 + test clips are held out, test clips are the steering carriers).

Spaces (all fit on knot clips at all 64 values):
  full      the 1024-d meanpool activation
  pca64     PCA-64 (run_two_route.build_all_values)
  leace2w   LEACE-style whitened direction plane: in PCA-64, W = Sigma^-1/2 (total covariance), basis = orth(W Sigma_zy)
            for y = (sin, cos); coordinates U^T W (z - mu). The plane any linear (sin, cos) reader uses, in units where
            every direction of the data has unit variance
  lda8w     whitened discriminant subspace: within-value covariance whitened (Sw^-1/2), top 8 PCs of the 64 whitened
            value means; coordinates V^T Sw^-1/2 (z - mu). Where direction information lives, including harmonics,
            with nuisance variance equalised
  chart2    the circular chart plane of geometry_checks.fit_circular_chart (chart units; for continuity with
            p2_ring_occupancy)

A. Levina-Bickel MLE intrinsic dimension (Levina & Bickel 2004; pooled with MacKay & Ghahramani's inverse-mean
   correction) per 45-degree direction bin (8 bins, by true label) on held-out clips (probe + test roles), neighbours
   drawn from all held-out clips; k = 5, 10, 20 (10 primary); 95% bootstrap CI over the clips in the bin.

B. 5-nearest-neighbour density of steering midpoints vs real clips at the same angle. For every test clip (source
   theta), shift delta in {90, 135, 180} and both senses: spline midpoint = z + curve(t(theta + s delta/2)) -
   curve(t(theta)) (the spline point at the nominal midpoint angle, shift mode, residual kept), chord midpoint = z +
   (P(theta) + P(theta + s delta))/2 - P(theta) with P the smoothed-knot polyline (run_two_route's chord). Real clips
   at the same angle = the other test clips whose label is the nominal midpoint angle theta + s delta/2 (snapped to
   the 64-value grid). Reference = real train clips: "train" (knot + probe folds; primary) and "probe" (probe folds
   only, disjoint from the spline's knots). Reported per space: the mean 5-NN distance of each midpoint divided by
   the mean 5-NN distance of the real clips at that angle (geometric mean over carriers; 1 = as dense as real clips
   there), the paired chord-minus-spline log ratio with a carrier-grouped bootstrap CI, and the fraction of midpoints
   beyond the 95th percentile of all real test clips' 5-NN distance.

  python scripts/run_local_density.py --layers 8 12 22
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import layer_role, provenance

SPACES = ("full", "pca64", "leace2w", "lda8w", "chart2")
SHIFTS = (90.0, 135.0, 180.0)
KS = (5, 10, 20)
N_BINS = 8


# ------------------------------------------------------------------ estimators (tested on planted data) -------------

def pairwise_dist(A, B=None):
    A = np.asarray(A, float)
    B = A if B is None else np.asarray(B, float)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def lb_inverse(X, k, query=None):
    """Levina-Bickel per-point inverse dimension estimates m_k(x)^-1 = 1/(k-1) sum_{j<k} log(T_k / T_j), neighbours
    from X (self excluded). query: row indices of X to estimate at (default all)."""
    X = np.asarray(X, float)
    q = np.arange(len(X)) if query is None else np.asarray(query)
    D = pairwise_dist(X[q], X)
    D[np.arange(len(q)), q] = np.inf                                   # exclude self
    T = np.sort(np.partition(D, k - 1, axis=1)[:, :k], axis=1)
    T = np.maximum(T, 1e-12)
    return np.log(T[:, -1:] / T[:, :-1]).mean(1)


def lb_dimension(inv):
    """Pooled Levina-Bickel dimension (MacKay-Ghahramani: invert the mean of the inverses)."""
    return float(1.0 / np.mean(inv))


def lb_boot_ci(inv, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    inv = np.asarray(inv, float)
    b = 1.0 / inv[rng.integers(0, len(inv), (n, len(inv)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def knn_mean_dist(A, ref, k=5):
    return mf.knn_distance(np.asarray(A, float), np.asarray(ref, float), k)


def grouped_boot(v, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in ug]
    sums = np.array([v[i].sum() for i in idx])
    cnts = np.array([len(i) for i in idx])
    s = rng.integers(0, len(ug), (n, len(ug)))
    means = sums[s].sum(1) / cnts[s].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ------------------------------------------------------------------ spaces ------------------------------------------

def inv_sqrt(S, eps=1e-6):
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps * w.max())
    return (V / np.sqrt(w)) @ V.T


def fit_spaces(X_knot, y_knot, pca, chart, n_lda=8):
    """Projection functions full activation [n, D] -> coordinates, for every space."""
    Zk = pca.project(X_knot)
    mu = Zk.mean(0)
    Zc = Zk - mu
    Y = np.stack([np.sin(np.radians(y_knot)), np.cos(np.radians(y_knot))], 1)
    W = inv_sqrt(np.cov(Zc, rowvar=False))
    Szy = Zc.T @ (Y - Y.mean(0)) / len(Zc)
    U, _ = np.linalg.qr(W @ Szy)                                          # [k, 2]
    vals = np.unique(y_knot)
    M = np.array([Zk[y_knot == v].mean(0) for v in vals])
    R = Zk - M[np.searchsorted(vals, y_knot)]
    Ws = inv_sqrt(R.T @ R / (len(R) - len(vals)))
    Mw = (M - mu) @ Ws
    _, _, Vt = np.linalg.svd(Mw - Mw.mean(0), full_matrices=False)
    V = Vt[:n_lda].T
    Ainv = np.linalg.pinv(chart["A"])
    return {"full": lambda X: np.asarray(X, float),
            "pca64": lambda X: pca.project(X),
            "leace2w": lambda X: (pca.project(X) - mu) @ W @ U,
            "lda8w": lambda X: (pca.project(X) - mu) @ Ws @ V,
            "chart2": lambda X: (np.asarray(X, float) - chart["mu"]) @ Ainv.T}


# ------------------------------------------------------------------ A. intrinsic dimension --------------------------

def intrinsic_dimension(Xh, yh, to_space, seed=0):
    out = {}
    edges = np.arange(N_BINS + 1) * 360.0 / N_BINS
    for sp, f in to_space.items():
        Zh = f(Xh)
        e = {"all": {}, "bins": []}
        inv = {k: lb_inverse(Zh, k) for k in KS}
        for k in KS:
            e["all"][f"k{k}"] = {"id": lb_dimension(inv[k]), "ci95": lb_boot_ci(inv[k], seed=seed)}
        for b in range(N_BINS):
            sel = (yh % 360.0 >= edges[b]) & (yh % 360.0 < edges[b + 1])
            row = {"bin_deg": [float(edges[b]), float(edges[b + 1])], "n": int(sel.sum())}
            for k in KS:
                row[f"k{k}"] = {"id": lb_dimension(inv[k][sel]), "ci95": lb_boot_ci(inv[k][sel], seed=seed)}
            e["bins"].append(row)
        ids = np.array([r["k10"]["id"] for r in e["bins"]])
        e["k10_bin_range"] = [float(ids.min()), float(ids.max())]
        e["dims"] = int(Zh.shape[1])
        out[sp] = e
    return out


# ------------------------------------------------------------------ B. midpoint density -----------------------------

def midpoints(pca, curve, X, y, delta, sign):
    """Spline and chord midpoints (full space) for clips X at labels y steered by sign * delta."""
    Z, resid = pca.project(X), pca.complement(X)
    tgt = (y + sign * delta) % 360.0
    mid_v = (y + sign * delta / 2.0) % 360.0
    ta, tm = curve.coord_of_value(y), curve.coord_of_value(mid_v)
    Zs = Z + curve(tm) - curve(ta)
    pa, pb = mf.piecewise_linear_point(curve, y), mf.piecewise_linear_point(curve, tgt)
    Zc = Z + (pa + pb) / 2.0 - pa
    return pca.lift(Zs) + resid, pca.lift(Zc) + resid, mid_v


def density_contrast(d_chord, d_spline, d_real, groups, p95):
    """Log-ratio summaries of chord / spline midpoint 5-NN distance vs real clips at the same angle."""
    lc, ls = np.log(d_chord / d_real), np.log(d_spline / d_real)
    diff = lc - ls
    return {"chord_over_real_geomean": float(np.exp(lc.mean())), "chord_over_real_ci95": list(
                np.exp(grouped_boot(lc, groups))),
            "spline_over_real_geomean": float(np.exp(ls.mean())), "spline_over_real_ci95": list(
                np.exp(grouped_boot(ls, groups))),
            "chord_minus_spline_logratio": float(diff.mean()), "chord_minus_spline_ci95": grouped_boot(diff, groups),
            "frac_chord_beyond_real_p95": float(np.mean(d_chord > p95)),
            "frac_spline_beyond_real_p95": float(np.mean(d_spline > p95)),
            "frac_carriers_chord_farther": float(np.mean(diff > 0)), "n": int(len(diff))}


def density(d, pca, curve, to_space, k=5):
    test = np.flatnonzero(d["role"] == "test")
    refs = {"train": d["role"] != "test", "probe": d["role"] == "probe"}
    Xt, yt = d["X"][test].astype(float), d["y"][test]
    vals = np.unique(d["y"])
    step = 360.0 / len(vals)
    out = {}
    for rname, rmask in refs.items():
        out[rname] = {}
        for sp, f in to_space.items():
            ref = f(d["X"][rmask])
            real = knn_mean_dist(f(Xt), ref, k)                          # [n_test]
            by_val = {v: real[np.isclose(yt, v)] for v in vals}
            p95 = float(np.percentile(real, 95))
            e = {"real_test_5nn_median": float(np.median(real)), "real_test_5nn_p95": p95}
            for delta in SHIFTS:
                dc, ds, dr, g = [], [], [], []
                for sign in (1.0, -1.0):
                    Ws, Wc, mid_v = midpoints(pca, curve, Xt, yt, delta, sign)
                    ds.append(knn_mean_dist(f(Ws), ref, k))
                    dc.append(knn_mean_dist(f(Wc), ref, k))
                    snap = vals[np.rint(((mid_v - vals[0]) % 360.0) / step).astype(int) % len(vals)]
                    dr.append(np.array([by_val[v].mean() for v in snap]))
                    g.append(test)
                e[f"{delta:.0f}"] = density_contrast(np.concatenate(dc), np.concatenate(ds), np.concatenate(dr),
                                                     np.concatenate(g), p95)
            out[rname][sp] = e
    return out


# ------------------------------------------------------------------ driver ------------------------------------------

def plot(res, path):
    layers = sorted(res, key=int)
    col = {"full": "#888888", "pca64": "#2a6fdb", "leace2w": "#d0421b", "lda8w": "#7a3fbf", "chart2": "#2e9d57"}
    ls = {layers[i]: s for i, s in enumerate(["-", "--", ":", "-."][:len(layers)])}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    centres = (np.arange(N_BINS) + 0.5) * 360.0 / N_BINS
    for L in layers:
        for sp in SPACES:
            b = res[L]["intrinsic_dimension"][sp]["bins"]
            axes[0].plot(centres, [r["k10"]["id"] for r in b], ls[L], color=col[sp],
                         label=f"{sp}" if L == layers[0] else None)
    axes[0].set_title("Levina-Bickel ID (k = 10) per 45-deg bin, held-out clips\n(line style = point: " +
                      ", ".join(f"{L} {ls[L]}" for L in layers) + ")", fontsize=8)
    axes[0].set_xlabel("direction bin centre (deg)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    w = 0.8 / len(layers)
    x = np.arange(len(SPACES))
    for i, L in enumerate(layers):
        for j, (key, mk) in enumerate((("chord", "o"), ("spline", "s"))):
            v = [res[L]["density"]["train"][sp]["180"][f"{key}_over_real_geomean"] for sp in SPACES]
            ci = np.array([res[L]["density"]["train"][sp]["180"][f"{key}_over_real_ci95"] for sp in SPACES])
            xx = x - 0.4 + w * (i + 0.5) + (0.12 if key == "spline" else -0.12) * w
            axes[1].errorbar(xx, v, yerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], fmt=mk, ms=4,
                             color=["#2a6fdb", "#e07b00", "#555555"][i % 3], mfc="white" if key == "spline" else None,
                             label=f"point {L} {key}")
    axes[1].axhline(1.0, color="#bbbbbb", lw=0.8)
    axes[1].set_xticks(x, SPACES, fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_title("180-deg midpoint 5-NN distance / real clips at that angle\n(train reference; filled chord, open "
                      "spline; 95% CI)", fontsize=8)
    axes[1].legend(fontsize=6, frameon=False)
    for i, L in enumerate(layers):
        for sp in SPACES:
            v = [res[L]["density"]["train"][sp][f"{dl:.0f}"]["chord_minus_spline_logratio"] for dl in SHIFTS]
            axes[2].plot(SHIFTS, v, ls[L], marker="o", ms=3, color=col[sp], label=sp if L == layers[0] else None)
    axes[2].axhline(0.0, color="#bbbbbb", lw=0.8)
    axes[2].set_xticks(SHIFTS)
    axes[2].set_xlabel("shift (deg)", fontsize=8)
    axes[2].set_title("chord minus spline midpoint log(5-NN distance ratio)\n(> 0: chord midpoint sparser)", fontsize=8)
    axes[2].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    rtr_spec = importlib.util.spec_from_file_location("run_two_route", PROJECT_ROOT / "scripts" / "run_two_route.py")
    rtr = importlib.util.module_from_spec(rtr_spec)
    rtr_spec.loader.exec_module(rtr)
    out = {"dataset": "direction", "variable": "direction",
           "provenance": provenance(args.split, seeds={"bootstrap": 0}, layers=args.layers, k=args.k, knn=5,
                                    lb_k=list(KS), spline="smooth", pool="meanpool"),
           "question": "is the chord midpoint in a low-density region of the space the model's direction readout uses "
                       "(whitened direction subspaces), not only in the 2-D ring plane; and what is the local "
                       "intrinsic dimension of the direction cloud?",
           "metric_notes": {
               "intrinsic_dimension": "Levina-Bickel MLE, MacKay-Ghahramani pooling, per 45-degree bin of held-out "
                                      "clips (probe + test roles), neighbours among all held-out clips; ci95 = "
                                      "bootstrap over the bin's clips. leace2w and chart2 are 2-D, so their ID is "
                                      "capped near 2 (a thin ring reads ~1, a filled disc ~2)",
               "density": "mean distance to the 5 nearest reference clips; *_over_real = midpoint distance / mean "
                          "distance of the real test clips at the nominal midpoint angle (geometric mean over "
                          "carriers x both senses); chord_minus_spline_logratio > 0 means the chord midpoint is "
                          "sparser; ci95 = bootstrap over carriers",
               "reference": "train = knot + probe folds (primary); probe = probe folds only (disjoint from the "
                            "spline's knots)",
               "spline": "smoothing spline on all 64 values from knot clips (run_two_route.build_all_values): a "
                         "density question about where the midpoints land, not a held-out steering test"},
           "spaces": {"full": "1024-d meanpool", "pca64": "PCA-64 (knot clips)",
                      "leace2w": "LEACE-style whitened (sin, cos) plane in PCA-64",
                      "lda8w": "top-8 within-value-whitened discriminant subspace in PCA-64",
                      "chart2": "circular chart plane (chart units)"},
           "layers": {}}
    for L in args.layers:
        d = load_inputs("direction", L, "direction", args.act_dir, args.table, args.split)
        knot = d["role"] == "knot"
        pca, curve, choice = rtr.build_all_values(d, args.k, "smooth")
        chart = gc.fit_circular_chart(d["X"][knot], d["y"][knot])
        to_space = fit_spaces(d["X"][knot], d["y"][knot], pca, chart)
        held = d["role"] != "knot"
        idim = intrinsic_dimension(d["X"][held].astype(float), d["y"][held], to_space)
        dens = density(d, pca, curve, to_space)
        e = {"layer_role": layer_role("direction", L)["layer_role"], "angle_source": curve.coord_source,
             "n_heldout_clips": int(held.sum()), "n_test_carriers": int((d["role"] == "test").sum()),
             "intrinsic_dimension": idim, "density": dens}
        t = dens["train"]
        e["verdict"] = (f"point {L}: ID (k=10, all held-out) " + ", ".join(
            f"{sp} {idim[sp]['all']['k10']['id']:.1f}" for sp in SPACES) + "; per-bin range " + ", ".join(
            f"{sp} {idim[sp]['k10_bin_range'][0]:.1f}-{idim[sp]['k10_bin_range'][1]:.1f}" for sp in SPACES) +
            ". 180-deg midpoint 5-NN distance / real clips at that angle (train ref), chord vs spline: " + ", ".join(
            f"{sp} {t[sp]['180']['chord_over_real_geomean']:.2f} vs {t[sp]['180']['spline_over_real_geomean']:.2f} "
            f"(diff {t[sp]['180']['chord_minus_spline_logratio']:+.3f} [{t[sp]['180']['chord_minus_spline_ci95'][0]:+.3f}, "
            f"{t[sp]['180']['chord_minus_spline_ci95'][1]:+.3f}])" for sp in SPACES) + ".")
        print(e["verdict"], flush=True)
        out["layers"][str(L)] = e
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    plot(out["layers"], args.figure)
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[8, 12, 22])
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "p2_local_density.json"))
    p.add_argument("--figure", default=str(PROJECT_ROOT / "figures" / "fig_local_density.png"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

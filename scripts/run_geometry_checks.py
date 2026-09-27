"""Cheap geometry checks on one layer, before any steering (nonobvious_components.md §7).

Uses train clips only. Writes results/p2_geometry_{dataset}_L{layer}.json and figures/fig4_centroid_plane_*.png,
figures/fig4_loo_stride_*.png.

  python scripts/run_geometry_checks.py --dataset direction --layer 12
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs

STRIDES = (1, 2, 4, 8, 16)


def plane_coords(C, plane):
    """2-D coordinates for plotting: first two activation PCs, or the centroid plane."""
    if plane == "activation":
        return lambda P: np.asarray(P)[..., :2]
    mean = C.mean(0)
    Vt = np.linalg.svd(C - mean, full_matrices=False)[2][:2]
    return lambda P: (np.asarray(P) - mean) @ Vt.T


def direction_groups(df):
    """Motion type x tercile of its magnitude (speed for constant-velocity clips, acceleration otherwise)."""
    mag = np.where(df["motion"] == "velocity", df["speed_mps"], df["acceleration_mps2"])
    group = np.empty(len(df), dtype=object)
    for m in df["motion"].unique():
        rows = (df["motion"] == m).to_numpy()
        edges = np.quantile(mag[rows], [1 / 3, 2 / 3])
        group[rows] = [f"{m}_t{int(np.searchsorted(edges, v, side='right'))}" for v in mag[rows]]
    return group


def run(args):
    d = load_inputs(args.dataset, args.layer, args.variable, args.act_dir, args.table, args.split)
    tr = d["is_train"]
    X, y, periodic = d["X"][tr], d["y"][tr], d["periodic"]
    pca = mf.fit_pca(X, args.k)
    Z = pca.project(X)
    cent = mf.centroids(Z, y)
    out = {"dataset": args.dataset, "layer": args.layer, "variable": args.variable or args.dataset,
           "n_train": int(tr.sum()), "k": int(pca.components.shape[0]),
           "pca_explained_frac_top10": (pca.explained[:10] / pca.explained.sum()).tolist(),
           "n_values": len(cent["values"]), "min_count": int(cent["count"].min())}

    angle, plane = "unsupervised", "activation"
    if periodic:
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        angle, plane = choice["angle"], choice["plane"]
        out["angle"] = {"source": "labels_angle (fallback, flagged)" if angle == "labels" else f"unsupervised, {plane} plane",
                        **choice["checks"]}
    curve = mf.fit_curve(cent, periodic, angle=angle, plane=plane)
    spread = gc.align(cent["spread"], cent["values"], curve.values)
    count = gc.align(cent["count"], cent["values"], curve.values)

    out["loo"] = {}
    for s in STRIDES:
        pc = gc.precision_check(X, y, periodic, args.k, angle, plane, stride=s)
        noise_rel = gc.loo_reconstruction(curve, spread / np.sqrt(count), stride=s)
        out["loo"][str(s)] = {**pc, "fp32_line_err_over_noise": noise_rel["line_err_over_noise"]}
    rng = np.random.default_rng(0)
    out["loo_shuffled_control"] = {str(s): {q: gc.loo_reconstruction(gc.shuffled_curve(curve, rng), stride=s)[q]
                                            for q in ("mean_err_cubic", "mean_err_line", "winner")} for s in (1, 16)}
    out["spread_vs_curvature"] = gc.spread_vs_curvature(curve, spread, count)
    out["participation"] = gc.participation_check(Z, y)
    if not periodic:
        out["knot_spacing"] = gc.knot_spacing(curve)
    else:
        out["ring_radius_by_group"] = gc.ring_radius_by_group(Z, y, direction_groups(d["df"][tr]))
    if args.basis:
        V = np.load(args.basis)                       # [D, r], train-standardised space (Part 1 convention)
        mu, sd = X.mean(0), X.std(0) + 1e-8
        C_std = np.array([((X[y == v] - mu) / sd).mean(0) for v in cent["values"]])
        out["principal_angles_basis_vs_centroid_plane"] = gc.principal_angles_to_centroid_plane(V, C_std)

    tag = f"{args.dataset}_L{args.layer}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_geometry_{tag}.json").write_text(json.dumps(out, indent=1))
    plot_plane(curve, cent, plane, Path(args.figures_dir) / f"fig4_centroid_plane_{tag}.png", tag)
    plot_loo(out["loo"], Path(args.figures_dir) / f"fig4_loo_stride_{tag}.png", tag, periodic, len(cent["values"]))
    return out


def plot_plane(curve, cent, plane, path, tag):
    to2d = plane_coords(cent["C"], plane)
    _, dense = curve.dense(1000)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(*to2d(dense).T, color="0.3", lw=1, label="spline")
    sc = ax.scatter(*to2d(cent["C"]).T, c=cent["values"], cmap="twilight" if curve.periodic else "viridis", s=18, zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="label value")
    ax.set_xlabel("PC1" if plane == "activation" else "centroid axis 1")
    ax.set_ylabel("PC2" if plane == "activation" else "centroid axis 2")
    ax.set_title(f"Centroids and spline, {tag}")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loo(loo, path, tag, periodic, m):
    s = np.array([int(k) for k in loo])
    step = 360.0 / m if periodic else 1.0
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for prec, style in (("fp32", "-"), ("bf16", "--")):
        ax.plot(s * step, [loo[k][prec]["mean_err_line"] for k in loo], style, color="C0", label=f"line {prec}")
        ax.plot(s * step, [loo[k][prec]["mean_err_cubic"] for k in loo], style, color="C1", label=f"cubic {prec}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("knot spacing (degrees)" if periodic else "knot spacing (label steps)")
    ax.set_ylabel("reconstruction error (PCA units)")
    ax.set_title(f"Spline vs line at unseen knots, {tag}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("direction", "speed", "acceleration"))
    p.add_argument("--layer", type=int, required=True, help="index into the 26 stored points (0 = embedding)")
    p.add_argument("--variable", default=None)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--basis", default=None, help="optional .npy [D, r] basis (e.g. INLP) for principal angles")
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None, help="CSV overriding the manifest table (tests)")
    p.add_argument("--split", default=None, help="JSON overriding this dataset's split (tests)")
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

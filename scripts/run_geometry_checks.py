"""Cheap geometry checks on one layer, before any steering (nonobvious_components.md §7).

Uses train clips only. Writes results/p2_geometry_{dataset}_L{layer}.json and figures/fig4_centroid_plane_*.png,
figures/fig4_loo_stride_*.png. For direction it also fits the supervised circular chart X ~ mu + A[cos, sin] and
compares it with Goodfire's unsupervised atan2(PC2, PC1) ("circular_chart").

--planted-ring adds a synthetic ring of radius r (in within-value SDs, r in --planted-radii) along two random
orthogonal directions to the real train activations and checks that the pipeline recovers the angle and the
knot-subsampling cubic gain; writes results/p2_planted_ring_{dataset}_L{layer}.json (r vs recovery table).

  python scripts/run_geometry_checks.py --dataset direction --layer 12
  python scripts/run_geometry_checks.py --dataset direction --layer 12 --planted-ring
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
from wm.provenance import layer_role, provenance

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


def analyse(X, y, periodic, df, args, basis=None, full=True):
    """All checks on one activation matrix (train rows). full=False: the core subset reported for a variant
    (angle, curvature, held-out reconstruction). Returns (out, curve, cent, plane)."""
    sub = gc.fit_subspace(X, y, args.plane, args.k, basis)
    Z = sub.project(X)
    cent = mf.centroids(Z, y)
    out = {"n_train": int(len(X)), "subspace": args.plane, "k": int(sub.components.shape[0]),
           "explained_frac_top10": (sub.explained[:10] / sub.explained.sum()).tolist(),
           "n_values": len(cent["values"]), "min_count": int(cent["count"].min())}
    angle, plane = "unsupervised", "activation"
    if periodic:
        choice = mf.choose_angle_source(cent["C"], cent["values"], max_dev_deg=args.angle_max_dev)
        angle, plane = choice["angle"], choice["plane"]
        out["angle"] = {"source": "labels_angle (fallback, flagged)" if angle == "labels" else f"unsupervised, {plane} plane",
                        "plane_used": plane if angle == "unsupervised" else None,
                        "orientation": choice["orientation"], "max_dev_deg": choice["max_dev_deg"],
                        "circular_corr": choice["circular_corr"], "max_dev_cap_deg": args.angle_max_dev,
                        **choice["checks"]}
    curve = mf.fit_curve(cent, periodic, angle=angle, plane=plane)
    spread = gc.align(cent["spread"], cent["values"], curve.values)
    count = gc.align(cent["count"], cent["values"], curve.values)
    out["loo"] = {}
    for st in STRIDES:
        noise_rel = gc.loo_reconstruction(curve, spread / np.sqrt(count), stride=st)
        if full:
            pc = gc.precision_check(X, y, periodic, args.k, angle, plane, stride=st) if args.plane == "pca" else {}
            out["loo"][str(st)] = {**pc, "fp32_line_err_over_noise": noise_rel["line_err_over_noise"],
                                   "winner": noise_rel["winner"], "mean_err_cubic": noise_rel["mean_err_cubic"],
                                   "mean_err_line": noise_rel["mean_err_line"]}
        else:
            out["loo"][str(st)] = {q: noise_rel[q] for q in ("mean_err_cubic", "mean_err_line", "winner",
                                                             "line_err_over_noise")}
    out["spread_vs_curvature"] = gc.spread_vs_curvature(curve, spread, count)
    if args.plane == "pca":
        out["heldout_reconstruction"] = gc.heldout_reconstruction(X, y, periodic, args.k, angle, plane)
    if periodic:
        out["circular_chart"] = gc.chart_check(X, y, sub if args.plane == "pca" else gc.fit_pca(X, args.k), plane)
        out["expected_sagitta_by_stride"] = gc.expected_sagitta(
            out["circular_chart"]["chart_radius"], len(cent["values"]), STRIDES,
            out["spread_vs_curvature"]["median_centroid_noise"])
    if not periodic:
        out["knot_spacing"] = gc.knot_spacing(curve)
    if full:
        rng = np.random.default_rng(0)
        out["loo_shuffled_unmatched_control"] = {
            str(st): {q: gc.loo_reconstruction(gc.shuffled_curve(curve, rng), stride=st)[q]
                      for q in ("mean_err_cubic", "mean_err_line", "winner")} for st in (1, 16)}
        out["participation"] = gc.participation_check(Z, y)
        if periodic:
            out["ring_radius_by_group"] = gc.ring_radius_by_group(Z, y, direction_groups(df))
    return out, curve, cent, plane


def summary_lines(out, periodic):
    """Rule-based one-line readings of the checks (no hand-written conclusions)."""
    lines = []
    loo = out["loo"]
    small = [st for st in STRIDES if st <= 8]
    line_wins_small = all(loo[str(st)]["winner"] == "line" for st in small if str(st) in loo)
    if periodic:
        a = out["angle"]
        lines.append(f"angle: {a['source']}, orientation {a['orientation']}" if a["plane_used"]
                     else "angle: unsupervised atan2 failed in both planes; labels fallback (flagged)")
        es = out["expected_sagitta_by_stride"]
        det = [st for st in STRIDES if es[str(st)]["sagitta_over_centroid_noise"] > 1]
        lines.append(f"curvature detectable only where expected sagitta > centroid noise: strides {det} "
                     f"(knot gaps {[es[str(st)]['gap_deg'] for st in det]} deg); cubic wins at strides "
                     f"{[st for st in STRIDES if loo[str(st)]['winner'] == 'cubic']}")
    else:
        ks = out["knot_spacing"]
        if ks["better"] == "linear" and ks["r2_linear"] > 0.99 and line_wins_small:
            lines.append("pre-registered negative: straight, evenly spaced line (knot spacing linear R2 "
                         f"{ks['r2_linear']:.3f}, line beats cubic at every stride <= 8); the spline gives no "
                         "advantage for this variable at this layer")
    if "heldout_reconstruction" in out:
        lines.append("held-out reconstruction best: " +
                     ", ".join(f"{k} = {v['best']}" for k, v in out["heldout_reconstruction"].items()))
    return lines


def run(args):
    variable = args.variable or args.dataset
    d = load_inputs(args.dataset, args.layer, variable, args.act_dir, args.table, args.split)
    tr = d["is_train"]
    X, y, periodic = d["X"][tr], d["y"][tr], d["periodic"]
    basis = gc.load_basis_matrix(args.basis, X) if args.basis else None
    if args.plane == "inlp" and basis is None:
        raise SystemExit("--plane inlp needs --basis")
    if args.plane == "chart" and not periodic:
        raise SystemExit("--plane chart is for direction (a circular variable)")
    core, curve, cent, plane = analyse(X, y, periodic, d["df"][tr], args, basis)
    out = {"dataset": args.dataset, "layer": args.layer, "variable": variable,
           **layer_role(args.dataset, args.layer, variable),
           "provenance": provenance(args.split, seeds={"seed": args.seed}, layer=args.layer, pool="meanpool",
                                    subspace=args.plane, k=args.k), **core}
    out["summary"] = summary_lines(core, periodic)
    out["heldout_reconstruction_note"] = ("held-out centroids rebuilt by the interpolating spline (Goodfire A.3), the "
                                          "count-weighted smoothing spline (B.1) and the chord; pick run_part2 "
                                          "--spline from this train-only check")
    if args.nuisance_regress:
        nm, R = gc.fit_nuisance(d["X"], d["df"], variable, tr)
        names, r2 = nm["names"], nm["r2_train"]
        nr, *_ = analyse(R[tr], y, periodic, d["df"][tr], args, None if args.plane == "inlp" else basis, full=False)
        out["nuisance_regressed"] = {"covariates": names, "variance_explained_by_nuisance_train": r2,
                                     "space": "train-standardised activations minus their least-squares fit on the "
                                              "covariates (train rows), same projection for all clips",
                                     **nr, "summary": summary_lines(nr, periodic)}
    if args.basis and args.plane == "pca":
        mu, sd = X.mean(0), X.std(0) + 1e-8
        V = np.load(args.basis)["Q"] if str(args.basis).endswith(".npz") else np.load(args.basis)
        C_std = np.array([((X[y == v] - mu) / sd).mean(0) for v in cent["values"]])
        out["principal_angles_basis_vs_centroid_plane"] = gc.principal_angles_to_centroid_plane(V, C_std)

    tag = f"{args.dataset}" + ("" if variable == args.dataset else f"_{variable}") + f"_L{args.layer}"
    tag += "" if args.plane == "pca" else f"_{args.plane}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_geometry_{tag}.json").write_text(json.dumps(out, indent=1))
    plot_plane(curve, cent, plane, Path(args.figures_dir) / f"fig4_centroid_plane_{tag}.png", tag)
    plot_loo(out["loo"], Path(args.figures_dir) / f"fig4_loo_stride_{tag}.png", tag, periodic, len(cent["values"]))
    for line in out["summary"]:
        print(line)
    if args.planted_ring:
        pr = gc.planted_ring_control(X, y, radii=args.planted_radii, k=args.k, seed=args.seed, periodic=periodic)
        pr.update({"dataset": args.dataset, "layer": args.layer, "n_train": int(tr.sum()),
                   "labels_used_for_sd_and_stratification": variable, "provenance": out["provenance"],
                   "layer_role": out["layer_role"]})
        (Path(args.results_dir) / f"p2_planted_ring_{tag}.json").write_text(json.dumps(pr, indent=1))
        print("planted ring (r in within-value SD):")
        print("   r  radius  |corr|  cubic_gain  recovered")
        for r in pr["rows"]:
            print(f"{r['r_within_sd']:5.2f}  {r['ring_radius']:6.2f}  {r['circular_corr']:.3f}  {r['cubic_gain']:10.3f}"
                  f"  {r['recovered']}")
        if pr["real_ring"]:
            rr = pr["real_ring"]
            print(f"real ring: radius {rr['radius']:.2f}, r = {rr['r_within_sd']:.2f} in its own plane's within-SD")
        out["planted_ring"] = pr
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
    first = next(iter(loo.values()))
    precs = (("fp32", "-"), ("bf16", "--")) if "fp32" in first else (("", "-"),)
    for prec, style in precs:
        if not prec:
            ax.plot(s * step, [loo[k]["mean_err_line"] for k in loo], style, color="C0", label="line")
            ax.plot(s * step, [loo[k]["mean_err_cubic"] for k in loo], style, color="C1", label="cubic")
            continue
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
    p.add_argument("--basis", default=None, help=".npy [D, r] or INLP .npz basis: principal angles; --plane inlp")
    p.add_argument("--plane", default="pca", choices=("pca", "chart", "inlp"),
                   help="subspace for centroids and splines: top-k PCA, the circular-chart plane, or the INLP basis")
    p.add_argument("--angle-max-dev", type=float, default=None,
                   help="optional cap (degrees) on the unsupervised angle's max deviation; default: reported only")
    p.add_argument("--nuisance-regress", action="store_true",
                   help="also report the core checks after regressing nuisance covariates out (labelled variant)")
    p.add_argument("--planted-ring", action="store_true", help="planted-ring positive control (r vs recovery)")
    p.add_argument("--planted-radii", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None, help="CSV overriding the manifest table (tests)")
    p.add_argument("--split", default=None, help="JSON overriding this dataset's split (tests)")
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

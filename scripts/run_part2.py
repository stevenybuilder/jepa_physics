"""Part 2: spline (manifold) steering vs straight-line steering toward held-out label values, with controls.

Roles (all from splits/split_v1.json, no clip in two roles):
  knot  = train folds 0-2 at the 48 kept values: PCA-64, centroids, splines, linear endpoints, nearest-real density.
  probe = train folds 3-4, all 64 values: the evaluation probe (never built the intervention).
  test  = steered clips, and the real clips at the target value that the nearest-real readout compares with.
Targets are the 16 held-out values (every 4th, interior for scalars). Arms: manifold (spline), linear (straight
line between the polyline points at the source and target values, in full space), and two controls on the same
knots: a spline through shuffled centroids and a random smooth curve of matched length. Every arm is an additive
edit x + delta that starts from the clip's true source value.

Writes results/p2_steer_{dataset}_{variable}_L{layer}.json and figures/fig4_gap_vs_shift_*.png,
figures/fig4_path_energy_*.png.

  python scripts/run_part2.py --dataset direction --layer 12 --variable direction
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

ARMS = ("manifold", "linear", "shuffled_control", "random_control")


def shift_of(source, target, periodic):
    return mf.value_error(source, target, periodic)


def build(d, k, angle):
    """PCA, centroids and all curves from the knot clips at kept values.
    angle: "unsupervised" (choose plane automatically, fall back to labels if the ring is not found) or "labels"."""
    knot_all = d["role"] == "knot"
    values = np.unique(d["y"])
    held = values[mf.heldout_mask(values, periodic=d["periodic"])]
    knot = knot_all & ~np.isin(d["y"], held)
    pca = mf.fit_pca(d["X"][knot], k)
    cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
    choice = None
    plane = "activation"
    if d["periodic"] and angle == "unsupervised":
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        angle, plane = choice["angle"], choice["plane"]
    curve = mf.fit_curve(cent, d["periodic"], angle=angle, plane=plane)
    # full-space centroids (in the curve's knot order) for the straight-line arm
    full_C = np.array([d["X"][knot & (d["y"] == v)].mean(0) for v in curve.values])
    full_curve = mf.Curve(spline=None, values=curve.values, coords=curve.coords, points=full_C,
                          periodic=curve.periodic, coord_source=curve.coord_source)
    rng = np.random.default_rng(0)
    return {"pca": pca, "cent": cent, "curve": curve, "angle_choice": choice, "plane": plane,
            "full_curve": full_curve, "held": held, "knot": knot,
            "controls": {"shuffled_control": gc.shuffled_curve(curve, rng),
                         "random_control": gc.random_smooth_curve(curve, rng)}}


def steer_arm(arm, x, src, tgt, m, K):
    """[n, K, D] waypoints for one arm, from the clips' true source values to one target value."""
    if arm == "linear":
        ca = mf.piecewise_linear_point(m["full_curve"], src)          # [n, D]
        cb = mf.piecewise_linear_point(m["full_curve"], tgt)          # [D]
        return mf.linear_path(x, ca, cb, K)
    curve = m["curve"] if arm == "manifold" else m["controls"][arm]
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(np.full(len(src), tgt))
    return mf.manifold_path(x, m["pca"], curve, ta, tb, K)


def run(args):
    d = load_inputs(args.dataset, args.layer, args.variable, args.act_dir, args.table, args.split)
    periodic = d["periodic"]
    angle = "labels" if args.labels_angle else "unsupervised"
    m = build(d, args.k, angle)
    X_knot = d["X"][m["knot"]]
    probe_rows = d["role"] == "probe"
    probe = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic)
    test = np.flatnonzero(d["role"] == "test")
    near = mf.NearestRealReadout(d["X"][test], d["y"][test])
    rng = np.random.default_rng(args.seed)

    rows, shared, energy = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for tgt in m["held"]:
        pool = test[~np.isclose(d["y"][test], tgt)]
        pick = rng.choice(pool, size=min(args.n_clips, len(pool)), replace=False)
        x, src = d["X"][pick].astype(float), d["y"][pick]
        pred0 = probe.predict(x)
        for arm in ARMS:
            W = steer_arm(arm, x, src, tgt, m, args.K)                 # [n, K, D]
            xs = W[:, -1]
            e = mf.off_manifold_energy(W, m["pca"], m["curve"], X_knot)
            energy[arm].append(np.stack([e["to_curve"], e["to_nearest_real"]], -1))   # [n, K, 2]
            shared[arm].append(gc.shared_delta_fraction(xs - x))
            pred = probe.predict(xs)
            for j in range(len(pick)):
                rows.append({"arm": arm, "id": int(d["df"]["id"].iloc[pick[j]]), "source": float(src[j]),
                             "target": float(tgt), "shift": float(shift_of(src[j], tgt, periodic)),
                             "probe_err_to_target": float(mf.value_error(pred[j], tgt, periodic)),
                             "probe_err_to_true": float(mf.value_error(pred[j], src[j], periodic)),
                             "probe_err_to_target_before": float(mf.value_error(pred0[j], tgt, periodic)),
                             "nearest_real_R": float(near.score(xs[j:j + 1], x[j:j + 1], tgt)[0]),
                             "energy_to_curve": float(e["to_curve"][j].mean()),
                             "energy_to_nearest_real": float(e["to_nearest_real"][j].mean()),
                             # excess = mean along the path minus the unsteered clip's own distance (waypoint 0)
                             "excess_to_curve": float(e["to_curve"][j].mean() - e["to_curve"][j, 0]),
                             "excess_to_nearest_real": float(e["to_nearest_real"][j].mean() - e["to_nearest_real"][j, 0]),
                             "norm_ratio": float(np.linalg.norm(xs[j]) / np.linalg.norm(x[j]))})

    out = {"dataset": args.dataset, "variable": args.variable or args.dataset, "layer": args.layer,
           "k": int(m["pca"].components.shape[0]), "K": args.K, "angle_source": m["curve"].coord_source,
           "held_out_values": m["held"].tolist(), "n_knot_clips": int(m["knot"].sum()),
           "n_probe_clips": int(probe_rows.sum()), "n_test_clips": len(test),
           "readouts": {"probe": mf.ProbeReadout.plays_M_y, "nearest_real": mf.NearestRealReadout.plays_M_y},
           "probe_test_error": float(np.mean(mf.value_error(probe.predict(d["X"][test]), d["y"][test], periodic))),
           "isometry_probe": mf.isometry(m["curve"], probe.raw(m["pca"].lift(m["curve"].points))),
           "shared_delta_fraction": {a: float(np.mean(v)) for a, v in shared.items()},
           "summary": summarise(rows, periodic), "rows": rows}
    if m["angle_choice"]:
        out["angle_check"] = {"plane_used": m["plane"], **m["angle_choice"]["checks"]}

    tag = f"{args.dataset}_{args.variable or args.dataset}_L{args.layer}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_steer_{tag}.json").write_text(json.dumps(out, indent=1))
    plot_gap(out["summary"], Path(args.figures_dir) / f"fig4_gap_vs_shift_{tag}.png", tag, periodic)
    plot_energy({a: np.concatenate(v) for a, v in energy.items()}, rows,
                Path(args.figures_dir) / f"fig4_path_energy_{tag}.png", tag)
    return out


def shift_bins(periodic):
    return np.array([0, 30, 60, 90, 120, 150, 180.01]) if periodic else None


def summarise(rows, periodic, n_bins=5):
    """Per arm and shift bin: mean probe error to target / to true, nearest-real R, energies, count."""
    shifts = np.array([r["shift"] for r in rows])
    edges = shift_bins(periodic)
    if edges is None:
        edges = np.quantile(shifts, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
    out = {}
    for arm in ARMS:
        R = [r for r in rows if r["arm"] == arm]
        s = np.array([r["shift"] for r in R])
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r, v in zip(R, s) if lo <= v < hi]
            if not sel:
                continue
            mean = lambda q: float(np.mean([r[q] for r in sel]))
            bins.append({"shift_lo": float(lo), "shift_hi": float(hi), "n": len(sel),
                         **{q: mean(q) for q in ("probe_err_to_target", "probe_err_to_true", "nearest_real_R",
                                                  "energy_to_curve", "energy_to_nearest_real",
                                                  "excess_to_curve", "excess_to_nearest_real", "norm_ratio")}})
        out[arm] = {"overall": {q: float(np.mean([r[q] for r in R])) for q in
                                ("probe_err_to_target", "probe_err_to_true", "nearest_real_R", "energy_to_curve",
                                 "energy_to_nearest_real", "excess_to_curve", "excess_to_nearest_real", "norm_ratio")}, "by_shift": bins}
    return out


def plot_gap(summary, path, tag, periodic):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arm in ARMS:
        b = summary[arm]["by_shift"]
        mid = [(x["shift_lo"] + x["shift_hi"]) / 2 for x in b]
        style = "-" if arm in ("manifold", "linear") else ":"
        axes[0].plot(mid, [x["probe_err_to_target"] for x in b], style, marker="o", label=arm)
        axes[1].plot(mid, [x["nearest_real_R"] for x in b], style, marker="o", label=arm)
    unit = "degrees" if periodic else "label units"
    axes[0].set(xlabel=f"shift |source - target| ({unit})", ylabel=f"probe error to target ({unit})",
                title="Evaluation probe")
    axes[1].set(xlabel=f"shift ({unit})", ylabel="R = 1 - D(steered)/D(unsteered)", title="Agreement with real clips")
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Held-out targets, {tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_energy(energy, rows, path, tag):
    """Mean distance along the path (waypoint fraction 0..1) to the curve and to the nearest real activation,
    for all steers and for the largest-shift third."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arm in ARMS:
        E = energy[arm]                                        # [n_total, K, 2]
        s = np.array([r["shift"] for r in rows if r["arm"] == arm])
        far = s >= np.quantile(s, 2 / 3)
        frac = np.linspace(0, 1, E.shape[1])
        for i, name in enumerate(("to fitted curve (PCA)", "to nearest real activation")):
            line, = axes[i].plot(frac, E[:, :, i].mean(0), label=arm)
            axes[i].plot(frac, E[far, :, i].mean(0), "--", color=line.get_color(), lw=0.8)
            axes[i].set(xlabel="fraction of path", ylabel="distance", title=name)
    axes[0].legend(fontsize=7, title="dashed: largest-shift third", title_fontsize=7)
    fig.suptitle(f"Off-manifold energy along the path, {tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("direction", "speed", "acceleration"))
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--variable", default=None)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--K", type=int, default=16, help="waypoints per path")
    p.add_argument("--n-clips", type=int, default=48, help="steered test clips per target value")
    p.add_argument("--labels-angle", action="store_true", help="flagged fallback: true angle as spline coordinate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

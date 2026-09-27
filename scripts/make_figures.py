"""Part 1 figures, read only from results/*.json. A figure whose inputs are missing is skipped.

  python scripts/make_figures.py [--results results] [--figures figures]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"direction": "#2a78d6", "speed": "#eb6834", "acceleration": "#1baf7a",
          "velocity": "#4a3aa7", "random": "#9a9893", "text": "#52514e"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": COLORS["text"], "xtick.color": COLORS["text"], "ytick.color": COLORS["text"],
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "legend.frameon": False,
    "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def load(results, pattern):
    """The results file whose name fully matches `pattern` (peak layer first if several), or None."""
    hits = sorted(p for p in results.glob("*.json") if re.fullmatch(pattern, p.name))
    data = [json.loads(p.read_text()) for p in hits]
    data.sort(key=lambda d: not d.get("is_peak", True))
    return data[0] if data else None


def legend_top(ax, ncol=3):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=ncol, fontsize=9, handlelength=1.5)


def curve(ax, sweep, key_mean, key_sd, color, label, rows=None):
    rows = rows or sweep["layers"]
    x = np.array([r["frac"] for r in rows])
    m = np.array([r[key_mean] for r in rows])
    s = np.array([r[key_sd] for r in rows])
    body = [not r.get("post_ln", False) for r in sweep["layers"]]
    ax.plot(x[body], m[body], color=color, label=label)
    ax.fill_between(x[body], (m - s)[body], (m + s)[body], color=color, alpha=0.18, lw=0)
    if not all(body):   # the post-LN copy of the last block: a hollow marker, not part of the line
        ax.plot(x[~np.array(body)], m[~np.array(body)], "o", mfc="white", mec=color, ms=6, mew=1.5)
    return x, m


def fig1(results, out):
    sweeps = {v: load(results, rf"p1a_{v}_{v}_meanpool\.json") for v in ("direction", "speed", "acceleration")}
    if not any(sweeps.values()):
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for v, sw in sweeps.items():
        if sw is None:
            continue
        x, m = curve(a, sw, "cv_mean", "cv_sd", COLORS[v], v)
        av = sw["availability"]
        if av["onset"] is not None:
            a.plot(x[av["onset"]], m[av["onset"]], "v", color=COLORS[v], ms=9, mec="white", mew=1.5)
        a.plot(x[av["peak"]], m[av["peak"]], "*", color=COLORS[v], ms=13, mec="white", mew=1)
    shuf = load(results, r"p1a_direction_direction_meanpool_shuffled\.json")
    if shuf:
        curve(a, shuf, "cv_mean", "cv_sd", COLORS["random"], "direction, shuffled labels")
    a.plot([], [], "v", color=COLORS["text"], label="onset (90% of max)")
    a.plot([], [], "*", color=COLORS["text"], ms=11, label="peak")
    a.set(xlabel="layer fraction", ylabel="probe R² (5-fold mean ± SD)",
          title="Where each variable becomes decodable")
    legend_top(a)
    d = sweeps["direction"]
    if d and "by_motion" in d["layers"][0]:
        for motion in d["layers"][0]["by_motion"]:
            rows = [dict(r["by_motion"][motion], frac=r["frac"], post_ln=r["post_ln"]) for r in d["layers"]]
            color = COLORS["velocity"] if motion == "velocity" else COLORS["acceleration"]
            curve(b, {"layers": rows}, "cv_mean", "cv_sd", color, f"{motion} clips", rows)
        b.set(xlabel="layer fraction", title="Direction, by motion type")
        legend_top(b, 2)
    else:
        b.axis("off")
    a.set_ylim(min(-0.05, a.get_ylim()[0]), 1.02)
    fig.savefig(out / "fig1_layer_curves.png")
    plt.close(fig)


def fig1b(results, out):
    d = load(results, r"p1a_direction_direction_meanpool\.json")
    if not d or "test_pred" not in d:
        return
    av, layers = d["availability"], sorted(d["test_pred"], key=int)
    names = {str(av["onset"]): "onset", str(av["peak"]): "peak", "24": "last block"}
    fig, axes = plt.subplots(1, len(layers), figsize=(3.3 * len(layers), 3.6))
    for ax, p in zip(np.atleast_1d(axes), layers):
        pred = np.array(d["test_pred"][p])
        sc = ax.scatter(pred[:, 1], pred[:, 0], c=d["test_theta"], cmap="twilight", vmin=0, vmax=360,
                        s=10, lw=0)
        ax.set(aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xlabel="predicted cos θ",
               title=f"{names.get(p, '')} (layer {d['layers'][int(p)]['frac']:.2f})")
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=":", color=COLORS["random"]))
    np.atleast_1d(axes)[0].set_ylabel("predicted sin θ")
    fig.colorbar(sc, ax=axes, label="true θ (deg)", shrink=0.8)
    fig.suptitle("Test-clip readouts form a ring ordered by true direction", x=0.02, ha="left",
                 fontweight="bold", y=0.9)
    fig.savefig(out / "fig1b_direction_circle.png")
    plt.close(fig)


def fig1c(results, out):
    """Extra (spec 5.1 b): direction probe from the direction set, read on the speed and acceleration sets."""
    runs = {pool: load(results, rf"p1a_support_transfer_{pool}\.json") for pool in ("meanpool", "diskpool")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 3.8), sharey=True, squeeze=False)
    for ax, (pool, r) in zip(axes[0], runs.items()):
        body = [row for row in r["layers"] if not row["post_ln"]]
        x = [row["frac"] for row in body]
        ax.plot(x, [row["source_cv_mae"] for row in body], color=COLORS["direction"],
                label="direction set (5-fold CV)")
        for name in r["targets"]:
            ax.plot(x, [row["targets"][name]["mae"] for row in body], color=COLORS[name],
                    label=f"{name} set (all clips)")
        ax.axhline(90, color=COLORS["random"], lw=0.8, ls=":", label="chance (90°)")
        ax.set(xlabel="layer fraction", ylim=(0, 120), title=f"Direction probe transfer ({pool})")
        legend_top(ax, 2)
    axes[0][0].set_ylabel("circular MAE (deg)")
    fig.savefig(out / "fig1c_direction_transfer.png")
    plt.close(fig)


def fig1d(results, out):
    """Extra (spec 5.1 a): Cartesian (vx, vy) vs polar (sin, cos) onset on constant-speed clips."""
    runs = [(d, load(results, rf"p1a_support_onset_{d}_meanpool\.json")) for d in ("direction", "speed")]
    runs = [(d, r) for d, r in runs if r]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 3.8), sharey=True, squeeze=False)
    for ax, (d, r) in zip(axes[0], runs):
        for name, color, label in (("vxvy", COLORS["velocity"], "(vx, vy)"), ("sincos", COLORS["direction"], "(sin θ, cos θ)")):
            t = r["targets"][name]
            x, m = curve(ax, t, "cv_mean", "cv_sd", color, label)
            av = t["availability"]
            if av["onset"] is not None:
                ci = av.get("onset_ci") or [av["onset"]] * 2
                ax.plot(x[av["onset"]], m[av["onset"]], "v", color=color, ms=9, mec="white", mew=1.5)
                ax.errorbar(x[av["onset"]], -0.03, xerr=[[x[av["onset"]] - ci[0] / 24], [ci[1] / 24 - x[av["onset"]]]],
                            fmt="none", color=color, capsize=3)
        diff = r.get("onset_difference", {})
        ax.set(xlabel="layer fraction", ylim=(-0.08, 1.02),
               title=f"{d} set, constant speed: Δonset {diff.get('point_estimate')} (95% CI {diff.get('ci95')})")
        legend_top(ax, 2)
    axes[0][0].set_ylabel("probe R² (5-fold mean ± SD)")
    fig.savefig(out / "fig1d_cartesian_vs_polar.png")
    plt.close(fig)


def fig1e(results, out):
    """Extra (spec 5.1 c): direction probe fit on start x < 0, read on x > 0 and the reverse."""
    runs = {pool: load(results, rf"p1a_support_spatial_{pool}\.json") for pool in ("meanpool", "diskpool")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for pool, ls in (("meanpool", "-"), ("diskpool", "--")):
        if pool not in runs:
            continue
        body = [row for row in runs[pool]["layers"] if not row["post_ln"]]
        x = [row["frac"] for row in body]
        within = [np.mean([row[k]["within_cv_mae"] for k in ("neg_to_pos", "pos_to_neg")]) for row in body]
        cross = [np.mean([row[k]["cross_mae"] for k in ("neg_to_pos", "pos_to_neg")]) for row in body]
        ax.plot(x, within, ls, color=COLORS["direction"], label=f"{pool}: same side (CV)")
        ax.plot(x, cross, ls, color=COLORS["speed"], label=f"{pool}: other side")
    ax.axhline(90, color=COLORS["random"], lw=0.8, ls=":")
    ax.set(xlabel="layer fraction", ylabel="circular MAE (deg), mean of both directions", ylim=(0, 120),
           title="Spatial generalisation: start x < 0 vs x > 0")
    legend_top(ax, 2)
    fig.savefig(out / "fig1e_spatial_generalisation.png")
    plt.close(fig)


def fig2(results, out):
    runs = {v: load(results, rf"p1b_{v}_{v}_meanpool_L\d+\.json") for v in ("direction", "speed")}
    runs = {v: r for v, r in runs.items() if r}
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 3.8), sharey=True, squeeze=False)
    for ax, (v, r) in zip(axes[0], runs.items()):
        x = np.array([row["dims_removed"] for row in r["rounds"]])
        m = np.array([row["cv_r2"] for row in r["rounds"]])
        s = np.array([row["cv_r2_sd"] for row in r["rounds"]])
        if "random" in r:
            rx = np.array([row["dims_removed"] for row in r["random"]["rows"]])
            rm = np.array([row["cv_r2"] for row in r["random"]["rows"]])
            rs = np.array([row["cv_r2_seed_sd"] for row in r["random"]["rows"]])
            ax.fill_between(rx, rm - rs, rm + rs, color=COLORS["random"], alpha=0.3, lw=0)
            ax.plot(rx, rm, color=COLORS["random"], label=f"random subspace ({r['random']['seeds']} seeds)")
        ax.plot(x, m, "o-", color=COLORS[v], ms=3.5, label="orthogonal probe sequence")
        ax.fill_between(x, m - s, m + s, color=COLORS[v], alpha=0.18, lw=0)
        stop = 0.1 if r["kind"] == "circular" else 0.05
        ax.axhline(stop, color=COLORS["text"], lw=0.8, ls="--")
        ax.set(xlabel="dimensions removed", title=f"{v}: {r['dims']} dims (layer {r['frac']:.2f})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_top(ax, 1)
    axes[0][0].set_ylabel("probe R² (5-fold mean)")
    fig.savefig(out / "fig2_inlp.png")
    plt.close(fig)


def fig2b(results, out):
    runs = {v: load(results, rf"p1b_{v}_{v}_meanpool_dims\.json") for v in ("direction", "speed", "acceleration")}
    runs = {v: r for v, r in runs.items() if r}
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for v, r in runs.items():
        body = [row for row in r["layers"] if not row["post_ln"]]
        ax.plot([row["frac"] for row in body], [row["dims"] for row in body], "o-", ms=3.5,
                color=COLORS[v], label=v)
    ax.set(xlabel="layer fraction", ylabel="linear subspace dimension (2K or K)",
           title="Dimensions needed to remove each variable, by depth")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    legend_top(ax)
    fig.savefig(out / "fig2b_dim_vs_layer.png")
    plt.close(fig)


def fig2d(results, out):
    """Extra (spec §6 item 7): direction vs speed / acceleration subspaces from the INLP bases."""
    path = results / "step2_subspace_angles.json"
    if not path.exists():
        return
    r = json.loads(path.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for pair, rows in r["pairs"].items():
        other = pair.split("_vs_")[1]
        rows = [row for row in rows if not row["post_ln"] and "as_stored" in row]
        if not rows:
            continue
        key = "mapped" if all("mapped" in row for row in rows) else "as_stored"
        x = [row["frac"] for row in rows]
        o = [row[key] for row in rows]
        a.plot(x, [v["overlap"] for v in o], "o-", ms=3.5, color=COLORS[other], label=f"direction vs {other}")
        a.fill_between(x, [v["random_overlap_p05_p95"][0] for v in o], [v["random_overlap_p05_p95"][1] for v in o],
                       color=COLORS[other], alpha=0.15, lw=0)
        a.plot(x, [v["random_expectation"] for v in o], ":", color=COLORS[other], lw=1)
        b.plot(x, [v["min_angle_deg"] for v in o], "o-", ms=3.5, color=COLORS[other], label=f"direction vs {other}")
        b.fill_between(x, [v["random_min_angle_p05_p95"][0] for v in o], [v["random_min_angle_p05_p95"][1] for v in o],
                       color=COLORS[other], alpha=0.15, lw=0)
    a.plot([], [], ":", color=COLORS["text"], lw=1, label="random expectation k_A/d (band: 5–95%)")
    a.set(xlabel="layer fraction", ylabel="overlap |Q_A^T Q_B|²_F / dim B", title="Extra: shared readout subspace")
    b.set(xlabel="layer fraction", ylabel="smallest principal angle (deg)", ylim=(0, 92),
          title="Extra: closest pair of directions")
    for ax in (a, b):
        legend_top(ax, 1)
    fig.savefig(out / "fig2d_subspace_angles.png")
    plt.close(fig)


def mae_panel(ax, r, title):
    n = [row["n"] for row in r["single"]]
    ax.plot(n, [row["mae_to_target"] for row in r["single"]], "o-", ms=3.5, color=COLORS["direction"],
            label="to target θ* = 90°")
    ax.plot(n, [row["mae_to_true"] for row in r["single"]], "o-", ms=3.5, color=COLORS["speed"],
            label="to true θ")
    ax.plot(n, [row["mae_to_target"] for row in r["all_targets"]], "--", color=COLORS["direction"],
            label="to target, all 64 targets")
    ax.plot(n, [row["mae_to_true"] for row in r["all_targets"]], "--", color=COLORS["speed"],
            label="to true, all 64 targets")
    ax.set(xlabel="probes used for steering (N)", ylabel="held-out probe MAE (deg)", ylim=(0, 185), title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend_top(ax, 2)


def radius_band(ax, rows, color, label, ls="-"):
    n = [row["n"] for row in rows]
    med = [row["radius"]["median"] for row in rows]
    ax.plot(n, med, ls, color=color, marker="o", ms=3, label=label)
    ax.fill_between(n, [row["radius"]["p05"] for row in rows], [row["radius"]["p95"] for row in rows],
                    color=color, alpha=0.15, lw=0)


def fig3(results, out):
    r = load(results, r"p1c_direction_L\d+\.json")
    if not r:
        return
    has_radius = "radius" in r["single"][0]
    fig, axes = plt.subplots(1, 2 if has_radius else 1, figsize=(11 if has_radius else 5.8, 3.8), squeeze=False)
    mae_panel(axes[0][0], r, f"Steering moves a held-out readout to the target (layer {r['frac']:.2f})")
    if has_radius:   # readout radius ‖(ŝ, ĉ)‖ of the evaluation probe beside the angle (spec §4)
        ax = axes[0][1]
        radius_band(ax, r["single"], COLORS["direction"], "paper: unit target (median, 5–95%)")
        if "radius_matched" in r:
            radius_band(ax, r["radius_matched"]["single"], COLORS["acceleration"],
                        "extra: radius-matched target", "--")
        ax.axhline(r["single"][0]["radius"]["median"], color=COLORS["text"], lw=0.8, ls=":",
                   label="unsteered test clips (median)")
        ax.set(xlabel="probes used for steering (N)", ylabel="held-out readout radius ‖(ŝ, ĉ)‖",
               title="Readout radius during steering (θ* = 90°)", ylim=(0, None))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_top(ax, 1)
    fig.savefig(out / "fig3_steering.png")
    plt.close(fig)


def fig3b(results, out):
    r = load(results, r"p1c_direction_L\d+\.json")
    if not r:
        return
    arms = [("paper: unit target", r)] + ([("extra: radius-matched target", r["radius_matched"])]
                                          if "radius_matched" in r else [])
    fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 3.8), squeeze=False)
    for ax, (name, arm) in zip(axes[0], arms):
        H = np.array([[np.nan if v is None else v for v in row] for row in arm["shift_bins"]["mae_to_target"]])
        edges = arm["shift_bins"]["edges"]
        im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=180,
                       extent=(-0.5, H.shape[0] - 0.5, edges[0], edges[-1]))
        ax.grid(False)
        ax.set(xlabel="probes used for steering (N)", ylabel="requested shift |θ − θ*| (deg)",
               title=f"Error to target by shift and N ({name})")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.colorbar(im, ax=ax, label="MAE to target (deg)")
    fig.savefig(out / "fig3b_shift_heatmap.png")
    plt.close(fig)


def fig3c(results, out):
    """Extras: learned steer vs the two random nulls, and the radius-matched arm (spec 5.3)."""
    r = load(results, r"p1c_direction_L\d+\.json")
    if not r or "random_nulls" not in r:
        return
    nulls = r["random_nulls"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    n = [row["n"] for row in nulls["rows"]]
    for key, color, label in (("random_basis", COLORS["random"], "random basis, own solve"),
                              ("random_orientation", COLORS["velocity"], "learned dose, random orientation")):
        band = [row[key]["mae_to_target"] for row in nulls["rows"]]
        a.fill_between(n, [x["p05"] for x in band], [x["p95"] for x in band], color=color, alpha=0.25, lw=0)
        a.plot(n, [x["mean"] for x in band], color=color, lw=1.2, label=f"null: {label} (5–95%)")
        b.plot(n, [row[key]["empirical_p_to_target"] for row in nulls["rows"]], "o-", ms=3.5, color=color,
               label=label)
    a.plot(n, [row["learned"]["mae_to_target"] for row in nulls["rows"]], "o-", ms=3.5,
           color=COLORS["direction"], label="learned subspace, unit target")
    if "radius_matched" in r:
        rm = r["radius_matched"]["single"]
        a.plot([row["n"] for row in rm], [row["mae_to_target"] for row in rm], "--", color=COLORS["acceleration"],
               label="learned subspace, radius-matched target")
    a.set(xlabel="probes used for steering (N)", ylabel="held-out MAE to θ* = 90° (deg)", ylim=(0, 185),
          title=f"Extra: learned subspace vs random nulls ({nulls['n_draws']} draws each)")
    b.axhline(1 / (nulls["n_draws"] + 1), color=COLORS["text"], lw=0.8, ls=":", label="resolution 1/(m+1)")
    b.set(xlabel="probes used for steering (N)", ylabel="empirical rank of learned MAE", yscale="log",
          ylim=(0.7 / (nulls["n_draws"] + 1), 1.2), title="Rank of the learned steer among null draws")
    for ax in (a, b):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        legend_top(ax, 1)
    fig.savefig(out / "fig3c_steering_nulls.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    for make in (fig1, fig1b, fig1c, fig1d, fig1e, fig2, fig2b, fig2d, fig3, fig3b, fig3c):
        make(args.results, args.figures)
    print("figures:", sorted(p.name for p in args.figures.glob("*.png")))

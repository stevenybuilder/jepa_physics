"""Part 2: spline (manifold) steering vs straight-line steering toward held-out label values, with controls.

Roles (all from splits/split_v1.json, no clip in two roles):
  knot  = train folds 0-2 at the kept values: PCA-64, centroids, splines, chord endpoints, nearest-real density.
  probe = train folds 3-4, all 64 values: the evaluation probe (never built the intervention).
  test  = steered clips, and the real clips at the target value that the nearest-real readout compares with.
Targets are the held-out values of --holdout: scattered (every 4th, 16 values), contiguous (one 8-value block: a
45 degree arc for direction, chosen by --seed; an interior block for scalars) or extrapolation (top 8, scalars only).

Matched-support arms (all edit only the PCA-k subspace and add back each clip's identical off-subspace residual):
manifold (spline walk), linear (straight line between the chord points at source and target), linear_dose_matched
(linear delta rescaled per clip and waypoint to the spline's ||delta||), projected (the spline's chord traversed with
its arc-length spacing) and reflected (2 * chord - spline). Controls: --n-controls draws each of a random smooth
curve of matched length and a spline through shuffled centroids; reported as mean, 5-95% band and the spline's rank.
--goodfire-baseline adds Goodfire's own comparison (linear replaces the whole activation), labelled separately.
--context-dataset speed steers speed-set test clips with the spline built on --dataset (held-out context).
The BF16 repeat of the spline-vs-line energy comparison runs unless --no-bf16.

Writes results/p2_steer_{dataset}_{variable}_L{layer}_{holdout}[_ctx-{context}].json and
figures/fig4_gap_vs_shift_*.png, figures/fig4_path_energy_*.png.

  python scripts/run_part2.py --dataset direction --layer 12 --variable direction --holdout contiguous
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

MAIN_ARMS = ("manifold", "linear", "linear_dose_matched", "projected", "reflected")
GOODFIRE_ARMS = ("goodfire_linear", "goodfire_manifold")
CONTROLS = ("random_control", "shuffled_control")
METRICS = ("probe_err_to_target", "probe_err_to_true", "nearest_real_R", "energy_to_curve", "energy_to_nearest_real",
           "excess_to_curve", "excess_to_nearest_real", "norm_ratio", "delta_norm", "delta_norm_path")
RANKED = {"nearest_real_R": True, "probe_err_to_target": False, "excess_to_curve": False,
          "excess_to_nearest_real": False}                       # metric -> higher is better
ARM_NOTES = {
    "manifold": "spline walk in the PCA-k subspace, additive (x + curve(t_k) - curve(t_src)), residual kept",
    "linear": "straight line between the polyline (chord) points at source and target, same subspace, residual kept",
    "linear_dose_matched": "linear arm with its delta rescaled, per clip and per waypoint, to the spline's ||delta||",
    "projected": "the spline's own chord (same endpoints) traversed with the spline's arc-length spacing",
    "reflected": "2 * projected - spline: the bend flipped, same endpoints, dose and waypoint count",
    "goodfire_linear": "Goodfire's linear baseline: the whole activation replaced by a full-space chord point",
    "goodfire_manifold": "Goodfire's manifold arm: PCA part replaced by the curve point, residual kept (A.6)",
}


def shift_of(source, target, periodic):
    return mf.value_error(source, target, periodic)


def build(d, k, angle, design="scattered", seed=0, n_controls=20):
    """PCA, centroids and all curves from the knot clips at kept values (the held-out design decides which).
    angle: "unsupervised" (choose plane automatically, fall back to labels if the ring is not found) or "labels"."""
    knot_all = d["role"] == "knot"
    values = np.unique(d["y"])
    mask, design_info = mf.heldout_design(values, design, d["periodic"], seed=seed)
    held = values[mask]
    knot = knot_all & ~np.isin(d["y"], held)
    pca = mf.fit_pca(d["X"][knot], k)
    cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
    choice = None
    plane = "activation"
    if d["periodic"] and angle == "unsupervised":
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        angle, plane = choice["angle"], choice["plane"]
    curve = mf.fit_curve(cent, d["periodic"], angle=angle, plane=plane)
    # full-space centroids (in the curve's knot order): only Goodfire's whole-activation linear baseline uses them
    full_C = np.array([d["X"][knot & (d["y"] == v)].mean(0) for v in curve.values])
    full_curve = mf.Curve(spline=None, values=curve.values, coords=curve.coords, points=full_C,
                          periodic=curve.periodic, coord_source=curve.coord_source)
    noise = gc.align(cent["spread"] / np.sqrt(cent["count"]), cent["values"], curve.values)
    spread = gc.align(cent["spread"], cent["values"], curve.values)
    sag = mf.sagitta(curve, held)
    for s in sag:
        s["sagitta_over_centroid_noise"] = s["sagitta"] / float(np.median(noise))
        s["sagitta_over_spread"] = s["sagitta"] / float(np.median(spread))
    rng = np.random.default_rng(seed)
    controls = {"random_control": [gc.random_smooth_curve(curve, rng) for _ in range(n_controls)],
                "shuffled_control": [gc.shuffled_curve(curve, rng) for _ in range(n_controls)]}
    return {"pca": pca, "cent": cent, "curve": curve, "angle_choice": choice, "plane": plane,
            "full_curve": full_curve, "held": held, "knot": knot, "design": design_info, "sagitta": sag,
            "controls": controls}


def curve_coords(curve, Z, src, tgt, K):
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(np.full(len(src), tgt))
    return mf.manifold_coords(Z, curve, ta, tb, K)


def subspace_arms(Z, src, tgt, m, K):
    """PCA-coordinate waypoints [n, K, k] of every matched-support arm. All are lifted with the same residual."""
    Zs = curve_coords(m["curve"], Z, src, tgt, K)
    Zl = mf.linear_coords(Z, mf.piecewise_linear_point(m["curve"], src),
                          mf.piecewise_linear_point(m["curve"], tgt), K)
    dl, ds = Zl - Z[:, None], Zs - Z[:, None]
    nl, ns = np.linalg.norm(dl, axis=-1, keepdims=True), np.linalg.norm(ds, axis=-1, keepdims=True)
    Zd = Z[:, None] + dl * np.where(nl > 0, ns / np.where(nl > 0, nl, 1.0), 0.0)
    projected, reflected = mf.chord_coords(Zs)
    return {"manifold": Zs, "linear": Zl, "linear_dose_matched": Zd, "projected": projected, "reflected": reflected}


def compose(pca, Zk, resid):
    """Full-space waypoints: the edited in-subspace part plus the clip's untouched off-subspace residual."""
    return pca.lift(Zk) + resid[:, None, :]


def goodfire_arms(x, src, tgt, m, K):
    """Goodfire's own comparison (full space): linear replaces the whole activation, manifold keeps the residual."""
    s = np.linspace(0.0, 1.0, K)[None, :, None]
    pa = mf.piecewise_linear_point(m["full_curve"], src)
    pb = np.broadcast_to(mf.piecewise_linear_point(m["full_curve"], tgt), pa.shape)
    curve = m["curve"]
    man = mf.manifold_path(x, m["pca"], curve, curve.coord_of_value(src),
                           curve.coord_of_value(np.full(len(src), tgt)), K, mode="replace")
    return {"goodfire_linear": (1 - s) * pa[:, None] + s * pb[:, None], "goodfire_manifold": man}


def evaluate(W, x, src, tgt, m, ctx):
    """Per-clip metrics of waypoints W [n, K, D] steering clips x [n, D] from src to tgt."""
    periodic = ctx["periodic"]
    xs = W[:, -1]
    e = mf.off_manifold_energy(W, m["pca"], m["curve"], ctx["X_knot"])
    pred = ctx["probe"].predict(xs)
    dn = np.linalg.norm(W - x[:, None], axis=-1)                           # [n, K] delivered dose per waypoint
    out = {"probe_err_to_target": mf.value_error(pred, tgt, periodic),
           "probe_err_to_true": mf.value_error(pred, src, periodic),
           "nearest_real_R": ctx["near"].score(xs, x, tgt),
           "energy_to_curve": e["to_curve"].mean(1), "energy_to_nearest_real": e["to_nearest_real"].mean(1),
           # excess = mean along the path minus the unsteered clip's own distance (waypoint 0)
           "excess_to_curve": e["to_curve"].mean(1) - e["to_curve"][:, 0],
           "excess_to_nearest_real": e["to_nearest_real"].mean(1) - e["to_nearest_real"][:, 0],
           "norm_ratio": np.linalg.norm(xs, axis=1) / np.linalg.norm(x, axis=1),
           "delta_norm": dn[:, -1], "delta_norm_path": dn.mean(1),
           "_dose": dn, "_energy": np.stack([e["to_curve"], e["to_nearest_real"]], -1), "_delta_end": xs - x}
    if ctx.get("near_ctx") is not None:
        out["nearest_real_R_context"] = ctx["near_ctx"].score(xs, x, tgt)
    return out


def pick_clips(d_steer, targets, n_clips, seed):
    """Steered clips per target: test clips of the steered dataset whose true value differs from the target."""
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(d_steer["role"] == "test")
    picks = {}
    for tgt in targets:
        pool = test[~np.isclose(d_steer["y"][test], tgt)]
        picks[float(tgt)] = rng.choice(pool, size=min(n_clips, len(pool)), replace=False)
    return picks


def bf16_steering_energy(d, d_steer, args, angle, picks):
    """Repeat the spline-vs-line off-manifold energy comparison with every activation rounded to bfloat16 (knots,
    steered clips, and the steered waypoints, as a BF16 pipeline would store them)."""
    db = {**d, "X": gc.bf16_round(d["X"])}
    Xs = gc.bf16_round(d_steer["X"])
    mb = build(db, args.k, angle, args.holdout, args.seed, n_controls=0)
    X_knot = db["X"][mb["knot"]]
    acc = {a: {"excess_to_curve": [], "excess_to_nearest_real": []} for a in ("manifold", "linear")}
    for tgt, pick in picks.items():
        x, src = Xs[pick].astype(float), d_steer["y"][pick]
        Z, resid = mb["pca"].project(x), mb["pca"].complement(x)
        arms = subspace_arms(Z, src, tgt, mb, args.K)
        for a in acc:
            W = gc.bf16_round(compose(mb["pca"], arms[a], resid)).astype(float)
            e = mf.off_manifold_energy(W, mb["pca"], mb["curve"], X_knot)
            acc[a]["excess_to_curve"].append(e["to_curve"].mean(1) - e["to_curve"][:, 0])
            acc[a]["excess_to_nearest_real"].append(e["to_nearest_real"].mean(1) - e["to_nearest_real"][:, 0])
    return {a: {q: float(np.mean(np.concatenate(v))) for q, v in qs.items()} for a, qs in acc.items()}


def band(values, spline_value, higher_better):
    v = np.asarray(values, dtype=float)
    better = v > spline_value if higher_better else v < spline_value
    return {"mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "spline": float(spline_value), "spline_rank": int(better.sum()) + 1, "n_draws": len(v),
            "frac_draws_spline_beats": float(np.mean(~better & (v != spline_value)))}


def run(args):
    d = load_inputs(args.dataset, args.layer, args.variable, args.act_dir, args.table, args.split)
    periodic = d["periodic"]
    angle = "labels" if args.labels_angle else "unsupervised"
    m = build(d, args.k, angle, args.holdout, args.seed, args.n_controls)
    probe_rows = d["role"] == "probe"
    probe = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic)
    test = np.flatnonzero(d["role"] == "test")
    ctx = {"periodic": periodic, "probe": probe, "X_knot": d["X"][m["knot"]],
           "near": mf.NearestRealReadout(d["X"][test], d["y"][test]), "near_ctx": None}
    d_steer = d
    if args.context_dataset:
        c = load_inputs(args.context_dataset, args.layer, args.variable or args.dataset, args.context_act_dir,
                        args.context_table, args.context_split)
        assert c["X"].shape[1] == d["X"].shape[1], "primary and context activations differ in width"
        ctest = np.flatnonzero(c["role"] == "test")
        ctx["near_ctx"] = mf.NearestRealReadout(c["X"][ctest], c["y"][ctest])
        d_steer = c
    arms = MAIN_ARMS + (GOODFIRE_ARMS if args.goodfire_baseline else ())
    picks = pick_clips(d_steer, m["held"], args.n_clips, args.seed)

    rows, shared = [], {a: [] for a in arms}
    per_arm = {a: {q: [] for q in (*METRICS, "nearest_real_R_context", "_dose", "_energy", "shift")} for a in arms}
    ctrl = {kind: [{q: [] for q in RANKED} for _ in m["controls"][kind]] for kind in CONTROLS}
    for tgt, pick in picks.items():
        x, src = d_steer["X"][pick].astype(float), d_steer["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Ws = {a: compose(m["pca"], Zk, resid) for a, Zk in subspace_arms(Z, src, tgt, m, args.K).items()}
        if args.goodfire_baseline:
            Ws.update(goodfire_arms(x, src, tgt, m, args.K))
        shift = shift_of(src, tgt, periodic)
        for arm in arms:
            ev = evaluate(Ws[arm], x, src, tgt, m, ctx)
            shared[arm].append(gc.shared_delta_fraction(ev["_delta_end"]))
            for q in per_arm[arm]:
                if q == "shift":
                    per_arm[arm][q].append(shift)
                elif q in ev:
                    per_arm[arm][q].append(ev[q])
            for j in range(len(pick)):
                rows.append({"arm": arm, "id": int(d_steer["df"]["id"].iloc[pick[j]]), "source": float(src[j]),
                             "target": float(tgt), "shift": float(shift[j]),
                             **{q: float(ev[q][j]) for q in METRICS},
                             **({"nearest_real_R_context": float(ev["nearest_real_R_context"][j])}
                                if "nearest_real_R_context" in ev else {})})
        for kind in CONTROLS:
            for i, cv in enumerate(m["controls"][kind]):
                ev = evaluate(compose(m["pca"], curve_coords(cv, Z, src, tgt, args.K), resid), x, src, tgt, m, ctx)
                for q in RANKED:
                    ctrl[kind][i][q].append(ev[q])
    cat = {a: {q: np.concatenate(v) for q, v in qs.items() if v} for a, qs in per_arm.items()}

    out = {"dataset": args.dataset, "variable": args.variable or args.dataset, "layer": args.layer,
           "k": int(m["pca"].components.shape[0]), "K": args.K, "angle_source": m["curve"].coord_source,
           "holdout": m["design"], "held_out_values": m["held"].tolist(),
           "sagitta_per_target": m["sagitta"],
           "sagitta_note": ("at a held-out target the linear arm aims at the chord point between the neighbouring kept "
                            "centroids and the spline at the curve point; sagitta = their distance (PCA units)"),
           "held_out_senses": {"fitting": "held-out values are never knots; probe clips (folds 3-4) never build or "
                                          "receive the edit; steered and nearest-real clips are test",
                               "development": "no design choice may be made on this output's test read",
                               "untouched": "test is read here: anything chosen after this read is exploratory"},
           "context": ({"steered_dataset": args.context_dataset, "spline_built_on": args.dataset,
                        "readouts": "probe and nearest_real from the primary dataset; nearest_real_R_context from "
                                    "real context-dataset test clips at the target value"}
                       if args.context_dataset else None),
           "n_knot_clips": int(m["knot"].sum()), "n_probe_clips": int(probe_rows.sum()), "n_test_clips": len(test),
           "n_steered_per_target": {str(t): len(p) for t, p in picks.items()},
           "arms": {a: ARM_NOTES[a] for a in arms},
           "matched_support": ("manifold, linear, linear_dose_matched, projected and reflected all edit only the "
                               "PCA-k subspace and add back the identical off-subspace residual of each clip"),
           "dose_matched_note": ("linear_dose_matched = the linear arm's delta rescaled per clip and per waypoint to "
                                 "the spline arm's ||delta|| at the same waypoint (dose-matched comparison)"),
           "readouts": {"probe": mf.ProbeReadout.plays_M_y, "nearest_real": mf.NearestRealReadout.plays_M_y},
           "probe_test_error": float(np.mean(mf.value_error(probe.predict(d["X"][test]), d["y"][test], periodic))),
           "isometry_probe": mf.isometry(m["curve"], probe.raw(m["pca"].lift(m["curve"].points))),
           "shared_delta_fraction": {a: float(np.mean(v)) for a, v in shared.items()},
           "delta_norm_per_waypoint": {a: cat[a]["_dose"].mean(0).tolist() for a in arms},
           "rescue_harm": rescue_harm(cat, arms),
           "controls": {kind: {q: band([np.concatenate(dr[q]).mean() for dr in ctrl[kind]],
                                       cat["manifold"][q].mean(), hb) for q, hb in RANKED.items()}
                        for kind in CONTROLS if m["controls"][kind]},
           "summary": summarise(rows, periodic, [a for a in arms if a not in GOODFIRE_ARMS]),
           "rows": rows}
    out["controls_note"] = (f"{args.n_controls} draws each; spline_rank = 1 + number of draws better than the spline "
                            "arm (1 = spline best); band = 5-95% of the draws' means")
    if args.goodfire_baseline:
        out["goodfire_comparison"] = {
            "note": "Goodfire's own comparison: residual erased (linear) vs kept (manifold), so it mixes "
                    "'residual kept vs erased' with 'curved vs straight'. Not the matched-support result.",
            "summary": summarise(rows, periodic, list(GOODFIRE_ARMS))}
    if not args.no_bf16:
        bf = bf16_steering_energy(d, d_steer, args, angle, picks)
        fp = {a: {q: float(cat[a][q].mean()) for q in ("excess_to_curve", "excess_to_nearest_real")}
              for a in ("manifold", "linear")}
        win = lambda r, q: "manifold" if r["manifold"][q] < r["linear"][q] else "linear"
        out["precision_steering_energy"] = {
            "fp32": fp, "bf16": bf,
            "winner_same": {q: win(fp, q) == win(bf, q) for q in ("excess_to_curve", "excess_to_nearest_real")}}
    if m["angle_choice"]:
        out["angle_check"] = {"plane_used": m["plane"], **m["angle_choice"]["checks"]}

    tag = f"{args.dataset}_{args.variable or args.dataset}_L{args.layer}_{args.holdout}"
    if args.context_dataset:
        tag += f"_ctx-{args.context_dataset}"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_steer_{tag}.json").write_text(json.dumps(out, indent=1))
    main = [a for a in arms if a not in GOODFIRE_ARMS]
    plot_gap(out["summary"], Path(args.figures_dir) / f"fig4_gap_vs_shift_{tag}.png", tag, periodic, main)
    plot_energy({a: cat[a]["_energy"] for a in main}, {a: cat[a]["shift"] for a in main},
                Path(args.figures_dir) / f"fig4_path_energy_{tag}.png", tag)
    return out


def rescue_harm(cat, arms):
    """Per-clip counts for the nearest-real readout, beside the mean. vs_unsteered: clips whose agreement with real
    target clips rose (R > 0) or fell (R < 0). vs_linear: clips where the arm's R beat or lost to the linear arm's
    R on the same (clip, target)."""
    out = {}
    lin = cat["linear"]["nearest_real_R"]
    for a in arms:
        R = cat[a]["nearest_real_R"]
        out[a] = {"mean_R": float(R.mean()), "n": int(len(R)),
                  "vs_unsteered": {"improved": int((R > 0).sum()), "worsened": int((R < 0).sum())}}
        if a != "linear":
            diff = R - lin
            out[a]["vs_linear"] = {"improved": int((diff > 0).sum()), "worsened": int((diff < 0).sum()),
                                   "tied": int((diff == 0).sum()), "mean_diff": float(diff.mean())}
    return out


def shift_bins(periodic):
    return np.array([0, 30, 60, 90, 120, 150, 180.01]) if periodic else None


def summarise(rows, periodic, arms, n_bins=5):
    """Per arm and shift bin: mean of every metric (probe errors, nearest-real R, energies, delivered dose), count."""
    shifts = np.array([r["shift"] for r in rows])
    edges = shift_bins(periodic)
    if edges is None:
        edges = np.quantile(shifts, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
    out = {}
    metrics = METRICS + (("nearest_real_R_context",) if "nearest_real_R_context" in rows[0] else ())
    for arm in arms:
        R = [r for r in rows if r["arm"] == arm]
        s = np.array([r["shift"] for r in R])
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [r for r, v in zip(R, s) if lo <= v < hi]
            if not sel:
                continue
            mean = lambda q: float(np.mean([r[q] for r in sel]))
            bins.append({"shift_lo": float(lo), "shift_hi": float(hi), "n": len(sel), **{q: mean(q) for q in metrics}})
        out[arm] = {"overall": {q: float(np.mean([r[q] for r in R])) for q in metrics}, "by_shift": bins}
    return out


def plot_gap(summary, path, tag, periodic, arms):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arm in arms:
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


def plot_energy(energy, shifts, path, tag):
    """Mean distance along the path (waypoint fraction 0..1) to the curve and to the nearest real activation,
    for all steers and for the largest-shift third."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arm in energy:
        E = energy[arm]                                        # [n_total, K, 2]
        s = shifts[arm]
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
    p.add_argument("--holdout", default="scattered", choices=mf.HOLDOUT_DESIGNS,
                   help="held-out label values: every 4th / one 8-value block (arc for direction) / top 8")
    p.add_argument("--goodfire-baseline", action="store_true",
                   help="also run Goodfire's comparison (linear replaces the whole activation), labelled")
    p.add_argument("--n-controls", type=int, default=20, help="draws per random-curve / shuffled-centroid control")
    p.add_argument("--no-bf16", action="store_true", help="skip the BF16 repeat of the steering-energy comparison")
    p.add_argument("--context-dataset", default=None, choices=("direction", "speed", "acceleration"),
                   help="held-out context: steer this dataset's test clips with the spline built on --dataset")
    p.add_argument("--context-act-dir", default=None)
    p.add_argument("--context-table", default=None)
    p.add_argument("--context-split", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-dir", default=None)
    p.add_argument("--table", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())

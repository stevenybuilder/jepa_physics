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
from wm.provenance import layer_role, provenance

MAIN_ARMS = ("manifold", "linear", "linear_dose_matched", "projected", "reflected")
GOODFIRE_ARMS = ("goodfire_linear", "goodfire_manifold")
CONTROLS = ("random_endpoint_matched", "random_unmatched", "shuffled_unmatched")
METRICS = ("probe_err_to_target", "probe_err_to_true", "nearest_real_R", "energy_to_curve", "energy_to_nearest_real",
           "excess_to_curve", "excess_to_nearest_real", "behaviour_energy", "behaviour_energy_mean", "norm_ratio",
           "delta_norm", "delta_norm_path", "probe_err_path", "probe_radius_min")
RANKED = {"nearest_real_R": True, "probe_err_to_target": False, "excess_to_curve": False,
          "excess_to_nearest_real": False, "behaviour_energy": False}   # metric -> higher is better
WAYPOINT_SERIES = ("_wp_err", "_wp_radius", "_wp_bc")
ARM_NOTES = {
    "manifold": "spline walk in the PCA-k subspace, additive (x + curve(t_k) - curve(t_src)), residual kept",
    "linear": "straight line between the polyline (chord) points at source and target, same subspace, residual kept",
    "linear_dose_matched": "linear arm with its delta rescaled, per clip and per waypoint, to the spline's ||delta||",
    "projected": "the spline's own chord (same endpoints) traversed with the spline's arc-length spacing",
    "reflected": "2 * projected - spline: the bend flipped, same endpoints, dose and waypoint count",
    "manifold_transport": "spline walk that rotates the clip's offset from the loop with the loop's frame "
                          "(direction only); residual kept",
    "goodfire_linear": "Goodfire's linear baseline: the whole activation replaced by a full-space chord point",
    "goodfire_manifold": "Goodfire's manifold arm (A.6, mode=replace): PCA part replaced by the curve point, "
                         "residual kept; runs by default",
}


def shift_of(source, target, periodic):
    return mf.value_error(source, target, periodic)


def build(d, k, angle, design="scattered", seed=0, n_controls=20, spline="interp", behaviour_mode="spline",
          subspace="pca", basis=None):
    """PCA, centroids and all curves from the knot clips at kept values (the held-out design decides which).
    angle: "unsupervised" (choose plane automatically, fall back to labels if the ring is not found) or "labels"."""
    knot_all = d["role"] == "knot"
    values = np.unique(d["y"])
    mask, design_info = mf.heldout_design(values, design, d["periodic"], seed=seed)
    held = values[mask]
    knot = knot_all & ~np.isin(d["y"], held)
    pca = gc.fit_subspace(d["X"][knot], d["y"][knot], subspace, k, basis)      # PCA-k unless --plane chart|inlp
    cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
    choice = None
    plane = "activation"
    if d["periodic"] and angle == "unsupervised":
        choice = mf.choose_angle_source(cent["C"], cent["values"])
        angle, plane = choice["angle"], choice["plane"]
    curve = mf.fit_curve(cent, d["periodic"], angle=angle, plane=plane, spline=spline)
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
    # reference manifold that NEITHER arm was built from (Goodfire A.7): probe-fold clips at all values, same PCA
    ref = d["role"] == "probe"
    ref_cent = mf.centroids(pca.project(d["X"][ref]), d["y"][ref])
    ref_curve = mf.fit_curve(ref_cent, d["periodic"], angle="labels", spline="smooth")
    behaviour = mf.BehaviourManifold(d["X"][ref], d["y"][ref], d["periodic"], mode=behaviour_mode, curve=ref_curve,
                                     pca=pca)
    rng = np.random.default_rng(seed)
    controls = {"random_endpoint_matched": [rng.standard_normal((3, pca.components.shape[0])) / np.arange(1, 4)[:, None]
                                            for _ in range(n_controls)],
                "random_unmatched": [gc.random_smooth_curve(curve, rng) for _ in range(n_controls)],
                "shuffled_unmatched": [gc.shuffled_curve(curve, rng) for _ in range(n_controls)]}
    return {"pca": pca, "cent": cent, "curve": curve, "angle_choice": choice, "plane": plane,
            "full_curve": full_curve, "held": held, "knot": knot, "design": design_info, "sagitta": sag,
            "controls": controls, "ref_curve": ref_curve, "X_ref": d["X"][ref], "behaviour": behaviour}


def curve_coords(curve, Z, src, tgt, K, mode="shift"):
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(np.full(len(src), tgt))
    return mf.manifold_coords(Z, curve, ta, tb, K, mode)


def subspace_arms(Z, src, tgt, m, K):
    """PCA-coordinate waypoints [n, K, k] of every matched-support arm. All are lifted with the same residual."""
    Zs = curve_coords(m["curve"], Z, src, tgt, K)
    Zl = mf.linear_coords(Z, mf.piecewise_linear_point(m["curve"], src),
                          mf.piecewise_linear_point(m["curve"], tgt), K)
    dl, ds = Zl - Z[:, None], Zs - Z[:, None]
    nl, ns = np.linalg.norm(dl, axis=-1, keepdims=True), np.linalg.norm(ds, axis=-1, keepdims=True)
    Zd = Z[:, None] + dl * np.where(nl > 0, ns / np.where(nl > 0, nl, 1.0), 0.0)
    projected, reflected = mf.chord_coords(Zs)
    out = {"manifold": Zs, "linear": Zl, "linear_dose_matched": Zd, "projected": projected, "reflected": reflected}
    if m["curve"].periodic:
        out["manifold_transport"] = curve_coords(m["curve"], Z, src, tgt, K, mode="transport")
    return out


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
    n, K = W.shape[:2]
    xs = W[:, -1]
    e = mf.off_manifold_energy(W, m["pca"], m["ref_curve"], m["X_ref"], knn=5)
    raw = ctx["probe"].raw(W.reshape(n * K, -1)).reshape(n, K, -1)         # probe read at every waypoint
    if periodic:
        wp_pred = np.degrees(np.arctan2(raw[..., 0], raw[..., 1])) % 360.0
        wp_radius = np.linalg.norm(raw, axis=-1)
    else:
        wp_pred, wp_radius = raw[..., 0], np.full((n, K), np.nan)
    wp_err = mf.value_error(wp_pred, tgt, periodic)
    pred = wp_pred[:, -1]
    bc = m["behaviour"].bhattacharyya(W)                                   # [n, K]
    dn = np.linalg.norm(W - x[:, None], axis=-1)                           # [n, K] delivered dose per waypoint
    out = {"probe_err_to_target": mf.value_error(pred, tgt, periodic),
           "probe_err_to_true": mf.value_error(pred, src, periodic),
           "nearest_real_R": ctx["near"].score(xs, x, tgt),
           "energy_to_curve": e["to_curve"].mean(1), "energy_to_nearest_real": e["to_nearest_real"].mean(1),
           # excess = mean along the path minus the unsteered clip's own distance (waypoint 0)
           "excess_to_curve": e["to_curve"].mean(1) - e["to_curve"][:, 0],
           "excess_to_nearest_real": e["to_nearest_real"].mean(1) - e["to_nearest_real"][:, 0],
           "behaviour_energy": bc.sum(1), "behaviour_energy_mean": bc.mean(1),
           "norm_ratio": np.linalg.norm(xs, axis=1) / np.linalg.norm(x, axis=1),
           "delta_norm": dn[:, -1], "delta_norm_path": dn.mean(1),
           "probe_err_path": wp_err.mean(1),
           "probe_radius_min": wp_radius.min(1) if periodic else np.full(n, np.nan),
           "_wp_err": wp_err, "_wp_radius": wp_radius, "_wp_bc": bc,
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
    mb = build(db, args.k, angle, args.holdout, args.seed, n_controls=0, spline=args.spline, behaviour_mode=args.behaviour,
               subspace=args.plane, basis=args.basis_matrix)
    acc = {a: {"excess_to_curve": [], "excess_to_nearest_real": []} for a in ("manifold", "linear")}
    for tgt, pick in picks.items():
        x, src = Xs[pick].astype(float), d_steer["y"][pick]
        Z, resid = mb["pca"].project(x), mb["pca"].complement(x)
        arms = subspace_arms(Z, src, tgt, mb, args.K)
        for a in acc:
            W = gc.bf16_round(compose(mb["pca"], arms[a], resid)).astype(float)
            e = mf.off_manifold_energy(W, mb["pca"], mb["ref_curve"], mb["X_ref"], knn=5)
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
    variable = args.variable or args.dataset
    d = load_inputs(args.dataset, args.layer, args.variable, args.act_dir, args.table, args.split)
    periodic = d["periodic"]
    angle = "labels" if args.labels_angle else "unsupervised"
    nuisance = None
    if args.nuisance_regress:
        N, names = gc.nuisance_matrix(d["df"], variable)
        X_res, r2 = gc.regress_out(d["X"], N, d["is_train"])
        d = {**d, "X": X_res}
        nuisance = {"covariates": names, "variance_explained_by_nuisance_train": r2,
                    "space": "train-standardised activations minus their least-squares fit on the covariates"}
    # an INLP .npz Q lives in train-standardised coordinates; the nuisance residuals already do too, so it is used
    # as is there (dividing by the residuals' SD would distort it); raw activations get the covector map Q / SD
    basis = (gc.load_basis_matrix(args.basis, None if nuisance else d["X"][d["is_train"]]) if args.basis else None)
    args.basis_matrix = basis
    if args.plane == "inlp" and basis is None:
        raise SystemExit("--plane inlp needs --basis")
    m = build(d, args.k, angle, args.holdout, args.seed, args.n_controls, args.spline, args.behaviour, args.plane,
              basis)
    probe_rows = d["role"] == "probe"
    probe = mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic)
    test = np.flatnonzero(d["role"] == "test")
    ctx = {"periodic": periodic, "probe": probe,
           "near": mf.NearestRealReadout(d["X"][test], d["y"][test]), "near_ctx": None}
    d_steer = d
    if args.context_dataset:
        c = load_inputs(args.context_dataset, args.layer, args.variable or args.dataset, args.context_act_dir,
                        args.context_table, args.context_split)
        if args.nuisance_regress:           # the context set's own covariates, fit on its own train rows
            c = {**c, "X": gc.regress_out(c["X"], gc.nuisance_matrix(c["df"], variable)[0], c["is_train"])[0]}
        assert c["X"].shape[1] == d["X"].shape[1], "primary and context activations differ in width"
        ctest = np.flatnonzero(c["role"] == "test")
        ctx["near_ctx"] = mf.NearestRealReadout(c["X"][ctest], c["y"][ctest])
        d_steer = c
    arms = (MAIN_ARMS + (("manifold_transport",) if periodic else ()) + ("goodfire_manifold",)
            + (("goodfire_linear",) if args.goodfire_baseline else ()))
    picks = pick_clips(d_steer, m["held"], args.n_clips, args.seed)

    rows, shared = [], {a: [] for a in arms}
    per_arm = {a: {q: [] for q in (*METRICS, "nearest_real_R_context", "_dose", "_energy", *WAYPOINT_SERIES, "shift",
                                   "_target")} for a in arms}
    ctrl = {kind: [{q: [] for q in RANKED} for _ in m["controls"][kind]] for kind in CONTROLS}
    ctrl_ref = {q: [] for q in RANKED}         # the spline arm on the same clip subset as the control draws
    for tgt, pick in picks.items():
        x, src = d_steer["X"][pick].astype(float), d_steer["y"][pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Zarms = subspace_arms(Z, src, tgt, m, args.K)
        Ws = {a: compose(m["pca"], Zk, resid) for a, Zk in Zarms.items()}
        Ws.update(goodfire_arms(x, src, tgt, m, args.K))
        shift = shift_of(src, tgt, periodic)
        for arm in arms:
            ev = evaluate(Ws[arm], x, src, tgt, m, ctx)
            shared[arm].append(gc.shared_delta_fraction(ev["_delta_end"]))
            for q in per_arm[arm]:
                if q == "shift":
                    per_arm[arm][q].append(shift)
                elif q == "_target":
                    per_arm[arm][q].append(np.full(len(pick), tgt))
                elif q in ev:
                    per_arm[arm][q].append(ev[q])
            for j in range(len(pick)):
                rows.append({"arm": arm, "id": int(d_steer["df"]["id"].iloc[pick[j]]), "source": float(src[j]),
                             "target": float(tgt), "shift": float(shift[j]),
                             **{q: float(ev[q][j]) for q in METRICS},
                             **({"nearest_real_R_context": float(ev["nearest_real_R_context"][j])}
                                if "nearest_real_R_context" in ev else {})})
        sub = slice(0, args.n_control_clips)
        Zs_sub = Zarms["manifold"][sub]
        ev = evaluate(Ws["manifold"][sub], x[sub], src[sub], tgt, m, ctx)
        for q in RANKED:
            ctrl_ref[q].append(ev[q])
        for kind in CONTROLS:
            for i, cv in enumerate(m["controls"][kind]):
                Zc = (mf.endpoint_matched_random(Zs_sub, cv) if kind == "random_endpoint_matched"
                      else curve_coords(cv, Z[sub], src[sub], tgt, args.K))
                ev = evaluate(compose(m["pca"], Zc, resid[sub]), x[sub], src[sub], tgt, m, ctx)
                for q in RANKED:
                    ctrl[kind][i][q].append(ev[q])
    cat = {a: {q: np.concatenate(v) for q, v in qs.items() if v} for a, qs in per_arm.items()}

    out = {"dataset": args.dataset, "variable": args.variable or args.dataset, "layer": args.layer,
           **layer_role(args.dataset, args.layer, variable),
           "provenance": provenance(args.split, seeds={"seed": args.seed, "controls": args.seed}, layer=args.layer,
                                    pool="meanpool", holdout=args.holdout, spline=args.spline, k=args.k,
                                    subspace=args.plane, K=args.K, nuisance_regressed=bool(nuisance)),
           "subspace": args.plane, "nuisance_regressed": nuisance,
           "k": int(m["pca"].components.shape[0]), "K": args.K, "angle_source": m["curve"].coord_source,
           "spline": m["curve"].kind,
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
           "deviations_from_goodfire": {
               "additive_path": "main arms are additive (x + curve(t_k) - curve(t_src)); Goodfire A.6 replaces the "
                                "PCA-64 part with the curve point and keeps the complement. That is run as "
                                "goodfire_manifold under --goodfire-baseline.",
               "matched_linear": "Goodfire's linear arm replaces the whole activation with a full-space chord point; "
                                 "the main linear arm edits the same PCA-k subspace as the spline and keeps the "
                                 "residual (goodfire_linear reproduces theirs).",
               "behaviour_space": "Eq. 9 distances are taken in the PCA-k coordinates of the reference spline "
                                  "(Goodfire's encoder output is itself 64-d); --behaviour centroid_sq is the "
                                  "squared, spread-normalised variant"},
           "energy_reference": ("energy_to_curve / excess_to_curve: distance (PCA-k) to a count-weighted smoothing "
                                "spline through probe-fold centroids at all values, which no arm was built from; "
                                "energy_to_nearest_real: mean distance to the 5 nearest probe-fold clips (full space)"),
           "behaviour_manifold": {"source": mf.BehaviourManifold.source, "tau": m["behaviour"].tau,
                                  "mode": m["behaviour"].mode, "distance_scale": m["behaviour"].scale,
                                  "bins": ("B = 128 points along the reference spline (probe folds), distances in "
                                           "PCA-k coordinates" if m["behaviour"].mode == "spline"
                                           else "64 per-value centroids, full space"),
                                  "caveat": mf.BehaviourManifold.caveat, "clips": "probe folds at the read layer",
                                  "metric": "behaviour_energy = E_BC, sum over the K waypoints of the Bhattacharyya "
                                            "distance to M_y (A.7); behaviour_energy_mean = per waypoint"},
           "waypoint_readout": waypoint_summary(cat, arms, periodic),
           "matched_support": ("manifold, linear, linear_dose_matched, projected and reflected all edit only the "
                               "PCA-k subspace and add back the identical off-subspace residual of each clip"),
           "dose_matched_note": ("linear_dose_matched = the linear arm's delta rescaled per clip and per waypoint to "
                                 "the spline arm's ||delta|| at the same waypoint (dose-matched comparison)"),
           "readouts": {"probe": mf.ProbeReadout.plays_M_y, "nearest_real": mf.NearestRealReadout.plays_M_y},
           "probe_test_error": float(np.mean(mf.value_error(probe.predict(d["X"][test]), d["y"][test], periodic))),
           "isometry_probe": mf.isometry(m["curve"], probe.raw(m["pca"].lift(m["curve"].points)),
                                         "angular" if periodic else "euclidean"),
           "isometry_behaviour": mf.isometry(m["curve"], m["behaviour"].geodesic_matrix(m["curve"].values),
                                             "precomputed"),
           "isometry_note": ("Pearson r between knot-pair geodesics on the activation spline and, for the probe, the "
                             "angle between (sin, cos) readouts (direction) or |difference| (scalars); for the "
                             "behaviour manifold, Hellinger geodesics along M_y (Goodfire A.5, geodesic vs geodesic)"),
           "shared_delta_fraction": {a: float(np.mean(v)) for a, v in shared.items()},
           "delta_norm_per_waypoint": {a: cat[a]["_dose"].mean(0).tolist() for a in arms},
           "rescue_harm": rescue_harm(cat, arms),
           "controls": {kind: {q: band([np.concatenate(dr[q]).mean() for dr in ctrl[kind]],
                                       np.concatenate(ctrl_ref[q]).mean(), hb) for q, hb in RANKED.items()}
                        for kind in CONTROLS if m["controls"][kind]},
           "summary": summarise(rows, periodic, [a for a in arms if a != "goodfire_linear"]),
           "gaps": paired_gaps(cat, [a for a in arms if a != "manifold"], periodic, args.seed),
           "gaps_note": ("manifold minus each other arm on the same (clip, target) rows: 95% CI from a paired "
                         "bootstrap over rows (1000 draws), and mean +/- SE across steer targets (Goodfire A.7 "
                         "reports mean +/- SE over pairs); overall and by shift bin"),
           "rows": rows}
    out["controls_note"] = (f"{args.n_controls} draws each, on the first {args.n_control_clips} steered clips per "
                            "target (the spline value in each band is on the same clips); spline_rank = 1 + number "
                            "of draws better than the spline arm (1 = spline best); band = 5-95% of the draws' means")
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
    tag += ("" if args.plane == "pca" else f"_{args.plane}") + ("_nuis" if nuisance else "")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.results_dir) / f"p2_steer_{tag}.json").write_text(json.dumps(out, indent=1))
    main = [a for a in arms if a != "goodfire_linear"]
    plot_gap(out["summary"], Path(args.figures_dir) / f"fig4_gap_vs_shift_{tag}.png", tag, periodic, main)
    plot_energy({a: cat[a]["_energy"] for a in main}, {a: cat[a]["shift"] for a in main},
                Path(args.figures_dir) / f"fig4_path_energy_{tag}.png", tag)
    plot_waypoints(out["waypoint_readout"], Path(args.figures_dir) / f"fig4_waypoint_readout_{tag}.png", tag,
                   periodic, main)
    return out


def waypoint_summary(cat, arms, periodic, n_bins=4):
    """Per arm and shift bin (the largest bin is the one near 180 degrees for direction): the evaluation probe's
    error to target and readout radius, and the Bhattacharyya distance to M_y, at every waypoint (mean over clips)."""
    shifts = np.concatenate([cat[a]["shift"] for a in arms])
    edges = (np.array([0, 45, 90, 135, 180.01]) if periodic
             else np.quantile(shifts, np.linspace(0, 1, n_bins + 1)) + np.r_[np.zeros(n_bins), 1e-9])
    out = {}
    for a in arms:
        s = cat[a]["shift"]
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (s >= lo) & (s < hi)
            if not sel.any():
                continue
            b = {"shift_lo": float(lo), "shift_hi": float(hi), "n": int(sel.sum()),
                 "err_to_target": cat[a]["_wp_err"][sel].mean(0).tolist(),
                 "bhattacharyya": cat[a]["_wp_bc"][sel].mean(0).tolist()}
            if periodic:
                b["radius"] = cat[a]["_wp_radius"][sel].mean(0).tolist()
            bins.append(b)
        out[a] = bins
    return out


def plot_waypoints(wp, path, tag, periodic, arms):
    """Largest-shift bin: probe readout radius (direction) or error, and behaviour distance, along the path."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for a in arms:
        b = wp[a][-1]
        frac = np.linspace(0, 1, len(b["err_to_target"]))
        style = "-" if a in ("manifold", "linear") else ":"
        axes[0].plot(frac, b["radius"] if periodic else b["err_to_target"], style, label=a)
        axes[1].plot(frac, b["bhattacharyya"], style, label=a)
    b = wp[arms[0]][-1]
    axes[0].set(xlabel="fraction of path", ylabel="probe readout radius" if periodic else "probe error to target",
                title=f"Evaluation probe, shift {b['shift_lo']:.0f}-{b['shift_hi']:.0f}")
    axes[1].set(xlabel="fraction of path", ylabel="Bhattacharyya distance to M_y", title="Behaviour manifold (Eq. 9)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Every waypoint scored, {tag}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paired_bootstrap(diff, n_boot=1000, seed=0):
    """Mean of paired differences with a 95% percentile-bootstrap CI over rows."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)], "n": int(len(diff))}


def paired_gaps(cat, others, periodic, seed=0):
    """manifold - other, per metric: paired bootstrap CI over rows and mean +/- SE across targets; by shift bin."""
    metrics = [q for q in (*RANKED, "probe_err_path", "probe_radius_min", "delta_norm")
               if not (q == "probe_radius_min" and not periodic)]
    s, tgt = cat["manifold"]["shift"], cat["manifold"]["_target"]
    edges = shift_bins(periodic)
    if edges is None:
        edges = np.quantile(s, np.linspace(0, 1, 6))
        edges[-1] += 1e-9
    out = {}
    for o in others:
        out[f"manifold_minus_{o}"] = res = {}
        for q in metrics:
            diff = cat["manifold"][q] - cat[o][q]
            per_t = np.array([diff[tgt == t].mean() for t in np.unique(tgt)])
            res[q] = {**paired_bootstrap(diff, seed=seed),
                      "mean_over_targets": float(per_t.mean()),
                      "se_over_targets": float(per_t.std(ddof=1) / np.sqrt(len(per_t))) if len(per_t) > 1 else None,
                      "by_shift": [{"shift_lo": float(lo), "shift_hi": float(hi),
                                    **paired_bootstrap(diff[(s >= lo) & (s < hi)], seed=seed)}
                                   for lo, hi in zip(edges[:-1], edges[1:]) if ((s >= lo) & (s < hi)).any()]}
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
    p.add_argument("--K", type=int, default=50, help="waypoints per path (Goodfire A.6 uses 50)")
    p.add_argument("--n-clips", type=int, default=48, help="steered test clips per target value")
    p.add_argument("--labels-angle", action="store_true", help="flagged fallback: true angle as spline coordinate")
    p.add_argument("--holdout", default="scattered", choices=mf.HOLDOUT_DESIGNS,
                   help="held-out label values: every 4th / one 8-value block (arc for direction) / top 8")
    p.add_argument("--goodfire-baseline", action="store_true",
                   help="also run Goodfire's comparison (linear replaces the whole activation), labelled")
    p.add_argument("--spline", default="interp", choices=("interp", "smooth"),
                   help="interpolating (Goodfire A.3) or count-weighted smoothing spline (B.1); choose from the "
                        "geometry check's heldout_reconstruction")
    p.add_argument("--n-controls", type=int, default=20, help="draws per random-curve / shuffled-centroid control")
    p.add_argument("--behaviour", default="spline", choices=("spline", "centroid_sq"),
                   help="Eq. 9 behaviour: literal (unsquared, tau=0.5, 128 spline bins) or squared/spread-normalised")
    p.add_argument("--n-control-clips", type=int, default=16, help="steered clips per target used by control draws")
    p.add_argument("--no-bf16", action="store_true", help="skip the BF16 repeat of the steering-energy comparison")
    p.add_argument("--plane", default="pca", choices=("pca", "chart", "inlp"),
                   help="steering subspace: top-k PCA (Goodfire), circular-chart plane, or INLP basis (--basis)")
    p.add_argument("--basis", default=None, help=".npy [D, r] or INLP .npz, for --plane inlp")
    p.add_argument("--nuisance-regress", action="store_true",
                   help="regress nuisance covariates out of the activations first (labelled variant)")
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

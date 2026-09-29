"""Aggregate run_endpoint_diagnosis.py outputs into results/p2_endpoint_diagnosis.json and
figures/fig_p2_endpoint_diagnosis.png.

  python scripts/summarize_endpoint_diagnosis.py --inputs L22_labels=... L12_unsup=... L12_labels=... L12_unsup_arc=...
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm.data import PROJECT_ROOT

RES = PROJECT_ROOT / "results"
READERS = {"probe": "probe_err", "mlp": "mlp_err", "R": "nearest_real_R"}


def stored_arc(L, seed):
    """Stored run_part2 raw-chord arc file (the REPORT table's source) for reproduction."""
    if seed == 0:
        p = RES / f"p2_steer_direction_direction_L{L}_contiguous_rawchord.json"
    else:
        sub = f"L{L}_s{seed}" + ("_labels" if L == 22 and seed in (4, 8, 9, 10) else "")
        p = RES / "arcs_rawchord" / sub / f"p2_steer_direction_direction_L{L}_contiguous.json"
    d = json.loads(p.read_text())
    s = d["summary"]
    return {"file": str(p.relative_to(PROJECT_ROOT)), "manifold": s["manifold"]["overall"]["probe_err_to_target"],
            "linear_raw": s["linear_raw"]["overall"]["probe_err_to_target"],
            "gap": d["gaps"]["manifold_minus_linear_raw"]["probe_err_to_target"]["mean"],
            "angle_source": d["angle_source"]}


def ms(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x)),
            "min": float(x.min()), "max": float(x.max())}


def summarise(runs, check_stored=None):
    """runs: list of per-arc dicts (seed 0 = headline, 1-16 = the 16 arcs)."""
    head = [r for r in runs if r["seed"] == 0]
    arcs = [r for r in runs if r["seed"] != 0]
    arms = sorted(runs[0]["arms"])
    out = {"headline_seed0": {}, "arcs16": {}, "n_arcs": len(arcs)}
    for name, group in (("headline_seed0", head), ("arcs16", arcs)):
        if not group:
            continue
        o = out[name]
        o["arms"] = {}
        for a in arms:
            e = {}
            for r, key in READERS.items():
                vals = [g["arms"][a][key] for g in group]
                gaps = [g["gap_minus_add_chord_raw"][a][r] for g in group]
                e[r] = ms(vals)
                e[f"{r}_gap_vs_add_chord_raw"] = ms(gaps)
                # arm better than the raw chord: lower error, or higher R
                better = [(gp < 0) if r != "R" else (gp > 0) for gp in gaps]
                e[f"{r}_n_better_than_chord_raw"] = int(np.sum(better))
            e["delta_norm"] = ms([g["arms"][a]["delta_norm"] for g in group])
            o["arms"][a] = e
        o["aim_geometry"] = {k: {q: ms([g["aim_geometry"][k][q] for g in group])
                                 for q in ("dist_to_true_centroid", "probe_err_of_aim_point")}
                             for k in group[0]["aim_geometry"] if all(k in g["aim_geometry"] for g in group)}
        ag = [g["arc_geometry"] for g in group]
        o["arc_geometry"] = {
            "bulge_beta_interp": ms([np.mean(a["bulge_beta_interp"]) for a in ag]),
            "bulge_beta_smooth": ms([np.mean(a["bulge_beta_smooth"]) for a in ag]),
            "sagitta_interp_vs_rawchord": ms([a["sagitta_interp_vs_rawchord"] for a in ag]),
            "sagitta_smooth_vs_rawchord": ms([a["sagitta_smooth_vs_rawchord"] for a in ag]),
            "smoothing_knot_displacement_mean": ms([a["smoothing_knot_displacement_mean"] for a in ag]),
            "sagitta_lam_cv_vs_rawchord": ms([a["sagitta_lam_cv_vs_rawchord"] for a in ag]),
            "sagitta_lam_cv_vs_its_own_chord": ms([a["sagitta_lam_cv_vs_its_own_chord"] for a in ag]),
            "bulge_beta_lam_cv": ms([np.mean(a["bulge_beta_lam_cv"]) for a in ag]),
            "centroid_noise_true_heldout": ms([a["centroid_noise_true_heldout"] for a in ag]),
            "causalab_vs_scipy_interp_max_abs_diff_max": float(max(a["causalab_vs_scipy_interp_max_abs_diff"]
                                                                   for a in ag)),
            "foreign_knots_total": int(sum(sum(a["foreign_knots_per_target"]) for a in ag)),
            "arcs_with_foreign_knots": [g["seed"] for g, a in zip(group, ag) if sum(a["foreign_knots_per_target"])],
            "lambda_cv_values": [a["lambda_cv"]["lambda_cv"] for a in ag],
            "cv_err_best_spline": ms([min(a["lambda_cv"]["cv_err_by_lambda"].values()) for a in ag]),
            "cv_err_interp": ms([a["lambda_cv"]["cv_err_by_lambda"]["0"] for a in ag]),
            "cv_err_chord": ms([a["lambda_cv"]["cv_err_chord"] for a in ag]),
            "kept_knots": {k: {q: ms([a["kept_knots"][k][q] for a in ag])
                               for q in ("probe_err", "dist_to_probe_fold_centroid")} for k in ag[0]["kept_knots"]},
            "causalab_edit_vs": {o_: {q: ms([g["causalab_edit_vs"][o_][q] for g in group])
                                      for q in ("cos", "norm_ratio", "endpoint_dist")}
                                 for o_ in group[0]["causalab_edit_vs"]},
        }
    if check_stored:
        rep = []
        for r in runs:
            try:
                s = stored_arc(r["layer"], r["seed"])
            except FileNotFoundError:
                continue
            rep.append({"seed": r["seed"], "stored": s, "ours_spline": r["arms"]["add:spline_smooth"]["probe_err"],
                        "ours_chord_raw": r["arms"]["add:chord_raw"]["probe_err"],
                        "ours_gap": r["gap_minus_add_chord_raw"]["add:spline_smooth"]["probe"],
                        "abs_diff_gap": abs(r["gap_minus_add_chord_raw"]["add:spline_smooth"]["probe"] - s["gap"])})
        out["reproduction_vs_stored"] = {"max_abs_diff_gap_deg": float(max(x["abs_diff_gap"] for x in rep)),
                                         "n": len(rep), "per_arc": rep}
    out["per_arc_gap_probe"] = {a: {r["seed"]: r["gap_minus_add_chord_raw"][a]["probe"] for r in runs}
                                for a in ("add:spline_smooth", "add:spline_interp", "add:causalab_lam_cv",
                                          "replace:causalab_interp", "add:oracle")}
    return out


def _g(res, cfg, grp, arm, reader="probe", what="gap"):
    e = res["configs"][cfg][grp]["arms"][arm]
    if what == "gap":
        return {"value": e[f"{reader}_gap_vs_add_chord_raw"]["mean"], "sd": e[f"{reader}_gap_vs_add_chord_raw"]["sd"],
                "n_arm_better": e[f"{reader}_n_better_than_chord_raw"],
                "key": f"configs.{cfg}.{grp}.arms.{arm}.{reader}_gap_vs_add_chord_raw.mean"}
    return {"value": e[reader]["mean"], "key": f"configs.{cfg}.{grp}.arms.{arm}.{reader}.mean"}


def decomposition(res):
    """The decomposition table: every cell a number with its key path in this file."""
    out = {}
    for cfg in ("L22_labels", "L12_unsup", "L12_labels"):
        if cfg not in res["configs"]:
            continue
        for grp in ("headline_seed0", "arcs16"):
            c = res["configs"][cfg][grp]
            ag, ar = c["aim_geometry"], c["arc_geometry"]
            aim = {k: {"dist_to_true_centroid": ag[k]["dist_to_true_centroid"]["mean"],
                       "probe_err_of_aim_point": ag[k]["probe_err_of_aim_point"]["mean"],
                       "key": f"configs.{cfg}.{grp}.aim_geometry.{k}"}
                   for k in ("spline_smooth", "chord_raw", "chord_smooth", "spline_interp", "causalab_lam_cv",
                             "chord_lam_cv_knots", "oracle")}
            out[f"{cfg}.{grp}"] = {
                "baseline_gap_ours_smooth_minus_raw_chord": _g(res, cfg, grp, "add:spline_smooth"),
                "a_aim": {"aim_points": aim,
                          "target_only_gap (tgt=spline_smooth, src=raw centroid)":
                              _g(res, cfg, grp, "hybrid:tgt=spline_smooth,src=chord_raw"),
                          "source_only_gap (tgt=raw chord, src=smoothed knot)":
                              _g(res, cfg, grp, "hybrid:tgt=chord_raw,src=spline_smooth"),
                          "oracle_true_centroid_gap": _g(res, cfg, grp, "add:oracle"),
                          "bulge_beta_smooth": ar["bulge_beta_smooth"]["mean"],
                          "bulge_beta_lam_cv": ar["bulge_beta_lam_cv"]["mean"],
                          "centroid_noise_true_heldout (sqrt(trace cov / n) of the true held-out centroid)":
                              ar["centroid_noise_true_heldout"]["mean"],
                          "sagitta_lam_cv_vs_rawchord": ar["sagitta_lam_cv_vs_rawchord"]["mean"],
                          "raw_centroid_vs_independent_probe_fold_centroid":
                              ar["kept_knots"]["raw_centroid"]["dist_to_probe_fold_centroid"]["mean"],
                          "bulge_beta_interp": ar["bulge_beta_interp"]["mean"]},
                "b_smoothing": {**{f"fitpack_s{f}": _g(res, cfg, grp, f"add:splrep_s{f}")
                                   for f in ("0", "0.03", "0.1", "0.3", "1", "3")},
                                "interp_causalab_lam0": _g(res, cfg, grp, "add:causalab_interp"),
                                **{f"reinsch_lam_{l}": _g(res, cfg, grp, f"add:causalab_lam_{l}")
                                   for l in ("1e-10", "1e-08", "1e-07", "1e-06", "1e-05", "0.0001")},
                                "reinsch_lam_cv": _g(res, cfg, grp, "add:causalab_lam_cv"),
                                "chord_through_reinsch_knots (denoised anchors, no bend)":
                                    _g(res, cfg, grp, "add:chord_lam_cv_knots"),
                                "reinsch_target_only (tgt=lam_cv, src=raw centroid)":
                                    _g(res, cfg, grp, "hybrid:tgt=causalab_lam_cv,src=chord_raw"),
                                "reinsch_source_only (tgt=raw chord, src=lam_cv knot)":
                                    _g(res, cfg, grp, "hybrid:tgt=chord_raw,src=causalab_lam_cv"),
                                "lambda_cv_values": ar["lambda_cv_values"],
                                "knot_cv_err": {"best_spline": ar["cv_err_best_spline"]["mean"],
                                                "interp": ar["cv_err_interp"]["mean"],
                                                "chord": ar["cv_err_chord"]["mean"]},
                                "kept_knot_probe_err": {k: v["probe_err"]["mean"]
                                                        for k, v in ar["kept_knots"].items()}},
                "c_norm": {"delta_norm_spline_smooth": c["arms"]["add:spline_smooth"]["delta_norm"]["mean"],
                           "delta_norm_chord_raw": c["arms"]["add:chord_raw"]["delta_norm"]["mean"],
                           "delta_norm_spline_interp": c["arms"]["add:spline_interp"]["delta_norm"]["mean"],
                           "chord_raw_at_spline_smooth_norm": _g(res, cfg, grp, "add_norm:chord_raw@spline_smooth"),
                           "spline_smooth_at_chord_raw_norm": _g(res, cfg, grp, "add_norm:spline_smooth@chord_raw"),
                           "spline_interp_at_chord_raw_norm": _g(res, cfg, grp, "add_norm:spline_interp@chord_raw"),
                           "chord_raw_at_spline_interp_norm": _g(res, cfg, grp, "add_norm:chord_raw@spline_interp")},
                "e_readers": {arm: {r: _g(res, cfg, grp, arm, r) for r in ("probe", "mlp", "R")}
                              for arm in ("add:spline_smooth", "add:spline_interp", "add:causalab_lam_cv",
                                          "add:oracle")},
                "paper_protocol": {
                    "replace_interp (A.3 + A.6 manifold)": _g(res, cfg, grp, "replace:causalab_interp", what="err"),
                    "replace_chord_raw_pca64 (A.9 chord, replace)": _g(res, cfg, grp, "replace:chord_raw", what="err"),
                    "replace_full_chord (A.6 linear)": _g(res, cfg, grp, "replace_full:chord_raw_fullspace",
                                                          what="err"),
                    "add_interp_gap": _g(res, cfg, grp, "add:causalab_interp"),
                    "replace_interp_gap_vs_add_chord": _g(res, cfg, grp, "replace:causalab_interp"),
                    "replace_lam_cv (exploratory)": _g(res, cfg, grp, "replace:causalab_lam_cv", what="err"),
                    "causalab_label_free_angle_interp_gap": _g(res, cfg, grp, "add:causalab_pcaangle_interp")},
            }
    return out


def foreign_split(res):
    """Point 12: per-arc probe gap on the 8 arcs whose label-free knot order puts foreign knots on the spline stretch
    vs the 8 clean arcs, for the stored (label-free) order and the value-ordered (labels) refit."""
    if "L12_unsup" not in res["configs"] or "L12_labels" not in res["configs"]:
        return None
    bad = res["configs"]["L12_unsup"]["arcs16"]["arc_geometry"]["arcs_with_foreign_knots"]
    out = {"affected_seeds": bad}
    for cfg in ("L12_unsup", "L12_labels", "L12_unsup_arc"):
        if cfg not in res["configs"]:
            continue
        pa = res["configs"][cfg]["per_arc_gap_probe"]
        for arm in ("add:spline_smooth", "add:causalab_lam_cv"):
            g = {int(k): v for k, v in pa[arm].items() if int(k) != 0}
            out[f"{cfg}.{arm}"] = {"affected": ms([g[s] for s in bad]),
                                   "clean": ms([g[s] for s in g if s not in bad])}
    return out


STEPS = [
    {"step": "PCA fit set and dimension",
     "paper": "64-D PCA over the activations of all prompts in the task (A.3, steering_paper.txt l.2621-2625)",
     "causalab": "configs/analysis/subspace.yaml k_features: 64; methods/pca.py collect_and_compute_PCA (centred SVD)",
     "ours": "PCA-64 by SVD on knot-fold clips (folds 0-2) at KEPT values only (run_part2.py:build l.85, "
             "manifold.py:fit_pca l.66)",
     "difference": "fit set excludes probe/test folds and the held-out arc (needed for a held-out test; the paper "
                   "has no held-out values)", "ledger": "known (design, #19/#213 context)"},
    {"step": "Whitening / standardisation",
     "paper": "none stated", "causalab": "activation_manifold.yaml intrinsic_mode: pca -> SplineManifoldConfig."
     "standardize_coords False (spline/train.py l.63-65; fitting_pipeline.py l.322-328: identity standardiser); PCA "
     "not whitened", "ours": "none (unwhitened PCA coordinates)", "difference": "none", "ledger": "-"},
    {"step": "Centroid rule",
     "paper": "mean of projected activations per ground-truth value (A.3 l.2629)",
     "causalab": "builders.py:compute_centroids l.95; path_steering/main.py l.548-560",
     "ours": "manifold.py:centroids l.77 (mean of PCA coordinates per value, knot clips at kept values)",
     "difference": "none", "ledger": "-"},
    {"step": "Intrinsic coordinate (cyclic)",
     "paper": "theta = atan2(PC2, PC1) of the centroids, unsupervised (A.3 l.2634-2637)",
     "causalab": "builders.py:detect_periodic_dims l.213 + remap_periodic_to_angle l.268 (atan2 of centred, "
                 "sqrt-eigenvalue-scaled first two PCA columns of the centroids; spline/train.py l.397-465)",
     "ours": "run_part2.py:build -> manifold.choose_angle_source l.354 (atan2 in activation or centroid plane, "
             "unscaled, else labels); point 22 runs on labels (label-free fails there), point 12 on the "
             "centroid-plane angle",
     "difference": "labels fallback; no sqrt-eigenvalue scaling; point-12 knot order not monotone on 8 arcs",
     "ledger": "known (#13, #213, #265)"},
    {"step": "Spline type and smoothing",
     "paper": "periodic cubic spline interpolating every centroid, no smoothing (A.3 l.2631-2633, l.2646); "
              "smoothing spline weighted by sqrt(count) only for the mountain car encoder (B.1 l.2836-2837)",
     "causalab": "cubic.py:CubicSpline1D._fit_periodic l.207 (Reinsch 1967, knot at every centroid, one lambda for "
                 "all coordinates, lambda=0 = interpolation); configs smoothness: 0.0 (activation_manifold.yaml l.7, "
                 "spline/train.py l.53)",
     "ours": "manifold.py:SmoothSpline l.215: FITPACK splrep per coordinate, weights sqrt(count)/sd_c, s = m, "
             "adaptive knot subset (a regression spline, not Reinsch), used for every headline/arc/bake-off run "
             "(--spline smooth)",
     "difference": "smoothing where the paper interpolates, and FITPACK per-coordinate chi-square smoothing instead "
                   "of causalab's Reinsch penalty; this diagnosis finds it is the main cause of the endpoint loss",
     "ledger": "known as a deviation (#14, #215 rated 'Low'); the impact is NEW: it is not low"},
    {"step": "Knot parameterisation / period",
     "paper": "atan2 angle as intrinsic coordinate; periodic spline", "causalab": "manifold.py:SplineManifold "
     "l.60-90: coordinates normalised by the period; period inferred as range*n/(n-1) when not passed (assumes "
     "equally spaced knots)", "ours": "radians, period 2 pi, scipy CubicSpline(bc_type='periodic') or splrep per=1",
     "difference": "none on full rings; with an arc removed causalab's inferred period is wrong (6.297 vs 6.283 at "
                   "pt 22 headline) unless periods=[2 pi] is passed; effect on the endpoint 0.01 deg",
     "ledger": "NEW (minor; only matters if causalab is run on held-out data)"},
    {"step": "Held-out target coordinate",
     "paper": "none: endpoints are centroids (A.6 l.2709; path_mode.py:_build_geodesic_path l.66 walks between "
              "knot coordinates)", "causalab": "same",
     "ours": "Curve.coord_of_value l.148 (value-order interpolation of knot coordinates; aim='arc' variant l.182)",
     "difference": "necessary extension; with a non-monotone label-free order it aims among foreign knots",
     "ledger": "known (#213, #265, #267)"},
    {"step": "Edit rule",
     "paper": "manifold: replace the top-64 PCA part with s(u), keep the complement; linear: replace the whole "
              "activation with the raw-centroid chord point (A.6 l.2715-2724; Eqs. 1-2 l.1066-1070)",
     "causalab": "steer/collect.py:collect_grid_distributions replace_fn l.241 (target replaces the featurised "
                 "coordinates; the PCA featurizer keeps the orthogonal complement); steer.py mode add|replace is the "
                 "generic API",
     "ours": "additive x + lift(P(tgt) - P(src)) with P(src) the curve point at the clip's TRUE source value "
             "(run_part2.py:subspace_arms l.135, bakeoff.arm_deltas l.100); replace arm goodfire_manifold exists "
             "but is not the headline",
     "difference": "additive with a curve-anchored source: a smoothing spline's knot displacement enters the edit "
                   "twice (target and source)", "ledger": "known as a deviation (deviations_from_goodfire); the "
                   "double-counting of smoothing through the source anchor is NEW"},
    {"step": "Residual handling", "paper": "complement preserved (A.6)", "causalab": "PCA featurizer error term",
     "ours": "complement preserved (manifold.PCA.complement l.61)", "difference": "none", "ledger": "-"},
    {"step": "Scale / norm", "paper": "none (no norm matching)", "causalab": "steer.py scale=1.0 default",
     "ours": "headline unmatched; bake-off matched to the spline's norm (bakeoff.match_norm l.125)",
     "difference": "bake-off only", "ledger": "known"},
    {"step": "Linear baseline",
     "paper": "A.6: raw full-space centroids, whole activation replaced; A.9: straight chord between c_a, c_b in "
              "PCA-64", "causalab": "path_mode.py:resolve_path_modes 'linear' (raw) / 'linear_subspace' (pca)",
     "ours": "linear (smoothed-knot chord, legacy), linear_raw (raw-centroid chord in PCA-64, additive), "
             "goodfire_linear (full replace, off by default)",
     "difference": "legacy line used smoothed knots", "ledger": "known (#173, #182-#184)"},
    {"step": "Readout", "paper": "forward pass to output distribution, 16 prompts per pair (A.6)",
     "causalab": "collect_grid_distributions", "ours": "pooled-vector probes/MLP/nearest-real at the edited layer",
     "difference": "no forward pass in 4.3", "ledger": "known (#16, #19)"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path")
    ap.add_argument("--out", default=str(RES / "p2_endpoint_diagnosis.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_p2_endpoint_diagnosis.png"))
    a = ap.parse_args()
    cfg = dict(x.split("=", 1) for x in a.inputs)
    res = {"what": "Why the smoothing spline loses the held-out endpoint to the raw-centroid chord (REPORT 4.3): "
                   "aim points, edit rules, smoothing, norm, knot order and readers, on the headline arc (seed 0) "
                   "and the 16 contiguous 45-degree arcs (seeds 1-16).",
           "script": "scripts/run_endpoint_diagnosis.py (per arc) + scripts/summarize_endpoint_diagnosis.py",
           "conventions": {
               "gap": "arm minus add:chord_raw, paired per clip, averaged over the 384 steered (clip, target) pairs "
                      "of an arc; + = arm worse on probe/mlp error; for R + = arm better",
               "add": "x + lift(P(target) - P(source)), residual kept (our main arms)",
               "replace": "PCA-64 part of x replaced by P(target), residual kept (Goodfire A.6 manifold rule)",
               "replace_full": "whole activation replaced by the full-space raw-centroid chord point (Goodfire A.6 "
                               "linear rule)",
               "add_norm:A@B": "arm A's additive edit rescaled per clip to arm B's norm",
               "hybrid": "target point from one rule, source anchor from another",
               "oracle": "true held-out centroid from the knot-fold clips at the held-out value (excluded from "
                         "every fit); raw centroid at kept values",
               "causalab_*": "refs/causalab SplineManifold/CubicSpline1D (periodic, period 2 pi given explicitly) "
                             "run on our knots and coordinates; lam = Reinsch smoothness; lam_cv = chosen by "
                             "leave-8-knot-block-out on kept knots only",
               "splrep_s<f>": "our SmoothSpline with s = f * m (f = 1 is the stored arm)",
               "readers": "probe = ridge on probe folds (the REPORT's endpoint reader); mlp = one-hidden-layer MLP "
                          "on probe folds (seed 0); R = nearest-real agreement against test clips at the target",
               "bulge_beta": "<mu - chord, spline - chord>/|spline - chord|^2 at each held-out target, averaged "
                             "over the 8 targets: 1 = true centroid where the spline puts it, 0 = on the raw chord"},
           "configs": {}}
    for name, path in cfg.items():
        runs = json.loads(Path(path).read_text())
        res["configs"][name] = {"angle": runs[0]["angle"], "aim": runs[0]["aim"], "layer": runs[0]["layer"],
                                **summarise(runs, check_stored=(name in ("L22_labels", "L12_unsup")))}
    res["decomposition"] = decomposition(res)
    res["foreign_knot_split"] = foreign_split(res)
    res["paper_vs_ours_steps"] = STEPS
    Path(a.out).write_text(json.dumps(res, indent=1))

    # figure: per-arc probe gap vs raw chord for the key spline variants, points 12 and 22
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    variants = [("add:spline_smooth", "ours: FITPACK smoothing spline (stored rule)", "#c0392b"),
                ("add:spline_interp", "interpolating (A.3 / causalab)", "#2471a3"),
                ("add:causalab_lam_cv", "causalab Reinsch, lambda by knot CV", "#27ae60"),
                ("replace:causalab_interp", "interp, replace edit (A.6) vs additive chord", "#8e44ad")]
    for ax, name in zip(axes[:2], ("L12_labels", "L22_labels")):
        if name not in res["configs"]:
            continue
        pa = res["configs"][name]["per_arc_gap_probe"]
        for j, (arm, lab, col) in enumerate(variants):
            seeds = sorted(int(s) for s in pa[arm])
            y = [pa[arm][s] for s in seeds]
            ax.scatter(np.array(seeds) + (j - 1.5) * 0.15, y, s=18, color=col, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"point {res['configs'][name]['layer']} (labels angle): spline - raw chord, probe")
        ax.set_xlabel("arc seed (0 = headline)")
        ax.set_ylabel("endpoint error gap (deg); + = chord better")
    axes[1].legend(fontsize=7, loc="upper right")
    # aim-point geometry panel
    ax = axes[2]
    names = [("spline_smooth", "smooth spline"), ("chord_smooth", "smoothed-knot chord"), ("chord_raw", "raw chord"),
             ("spline_interp", "interp spline"), ("causalab_lam_cv", "Reinsch lam_cv"), ("oracle", "true centroid")]
    for k, (name, mk) in enumerate((("L12_labels", "o"), ("L22_labels", "s"))):
        if name not in res["configs"]:
            continue
        ag = res["configs"][name]["arcs16"]["aim_geometry"]
        for j, (arm, lab) in enumerate(names):
            ax.scatter(ag[arm]["dist_to_true_centroid"]["mean"], ag[arm]["probe_err_of_aim_point"]["mean"],
                       marker=mk, s=40, color=plt.cm.tab10(j), label=lab if k == 0 else None)
    ax.set_xlabel("aim point distance to true held-out centroid (PCA units)")
    ax.set_ylabel("probe error of the aim point itself (deg)")
    ax.set_title("aim points, 16-arc mean (o = pt 12, s = pt 22)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=130)
    print("wrote", a.out, a.fig)


if __name__ == "__main__":
    main()

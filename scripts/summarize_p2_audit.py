"""Three audit checks of undisclosed Part 2 choices (against Goodfire A.3), summarised from run_part2 outputs with the
frozen verdict rule and the existing metrics only. Runs (made beforehand with run_part2.py; see each output's
"commands"):

  1. angle source at point 22: which angle each of the 32 arc runs used; point-22 unsupervised-angle arcs rerun with
     --labels-angle into results/arcs/L22_s{s}_labels/          -> results/p2_angle_source_audit.json
  2. interpolating spline (Goodfire A.3) on the labels angle (monotone knot order): contiguous design at points 12 / 22
     into results/p2_interp_labels/, arc seeds into results/arcs_interp/L{L}_s{s}/, plus the train-only held-out
     reconstruction (geometry_checks.heldout_reconstruction) on the labels angle -> results/p2_interp_labels_summary.json
  3. scalar extrapolation with the spline continued linearly along its end tangent (--extend linear,
     mf.LinearExtension) into results/p2_linear_ext/            -> results/p2_extrapolation_linear_ext.json

  PYTHONPATH=src python scripts/summarize_p2_audit.py [--item 1|2|3|all]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wm import geometry_checks as gc
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

RES = PROJECT_ROOT / "results"
DIR = "p2_steer_direction_direction_L{L}_contiguous.json"
GAP_NOTE = ("gaps are manifold (spline) minus linear (chord), same clip and target: 'mean' with 95% paired clip "
            "bootstrap CI (as REPORT 4.3); endpoint gap + = spline worse, min-radius gap + = spline better")


def run_row(r):
    """One run_part2 output -> the numbers these checks compare (all stored fields; no new metric)."""
    g = r["gaps"]["manifold_minus_linear"]
    s = r["summary"]
    return {"angle_source": r["angle_source"], "spline": r["spline"],
            "curve_extension": r.get("curve_extension", "cubic"),
            "held_out": [r["holdout"].get("block_first_value", r["held_out_values"][0]),
                         r["holdout"].get("block_last_value", r["held_out_values"][-1])],
            "endpoint_gap": g["probe_err_to_target"]["mean"], "endpoint_gap_ci95": g["probe_err_to_target"]["ci95"],
            "min_radius_gap": g["probe_radius_min"]["mean"] if "probe_radius_min" in g else None,
            "min_radius_gap_ci95": g["probe_radius_min"]["ci95"] if "probe_radius_min" in g else None,
            "endpoint_err_spline": s["manifold"]["overall"]["probe_err_to_target"],
            "endpoint_err_chord": s["linear"]["overall"]["probe_err_to_target"],
            "delta_norm_spline": s["manifold"]["overall"]["delta_norm"],
            "delta_norm_chord": s["linear"]["overall"]["delta_norm"],
            "sagitta_over_noise_median": r["verdict"]["sagitta_over_noise_median"],
            "verdict": r["verdict"]["call"]}


def load(p):
    return run_row(json.loads(Path(p).read_text()))


def agg(rows):
    e = np.array([r["endpoint_gap"] for r in rows])
    m = np.array([r["min_radius_gap"] for r in rows])
    calls = [r["verdict"] for r in rows]
    return {"n_runs": len(rows), "endpoint_gap_mean": float(e.mean()), "endpoint_gap_sd": float(e.std(ddof=1)),
            "min_radius_gap_mean": float(m.mean()), "min_radius_gap_sd": float(m.std(ddof=1)),
            "min_radius_gap_range": [float(m.min()), float(m.max())],
            "verdict_counts": {c: calls.count(c) for c in sorted(set(calls))}}


def item1():
    arcs = {L: {s: load(RES / "arcs" / f"L{L}_s{s}" / DIR.format(L=L)) for s in range(1, 17)} for L in (12, 22)}
    head = {L: json.loads((RES / DIR.format(L=L)).read_text()) for L in (12, 22)}
    unsup22 = [s for s, r in arcs[22].items() if r["angle_source"] == "unsupervised_angle"]
    reruns = {s: load(p) for s in unsup22 if (p := RES / "arcs" / f"L22_s{s}_labels" / DIR.format(L=22)).exists()}
    consistent = [reruns.get(s, r) for s, r in arcs[22].items()]
    return {"provenance": provenance(None, seeds={"arcs": "1-16", "reruns": sorted(reruns)}), "gap_note": GAP_NOTE,
            "commands": "run_part2.py --dataset direction --layer 22 --holdout contiguous --spline smooth --seed S "
                        "--labels-angle --results-dir results/arcs/L22_sS_labels (stored arcs: same without "
                        "--labels-angle; parity of the stored L22_s4 rerun at HEAD: gains equal to ~1e-12)",
            "headline_angle_source": {str(L): head[L]["angle_source"] for L in (12, 22)},
            "headline_L22_angle_check": head[22].get("angle_check"),
            "arcs": {str(L): {str(s): r for s, r in arcs[L].items()} for L in (12, 22)},
            "angle_source_counts": {str(L): {a: sum(r["angle_source"] == a for r in arcs[L].values())
                                             for a in ("unsupervised_angle", "labels_angle")} for L in (12, 22)},
            "L22_unsupervised_seeds": unsup22,
            "L22_labels_reruns": {str(s): {"original": arcs[22][s], "labels": r} for s, r in reruns.items()},
            "L22_aggregate_stored": agg(list(arcs[22].values())),
            "L22_aggregate_all_labels": agg(consistent),
            "L22_aggregate_note": "all_labels = stored arcs with every unsupervised-angle seed replaced by its "
                                  "--labels-angle rerun; seeds 4 and 8 are the same arc (both kept, as in REPORT)"}


def recon_labels(L):
    d = load_inputs("direction", L, "direction")
    tr = d["is_train"]
    return gc.heldout_reconstruction(d["X"][tr], d["y"][tr], True, 64, angle="labels")


def item2(seeds):
    out = {"provenance": provenance(None, seeds={"arcs": seeds}), "gap_note": GAP_NOTE,
           "commands": "run_part2.py --dataset direction --layer L --holdout contiguous --spline interp --labels-angle "
                       "[--goodfire-baseline for the seed-0 run, as stored] [--seed S]",
           "edit_size_note": ("run_part2 does not emit a delta-over-natural-centroid-change number (that is "
                              "run_session2); delta_norm = mean full-space ||x_steered - x|| at the path end, so "
                              "delta_norm_spline / delta_norm_chord is the edit size relative to the chord's, whose "
                              "endpoint is the kept-centroid chord point"),
           "reconstruction_note": ("geometry_checks.heldout_reconstruction on train folds (as run_geometry_checks): "
                                   "mean PCA-unit error rebuilding held-out centroids, contiguous = mean over arc "
                                   "seeds 0-3; stored = unsupervised atan2 knot order (p2_geometry_direction_L*.json)"),
           "points": {}}
    for L in (12, 22):
        geo = json.loads((RES / f"p2_geometry_direction_L{L}.json").read_text())
        stored = load(RES / DIR.format(L=L))
        new = load(RES / "p2_interp_labels" / DIR.format(L=L))
        for r in (stored, new):
            r["delta_ratio_spline_over_chord"] = r["delta_norm_spline"] / r["delta_norm_chord"]
        sa = [load(RES / "arcs" / f"L{L}_s{s}" / DIR.format(L=L)) for s in seeds]
        na = [load(p) for s in seeds if (p := RES / "arcs_interp" / f"L{L}_s{s}" / DIR.format(L=L)).exists()]
        out["points"][str(L)] = {
            "heldout_reconstruction": {"stored_unsupervised_order": geo["heldout_reconstruction"],
                                       "labels_order": recon_labels(L)},
            "seed0_stored_smoothing": stored, "seed0_interp_labels": new,
            "arcs_stored_smoothing_same_seeds": agg(sa), "arcs_interp_labels": agg(na) if len(na) > 1 else None,
            "arcs_interp_labels_rows": {str(s): r for s, r in zip(seeds, na)}}
    return out


def item3():
    runs = [("speed", 12), ("speed", 19), ("acceleration", 12), ("acceleration", 21)]
    out = {"provenance": provenance(None, seeds={"seed": 0}), "commands":
           "run_part2.py --dataset V --layer L --holdout extrapolation --spline smooth --extend linear "
           "--results-dir results/p2_linear_ext",
           "extension_rule": ("explicit linear continuation along the end tangent (mf.LinearExtension): "
                              "f(t_end) + (t - t_end) f'(t_end) past the last knot; not splev ext=3, which clamps "
                              "to the end value. Stored runs: splev ext=0, the end cubic piece extended"),
           "bend_note": "sagitta_over_noise_median = verdict's median over targets of spline-vs-chord target distance "
                        "/ centroid noise", "runs": {}}
    for v, L in runs:
        f = f"p2_steer_{v}_{v}_L{L}_extrapolation.json"
        out["runs"][f"{v}_L{L}"] = {"stored_cubic_ext": load(RES / f), "linear_ext": load(RES / "p2_linear_ext" / f)}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--item", default="all", choices=("1", "2", "3", "all"))
    p.add_argument("--arc-seeds", default="1-8", help="seed range of the interpolating-spline arc sweep")
    a = p.parse_args(argv)
    lo, hi = map(int, a.arc_seeds.split("-"))
    todo = {"1": (item1, "p2_angle_source_audit.json"),
            "2": (lambda: item2(list(range(lo, hi + 1))), "p2_interp_labels_summary.json"),
            "3": (item3, "p2_extrapolation_linear_ext.json")}
    for k, (fn, name) in todo.items():
        if a.item in (k, "all"):
            (RES / name).write_text(json.dumps(fn(), indent=1))
            print("wrote", RES / name)


if __name__ == "__main__":
    main()

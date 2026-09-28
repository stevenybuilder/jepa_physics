"""Aggregate scripts/run_bakeoff_unified_16arc.py raw outputs into results/p2_bakeoff_unified_16arc.json
(`table`: arm -> metric -> point -> {mean, ci95}; `paired_vs_chord` likewise)."""
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "endpoint_diagnosis_raw"
_s = importlib.util.spec_from_file_location("agg", ROOT / "scripts" / "aggregate_endpoint_diagnosis_path.py")
agg = importlib.util.module_from_spec(_s)
_s.loader.exec_module(agg)
BASE = ("chord_raw", "interp_causalab_lam0", "causalab_lam_cv", "fitpack_smooth", "fourier2_4d", "probe_qr")
ARMS = BASE + tuple(f"{a}@chord_norm" for a in BASE[1:])
MET = ("err_end", "R_end_test", "radius_min", "radius_mean", "monotone_frac", "ordering_spearman", "R_wp_mean",
       "delta_norm_end")


def main():
    rng = np.random.default_rng(0)
    table, paired, head, inputs, extra = {a: {} for a in ARMS}, {}, {}, {}, {}
    for L in (12, 22):
        arcs = []
        for h in ("a", "b"):
            p = RAW / f"unified_L{L}_{h}.json"
            arcs += json.loads(p.read_text())
            inputs[str(p.relative_to(ROOT))] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
        arcs = sorted(arcs, key=lambda r: r["seed"])
        assert [r["seed"] for r in arcs] == list(range(16)), [r["seed"] for r in arcs]
        g = [np.asarray(r["clip"]) for r in arcs]
        for a in ARMS:
            for q in MET:
                table[a].setdefault(q, {})[str(L)] = agg.boot_mean([np.asarray(r["arms"][a][q], float) for r in arcs],
                                                                   g, rng)
        for a in ARMS[1:]:
            pa = paired.setdefault(f"{a} - chord_raw", {})
            for q in MET[:-1]:
                pa.setdefault(q, {})[str(L)] = agg.boot_mean([np.asarray(r["arms"][a][q], float)
                                                              - np.asarray(r["arms"]["chord_raw"][q], float)
                                                              for r in arcs], g, rng)
        h0 = arcs[0]
        head[str(L)] = {a: {q: agg.boot_mean([np.asarray(h0["arms"][a][q], float)], g[:1], rng) for q in MET}
                        for a in ARMS}
        extra[str(L)] = {"lambda_cv_per_arc": [r["lambda_cv"] for r in arcs],
                         "probe_qr_n_probes_per_arc": [r["probe_basis"].get("n_probes") for r in arcs],
                         "radius_wp_mean_over_arcs": {a: np.mean([r["arms"][a]["radius_wp_mean"] for r in arcs], 0)
                                                      .tolist() for a in ARMS}}
    sp = ROOT / "scripts" / "run_bakeoff_unified_16arc.py"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    out = {"keys": {
        "points": "12 = layer 12 (unsupervised angle), 22 = layer 22 (label angle); 16 contiguous held-out 45-deg arcs "
                  "(seed = arc), 8 held-out targets x 48 picked test clips per arc (p2.pick_clips)",
        "arms": {"chord_raw": "raw-centroid straight edit (run_part2 linear_raw)",
                 "interp_causalab_lam0": "the paper's interpolating periodic cubic (causalab code, smoothness 0) on our "
                                         "value-ordered knots",
                 "causalab_lam_cv": "knot-CV Reinsch smoother (EXPLORATORY: lambda chosen by leave-block-out on kept knots)",
                 "fitpack_smooth": "our stored FITPACK count-weighted smoothing spline",
                 "fourier2_4d": "straight edit in the fitted 4-D Fourier encoding Z ~ a + B[cos t, sin t, cos 2t, sin 2t] "
                                "(OLS on knot-fold rows at kept values in PCA-64; run_motion_geometry exp3 fourier2_4d)",
                 "probe_qr": "Part 1 multi-probe steer (wm.bakeoff.probe_qr_delta), basis refit per arc on knot rows at "
                             "kept values (wm.bakeoff.refit_probe_basis); computed here, no stored 16-arc per-arc "
                             "probe-QR numbers existed (p2_bakeoff_*_rawchord.json are the headline arc only)",
                 "<arm>@chord_norm": "full-space delta rescaled per clip and per waypoint to the raw chord's ||delta||"},
        "walk": "spline arms: mf.manifold_coords shift walk (K = 11) on the stored curve's coordinates; straight arms: "
                "x + s * delta, s = linspace(0, 1, 11)",
        "err_end": "ridge probe (probe folds) endpoint error to target, deg",
        "R_end_test": "nearest-real R at the endpoint vs test-clip mean at the target (run_part2.evaluate rule)",
        "radius_min/radius_mean": "min / mean over the 11 waypoints of |probe (sin, cos)| (run_part2.evaluate wp_radius)",
        "monotone_frac": "fraction of the 10 steps in which the probe's decoded angle advances along the traversed arc",
        "R_wp_mean": "mean over waypoints of nearest-real R vs the probe-fold centroid at the waypoint's intended angle",
        "ci95": "95% clip-bootstrap (clips resampled within each arc, all targets of a clip together, 1000 draws); "
                "statistic = mean over 16 arcs of per-arc means",
        "table": "arm -> metric -> point -> {mean, ci95, n_arcs}", "paired_vs_chord": "arm - chord_raw, same format",
        "headline_arc": "layer 12 / 22 seed 0 only, clip bootstrap"},
        "table": table, "paired_vs_chord": paired, "headline_arc": head, "extra": extra,
        "provenance": {"commit": commit, "dirty": dirty, "script_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                       "path_stage_sha256": hashlib.sha256((ROOT / "scripts" / "run_endpoint_diagnosis_path.py")
                                                           .read_bytes()).hexdigest(),
                       "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "raw_inputs_mtime_utc": inputs, "timestamp_utc": datetime.now(timezone.utc).isoformat()}}
    (ROOT / "results" / "p2_bakeoff_unified_16arc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

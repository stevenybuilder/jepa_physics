"""Aggregate scripts/run_endpoint_diagnosis_path.py raw outputs into results/p2_endpoint_diagnosis_path.json, compact
the raw per-arc outputs (per-clip scalars + per-waypoint means; K=11 per-clip radius), draw
figures/fig_p2_path_by_spline.png.

  python scripts/aggregate_endpoint_diagnosis_path.py
"""
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "endpoint_diagnosis_raw"
ARMS = ("chord_raw", "fitpack_smooth", "interp_causalab_lam0", "causalab_lam_cv")
NORMED = tuple(f"{a}@chord_norm" for a in ARMS[1:])
SCAL = ("radius_min", "radius_mean", "monotone_frac", "ordering_spearman", "R_end_test", "R_wp_mean", "err_end")
WPS = ("radius", "R_wp", "progress", "delta_norm_pca")
N_BOOT = 1000


def load(L):
    arcs = []
    for part in ("a", "b"):
        p = RAW / f"path_L{L}_{part}.json.gz"
        arcs += json.loads(gzip.decompress(p.read_bytes()))
    return sorted(arcs, key=lambda r: r["seed"])


def compact(arcs, L):
    out = []
    for r in arcs:
        c = {k: r[k] for k in r if k != "K"}
        c["K"] = {}
        for K, arms in r["K"].items():
            c["K"][K] = {}
            for a, dd in arms.items():
                e = {q: np.round(np.asarray(dd[q], float), 4).tolist() for q in SCAL}
                e["delta_norm_end"] = np.round(np.asarray(dd["delta_norm_pca"])[:, -1], 4).tolist()
                e["wp_mean"] = {q: np.round(np.asarray(dd[q], float).mean(0), 4).tolist() for q in WPS}
                if K == "11":
                    e["radius_per_clip_wp"] = np.round(np.asarray(dd["radius"], float), 3).tolist()
                c["K"][K][a] = e
        out.append(c)
    return out


def boot_mean(vals, groups, rng):
    """vals: list over arcs of per-row arrays; groups: per-arc clip ids. Mean over arcs of per-arc means, 95% CI from
    resampling clips (all of a clip's rows together) within every arc."""
    point = float(np.mean([v.mean() for v in vals]))
    boots = np.zeros(N_BOOT)
    for v, g in zip(vals, groups):
        u, inv = np.unique(g, return_inverse=True)
        s, n = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
        idx = rng.integers(0, len(u), (N_BOOT, len(u)))
        boots += s[idx].sum(1) / n[idx].sum(1)
    boots /= len(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": point, "ci95": [float(lo), float(hi)], "n_arcs": len(vals)}


def main():
    res = {"keys": {
        "arms": {"chord_raw": "raw-centroid chord, run_part2.subspace_arms linear_raw rule (mf.linear_coords)",
                 "fitpack_smooth": "stored FITPACK count-weighted smoothing spline (p2.build 'smooth'), "
                                   "mf.manifold_coords shift walk",
                 "interp_causalab_lam0": "causalab SplineManifold periodic cubic smoothness 0 on the raw kept "
                                         "centroids at the stored coordinates (the endpoint arm 'causalab_interp' / "
                                         "interp_causalab_lam0 of p2_endpoint_diagnosis.json)",
                 "causalab_lam_cv": "causalab Reinsch smoother, lambda by leave-block-out on kept knots (exploratory)",
                 "<arm>@chord_norm": "that arm's PCA delta rescaled per clip and per waypoint to the chord's ||delta||"},
        "walk": "t_k = ta + s_k * curve.step(ta, tb), s = linspace(0,1,K), z_k = z + f(t_k) - f(ta) "
                "(src/wm/manifold.py manifold_coords, mode='shift'); residual kept (run_part2.compose); all spline arms "
                "use the stored curve's coordinates and step, so they traverse the same arc",
        "radius": "|ProbeReadout.raw(W)|, norm of the ridge probe's (sin, cos) output (scripts/run_part2.py evaluate, "
                  "wp_radius); probe fit on probe folds",
        "radius_min": "per clip: min radius over the K waypoints; aggregated mean over clips per arc, then mean over arcs",
        "radius_mean": "per clip: mean radius over the K waypoints (waypoint 0 = the unsteered clip included)",
        "monotone_frac": "per clip: fraction of the K-1 consecutive steps in which the probe's decoded angle advances "
                         "along the traversed arc (signed, unwrapped)",
        "ordering_spearman": "per clip: Spearman(waypoint index, signed decoded progress)",
        "R_end_test": "nearest-real R at the endpoint vs the mean of TEST clips at the target (run_part2.evaluate's "
                      "nearest_real_R; comparable to stored nearest_real_R)",
        "R_wp_mean": "per clip: mean over waypoints of nearest-real R vs the PROBE-FOLD full-space centroid at the "
                     "waypoint's intended angle (src + s_k * traversed shift, snapped to nearest probe-fold value)",
        "err_end": "probe endpoint error (deg) to target",
        "ci95": "95% clip-bootstrap: clips resampled (all of a clip's targets together) within each arc, "
                "1000 draws, statistic = mean over the 16 arcs of per-arc means",
        "paired_diff": "arm - chord_raw computed per row, same bootstrap",
        "points": "12 = layer 12 (unsupervised angle), 22 = layer 22 (label angle), as run_endpoint_diagnosis 'auto'",
        "headline_arc": "layer 12, seed 0 (held-out 303.75..343.125), the arc of p2_conceptor_direction_L12.json"}}
    rng = np.random.default_rng(0)
    agg, head, fig = {}, {}, {}
    inputs = {}
    for L in (12, 22):
        arcs = load(L)
        for part in ("a", "b"):
            p = RAW / f"path_L{L}_{part}.json.gz"
            inputs[str(p.relative_to(ROOT))] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
        (RAW / f"path_L{L}_compact.json").write_text(json.dumps(compact(arcs, L)))
        agg[str(L)], head[str(L)], fig[str(L)] = {}, {}, {}
        for K in ("11", "50"):
            A = {}
            groups = [np.asarray(r["clip"]) for r in arcs]
            for a in ARMS + NORMED:
                A[a] = {q: boot_mean([np.asarray(r["K"][K][a][q], float) for r in arcs], groups, rng) for q in SCAL}
                A[a]["delta_norm_end"] = boot_mean([np.asarray(r["K"][K][a]["delta_norm_pca"], float)[:, -1]
                                                    for r in arcs], groups, rng)
            D = {}
            for a in ARMS[1:] + NORMED:
                D[f"{a} - chord_raw"] = {q: boot_mean([np.asarray(r["K"][K][a][q], float)
                                                       - np.asarray(r["K"][K]["chord_raw"][q], float) for r in arcs],
                                                      groups, rng) for q in SCAL}
            agg[str(L)][K] = {"per_arm": A, "paired_vs_chord": D, "lambda_cv_per_arc": [r["lambda_cv"] for r in arcs],
                              "n_rows_per_arc": [len(r["clip"]) for r in arcs]}
            h = arcs[0]
            assert h["seed"] == 0
            g = [np.asarray(h["clip"])]
            head[str(L)][K] = {
                "per_arm": {a: {q: boot_mean([np.asarray(h["K"][K][a][q], float)], g, rng) for q in SCAL}
                            for a in ARMS + NORMED},
                "paired_vs_chord": {f"{a} - chord_raw": {q: boot_mean([np.asarray(h["K"][K][a][q], float)
                                                                       - np.asarray(h["K"][K]["chord_raw"][q], float)],
                                                                      g, rng) for q in SCAL} for a in ARMS[1:] + NORMED}}
            if K == "11":
                fig[str(L)] = {a: np.mean([np.asarray(r["K"][K][a]["radius"], float).mean(0) for r in arcs], 0)
                               for a in ARMS + NORMED}
    stored = json.loads((ROOT / "results" / "p2_conceptor_direction_L12.json").read_text())["arms"]
    res["headline_reproduction"] = {
        "stored_file": "results/p2_conceptor_direction_L12.json arms.{manifold,linear_raw} (K=50, 48 clips/target, "
                       "layer 12 seed 0)",
        "stored": {"fitpack_smooth": {"probe_radius_min": stored["manifold"]["probe_radius_min"],
                                      "nearest_real_R": stored["manifold"]["nearest_real_R"]},
                   "chord_raw": {"probe_radius_min": stored["linear_raw"]["probe_radius_min"],
                                 "nearest_real_R": stored["linear_raw"]["nearest_real_R"]}},
        "recomputed_K50": {a: {"probe_radius_min": head["12"]["50"]["per_arm"][a]["radius_min"]["mean"],
                               "nearest_real_R": head["12"]["50"]["per_arm"][a]["R_end_test"]["mean"]}
                           for a in ("fitpack_smooth", "chord_raw")},
        "recomputed_K11": {a: {"probe_radius_min": head["12"]["11"]["per_arm"][a]["radius_min"]["mean"],
                               "nearest_real_R": head["12"]["11"]["per_arm"][a]["R_end_test"]["mean"]}
                           for a in ("fitpack_smooth", "chord_raw")}}
    res["aggregate_16arc"] = agg
    res["headline_arc"] = head
    res["radius_along_path_K11_mean_over_arcs"] = {L: {a: v.tolist() for a, v in d.items()} for L, d in fig.items()}
    sp = ROOT / "scripts" / "run_endpoint_diagnosis_path.py"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)
    res["provenance"] = {"commit": commit, "dirty": dirty,
                         "script": str(sp.relative_to(ROOT)), "script_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                         "endpoint_script_sha256": hashlib.sha256((ROOT / "scripts" / "run_endpoint_diagnosis.py")
                                                                  .read_bytes()).hexdigest(),
                         "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "raw_inputs_mtime_utc": inputs,
                         "compact_raw": [f"results/endpoint_diagnosis_raw/path_L{L}_compact.json" for L in (12, 22)],
                         "data_inputs": "wm.p2_data.load_inputs('direction', L, 'direction'); split via p2.build",
                         "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(range(16)),
                         "n_clips_per_target": 48, "K": [11, 50]}
    (ROOT / "results" / "p2_endpoint_diagnosis_path.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"chord_raw": "#666666", "fitpack_smooth": "#1f77b4", "interp_causalab_lam0": "#d62728",
           "causalab_lam_cv": "#2ca02c"}
    f, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    s = np.linspace(0, 1, 11)
    for i, L in enumerate(("12", "22")):
        for a in ARMS:
            ax[i].plot(s, fig[L][a], "-o", ms=3, color=col[a], label=a)
            if a != "chord_raw":
                ax[i].plot(s, fig[L][f"{a}@chord_norm"], "--", color=col[a], lw=1, alpha=0.7)
        ax[i].set_title(f"point {L}: probe radius along the path (16 arcs)")
        ax[i].set_xlabel("waypoint fraction source -> target")
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("probe (sin, cos) radius")
    ax[0].legend(fontsize=8, title="solid: own norm; dashed: chord's norm", title_fontsize=7)
    f.tight_layout()
    f.savefig(ROOT / "figures" / "fig_p2_path_by_spline.png", dpi=150)


if __name__ == "__main__":
    main()

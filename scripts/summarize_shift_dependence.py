"""Shift dependence of the spline-minus-chord gaps, pooled over the stored contiguous-arc direction runs (no new
compute: reads results/p2_steer_direction_direction_L{L}_contiguous.json and results/arcs/L{L}_s{1..16}/).

PART2_RATIONALE.md section 4 pre-registered that the spline advantage is ~0 below ~45 degrees of angular shift and
grows toward 180. run_part2 stores the paired gaps (manifold - linear, same clip and target) by shift bin in
gaps.manifold_minus_linear.<metric>.by_shift; this script collects them per run and reports, per shift bin, the mean
and SD across runs and the number of runs whose 95% CI excludes 0 in each direction. Seeds 4 and 8 drew the same arc;
both are kept (as in REPORT section 4.3) and the arc is listed.

  python scripts/summarize_shift_dependence.py
"""
import json
from pathlib import Path

import numpy as np

from wm.data import PROJECT_ROOT
from wm.provenance import provenance

METRICS = ("probe_radius_min", "probe_err_to_target", "nearest_real_R", "excess_to_nearest_real", "excess_to_curve",
           "behaviour_energy_rel_floor", "intermediate_mass", "ordering_spearman")
SIGN = {"probe_radius_min": "+ = spline higher (better)", "probe_err_to_target": "- = spline better",
        "nearest_real_R": "+ = spline better", "excess_to_nearest_real": "- = spline closer to real clips (full space, 5-NN)",
        "excess_to_curve": "- = spline closer to the probe-fold reference curve",
        "behaviour_energy_rel_floor": "- = spline better (Eq. 9, circular at the steered layer)",
        "intermediate_mass": "+ = spline better (Eq. 9, circular)", "ordering_spearman": "+ = spline better (Eq. 9)"}


def runs(layer):
    res = PROJECT_ROOT / "results"
    out = [("s0", res / f"p2_steer_direction_direction_L{layer}_contiguous.json")]
    out += [(f"s{s}", res / "arcs" / f"L{layer}_s{s}" / f"p2_steer_direction_direction_L{layer}_contiguous.json")
            for s in range(1, 17)]
    return [(n, p) for n, p in out if p.exists()]


def main():
    result = {"provenance": provenance(None, seeds={"seeds": "0-16 as stored"}), "sign_convention": SIGN,
              "gap": "manifold (smoothing spline, additive) minus linear (chord, same PCA-64 subspace, residual kept)",
              "layers": {}}
    for layer in (12, 22):
        rs = runs(layer)
        per = {}
        arcs = {}
        for name, path in rs:
            d = json.loads(path.read_text())
            arcs[name] = [d["holdout"]["block_first_value"], d["holdout"]["block_last_value"]]
            g = d["gaps"]["manifold_minus_linear"]
            for q in METRICS:
                if q not in g:
                    continue
                for b in g[q]["by_shift"]:
                    key = f"{b['shift_lo']:.0f}-{b['shift_hi']:.0f}"
                    per.setdefault(q, {}).setdefault(key, []).append((b["mean"], b["ci95"][0], b["ci95"][1]))
                per.setdefault(q, {}).setdefault("overall", []).append((g[q]["mean"], g[q]["ci95"][0], g[q]["ci95"][1]))
        table = {}
        for q, bins in per.items():
            table[q] = {}
            for key, v in bins.items():
                v = np.array(v)
                table[q][key] = {"mean_over_runs": float(v[:, 0].mean()), "sd_over_runs": float(v[:, 0].std(ddof=1)),
                                 "n_runs": len(v), "n_ci_above_0": int((v[:, 1] > 0).sum()),
                                 "n_ci_below_0": int((v[:, 2] < 0).sum())}
        result["layers"][str(layer)] = {"runs": [n for n, _ in rs], "arcs_deg": arcs, "by_shift": table}
    out = PROJECT_ROOT / "results" / "p2_shift_dependence.json"
    out.write_text(json.dumps(result, indent=1))
    for layer, v in result["layers"].items():
        print(f"point {layer}: {len(v['runs'])} runs")
        for q in ("probe_radius_min", "probe_err_to_target", "nearest_real_R", "excess_to_nearest_real"):
            row = "  ".join(f"{k}: {s['mean_over_runs']:+.3f}±{s['sd_over_runs']:.3f} ({s['n_ci_above_0']}+/{s['n_ci_below_0']}-)"
                            for k, s in v["by_shift"][q].items())
            print(f"  {q}: {row}")


if __name__ == "__main__":
    main()

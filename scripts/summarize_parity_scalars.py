"""Speed / acceleration parity with the direction arc sweep (REPORT section 4.3), no new compute: aggregates the
16 contiguous held-out blocks per cell (results/arcs_{var}/L{pt}_s{1..16}/, run_part2.py --holdout contiguous
--spline smooth --goodfire-baseline --seed s), the matched-norm bake-offs, and the shift dependence of the gaps.

Per cell and arm pair (spline - smoothed-knot chord = gaps.manifold_minus_linear, spline - raw-centroid chord =
gaps.manifold_minus_linear_raw) and metric: mean +/- SD over blocks of the per-clip `.mean` field (the field the
REPORT table uses), range, blocks where the spline is ahead, blocks whose clip-bootstrap ci95 excludes 0.
Direction's 16 arcs (results/arcs_rawchord/, point 22 on the all-labels set) are aggregated the same way beside them.

  python scripts/summarize_parity_scalars.py
writes results/p2_parity_scalars.json, results/p2_shift_dependence_scalars.json, figures/fig_parity_scalars.png
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm.data import PROJECT_ROOT
from wm.provenance import provenance

CELLS = (("speed", 12), ("speed", 19), ("acceleration", 12), ("acceleration", 21))
DIRECTION_CELLS = (("direction", 12), ("direction", 22))
LABELS22_SEEDS = (4, 8, 9, 10)
PAIRS = {"spline_minus_chord": "manifold_minus_linear", "spline_minus_raw_chord": "manifold_minus_linear_raw"}
# metric -> +1 if higher is better for the spline (gap > 0 = spline ahead), -1 if lower is better
BETTER = {"probe_err_to_target": -1, "nearest_real_R": +1, "probe_err_path": -1, "probe_radius_min": +1,
          "ordering_spearman": +1, "intermediate_mass": +1, "excess_to_curve": -1, "excess_to_nearest_real": -1,
          "behaviour_energy_rel_floor": -1}
ENDPOINT = ("probe_err_to_target", "nearest_real_R")
PATH = ("probe_err_path", "probe_radius_min", "ordering_spearman", "intermediate_mass", "excess_to_curve",
        "excess_to_nearest_real", "behaviour_energy_rel_floor")


def block_runs(var, pt, results_dir):
    """(seed, path) for the 16 blocks of a cell. Direction reads results/arcs_rawchord/ (point 22: seeds 4, 8, 9, 10
    from their _labels reruns, the all-labels set of REPORT section 4.3)."""
    res = Path(results_dir)
    out = []
    for s in range(1, 17):
        if var == "direction":
            sub = f"L{pt}_s{s}_labels" if pt == 22 and s in LABELS22_SEEDS else f"L{pt}_s{s}"
            p = res / "arcs_rawchord" / sub / f"p2_steer_direction_direction_L{pt}_contiguous.json"
        else:
            p = res / f"arcs_{var}" / f"L{pt}_s{s}" / f"p2_steer_{var}_{var}_L{pt}_contiguous.json"
        if p.exists():
            out.append((s, p))
    return out


def aggregate_blocks(docs):
    """docs: list of (seed, run_part2 JSON dict). Returns per arm pair and metric the across-block summary."""
    out = {"n_blocks": len(docs), "seeds": [s for s, _ in docs],
           "blocks": {str(s): [d["holdout"]["block_first_value"], d["holdout"]["block_last_value"]] for s, d in docs},
           "probe_test_error": [float(d["probe_test_error"]) for _, d in docs]}
    firsts = [tuple(v) for v in out["blocks"].values()]
    out["n_distinct_blocks"] = len(set(firsts))
    out["pairs"] = {}
    for name, key in PAIRS.items():
        res = out["pairs"][name] = {}
        for q, sign in BETTER.items():
            vals = [(d["gaps"][key][q]["mean"], *d["gaps"][key][q]["ci95"]) for _, d in docs
                    if key in d["gaps"] and isinstance(d["gaps"][key].get(q), dict)]
            if not vals:
                continue
            v = np.array(vals, float)
            m = v[:, 0]
            res[q] = {"mean": float(m.mean()), "sd": float(m.std(ddof=1)) if len(m) > 1 else None,
                      "min": float(m.min()), "max": float(m.max()), "n": int(len(m)),
                      "n_spline_ahead": int((sign * m > 0).sum()),
                      "n_ci_excl_0": int(((v[:, 1] > 0) | (v[:, 2] < 0)).sum()),
                      "n_ci_above_0": int((v[:, 1] > 0).sum()), "n_ci_below_0": int((v[:, 2] < 0).sum()),
                      "per_block": [float(x) for x in m], "per_block_ci95": v[:, 1:].tolist(),
                      "sign": "+ = spline better" if sign > 0 else "- = spline better"}
            if q in ("probe_err_to_target", "probe_err_path"):
                rel = m / np.array(out["probe_test_error"])[:len(m)]
                res[q]["mean_over_probe_test_error"] = float(rel.mean())
    out["verdicts"] = [d["verdict"]["call"] for _, d in docs]
    return out


def bakeoff_rows(var, pt, results_dir):
    out = {}
    for tag, fname in (("refit_knot_rows", f"p2_bakeoff_{var}_{var}_L{pt}_contiguous.json"),
                       ("stored_part1_basis", f"p2_bakeoff_{var}_{var}_L{pt}_contiguous_storedbasis.json")):
        p = Path(results_dir) / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[tag] = {"source": f"results/{fname}", "probe_basis": d["probe_basis"], "unsteered_err": d["unsteered_err"],
                    "arms": {a: {k: r[k] for k in ("err_probe_matched", "err_mlp_matched", "err_probe_unmatched",
                                                    "err_mlp_unmatched", "effective_rank", "mean_norm_unmatched")}
                             for a, r in d["arms"].items()}}
    return out


def shift_dependence(docs, n_bins=5, n_boot=1000):
    """Gap (manifold - linear and manifold - linear_raw, same clip and target) vs |delta value|, with FIXED equal-width
    bins over the pooled shift range of the cell (run_part2's scalar bins are per-run quantiles, not comparable across
    blocks). Per bin: mean and SD over blocks of the per-block mean, and blocks whose clip-bootstrap CI excludes 0."""
    boot = _paired_bootstrap()
    rows = {s: d["rows"] for s, d in docs}
    shifts = np.concatenate([[r["shift"] for r in R if r["arm"] == "manifold"] for R in rows.values()])
    edges = np.linspace(0, shifts.max() + 1e-9, n_bins + 1)
    out = {"edges": edges.tolist(), "pairs": {}}
    for name, arm in (("spline_minus_chord", "linear"), ("spline_minus_raw_chord", "linear_raw")):
        per = out["pairs"][name] = {}
        for q in ("probe_err_to_target", "nearest_real_R", "probe_err_path", "excess_to_nearest_real"):
            bins = [[] for _ in range(n_bins)]
            for s, R in rows.items():
                a = {(r["id"], r["target"]): r for r in R if r["arm"] == "manifold"}
                b = {(r["id"], r["target"]): r for r in R if r["arm"] == arm}
                keys = [k for k in a if k in b and a[k][q] is not None and b[k][q] is not None]
                sh = np.array([a[k]["shift"] for k in keys])
                diff = np.array([a[k][q] - b[k][q] for k in keys])
                ids = np.array([k[0] for k in keys])
                for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                    sel = (sh >= lo) & (sh < hi)
                    if sel.sum() >= 5:
                        r = boot(diff[sel], ids[sel], n_boot=n_boot)
                        bins[i].append((r["mean"], *r["ci95"]))
            per[q] = []
            for (lo, hi), v in zip(zip(edges[:-1], edges[1:]), bins):
                if not v:
                    continue
                v = np.array(v)
                per[q].append({"shift_lo": float(lo), "shift_hi": float(hi), "n_blocks": int(len(v)),
                               "mean_over_blocks": float(v[:, 0].mean()),
                               "sd_over_blocks": float(v[:, 0].std(ddof=1)) if len(v) > 1 else None,
                               "n_ci_above_0": int((v[:, 1] > 0).sum()), "n_ci_below_0": int((v[:, 2] < 0).sum())})
    return out


def _paired_bootstrap():
    spec = importlib.util.spec_from_file_location("run_part2", Path(__file__).with_name("run_part2.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.paired_bootstrap


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [*CELLS, *DIRECTION_CELLS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (var, pt) in zip(axes.ravel(), cells):
        c = result["cells"].get(f"{var}_L{pt}")
        if not c or not c["blocks"]["n_blocks"]:
            ax.set_axis_off()
            continue
        for j, (name, color) in enumerate((("spline_minus_chord", "C0"), ("spline_minus_raw_chord", "C3"))):
            g = c["blocks"]["pairs"][name].get("probe_err_to_target")
            if not g:
                continue
            x = np.arange(len(g["per_block"])) + (j - 0.5) * 0.25
            m, ci = np.array(g["per_block"]), np.array(g["per_block_ci95"])
            ax.errorbar(x, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o", ms=3, color=color, lw=0.8,
                        label=f"{'raw chord' if 'raw' in name else 'chord'}: {g['mean']:+.3g} ± {g['sd']:.2g}, "
                              f"ahead {g['n_spline_ahead']}/{g['n']}, CI≠0 {g['n_ci_excl_0']}/{g['n']}")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(np.arange(c["blocks"]["n_blocks"]), [str(s) for s in c["blocks"]["seeds"]], fontsize=7)
        unit = "deg" if var == "direction" else ("m/s" if var == "speed" else "m/s²")
        ax.set(title=f"{var}, point {pt}" + (" (comparison)" if var == "direction" else ""),
               xlabel="held-out block (seed)", ylabel=f"endpoint probe error gap ({unit})\n< 0: spline better")
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Spline minus chord (blue: smoothed-knot chord, red: raw-centroid chord), endpoint probe error per held-out block (clip-bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(results_dir=None, figures_dir=None):
    res = Path(results_dir) if results_dir else PROJECT_ROOT / "results"
    figs = Path(figures_dir) if figures_dir else PROJECT_ROOT / "figures"
    result = {"provenance": provenance(None, seeds={"blocks": "1-16"}),
              "estimators": "per block: gaps.<pair>.<metric>.mean (per-clip mean of paired spline - chord differences) "
                            "and its ci95 (clip bootstrap, 1000 draws); across blocks: mean and SD (ddof 1)",
              "pairs": PAIRS, "cells": {}}
    shift = {"provenance": result["provenance"], "gap": PAIRS, "cells": {}}
    for var, pt in (*CELLS, *DIRECTION_CELLS):
        runs = block_runs(var, pt, res)
        docs = [(s, json.loads(p.read_text())) for s, p in runs]
        cell = {"sources": [str(p.relative_to(res.parent)) if p.is_relative_to(res.parent) else str(p) for _, p in runs],
                "blocks": aggregate_blocks(docs)}
        if var != "direction":
            cell["bakeoff"] = bakeoff_rows(var, pt, res)
            if docs:
                shift["cells"][f"{var}_L{pt}"] = shift_dependence(docs)
        result["cells"][f"{var}_L{pt}"] = cell
        del docs
    (res / "p2_parity_scalars.json").write_text(json.dumps(result, indent=1))
    (res / "p2_shift_dependence_scalars.json").write_text(json.dumps(shift, indent=1))
    plot(result, figs / "fig_parity_scalars.png")
    return result


if __name__ == "__main__":
    main()

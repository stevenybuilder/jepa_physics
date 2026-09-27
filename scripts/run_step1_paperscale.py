"""Step 1 at the paper's scale (Part 1 extra, spec §6 item 2 / §7 outcomes). Stored mean-pool activations only.

  python scripts/run_step1_paperscale.py [--n-matched 240] [--seeds 10] [--n-boot 200]

For direction (direction set) and speed (speed set), every layer point, V-JEPA 2 and the random-init ViT-L:
  (a) full   : the step-1 sweep on our split (read from results/p1a_*_meanpool{,_random}.json + artifacts/oof)
  (b) dir8   : the paper's 8 directions (θ ∈ {0, 45, …, 315}), all clips at those angles, 5 random folds
  (c) dir8_grouped : same clips, 5 folds that each hold out whole directions (one reading of the paper's unspecified
                 "grouped CV"; α and the curve use pooled out-of-fold R², the fold-mean is undefined)
  (d) matched: random subsets of --n-matched clips over all 64 directions, 5 random folds, --seeds seeds (mean ± SD);
      also at n = |dir8| so (b) vs this isolates direction coverage from sample size
Onset = first point reaching 90% of the curve's max over points 0..24; CI = clip bootstrap of the OOF predictions
(for (d) pooled over seeds). Writes results/p1a_paperscale_{dataset}_{variable}.json and figures/fig1f_paperscale.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.paperscale import (ci_from_draws, direction_grouped_folds, eight_direction_rows, onset_draws,  # noqa: E402
                           probe_rows, random_folds, summarise_curve)
from wm.probes import (LAST_BLOCK, N_POINTS, RESULTS, availability, layer_fraction, layer_matrix,  # noqa: E402
                       load_activations, oof_path, result_name, targets, write_json)

RUNS = [("direction", "direction"), ("speed", "speed")]
MODELS = ("vjepa2", "random")
SHOW = (1, 8)   # block 1 (our onset) and block 8 (the paper's emergence layer)


def reference(dataset, variable, model, score_fn, n_boot, results):
    """(a): the full-data sweep as written by run_step1.py; the pooled-OOF curve is recomputed from artifacts/oof."""
    name = result_name("p1a", dataset, variable, "meanpool", model)
    sw = json.loads((results / f"{name}.json").read_text())
    z = np.load(oof_path(results, name))
    Y, oofs = z["Y"], z["oof"]
    pooled = [score_fn(Y, o)["r2"] for o in oofs]
    av = sw["availability"]
    return {"source": f"results/{name}.json", "n_clips": sw["n_train"], "n_fit": round(sw["n_train"] * 0.8),
            "curve": [r["cv_mean"] for r in sw["layers"]], "curve_sd": [r["cv_sd"] for r in sw["layers"]],
            "curve_pooled": pooled, "curve_kind": "fold_mean",
            "onset": av["onset"], "onset_ci": av["onset_ci"], "peak": av["peak"],
            "pooled_onset": summarise_curve(pooled, oofs, Y, score_fn, n_boot)}


def run(dataset, variable, args, results):
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    theta = df["theta_degrees"].to_numpy(float)
    n_all = len(df)
    rows8 = eight_direction_rows(theta)
    specs = {"dir8": (rows8, random_folds(len(rows8), 5, 0), "fold_mean"),
             "dir8_grouped": (rows8, direction_grouped_folds(theta[rows8], 5, 0), "pooled")}
    matched = {"matched": args.n_matched, "matched_dir8n": len(rows8)}
    for tag, n in matched.items():
        for s in range(args.seeds):
            rows = np.sort(np.random.default_rng(1000 + s).choice(n_all, n, replace=False))
            specs[f"{tag}/{s}"] = (rows, random_folds(n, 5, s), "fold_mean")
    grouped_check = {int(k): sorted(set(np.round(theta[rows8][specs["dir8_grouped"][1] == k], 3)))
                     for k in range(5)}

    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": "meanpool", "label": "PART 1 EXTRA (paper scale)",
           "paper_design": "8 directions x 7 speeds x 7 starts = 392 clips, Adam probes, 5-fold grouped CV",
           "n_dir8": int(len(rows8)), "dir8_theta": sorted(set(np.round(theta[rows8], 3))),
           "dir8_grouped_fold_directions": grouped_check, "n_matched": args.n_matched, "seeds": args.seeds,
           "motion_counts_dir8": df["motion"].iloc[rows8].value_counts().to_dict(),
           "rule": "onset = first point with CV R2 >= 90% of max over points 0..24; CI: clip bootstrap of OOF predictions "
                   f"({args.n_boot} draws; matched conditions pool {args.n_boot // args.seeds} draws per seed)",
           "curve_kind": {"full": "fold_mean (as run_step1)", "dir8": "fold_mean", "dir8_grouped": "pooled OOF R2",
                          "matched": "fold_mean, mean over seeds", "matched_dir8n": "fold_mean, mean over seeds"},
           "models": {}}
    for model in MODELS:
        acts = load_activations(dataset, "meanpool", model)
        fits = {k: [] for k in specs}
        for p in range(N_POINTS):
            X = layer_matrix(acts, p)
            for k, (rows, folds, select) in specs.items():
                fits[k].append(probe_rows(X, rows, Y, folds, score_fn, select))
            print(f"{dataset}/{variable} {model} point {p:2d}: dir8 {fits['dir8'][-1]['cv_mean']:.3f} "
                  f"grouped {fits['dir8_grouped'][-1]['pooled']:.3f} matched/0 {fits['matched/0'][-1]['cv_mean']:.3f}",
                  flush=True)
        res = {"full": reference(dataset, variable, model, score_fn, args.n_boot, results)}
        for k in ("dir8", "dir8_grouped"):
            rows, folds, select = specs[k]
            key = "cv_mean" if select == "fold_mean" else "pooled"
            curve = [f[key] for f in fits[k]]
            oofs = [f["oof"] for f in fits[k]]
            s = summarise_curve(curve, oofs, Y[rows], score_fn, args.n_boot)
            res[k] = {"n_clips": int(len(rows)), "n_fit": int(len(rows) - np.bincount(folds).max()),
                      "curve": curve, "curve_sd": [f["cv_sd"] for f in fits[k]],
                      "curve_pooled": [f["pooled"] for f in fits[k]], "alpha": [f["alpha"] for f in fits[k]], **s}
        for tag, n in matched.items():
            curves = np.array([[f["cv_mean"] for f in fits[f"{tag}/{s}"]] for s in range(args.seeds)])
            per_seed, draws = [], []
            for s in range(args.seeds):
                rows = specs[f"{tag}/{s}"][0]
                per_seed.append(availability(curves[s][:LAST_BLOCK + 1])["onset"])
                draws += onset_draws(Y[rows], [f["oof"] for f in fits[f"{tag}/{s}"]], score_fn,
                                     max(1, args.n_boot // args.seeds), seed=s)
            ci, n_none = ci_from_draws(draws)
            mean = curves.mean(axis=0)
            av = availability(mean[:LAST_BLOCK + 1])
            res[tag] = {"n_clips": n, "n_fit": int(n - np.ceil(n / 5)), "curve": mean.tolist(),
                        "curve_sd": curves.std(axis=0, ddof=1).tolist(), "curve_per_seed": curves.tolist(),
                        "onset": av["onset"], "peak": av["peak"], "peak_score": av["peak_score"],
                        "onset_per_seed": per_seed, "onset_ci": ci, "onset_boot_draws_without_onset": n_none}
        for c in res.values():
            c["at_point"] = {str(p): c["curve"][p] for p in SHOW}
        out["models"][model] = res
    out["summary"] = {m: {c: {"block1": r["at_point"]["1"], "block8": r["at_point"]["8"], "onset": r["onset"],
                              "onset_ci": r["onset_ci"], "n_clips": r["n_clips"]}
                          for c, r in out["models"][m].items()} for m in MODELS}
    write_json(results / f"p1a_paperscale_{dataset}_{variable}.json", out)
    return out


LABELS = {"full": "(a) full data, our folds", "dir8": "(b) 8 directions, random folds",
          "dir8_grouped": "(c) 8 directions, direction-grouped folds", "matched": "(d) {n} clips, 64 directions"}
COLORS = {"full": "#2a78d6", "dir8": "#eb6834", "dir8_grouped": "#4a3aa7", "matched": "#138a60"}


def figure(outs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                         "axes.grid": True, "grid.color": "#ebeae6", "legend.frameon": False, "lines.linewidth": 2,
                         "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})
    fig, axes = plt.subplots(len(outs), 2, figsize=(11, 3.6 * len(outs)), sharex=True, squeeze=False)
    x = np.array([layer_fraction(p) for p in range(LAST_BLOCK + 1)])
    for i, o in enumerate(outs):
        for j, model in enumerate(MODELS):
            ax, res = axes[i, j], o["models"][model]
            for c in ("full", "dir8", "dir8_grouped", "matched"):
                r = res[c]
                m = np.array(r["curve"][:LAST_BLOCK + 1], float)
                ax.plot(x, m, color=COLORS[c], label=LABELS[c].format(n=o["n_matched"]), lw=1.6)
                if c == "matched":
                    sd = np.array(r["curve_sd"][:LAST_BLOCK + 1])
                    ax.fill_between(x, m - sd, m + sd, color=COLORS[c], alpha=0.18, lw=0)
                if r["onset"] is not None:
                    ax.plot(x[r["onset"]], m[r["onset"]], "v", color=COLORS[c], ms=8, mec="white", mew=1.2)
                    if r["onset_ci"]:
                        lo, hi = r["onset_ci"]
                        ax.plot([x[lo], x[hi]], [m[r["onset"]]] * 2, color=COLORS[c], lw=4, alpha=0.35,
                                solid_capstyle="round")
            ax.axvline(layer_fraction(8), color="#9a9893", lw=0.8, ls=":")
            on = {c: res[c]["onset"] for c in ("full", "dir8", "dir8_grouped")}
            ci = {c: res[c]["onset_ci"] for c in ("dir8", "dir8_grouped")}
            ax.set_title(f"{o['variable']}, {'V-JEPA 2' if model == 'vjepa2' else 'random-init ViT-L'}: onset point "
                         f"{on['full']} (full) -> {on['dir8']} {ci['dir8']} (b) / {on['dir8_grouped']} "
                         f"{ci['dir8_grouped']} (c)", fontsize=8.5)
            ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1.02)
            if j == 0:
                ax.set_ylabel("probe R² (5-fold CV)")
            if i == len(outs) - 1:
                ax.set_xlabel("layer fraction (dotted: block 8, the paper's emergence layer)")
    axes[0, 0].plot([], [], "v", color="#52514e", label="onset (90% of max); bar = 95% bootstrap CI")
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, fontsize=8)
    fig.suptitle("Extra: Step 1 at the paper's scale (8 directions, ~190 clips) vs full data", x=0.01, ha="left",
                 y=1.06, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-matched", type=int, default=240)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    outs = [run(d, v, args, args.results) for d, v in RUNS]
    args.figures.mkdir(parents=True, exist_ok=True)
    figure(outs, args.figures / "fig1f_paperscale.png")
    for o in outs:
        print(f"\n{o['dataset']}/{o['variable']}  (n_dir8={o['n_dir8']})")
        for m, cs in o["summary"].items():
            for c, s in cs.items():
                print(f"  {m:7s} {c:14s} n={s['n_clips']:5d} block1={s['block1']:.3f} block8={s['block8']:.3f} "
                      f"onset={s['onset']} CI={s['onset_ci']}")

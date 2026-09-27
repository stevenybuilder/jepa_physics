"""Objective axis (spec §6 extra): V-JEPA 2 (latent prediction) vs VideoMAE (pixel reconstruction) vs the random-init
ViT-L, from results already on disk. Per variable and model: peak CV R2 (and its point), block-1 CV R2, selectivity
over the random-init model, nested K at the model's own peak layer, and probes needed to steer to within 10 deg.

  python scripts/run_objective_axis.py [--results results] [--figures figures]

Reads (any may be missing; the script says which and fills None):
  results/p1a_{dataset}_{variable}_meanpool[_videomae|_random].json      step 1
  results/p1b_{dataset}_{variable}_meanpool[_videomae|_random]_L{peak}.json   step 2 nested K
  results/p1c_{dataset}_L{peak}[_videomae|_random].json                   step 3 steering
  artifacts/oof/p1a_..._meanpool[_videomae|_random].npz                   out-of-fold predictions (selectivity onset)
Writes results/objective_axis.json and figures/fig5_objective_axis.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT  # noqa: E402
from wm.probes import LAST_BLOCK, layer_fraction, result_name, write_json  # noqa: E402

MODELS = ("vjepa2", "videomae", "random")
LABELS = {"vjepa2": "V-JEPA 2 ViT-L (latent prediction)", "videomae": "VideoMAE ViT-L (pixel reconstruction)",
          "random": "random-init ViT-L"}
COLORS = {"vjepa2": "C0", "videomae": "C3", "random": "0.5"}
PRIMARY = {"direction": "direction", "speed": "speed", "acceleration": "acceleration"}
TARGET_DEG = 10.0
TARGET_REL = 0.10


def suffix(model):
    return "" if model == "vjepa2" else f"_{model}"


def load(path, missing):
    if path.exists():
        return json.loads(path.read_text())
    missing.append(str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path))
    return None


def probes_to_reach(p1c, kind):
    """First probe count n whose steered test MAE-to-target reaches the bar: 10 deg (direction) or 10% of the
    unsteered (n = 0) MAE-to-target (scalars). None if never reached within the sweep."""
    rows = p1c.get("single") or []
    if not rows:
        return None, None
    bar = TARGET_DEG if kind == "circular" else TARGET_REL * rows[0]["mae_to_target"]
    hit = next((r["n"] for r in rows if r["mae_to_target"] <= bar), None)
    return hit, bar


def selectivity_onset_from_oof(results, dataset, variable, model):
    """Selectivity onset of `model` over the random-init sweep from saved out-of-fold predictions (wm.probes)."""
    from wm.probes import load_table, oof_path, selectivity_onset, targets
    pm = oof_path(results, result_name("p1a", dataset, variable, "meanpool", model))
    pr = oof_path(results, result_name("p1a", dataset, variable, "meanpool", "random"))
    if not (pm.exists() and pr.exists()):
        return None, f"missing {pm.name if not pm.exists() else pr.name}"
    zm, zr = np.load(pm), np.load(pr)
    if not np.array_equal(zm["Y"], zr["Y"]):
        return None, "train rows differ between the two sweeps"
    _, _, score_fn = targets(load_table(dataset), variable)
    onset, _ = selectivity_onset(zm["Y"], zm["oof"], zr["oof"], score_fn)
    return onset, "computed from out-of-fold predictions"


def row(results, dataset, variable, model, sweeps, missing):
    s = sweeps[model]
    if s is None:
        return {"available": False}
    layers = s["layers"][:LAST_BLOCK + 1]
    cv = np.array([r["cv_mean"] for r in layers])
    av = s["availability"]
    peak = av["peak"]
    out = {"available": True, "kind": s["kind"], "peak_point": peak, "peak_frac": layer_fraction(peak),
           "peak_cv_r2": float(cv[peak]), "peak_test_r2": layers[peak]["test_r2"],
           "peak_cv_mae": layers[peak]["cv_mae_mean"], "block1_cv_r2": float(cv[1]), "onset_point": av["onset"],
           "final_ln_cv_r2": s["layers"][-1]["cv_mean"]}
    rnd = sweeps["random"]
    if model != "random" and rnd is not None:
        rcv = np.array([r["cv_mean"] for r in rnd["layers"][:LAST_BLOCK + 1]])
        diff = cv - rcv
        out["selectivity_at_peak"] = float(diff[peak])
        out["selectivity_max"] = float(diff.max())
        out["selectivity_max_point"] = int(diff.argmax())
        out["selectivity_block1"] = float(diff[1])
        if av.get("selectivity_onset") is not None or model == "vjepa2":
            out["selectivity_onset"], out["selectivity_onset_source"] = av.get("selectivity_onset"), "step-1 file"
        else:
            out["selectivity_onset"], out["selectivity_onset_source"] = selectivity_onset_from_oof(
                results, dataset, variable, model)
    p1b = load(results / f"{result_name('p1b', dataset, variable, 'meanpool', model)}_L{peak}.json", missing)
    if p1b is not None:
        out.update({"nested_K": p1b.get("K"), "nested_K_folds": p1b.get("K_folds"),
                    "nested_K_dims": p1b.get("dims"), "paper_K": (p1b.get("paper") or {}).get("K")})
    else:
        have = sorted(int(p.stem.rsplit("_L", 1)[1]) for p in results.glob(
            f"{result_name('p1b', dataset, variable, 'meanpool', model)}_L*.json") if p.stem.rsplit("_L", 1)[1].isdigit())
        out["nested_K"], out["p1b_layers_on_disk"] = None, have
    if variable == PRIMARY[dataset]:
        p1c = load(results / f"p1c_{dataset}_L{peak}{suffix(model)}.json", missing)
        if p1c is not None:
            n, bar = probes_to_reach(p1c, s["kind"])
            out.update({"steer_probes_to_bar": n, "steer_bar": bar, "steer_bar_reached": n is not None,
                        "steer_bar_rule": (f"MAE-to-target <= {TARGET_DEG} deg" if s["kind"] == "circular" else
                                           f"MAE-to-target <= {TARGET_REL:.0%} of the unsteered MAE-to-target"),
                        "steer_K": p1c.get("K"), "steer_mae_at_K": (p1c.get("single") or [{}])[-1].get("mae_to_target")})
        else:
            out["steer_probes_to_bar"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures")
    ap.add_argument("--variables", nargs="+", default=["direction/direction", "speed/speed", "acceleration/acceleration"],
                    help="dataset/variable pairs")
    args = ap.parse_args()
    missing, table, curves = [], {}, {}
    for pair in args.variables:
        dataset, variable = pair.split("/")
        sweeps = {m: load(args.results / f"{result_name('p1a', dataset, variable, 'meanpool', m)}.json", missing)
                  for m in MODELS}
        table[pair] = {m: row(args.results, dataset, variable, m, sweeps, missing) for m in MODELS}
        curves[pair] = {m: [r["cv_mean"] for r in s["layers"]] if s else None for m, s in sweeps.items()}
    if missing:
        print("missing (filled with None):\n  " + "\n  ".join(missing))
    out = {"models": LABELS, "variables": table, "missing": missing,
           "notes": {"peak": "argmax of the 5-fold CV R2 over points 0..24 (post-LN point excluded), each model its own",
                     "selectivity": "model CV R2 minus random-init CV R2 at the same point (same split, folds, alphas)",
                     "nested_K": "step 2 nested (CV-faithful) K at the model's own step-1 peak",
                     "steer": "step 3 single-target sweep at the model's own peak; probes needed to reach the bar",
                     "videomae_input": "VideoMAE sees 224-px resized clips (1568 tokens) vs V-JEPA's native 256 px "
                                       "(2048 tokens); see wm.extract"},
           "cv_curves": curves}
    write_json(args.results / "objective_axis.json", out)
    for pair, rows in table.items():
        print(f"\n{pair}")
        print(f"  {'model':9s} {'peak':>4s} {'peakR2':>7s} {'blk1R2':>7s} {'sel@pk':>7s} {'K':>5s} {'n->bar':>6s}")
        for m, r in rows.items():
            if not r["available"]:
                print(f"  {m:9s} (no step-1 file)")
                continue
            f = lambda v, fmt: "   -" if v is None else format(v, fmt)  # noqa: E731
            print(f"  {m:9s} {r['peak_point']:4d} {r['peak_cv_r2']:7.3f} {r['block1_cv_r2']:7.3f} "
                  f"{f(r.get('selectivity_at_peak'), '7.3f'):>7s} {f(r.get('nested_K'), '5d'):>5s} "
                  f"{f(r.get('steer_probes_to_bar'), '6d'):>6s}")
    figure(table, curves, args.figures / "fig5_objective_axis.png")
    print(f"\nwrote {args.results / 'objective_axis.json'} and {args.figures / 'fig5_objective_axis.png'}")


def figure(table, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = list(table)
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.2 * len(pairs), 7), squeeze=False)
    for j, pair in enumerate(pairs):
        ax = axes[0, j]
        for m in MODELS:
            c = curves[pair][m]
            if c is None:
                continue
            x = [layer_fraction(p) for p in range(len(c))]
            ax.plot(x[:LAST_BLOCK + 1], c[:LAST_BLOCK + 1], color=COLORS[m], lw=1.6, label=LABELS[m])
            pk = table[pair][m]["peak_point"]
            ax.plot(layer_fraction(pk), c[pk], "o", color=COLORS[m], ms=5)
        ax.set_title(pair.split("/")[1])
        ax.set_xlabel("layer (block / 24)")
        ax.set_ylabel("5-fold CV R$^2$")
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1.02)
        if j == 0:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax = axes[1, j]
        metrics = [("nested_K", "nested K at peak"), ("steer_probes_to_bar", "probes to steer bar")]
        w = 0.25
        for i, m in enumerate(MODELS):
            r = table[pair][m]
            for k, (key, _) in enumerate(metrics):
                v = r.get(key) if r.get("available") else None
                xpos = k + (i - 1) * w
                if v is None:
                    txt = "not reached" if (key == "steer_probes_to_bar" and r.get("steer_bar_reached") is False) else "n/a"
                    ax.text(xpos, 0.5, txt, ha="center", va="bottom", fontsize=7, color=COLORS[m], rotation=90)
                else:
                    ax.bar(xpos, v, w, color=COLORS[m])
                    ax.text(xpos, v, str(v), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([lab for _, lab in metrics], fontsize=8)
        ax.set_ylabel("count")
        bar = "10 deg" if pair.startswith("direction") else "10% of unsteered MAE"
        ax.set_title(f"steer bar: {bar}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()

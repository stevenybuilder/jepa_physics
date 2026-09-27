"""Random-feature floor (Part 1 extra, spec §6 item 2 / §7 outcomes): 1024 fixed random ReLU features of the disk
centroid trajectory, probed with the step-1 ridge / α-CV / test-once on our split.

  python scripts/run_random_feature_floor.py [--seeds 3] [--dataset direction]

Per variable: random features of the 32-number centroid trajectory (the file's main row, seed 0, with seed mean ± SD
beside it), of its 30 frame differences, and a plain linear probe on the 32 centroids. Writes
results/p1a_{dataset}_{variable}_randfeat.json (p1a schema, one row, point = -1) and adds a randfeat row to
results/p1a_baseline_comparison.json (every baseline and the block-1 / peak model scores, CV R² side by side).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import load_table  # noqa: E402
from wm.pixels import load_pixels, probe_row  # noqa: E402
from wm.probes import ALPHAS, RESULTS, VARIABLES, result_name, split_rows, targets, write_json  # noqa: E402
from wm.randfeat import N_FEATURES, centroid_inputs, random_relu_features  # noqa: E402

KEYS = ("cv_mean", "cv_sd", "test_r2", "cv_mae_mean", "test_mae", "alpha")


def brief(row):
    out = {k: row[k] for k in KEYS}
    for k in ("cv_acc15_mean", "test_acc15"):
        if k in row:
            out[k] = row[k]
    return out


def run(dataset, variable, cen, seeds, results):
    df = load_table(dataset)
    tr, te, folds = split_rows(dataset, df)
    Y, kind, score_fn = targets(df, variable)
    motion = df["motion"].to_numpy()
    by_input = {}
    main = None
    for which in ("centroid", "diff"):
        Ztr, Zte = centroid_inputs(cen, tr, te, which)
        rows = []
        for s in range(seeds):
            Ftr, Fte = random_relu_features(Ztr, Zte, N_FEATURES, seed=s)
            row, Pte = probe_row(Ftr, Fte, Y[tr], Y[te], folds, kind, score_fn, motion[tr], motion[te])
            rows.append(row)
            if which == "centroid" and s == 0:
                main, main_pred = row, Pte
        by_input[which] = {"d_in": int(Ztr.shape[1]), "per_seed": [brief(r) for r in rows],
                           **{f"{k}_seed_mean": float(np.mean([r[k] for r in rows])) for k in ("cv_mean", "test_r2")},
                           **{f"{k}_seed_sd": float(np.std([r[k] for r in rows], ddof=1)) if seeds > 1 else 0.0
                              for k in ("cv_mean", "test_r2")}}
    Ztr, Zte = centroid_inputs(cen, tr, te, "centroid")
    lin, _ = probe_row(Ztr, Zte, Y[tr], Y[te], folds, kind, score_fn)
    by_input["linear_centroid"] = {"d_in": int(Ztr.shape[1]), **brief(lin)}
    main = {**main, "input": "centroid", "seed": 0,
            "cv_mean_seed_mean": by_input["centroid"]["cv_mean_seed_mean"],
            "cv_mean_seed_sd": by_input["centroid"]["cv_mean_seed_sd"],
            "test_r2_seed_mean": by_input["centroid"]["test_r2_seed_mean"]}
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": "randfeat", "model": "randfeat",
           "shuffled": False, "alphas": ALPHAS.tolist(), "n_train": len(tr), "n_test": len(te),
           "label": "PART 1 EXTRA (random-feature floor)",
           "features": f"relu(Z W + b), W ~ N(0, 1/d_in) [d_in, {N_FEATURES}], b ~ N(0, 1), seeds 0..{seeds - 1}; "
                       "Z = train-z-scored disk centroid (x,y)/256 per frame (32) [main row] or its 15 diffs (30)",
           "layers": [main], "by_input": by_input}
    if kind == "circular":
        out["test_theta"] = df["theta_degrees"].to_numpy()[te].tolist()
        out["test_pred"] = {"-1": main_pred.round(4).tolist()}
    write_json(results / (result_name("p1a", dataset, variable, "randfeat") + ".json"), out)
    return out


def comparison(results, randfeat):
    """results/p1a_baseline_comparison.json: CV (and test) R² of every Part 1 baseline per variable, read from the
    results files that exist, plus the randfeat rows just computed."""
    table = {}
    for dataset, variables in VARIABLES.items():
        for variable in variables:
            rows = {}
            for tag, name in (("vjepa2_block1", "meanpool"), ("random_vit_block1", "meanpool_random"),
                              ("vjepa2_peak", "meanpool"), ("shuffled_block1", "meanpool_shuffled"),
                              ("pixels", "pixels"), ("trajectory_linear", "trajectory")):
                p = results / f"p1a_{dataset}_{variable}_{name}.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                r = (d["layers"][d["availability"]["peak"]] if tag == "vjepa2_peak"
                     else d["layers"][1] if "block1" in tag else d["layers"][0])
                rows[tag] = {"cv_mean": r["cv_mean"], "test_r2": r["test_r2"], "point": r["point"], "source": p.name}
            rf = randfeat.get((dataset, variable))
            if rf:
                bi = rf["by_input"]
                rows["randfeat_centroid"] = {"cv_mean": bi["centroid"]["cv_mean_seed_mean"],
                                             "cv_sd_over_seeds": bi["centroid"]["cv_mean_seed_sd"],
                                             "test_r2": bi["centroid"]["test_r2_seed_mean"], "point": -1,
                                             "source": result_name("p1a", dataset, variable, "randfeat") + ".json"}
                rows["randfeat_diff"] = {"cv_mean": bi["diff"]["cv_mean_seed_mean"],
                                         "cv_sd_over_seeds": bi["diff"]["cv_mean_seed_sd"],
                                         "test_r2": bi["diff"]["test_r2_seed_mean"], "point": -1,
                                         "source": result_name("p1a", dataset, variable, "randfeat") + ".json"}
                rows["linear_centroid32"] = {"cv_mean": bi["linear_centroid"]["cv_mean"],
                                             "test_r2": bi["linear_centroid"]["test_r2"], "point": -1,
                                             "source": result_name("p1a", dataset, variable, "randfeat") + ".json"}
            table[f"{dataset}/{variable}"] = rows
    path = results / "p1a_baseline_comparison.json"
    old = json.loads(path.read_text()).get("table", {}) if path.exists() else {}
    for k, v in table.items():   # keep rows of datasets not rerun this time
        old[k] = {**old.get(k, {}), **v}
    write_json(path, {"label": "Part 1 baselines vs model, CV R2 (fold mean) and test R2", "table": old})
    return old


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    done = {}
    for dataset in [args.dataset] if args.dataset else VARIABLES:
        _, cen = load_pixels(dataset)
        for variable in VARIABLES[dataset]:
            o = run(dataset, variable, cen, args.seeds, args.results)
            done[(dataset, variable)] = o
            bi = o["by_input"]
            print(f"{dataset}/{variable}: randfeat(centroid) cv R2 {bi['centroid']['cv_mean_seed_mean']:.3f}"
                  f"±{bi['centroid']['cv_mean_seed_sd']:.3f} test {bi['centroid']['test_r2_seed_mean']:.3f} | "
                  f"randfeat(diff) cv {bi['diff']['cv_mean_seed_mean']:.3f} test {bi['diff']['test_r2_seed_mean']:.3f} | "
                  f"linear(centroid32) cv {bi['linear_centroid']['cv_mean']:.3f} test {bi['linear_centroid']['test_r2']:.3f}",
                  flush=True)
    table = comparison(args.results, done)
    for k, rows in table.items():
        print(k, {t: round(r["cv_mean"], 3) for t, r in rows.items()})

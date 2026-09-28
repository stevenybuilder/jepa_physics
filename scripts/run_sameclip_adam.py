"""Same-clip direction vs speed probe counts under the paper's C.11 Adam recipe (ledger #178).

scripts/run_sameclip.py runs the same-clip comparison with the ridge sequence only. Here the probe at every round is
the paper's C.11 Adam probe (refs/physics_paper.txt l.1204-1212: Adam, lr 1e-3, weight decay 1e-4, 100 epochs
direction / 50 speed), via wm.inlp.adam_sequence exactly as scripts/run_step2.py --recipe adam (paper protocol: fit on
all restricted train rows, score on the restricted test rows; patience 3; init seed 0 + round - 1). Clips, labels,
split and standardisation are run_sameclip's: the 750 constant-velocity clips (motion == 'velocity') of the supplied
direction set, splits/split_v1.json restricted to them (596 train / 154 test), speed_mps from the direction-set
metadata, per-feature z-score from the restricted train rows.

K under C.11: K_first (first at-chance round - 1, the paper's rule) and K_patience (start of the final 3-round
at-chance run - 1), as the stored _adam_b64.json files. K under Fig. 22 (caption l.1172: direction R2 < 0.3, speed
R2 < 0.1): first round below the threshold - 1, over the rounds the sequence ran; censored when the sequence stopped
first (then K_fig22 is a floor, set to K_patience). mae_rule_first flags a C.11 stop fired by the MAE clause while R2
was still above the C.11 R2 threshold. Writes results/p1b_sameclip_adam_direction_vs_speed.json.

  python scripts/run_sameclip_adam.py [--points 8 9 22 19] [--batches 64 0]    (0 = full batch)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_sameclip import VARS, restrict  # noqa: E402
from wm.inlp import adam_sequence, write_result  # noqa: E402
from wm.probes import RESULTS, load_activations, load_table, split_rows, standardized_layer, targets  # noqa: E402

LOOSE = {"circular": 0.3, "scalar": 0.1}          # Fig. 22 caption
C11_R2 = {"circular": 0.1, "scalar": 0.05}        # C.11 R2 clause


def sameclip_rows(df=None):
    """run_sameclip's clip restriction: (df, keep, tr, te, folds) on the direction set's constant-velocity clips."""
    df = load_table("direction") if df is None else df
    keep = (df["motion"] == "velocity").to_numpy()
    if not np.isfinite(df.loc[keep, "speed_mps"].to_numpy(float)).all():
        sys.exit("speed_mps missing for constant-velocity direction-set clips; stop")
    tr, te, folds = restrict(*split_rows("direction", df), keep)
    return df, keep, tr, te, folds


def k_rules(summary, kind):
    """C.11 (K_first, K_patience) and Fig. 22 K from one Adam sequence's per-round test curve."""
    rows = summary["rounds"]
    loose, c11 = LOOSE[kind], C11_R2[kind]
    k22 = next((r["round"] - 1 for r in rows if r["test_r2"] < loose), None)
    censored = k22 is None
    first = next((r for r in rows if r["round"] == summary["K_first"] + 1), None)
    mae_first = bool(first is not None and first["test_r2"] >= c11)
    return {"K_first": summary["K_first"], "K_patience": summary["K_patience"], "patience": summary["patience"],
            "hit_round_cap": summary["hit_round_cap"],
            "K_fig22": summary["K_patience"] if censored else k22, "K_fig22_censored": censored,
            "fig22_threshold": f"R2 < {loose}",
            "c11_first_trigger": None if first is None else ("mae" if mae_first else "r2"),
            "mae_rule_first": mae_first,
            "fig22_after_c11_stop": bool(not censored and k22 > summary["K_first"]),
            "rounds_run": len(rows)}


def stored_cross_set(var, point, results_dir):
    """Stored Adam K (cross-set: direction from the direction set, speed from the speed set), with source keys."""
    out = {}
    for tag in ("b64", "full"):
        f = Path(results_dir) / f"p1b_{var}_{var}_meanpool_L{point}_adam_{tag}.json"
        if f.exists():
            d = json.loads(f.read_text())
            kr = k_rules(d, d["kind"])
            out[f"adam_{tag}"] = {"file": f"results/{f.name}", "K_first [K_first]": d["K_first"],
                                  "K_patience [K_patience]": d["K_patience"],
                                  "K_fig22 (derived here from rounds[].test_r2)": kr["K_fig22"],
                                  "K_fig22_censored": kr["K_fig22_censored"],
                                  "batch": d["recipe"]["batch"], "epochs_used": d["recipe"]["epochs_used"]}
    return out or None


def comparison(cell, tag):
    cmp = {}
    for name, key in (("C11", "K_first"), ("C11_patience", "K_patience"), ("fig22", "K_fig22")):
        kd, ks = cell["direction"][tag][key], cell["speed"][tag][key]
        cmp[name] = {"K_direction": kd, "dims_direction_2K": 2 * kd, "K_speed": ks, "dims_speed_K": ks,
                     "speed_fewer_probes": ks < kd, "speed_fewer_dims": ks < 2 * kd}
    for v in VARS:
        cmp["fig22"][f"{v}_censored"] = cell[v][tag]["K_fig22_censored"]
    return cmp


def merge(paths, out_path, primary="b64", pending=()):
    """Merge shard files: cells by point, batch runs by tag; comparisons recomputed (primary batch first)."""
    shards = [json.loads(Path(f).read_text()) for f in paths]
    cells, runtime = {}, 0.0
    for d in shards:
        runtime += d["runtime_s_total"]
        for key, c in d["cells"].items():
            cell = cells.setdefault(key, {"point": c["point"]})
            for v in VARS:
                cell.setdefault(v, {"kind": c[v]["kind"]})
                cell[v].update({t: r for t, r in c[v].items() if t not in ("kind", "stored_cross_set")})
                cell[v]["stored_cross_set"] = c[v]["stored_cross_set"]
    for cell in cells.values():
        tags = sorted({t for t in cell["direction"] if t not in ("kind", "stored_cross_set")},
                      key=lambda t: t != primary)
        cell["comparison"] = comparison(cell, tags[0])
        cell["comparison_by_batch"] = {t: comparison(cell, t) for t in tags}
        cell["stored_cross_set"] = {v: cell[v]["stored_cross_set"] for v in VARS}
    out = {k: v for k, v in shards[0].items() if k not in ("cells", "provenance")}
    out["method"]["comparison_uses"] = f"{primary} (comparison_by_batch has every batch run)"
    out.update({"points_done": sorted(cells, key=lambda k: int(k[1:])), "runtime_s_total": round(runtime, 1),
                "runtime_note": "sum of shard runtimes (shards ran in parallel, 1 BLAS thread each)",
                "points_pending": [f"L{p}" for p in pending],
                "shards": [str(Path(f).name) for f in paths], "cells": cells})
    write_result(out_path, out, seeds_adam=shards[0]["provenance"]["seeds_adam"],
                 test_read=shards[0]["provenance"]["test_read"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, nargs="+", default=[8, 9, 22, 19])
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 0], help="0 = full batch")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--merge", type=Path, nargs="+", default=None,
                    help="combine shard outputs of this script (one per point / batch run in parallel) into --out")
    ap.add_argument("--pending", type=int, nargs="*", default=[], help="with --merge: points still running")
    args = ap.parse_args(argv)
    out_path = args.out or Path(args.results) / "p1b_sameclip_adam_direction_vs_speed.json"
    if args.merge:
        return merge(args.merge, out_path, pending=args.pending)

    df, keep, tr, te, folds = sameclip_rows()
    acts = load_activations("direction", "meanpool")
    Ys = {v: targets(df, v) for v in VARS}
    cells, t_all = {}, time.time()
    for p in args.points:
        Xtr, Xte = standardized_layer(acts, p, tr, te)
        if not (np.isfinite(Xtr).all() and np.isfinite(Xte).all()):
            sys.exit(f"non-finite activations at point {p} on the restricted clips; stop")
        cell = {"point": p}
        for v in VARS:
            Y, kind, score_fn = Ys[v]
            cell[v] = {"kind": kind}
            for b in args.batches:
                tag = "full" if b == 0 else f"b{b}"
                t0 = time.time()
                s, _ = adam_sequence(Xtr, Y[tr], Xte, Y[te], score_fn, kind, None, b or None, 0, args.patience)
                cell[v][tag] = {**k_rules(s, kind), "m": s["m"], "recipe": s["recipe"], "protocol": s["protocol"],
                                "test_r2_curve": [r["test_r2"] for r in s["rounds"]],
                                "test_mae_curve": [r["test_mae"] for r in s["rounds"]],
                                "base_mae_curve": [r["base_mae"] for r in s["rounds"]],
                                "train_r2_curve": [r["train_r2"] for r in s["rounds"]],
                                "failed_to_train_rounds": [r["round"] for r in s["rounds"] if r["failed_to_train"]],
                                "runtime_s": round(time.time() - t0, 1)}
                print(p, v, tag, {k: cell[v][tag][k] for k in ("K_first", "K_patience", "K_fig22",
                                                                "K_fig22_censored", "c11_first_trigger",
                                                                "runtime_s")}, flush=True)
            cell[v]["stored_cross_set"] = stored_cross_set(v, p, args.results)
        tags = [("full" if b == 0 else f"b{b}") for b in args.batches]
        cell["comparison"] = comparison(cell, tags[0])
        cell["comparison_by_batch"] = {t: comparison(cell, t) for t in tags}
        cell["stored_cross_set"] = {v: cell[v]["stored_cross_set"] for v in VARS}
        cells[f"L{p}"] = cell
        out = {"item": "Same-clip direction vs speed probe counts, C.11 Adam recipe (ledger #178)",
               "clips": {"dataset": "direction (supplied)", "filter": "motion == 'velocity' (constant velocity)",
                         "n": int(keep.sum()), "n_train": int(len(tr)), "n_test": int(len(te)),
                         "split": "splits/split_v1.json direction split restricted to these clips "
                                  "(scripts/run_sameclip.py restrict)",
                         "speed_labels": "speed_mps from the direction-set metadata"},
               "method": {"sequence": "wm.inlp.adam_sequence (scripts/run_step2.py --recipe adam): C.11 Adam probe "
                                      "each round, QR, project out span(W_k); paper protocol",
                          "standardisation": "per-feature z-score, stats from the restricted train rows",
                          "comparison_uses": f"{tags[0]} (comparison_by_batch has every batch run)",
                          "stop_rules": {"C11": "K_first: direction R2 < 0.1 or circular MAE > 80 deg; speed R2 < "
                                                "0.05 or MAE > 0.9 x mean-predictor MAE (l.1211-1212)",
                                         "C11_patience": f"K_patience: start of the final {args.patience}-round "
                                                         "C.11 at-chance run - 1 (stored Adam files' variant)",
                                         "fig22": "direction R2 < 0.3, speed R2 < 0.1 (Fig. 22 caption l.1172); "
                                                  "censored = never crossed before the sequence stopped (K_fig22 "
                                                  "then = K_patience, a floor)"},
                          "K": "probes kept; dims = 2K for direction (sin, cos), K for speed",
                          "ridge_counterpart": "results/p1b_sameclip_direction_vs_speed.json cells.L{pt}.comparison."
                                               "paper_{C11,fig22}"},
               "points_done": list(cells), "runtime_s_total": round(time.time() - t_all, 1), "cells": cells}
        write_result(out_path, out, seeds_adam={"split": 0, "adam_init": "0 + round - 1", "minibatch_order": "same rng "
                                           "as the init seed (wm.adam_probe.init_linear)"},
                     test_read="paper protocol scores the restricted test split at every round (as stored step 2)")
    return out


if __name__ == "__main__":
    main()

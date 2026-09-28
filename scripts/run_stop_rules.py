"""Paper-first QA of App. C.11 (refs/physics_paper.txt l.1197, 1211-1212, 1172, 1225) against wm.inlp, on stored
pooled activations. CPU only.

  python scripts/run_stop_rules.py --items 1 2

  1  Stopping-rule asymmetry: probe sequence (ridge, stored alpha, stored split) for direction, speed and acceleration
     at points 8, 9 and each variable's peak, K under the stored C.11 rule, an R2-only stop at 0.05 and at 0.1 for all
     three, and Fig. 22's thresholds (direction 0.3, scalars 0.1); nested and paper protocols.
     -> results/p1b_stop_rules.json
  2  One-column-per-round removal (wm.inlp.one_column 'alternate' and 'top_sv') for direction and speed at point 9,
     ridge nested / paper and Adam b64 / full; sawtooth block (wm.inlp.sawtooth) beside the stored 2-column values.
     -> results/p1b_one_column_removal.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import load_table  # noqa: E402
from wm.inlp import (adam_sequence, at_chance, nested_curve, one_column, paper_curve_and_basis,  # noqa: E402
                     probe_sequence, ridge_fit, sawtooth)
from wm.probes import (RESULTS, drop_nan_clips, load_activations, split_rows, standardized_layer,  # noqa: E402
                       targets, write_json)
from wm.robustness import k_below, lockstep_until  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--items", nargs="+", type=int, default=[1, 2], choices=[1, 2])
parser.add_argument("--results", type=Path, default=RESULTS)
args = parser.parse_args()
R = args.results
PEAK = {"direction": 22, "speed": 19, "acceleration": 21}


def stored(name):
    return json.loads((R / name).read_text())


def data(variable):
    df = load_table(variable)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(variable, df)
    acts = load_activations(variable, "meanpool", "vjepa2")
    tr, te, folds, _ = drop_nan_clips(acts, tr, te, folds)
    return Y, kind, score_fn, tr, te, folds, acts


def k_stored_rule(rows, kind):
    return next((r["round"] - 1 for r in rows if at_chance(r, kind)), None)


# ------------------------------------------------------------------ item 1

def item1():
    rules = {"stored_C11": "C.11 l.1211-1212: direction R2 < 0.1 or circular MAE > 80 deg; scalars R2 < 0.05 or MAE > "
                           "0.9 x mean-predictor MAE",
             "r2_lt_0.05": "R2 < 0.05, all variables", "r2_lt_0.1": "R2 < 0.1, all variables",
             "fig22": "Fig. 22 caption l.1172: direction R2 < 0.3, scalars R2 < 0.1"}
    out = {"item": "Stopping-rule asymmetry (C.11 l.1211-1212 vs Fig. 22 l.1172)", "rules": rules,
           "K": "probes kept before the first round meeting the rule (first such round - 1); null = never met within "
                "the rounds run (censored)",
           "sequence": "wm.inlp.probe_sequence, ridge, stored alpha (nested: stored alpha_folds per fold; paper: stored "
                       "all-train alpha), stored split and folds, train-standardised meanpool; run until the curve's "
                       "R2 < 0.05 or d / m rounds",
           "protocols": {"nested": "fold-mean curve over the 5 held-out folds (pooled K, as stored)",
                         "paper": "all-train sequence scored on test (as stored)"},
           "cells": {}}
    for variable in ("direction", "speed", "acceleration"):
        Y, kind, score_fn, tr, te, folds, acts = data(variable)
        fig22 = 0.3 if kind == "circular" else 0.1
        for point in (8, 9, PEAK[variable]):
            ref = stored(f"p1b_{variable}_{variable}_meanpool_L{point}.json")
            Xtr, Xte = standardized_layer(acts, point, tr, te)
            Ytr = Y[tr].reshape(len(tr), -1)
            cap = Xtr.shape[1] // Ytr.shape[1]
            seqs = {"nested": [probe_sequence(Xtr[folds != k], Ytr[folds != k], ridge_fit(a), score_fn,
                                              Xtr[folds == k], Ytr[folds == k])
                               for k, a in zip(np.unique(folds), ref["alpha_folds"])],
                    "paper": [probe_sequence(Xtr, Ytr, ridge_fit(ref["alpha"]), score_fn, Xte, Y[te])]}
            cell = {"point": point, "is_peak": point == PEAK[variable], "m": Ytr.shape[1]}
            for proto, gens in seqs.items():
                rows = lockstep_until(gens, 0.05, cap)
                r2 = [r["r2"] for r in rows]
                st_rows = ref["rounds"] if proto == "nested" else ref["paper"]["rounds"]
                st_r2 = [r["cv_r2" if proto == "nested" else "test_r2"] for r in st_rows]
                n = min(len(r2), len(st_r2))
                K = {"stored_C11": k_stored_rule(rows, kind), "r2_lt_0.05": k_below(r2, 0.05),
                     "r2_lt_0.1": k_below(r2, 0.1), "fig22": k_below(r2, fig22)}
                src = ref if proto == "nested" else ref["paper"]
                cell[proto] = {"K": K, "K_stored_file": src["K"],
                               "stored_rule_parity": K["stored_C11"] == src["K"],
                               "r2_at_stored_stop": r2[K["stored_C11"]] if K["stored_C11"] is not None else None,
                               "stored_stop_trigger": None if K["stored_C11"] is None else
                               ("r2" if r2[K["stored_C11"]] < (0.1 if kind == "circular" else 0.05) else "mae"),
                               "rounds_run": len(rows),
                               "parity_max_abs_diff_r2_vs_stored": float(np.max(np.abs(np.array(r2[:n]) - st_r2[:n]))),
                               "r2_curve": r2}
                print(f"{variable} L{point} {proto}: {K} (stored {src['K']}, rounds {len(rows)}, parity "
                      f"{cell[proto]['parity_max_abs_diff_r2_vs_stored']:.1e})")
            out["cells"][f"{variable}_L{point}"] = cell
    common = {}
    for thr in ("r2_lt_0.05", "r2_lt_0.1"):
        for proto in ("nested", "paper"):
            common[f"{thr}_{proto}"] = {f"L{p}": {v: out["cells"][f"{v}_L{p}"][proto]["K"][thr]
                                                  for v in ("direction", "speed", "acceleration")}
                                        for p in (8, 9)}
    out["common_r2_stop_points_8_9"] = common
    write_json(R / "p1b_stop_rules.json", out)


# ------------------------------------------------------------------ item 2

def block(st, prefix):
    """lag-1 drop autocorrelation and isolated-dip count for R2 and acc15 from a wm.inlp.sawtooth block."""
    out = {}
    for metric in ("r2", "acc15"):
        b = st["by_metric"].get(f"{prefix}_{metric}")
        out[metric] = None if b is None else {"drop_lag1_autocorr": b["drop_lag1_autocorr"],
                                              "n_isolated_dips": len(b["isolated_dips"]),
                                              "isolated_dips": b["isolated_dips"], "n_rounds": len(b["drops"]) + 1}
    return out


def item2():
    point = 9
    out = {"item": "One-column-per-round removal (C.11 l.1197 'project out the learned direction' vs W_k in R^{2 x d} "
                   "l.1225)", "point": point,
           "variants": {"two_column": "stored: span(W_k), 2 columns per direction round",
                        "one_col_alternate": "column (k - 1) mod 2 of W_k: sin readout on odd rounds, cos on even",
                        "one_col_top_sv": "top left-singular vector of W_k (largest-gain input direction)"},
           "speed_note": "speed probes have 1 output, so both one-column variants remove span(W_k), the stored "
                         "sequence; rerun as a parity check",
           "sequences": {"ridge_nested": "wm.inlp.nested_curve, stored alpha (alpha_folds all equal it), fold-mean "
                                         "curve, stored stop (every fold and the mean at chance)",
                         "ridge_paper": "wm.inlp.paper_curve_and_basis, stored alpha, scored on test, stop at the "
                                        "first at-chance round",
                         "adam_b64": "wm.inlp.adam_sequence, C.11 Adam, batch 64, patience 3 (as stored)",
                         "adam_full": "wm.inlp.adam_sequence, C.11 Adam, full batch, patience 3 (as stored)"},
           "statistics": "wm.inlp.sawtooth by_metric: lag-1 autocorrelation of per-round drops (negative = sawtooth) "
                         "and isolated dips (fall > 0.05 then recover > 0.05); acc15 exists for direction only",
           "cells": {}}
    for variable in ("direction", "speed"):
        Y, kind, score_fn, tr, te, folds, acts = data(variable)
        Xtr, Xte = standardized_layer(acts, point, tr, te)
        ref = stored(f"p1b_{variable}_{variable}_meanpool_L{point}.json")
        assert all(a == ref["alpha"] for a in ref["alpha_folds"])
        adam_ref = {t: stored(f"p1b_{variable}_{variable}_meanpool_L{point}_adam_{t}.json") for t in ("b64", "full")}
        cell = {"two_column_stored": {
            "ridge_nested": {"K": ref["K"], **block(ref["sawtooth"], "cv")},
            "ridge_paper": {"K": ref["paper"]["K"], **block(ref["paper"]["sawtooth"], "test")},
            **{f"adam_{t}": {"K_first": a["K_first"], **block(a["sawtooth"], "test")} for t, a in adam_ref.items()}}}
        modes = ("alternate", "top_sv") if kind == "circular" else ("top_sv",)
        for mode in modes:
            sel = one_column(mode)
            res = {}
            nested = nested_curve(Xtr, Y[tr], folds, ref["alpha"], score_fn, kind, alpha_per_fold=False, select=sel)
            res["ridge_nested"] = {"K_rounds": nested["K"],
                                   **block(sawtooth(nested, np.zeros((0, 1, 1))), "cv")}
            paper, _, _ = paper_curve_and_basis(Xtr, Y[tr], Xte, Y[te], ref["alpha"], score_fn, kind, 0,
                                                Xtr.shape[1], select=sel)
            res["ridge_paper"] = {"K_rounds": paper["K"], **block(sawtooth(paper, np.zeros((0, 1, 1)), "test"), "test")}
            for t, batch in (("b64", 64), ("full", None)):
                a, _ = adam_sequence(Xtr, Y[tr], Xte, Y[te], score_fn, kind, batch=batch, select=sel)
                res[f"adam_{t}"] = {"K_first": a["K_first"], "K_patience": a["K_patience"],
                                    **block(a["sawtooth"], "test")}
            cell[f"one_col_{mode}"] = res
            print(variable, mode, json.dumps({s: {m: (v[m] and (round(v[m]["drop_lag1_autocorr"], 2),
                                                                v[m]["n_isolated_dips"])) for m in ("r2", "acc15")}
                                              for s, v in res.items()}))
        out["cells"][variable] = cell
    write_json(R / "p1b_one_column_removal.json", out)


for item in args.items:
    {1: item1, 2: item2}[item]()

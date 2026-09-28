"""Audit robustness checks: four procedural deviations from the paper (Joseph et al. 2026) rerun on stored pooled
activations, each beside the stored value. CPU only.

  python scripts/run_audit_robustness.py --items 1 2 3 4

  1  CV grouping (App. B "5-fold grouped cross-validation"): step-1 layer curve with folds grouped by direction value
     and by start-position cell (1 m grid), V-JEPA and random-init meanpool; speed grouped by speed value.
     -> results/p1a_grouped_cv.json, figures/fig1i_grouped_cv.png
  2  INLP coordinates (C.11): probe sequence on raw (centred, not z-scored) meanpool features, points 8 and 9.
     -> results/p1b_raw_coordinates.json
  3  Sawtooth metric (Fig. 4c, C.10): points 8 and 9 sequences (ridge nested / paper, Adam b64 / full) scored on R²
     and 8-way binned accuracy for both variables. -> results/p1b_sawtooth_metrics.json
  4  Steering evaluation probe (C.12): point-9 direction steering read by an α = 1e-3 ridge and by an Adam
     (lr 1e-3, wd 1e-4) evaluation probe. -> results/p1c_direction_evalprobe_recipe.json
Outer 80/20 split, basis and train-side recipe are the stored ones (splits/split_v1.json, artifacts/inlp).
"""
import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.adam_probe import fit_adam  # noqa: E402
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.inlp import ADAM_C11, basis_path, inlp, load_basis, ridge_fit  # noqa: E402
from wm.paperscale import ci_from_draws, direction_grouped_folds, onset_draws, probe_rows  # noqa: E402
from wm.probes import (ALPHAS, LAST_BLOCK, RESULTS, availability, bootstrap_onset, cv_select_alpha,  # noqa: E402
                       drop_nan_clips, fit_ridge, layer_fraction, layer_matrix, load_activations, split_rows,
                       standardized_layer, targets, write_json)
from wm.robustness import (binned_score, decay_stats, equal_count_edges, in_sample_and_oof, run_rounds,  # noqa: E402
                           sector_folds, start_cells, theta_lookup, value_grouped_folds)
from wm.splits import load_split  # noqa: E402
from wm.steer import evaluate, grouped_folds, random_nulls, rank_matched_null  # noqa: E402
from wm.support import layer_cv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--items", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4, 5])
parser.add_argument("--results", type=Path, default=RESULTS)
parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures")
parser.add_argument("--n-boot", type=int, default=200)
args = parser.parse_args()
R = args.results


def stored(name):
    return json.loads((R / name).read_text())


def data(dataset, variable, model="vjepa2"):
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(dataset, df)
    acts = load_activations(dataset, "meanpool", model)
    tr, te, folds, _ = drop_nan_clips(acts, tr, te, folds)
    return df, Y, kind, score_fn, tr, te, folds, acts


def units_intact(dataset, df, tr, folds):
    """True if no byte-identical-clip unit (frame hash) spans two folds."""
    h = load_split(dataset)["frame_hash"]
    seen = {}
    for i, f in zip(df["id"].to_numpy()[tr], folds):
        if seen.setdefault(h[str(i)], f) != f:
            return False
    return True


# ------------------------------------------------------------------ item 1

def item1():
    out = {"item": "CV grouping (paper App. B l.693: '5-fold grouped cross-validation', group key not stated)",
           "outer_split": "stored 80/20 split (splits/split_v1.json), unchanged; only the 5 inner folds change",
           "schemes": {"stratified": "stored folds (stratified by label value; every value in every fold)",
                       "direction_grouped": "whole direction values held out (64 values dealt round-robin, seed 0)",
                       "start_grouped": "whole 1 m x 1 m start-position cells held out (16 cells on [-2, 2]^2, "
                                        "dealt round-robin, seed 0)",
                       "speed_grouped": "whole speed values held out (64 values dealt round-robin, seed 0)"},
           "onset_rule": "wm.probes.availability (first point with fold-mean CV R2 >= 90% of max over 0..24); CI: "
                         "wm.probes.bootstrap_onset, 200-draw clip bootstrap of the out-of-fold predictions",
           "sets": {}}
    for dataset, variable, extra in (("direction", "direction", ("direction_grouped", "start_grouped")),
                                     ("speed", "speed", ("speed_grouped",))):
        for model in ("vjepa2", "random"):
            df, Y, kind, score_fn, tr, te, folds, acts = data(dataset, variable, model)
            schemes = {"stratified": folds}
            if "direction_grouped" in extra:
                schemes["direction_grouped"] = direction_grouped_folds(df["theta_degrees"].to_numpy()[tr])
            if "start_grouped" in extra:
                schemes["start_grouped"] = value_grouped_folds(
                    start_cells(df["start_x"].to_numpy()[tr], df["start_y"].to_numpy()[tr]))
            if "speed_grouped" in extra:
                schemes["speed_grouped"] = value_grouped_folds(df["speed_mps"].to_numpy()[tr])
            get = lru_cache(maxsize=1)(lambda p: layer_matrix(acts, p))
            ref = stored(f"p1a_{dataset}_{variable}_meanpool{'' if model == 'vjepa2' else '_random'}.json")
            block = {"stored": {"file": f"p1a_{dataset}_{variable}_meanpool{'' if model == 'vjepa2' else '_random'}.json",
                                "block1_r2": ref["layers"][1]["cv_mean"], "block8_r2": ref["layers"][8]["cv_mean"],
                                "onset": ref["availability"]["onset"], "onset_ci": ref["availability"]["onset_ci"],
                                "peak": ref["availability"]["peak"]}}
            for name, f in schemes.items():
                rows, oofs = layer_cv(get, Y, tr, f, score_fn, points=range(26))
                curve = [r["cv_mean"] for r in rows]
                av = availability(curve[:LAST_BLOCK + 1])
                ci, n_none = bootstrap_onset(Y[tr], oofs[:LAST_BLOCK + 1], score_fn, args.n_boot, 0)
                block[name] = {"block1_r2": curve[1], "block8_r2": curve[8], "onset": av["onset"], "onset_ci": ci,
                               "onset_boot_draws_without_onset": n_none, "peak": av["peak"],
                               "peak_score": av["peak_score"], "units_intact": units_intact(dataset, df, tr, f),
                               "fold_sizes": np.bincount(f).tolist(),
                               "curve": curve, "curve_sd": [r["cv_sd"] for r in rows],
                               "alpha": [r["alpha"] for r in rows]}
                print(f"{dataset}/{model}/{name}: block1 {curve[1]:.3f} block8 {curve[8]:.3f} onset {av['onset']} {ci}")
            block["stratified"]["parity_max_abs_diff_vs_stored"] = float(
                np.max(np.abs(np.array(block["stratified"]["curve"]) - [r["cv_mean"] for r in ref["layers"]])))
            out["sets"][f"{dataset}_{model}"] = block
    write_json(R / "p1a_grouped_cv.json", out)
    figure1(out)


def figure1(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"stratified": ("#2a78d6", "-", "stratified (stored)"),
             "direction_grouped": ("#eb6834", "--", "direction-grouped"),
             "start_grouped": ("#1baf7a", ":", "start-cell-grouped"),
             "speed_grouped": ("#eb6834", "--", "speed-grouped"),
             "sector_grouped": ("#8b5cf6", "-.", "45° sector-grouped (pooled OOF R²)")}
    out = {**out, "sets": {k: dict(v) for k, v in out["sets"].items()}}   # plotting copy; the caller's dict is untouched
    for key, res in out.get("sector_grouped", {}).get("sets", {}).items():
        out["sets"][key]["sector_grouped"] = res
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    x = np.array([layer_fraction(p) for p in range(LAST_BLOCK + 1)])
    panels = [("direction_vjepa2", "direction_random", "direction set: direction (random-init faint)"),
              ("speed_vjepa2", "speed_random", "speed set: speed (random-init faint)")]
    for ax, (kv, kr, title) in zip(axes[:2], panels):
        for key, alpha, lw in ((kv, 1.0, 2.0), (kr, 0.45, 1.4)):
            for name, res in out["sets"][key].items():
                if name == "stored":
                    continue
                c, ls, lab = style[name]
                ax.plot(x, res["curve"][:LAST_BLOCK + 1], color=c, ls=ls, lw=lw, alpha=alpha,
                        label=f"{lab}, {'V-JEPA 2' if key == kv else 'random-init'}")
                if res["onset"] is not None:
                    ax.plot(x[res["onset"]], res["curve"][res["onset"]], "v", color=c, alpha=alpha, ms=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("block / 24")
        ax.set_ylabel("5-fold CV R²")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    names = [n for n in ("stratified", "direction_grouped", "start_grouped", "sector_grouped")
             if n in out["sets"]["direction_vjepa2"]]
    for i, key in enumerate(("direction_vjepa2", "direction_random")):
        for j, n in enumerate(names):
            r = out["sets"][key][n]
            ci = r["onset_ci"] or [r["onset"], r["onset"]]
            xpos = i * 5 + j
            ax.errorbar(xpos, r["onset"], yerr=[[r["onset"] - ci[0]], [ci[1] - r["onset"]]], fmt="o",
                        color=style[n][0], capsize=3)
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(["V-JEPA 2", "random-init"])
    ax.set_ylabel("direction onset point [95% CI]")
    ax.set_title("direction onset by fold scheme", fontsize=10)
    fig.tight_layout()
    args.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figures / "fig1i_grouped_cv.png", dpi=200, bbox_inches="tight")


# ------------------------------------------------------------------ item 2

def item2():
    out = {"item": "INLP coordinates (paper C.11 gives no normalisation; raw X implied)",
           "raw": "meanpool features centred with the train mean, no per-feature scaling; alpha re-chosen in raw "
                  "coordinates (nested: per fold by CV over the fold's training rows, as stored; paper protocol: "
                  "all-train CV over wm.probes.ALPHAS)",
           "stopping_rule": "wm.inlp.at_chance: direction R2 < 0.1 or MAE > 80 deg; speed R2 < 0.05 or MAE > 0.9 x "
                            "mean-predictor MAE (the stored rule)", "cells": {}}
    for dataset in ("direction", "speed"):
        df, Y, kind, score_fn, tr, te, folds, acts = data(dataset, dataset)
        for point in (8, 9):
            X = layer_matrix(acts, point)
            mu = X[tr].mean(axis=0)
            Xtr, Xte = X[tr] - mu, X[te] - mu
            cv = cv_select_alpha(Xtr, Y[tr], folds, ALPHAS, score_fn)
            summary, *_ = inlp(Xtr, Y[tr], Xte, Y[te], folds, cv["alpha"], score_fn, kind, protocols=("nested", "paper"))
            ref = stored(f"p1b_{dataset}_{dataset}_meanpool_L{point}.json")
            edge = [a for a in summary["alpha_folds"] + [cv["alpha"]] if a in (ALPHAS[0], ALPHAS[-1])]
            cell = {"raw": {"nested_K": summary["K"], "nested_K_folds": summary["K_folds"],
                            "K_loose": summary["K_loose"], "paper_K": summary["paper"]["K"],
                            "paper_K_loose": summary["paper"]["K_loose"], "alpha_all_train": cv["alpha"],
                            "alpha_folds": summary["alpha_folds"], "alpha_at_grid_edge": bool(edge),
                            "round1_cv_r2": summary["rounds"][0]["cv_r2"],
                            "cv_r2_curve": [r["cv_r2"] for r in summary["rounds"]],
                            "paper_test_r2_curve": [r["test_r2"] for r in summary["paper"]["rounds"]]},
                    "stored_zscored": {"file": f"p1b_{dataset}_{dataset}_meanpool_L{point}.json",
                                       "nested_K": ref["K"], "nested_K_folds": ref["K_folds"],
                                       "K_loose": ref["K_loose"], "paper_K": ref["paper"]["K"],
                                       "paper_K_loose": ref["paper"]["K_loose"], "alpha_all_train": ref["alpha"],
                                       "alpha_folds": ref["alpha_folds"]}}
            out["cells"][f"{dataset}_L{point}"] = cell
            print(f"raw {dataset} L{point}: nested K {summary['K']} (z {ref['K']}), paper K {summary['paper']['K']} "
                  f"(z {ref['paper']['K']}), alpha {cv['alpha']}")
    write_json(R / "p1b_raw_coordinates.json", out)


# ------------------------------------------------------------------ item 3

def adam_fit(kind, batch, seed=0):
    state = {"k": 0}

    def fit(X, Y):
        state["k"] += 1
        return fit_adam(X, Y, ADAM_C11["lr"], ADAM_C11["weight_decay"], ADAM_C11["epochs"][kind], batch,
                        seed + state["k"] - 1)
    return fit


def stats(rounds, keys):
    return {k: decay_stats([r[k] for r in rounds]) for k in keys}


def item3():
    out = {"item": "Sawtooth metric (paper Fig. 4c 'probe accuracy', C.10)",
           "metrics": {"r2": "mean per-column R2 (as stored)",
                       "bacc8": "8-way binned accuracy (v2 bins, wm.robustness): direction 45 deg bins with edges at "
                                "19.6875 + 45 k (label midpoints; 8 labels per bin), true bin from theta_degrees, "
                                "predicted bin from atan2(sin, cos); speed 8 equal-count bins with edges at the "
                                "midpoints between adjacent train-label values. No label sits on an edge. v1 (edges "
                                "0, 45, ... on labels; quantile speed edges on labels) kept in "
                                "p1b_sawtooth_metrics_v1_edgebins.json",
                       "acc15": "direction only: within-15 deg accuracy, the stored direction metric"},
           "statistics": "per metric, as wm.inlp.sawtooth: lag-1 autocorrelation of per-round drops (negative = "
                         "sawtooth) and isolated dips (fall > 0.05 and recover > 0.05 next round), over the same "
                         "rounds as the stored sequence",
           "sequences": {"ridge_nested": "stored ridge sequence, fold-mean over the 5 held-out folds (alpha_folds as "
                                         "stored)",
                         "ridge_paper": "stored ridge sequence fit on all train, scored on test (stored alpha)",
                         "adam_b64": "C.11 Adam sequence, batch 64 (as stored), scored on test",
                         "adam_full": "C.11 Adam sequence, full batch (as stored), scored on test"},
           "cells": {}}
    for dataset in ("direction", "speed"):
        df, Y, kind, _, tr, te, folds, acts = data(dataset, dataset)
        fn = binned_score(kind, None if kind == "circular" else equal_count_edges(Y[tr, 0]),
                          theta_lookup(Y, df["theta_degrees"].to_numpy(float)) if kind == "circular" else None)
        keys = ["r2", "bacc8"] + (["acc15"] if kind == "circular" else [])
        for point in (8, 9):
            Xtr, Xte = standardized_layer(acts, point, tr, te)
            ref = stored(f"p1b_{dataset}_{dataset}_meanpool_L{point}.json")
            cell = {}
            n = len(ref["rounds"])
            per_fold = [run_rounds(Xtr[folds != k], Y[tr][folds != k], ridge_fit(a), fn, Xtr[folds == k],
                                   Y[tr][folds == k], n) for k, a in zip(np.unique(folds), ref["alpha_folds"])]
            nested = [{"round": i + 1, **{k: float(np.mean([f[i][k] for f in per_fold])) for k in keys}}
                      for i in range(n)]
            cell["ridge_nested"] = {"stats": stats(nested, keys), "curve": nested,
                                    "parity_max_abs_diff_r2_vs_stored": float(np.max(np.abs(
                                        np.array([r["r2"] for r in nested]) - [r["cv_r2"] for r in ref["rounds"]])))}
            n = len(ref["paper"]["rounds"])
            paper = run_rounds(Xtr, Y[tr], ridge_fit(ref["alpha"]), fn, Xte, Y[te], n)
            cell["ridge_paper"] = {"stats": stats(paper, keys), "curve": [{k: r[k] for k in ["round"] + keys}
                                                                         for r in paper],
                                   "parity_max_abs_diff_r2_vs_stored": float(np.max(np.abs(
                                       np.array([r["r2"] for r in paper]) - [r["test_r2"] for r in ref["paper"]["rounds"]])))}
            for tag, batch in (("b64", 64), ("full", len(tr))):
                ra = stored(f"p1b_{dataset}_{dataset}_meanpool_L{point}_adam_{tag}.json")
                n = len(ra["rounds"])
                rounds = run_rounds(Xtr, Y[tr], adam_fit(kind, batch), fn, Xte, Y[te], n)
                cell[f"adam_{tag}"] = {"stats": stats(rounds, keys),
                                       "curve": [{k: r[k] for k in ["round"] + keys} for r in rounds],
                                       "parity_max_abs_diff_r2_vs_stored": float(np.max(np.abs(
                                           np.array([r["r2"] for r in rounds]) - [r["test_r2"] for r in ra["rounds"]])))}
            for seq, v in cell.items():
                print(f"{dataset} L{point} {seq}: " + ", ".join(
                    f"{k} ac {v['stats'][k]['drop_lag1_autocorr']:.2f} dips {v['stats'][k]['n_isolated_dips']}"
                    for k in keys) + f" (parity {v['parity_max_abs_diff_r2_vs_stored']:.1e})")
            out["cells"][f"{dataset}_L{point}"] = cell
    write_json(R / "p1b_sawtooth_metrics.json", out)


# ------------------------------------------------------------------ item 4

REPORT_N = (1, 2, 3, 5, 10, 20)


def item4():
    dataset = variable = "direction"
    point = 9
    df, Y, kind, _, tr, te, folds, acts = data(dataset, variable)
    probes = load_basis(basis_path(dataset, variable, point))
    K = len(probes["W"])
    Xtr, Xte = standardized_layer(acts, point, tr, te)
    labels = df["theta_degrees"].to_numpy(float)[te]
    all_targets = np.unique(df["theta_degrees"].to_numpy(float))
    hashes = load_split(dataset)["frame_hash"]
    groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
    gfolds = grouped_folds(groups, 5, 0)
    ref = stored(f"p1c_{dataset}_L{point}.json")
    ref_rm = stored(f"p1c_{dataset}_L{point}_rankmatched.json")
    Ns = list(REPORT_N) + [K]

    def pick(rows, key="mae_to_target"):
        return {str(r["n"]): r[key] for r in rows if r["n"] in Ns}

    def bar(rows):
        return next((r["n"] for r in rows if r["n"] > 0 and r["mae_to_target"] <= 10.0), None)

    recipes = {"ridge_alpha_1e-3": lambda X, Yv: fit_ridge(X, Yv, 1e-3),
               "adam_c11": lambda X, Yv: fit_adam(X, Yv, ADAM_C11["lr"], ADAM_C11["weight_decay"],
                                                  ADAM_C11["epochs"]["circular"], 64, 0)}
    out = {"item": "Steering evaluation probe (paper C.12 l.1245: probe fit on test, R2 = 0.99)",
           "dataset": dataset, "variable": variable, "point": point, "K": K, "single_target": 90.0,
           "n_test": len(te), "space": "train-standardised activations (as stored)",
           "recipes": {"ridge_alpha_1e-3": "closed-form ridge on the test activations, alpha = 1e-3 (near-unregularised)",
                       "adam_c11": "Adam lr 1e-3, coupled weight decay 1e-4, 100 epochs, batch 64, MSE, init seed 0, "
                                   "targets standardised for training (wm.adam_probe.fit_adam)",
                       "stored": "ridge, alpha by 5-fold grouped CV inside test (alpha = 100)"},
           "bar": "first N with single-target MAE-to-target <= 10 deg",
           "nulls": "random_basis: rank-2K random orthonormal basis, own least-squares solve, 20 draws (wm.steer."
                    "random_nulls); rank_matched: rank 2N nested draws (wm.steer.rank_matched_null); empirical p = "
                    "(1 + #draws <= learned) / 21",
           "stored": {"file": f"p1c_{dataset}_L{point}.json", "eval_probe": ref["eval_probe"],
                      "mae_to_target": pick(ref["single"]), "n_to_10deg": bar(ref["single"]),
                      "random_basis_mean": {str(r["n"]): r["random_basis"]["mae_to_target"]["mean"]
                                            for r in ref["random_nulls"]["rows"] if r["n"] in Ns},
                      "random_basis_p": {str(r["n"]): r["random_basis"]["empirical_p_to_target"]
                                         for r in ref["random_nulls"]["rows"] if r["n"] in Ns},
                      "rank_matched_mean": {str(r["n"]): r["rank_matched"]["mae_to_target"]["mean"]
                                            for r in ref_rm["rank_matched_null"]["rows"] if r["n"] in Ns},
                      "rank_matched_p": {str(r["n"]): r["rank_matched"]["empirical_p_to_target"]
                                         for r in ref_rm["rank_matched_null"]["rows"] if r["n"] in Ns}},
           "recipe_results": {}}
    for name, fit in recipes.items():
        W, b, rep = in_sample_and_oof(fit, Xte, Y[te], gfolds, kind)
        res = evaluate(Xte, labels, probes, W, b, kind, 90.0, all_targets)
        nulls = random_nulls(Xte, labels, probes, W, b, kind, 90.0, 20)
        rm = rank_matched_null(Xte, labels, probes, W, b, kind, 90.0, 20)
        out["recipe_results"][name] = {
            "eval_probe": rep, "mae_to_target": pick(res["single"]), "mae_to_true": pick(res["single"], "mae_to_true"),
            "all_targets_mae_to_target": pick(res["all_targets"]), "n_to_10deg": bar(res["single"]),
            "random_basis_mean": {str(r["n"]): r["random_basis"]["mae_to_target"]["mean"] for r in nulls["rows"]
                                  if r["n"] in Ns},
            "random_basis_p": {str(r["n"]): r["random_basis"]["empirical_p_to_target"] for r in nulls["rows"]
                               if r["n"] in Ns},
            "rank_matched_mean": {str(r["n"]): r["rank_matched"]["mae_to_target"]["mean"] for r in rm["rows"]
                                  if r["n"] in Ns},
            "rank_matched_p": {str(r["n"]): r["rank_matched"]["empirical_p_to_target"] for r in rm["rows"]
                               if r["n"] in Ns},
            "n0_mae_to_target": res["single"][0]["mae_to_target"],
            "single_full": [{"n": r["n"], "mae_to_target": r["mae_to_target"]} for r in res["single"]]}
        print(f"{name}: in-sample R2 {rep['in_sample']['r2']:.4f} oof {rep['out_of_fold']['r2_fold_mean']:.3f}; "
              f"MAE {pick(res['single'])}; N to 10 deg {bar(res['single'])}")
    write_json(R / f"p1c_{dataset}_evalprobe_recipe.json", out)


# ------------------------------------------------------------------ item 5

def item5():
    """Sector-grouped folds: direction layer curve with whole 45 deg sectors held out, appended to p1a_grouped_cv.json
    under 'sector_grouped' (existing keys rewritten byte-identically; no write_json, so the stored provenance stays)."""
    import subprocess
    path = R / "p1a_grouped_cv.json"
    text = path.read_text()
    out = json.loads(text)
    assert json.dumps(out, indent=1) == text, "stored file does not round-trip; refusing to rewrite it"
    sec = {"scheme": "whole contiguous 45 deg direction sectors held out: the 8 wm.robustness.direction_bin sectors "
                     "(edges at 19.6875 + 45 k, 8 directions each) dealt round-robin to 5 folds (2, 2, 2, 1, 1 "
                     "sectors per fold, seed 0); outer split as the other schemes",
           "score": "alpha chosen by, and curve / onset read from, the R2 of all out-of-fold predictions together "
                    "(wm.paperscale.probe_rows select='pooled', as the 8-direction grouped runs): a fold holding one "
                    "or two 45 deg sectors has little sin/cos variance of its own, so the fold-mean R2 the stored rows "
                    "use is not defined in any useful sense here (it reaches -25; kept under fold_mean_diagnostic). "
                    "For the stored schemes every fold spans the circle and fold-mean ~ pooled",
           "onset_rule": "wm.probes.availability on the pooled curve; CI: 200-draw clip bootstrap of the fixed "
                         "out-of-fold predictions (wm.paperscale.onset_draws, the same statistic as the stored CIs)",
           "sets": {}}
    for model in ("vjepa2", "random"):
        df, Y, kind, score_fn, tr, te, folds, acts = data("direction", "direction", model)
        f = sector_folds(df["theta_degrees"].to_numpy()[tr])
        fits = [probe_rows(layer_matrix(acts, p), tr, Y, f, score_fn, select="pooled") for p in range(26)]
        curve = [r["pooled"] for r in fits]
        oofs = [r["oof"] for r in fits]
        av = availability(curve[:LAST_BLOCK + 1])
        ci, n_none = ci_from_draws(onset_draws(Y[tr], oofs, score_fn, args.n_boot, 0))
        fm = [r["curve_fold_mean"][int(np.argmax(r["curve_pooled"]))] for r in fits]
        sec["sets"][f"direction_{model}"] = {
            "block1_r2": curve[1], "block8_r2": curve[8], "onset": av["onset"], "onset_ci": ci,
            "onset_boot_draws_without_onset": n_none, "peak": av["peak"], "peak_score": av["peak_score"],
            "units_intact": units_intact("direction", df, tr, f), "fold_sizes": np.bincount(f).tolist(),
            "curve": curve, "alpha": [r["alpha"] for r in fits],
            "cv_mae_pooled": [score_fn(Y[tr], o)["mae"] for o in oofs],
            "fold_mean_diagnostic": {"curve_at_pooled_alpha": fm, "block1": fm[1], "block8": fm[8]}}
        print(f"direction/{model}/sector_grouped: block1 {curve[1]:.3f} block8 {curve[8]:.3f} onset {av['onset']} {ci}")
    sec["provenance"] = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                  cwd=PROJECT_ROOT).stdout.strip(),
                         "timestamp_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}
    out["sector_grouped"] = sec
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    figure1(out)


for item in args.items:
    {1: item1, 2: item2, 3: item3, 4: item4, 5: item5}[item]()

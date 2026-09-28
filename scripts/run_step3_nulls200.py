"""Point-9 direction steering nulls at 200 draws (stored files: 20). wm.steer.rank_matched_null with n_draws = 200, seed 0,
under run_steering's paper protocol (same split, standardisation, evaluation probe, single target 90 deg, paper arm):
its rank_matched rows are the rank-2N null and its random_basis_full_rank rows are exactly random_nulls' rank-2K
random_basis null (same rng stream). Draws 1-20 of the 200 are the stored 20 draws (asserted), so nothing is redrawn.
Bases: the stored ridge basis (artifacts/inlp/direction_direction_L9.npz, K = 37) and the Adam C.11 basis
(artifacts/inlp/direction_direction_L9_adam_b64.npz, K = 84). Evaluation-probe variants on the ridge basis (as
results/p1c_direction_evalprobe_recipe.json and the split-half probes of p1c_direction_L9_rank2_covweighted.json), p at
every N. Writes results/p1c_direction_L9_nulls200.json; the stored 20-draw files are not touched.

  python scripts/run_step3_nulls200.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.adam_probe import fit_adam  # noqa: E402
from wm.inlp import ADAM_C11, INLP_DIR, basis_path, load_basis  # noqa: E402
from wm.probes import (RESULTS, drop_nan_clips, fit_ridge, load_activations, load_table, split_rows,  # noqa: E402
                       standardized_layer, targets, write_json)
from wm.provenance import sha256_file  # noqa: E402
from wm.robustness import in_sample_and_oof  # noqa: E402
from wm.splits import load_split  # noqa: E402
from wm.steer import empirical_p, eval_probe_cv, grouped_folds, rank_matched_null, stratified_halves  # noqa: E402

DATASET, POINT, SINGLE, N_DRAWS, SEED = "direction", 9, 90.0, 200, 0
N_REPORT = (1, 2, 3, 5, 10, 12, 15, 18, 20, 25, 30)
NULLS = {"rank_matched": "rank_matched", "rank_2K": "random_basis_full_rank"}

df = load_table(DATASET)
Y, kind, _ = targets(df, DATASET)
tr, te, folds = split_rows(DATASET, df)
acts = load_activations(DATASET, "meanpool", "vjepa2", None)
tr, te, folds, n_nan = drop_nan_clips(acts, tr, te, folds)
Xtr, Xte = standardized_layer(acts, POINT, tr, te)
labels = df["theta_degrees"].to_numpy(float)[te]
hashes = load_split(DATASET)["frame_hash"]
groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
eval_W, eval_b, eval_report = eval_probe_cv(Xte, Y[te], kind, groups=groups)

stored = {"ridge": json.loads((RESULTS / "p1c_direction_L9.json").read_text()),
          "adam": json.loads((RESULTS / "p1c_direction_L9_adam_basis.json").read_text())}
stored_rm = {"ridge": json.loads((RESULTS / "p1c_direction_L9_rankmatched.json").read_text())["rank_matched_null"],
             "adam": stored["adam"]["rank_matched_null"]}
for st in stored.values():
    assert st["eval_probe"]["alpha"] == eval_report["alpha"]
    assert abs(st["eval_probe"]["in_sample"]["r2"] - eval_report["in_sample"]["r2"]) < 1e-9
npz = {"ridge": basis_path(DATASET, DATASET, POINT), "adam": Path(INLP_DIR) / "direction_direction_L9_adam_b64.npz"}
bases = {name: load_basis(p) for name, p in npz.items()}


def first_n(rows, key, thr, sustained=False):
    """First N with p < thr; sustained = first N from which every later N of the grid also has p < thr."""
    ns = [r["n"] for r in rows]
    ok = [r[key] < thr for r in rows]
    if not sustained:
        return next((n for n, o in zip(ns, ok) if o), None)
    return next((n for i, n in enumerate(ns) if all(ok[i:])), None)


def summarise(null, learned_key=None):
    """Per-N rows from rank_matched_null output: learned, and per null mean / sd / median / p (200) / p (first 20)."""
    out = []
    for r in null["rows"]:
        lrn = r["learned"]["mae_to_target"]
        row = {"n": r["n"], "learned_mae_to_target": lrn}
        for short, key in NULLS.items():
            d = np.array(r[key]["mae_to_target"]["draws"])
            row[short] = {"mean": r[key]["mae_to_target"]["mean"], "sd": r[key]["mae_to_target"]["sd"],
                          "median": float(np.median(d)), "p": r[key]["empirical_p_to_target"],
                          "p_first20": empirical_p(lrn, d[:20]), "n_draws_le_learned": int((d <= lrn).sum())}
        out.append(row)
    return out


def thresholds(rows):
    return {short: {f"p_lt_{t}": {"first_n": first_n([{"n": r["n"], "p": r[short]["p"]} for r in rows], "p", t),
                                  "first_n_sustained": first_n([{"n": r["n"], "p": r[short]["p"]} for r in rows], "p", t,
                                                               sustained=True)}
                    for t in (0.01, 0.05)}
            for short in NULLS}


headline = {}
for name, probes in bases.items():
    K = len(probes["W"])
    null = rank_matched_null(Xte, labels, probes, eval_W, eval_b, kind, SINGLE, N_DRAWS, SEED)
    rows = summarise(null)
    # reproduction checks against the stored 20-draw files
    st_single = {r["n"]: r["mae_to_target"] for r in stored[name]["single"]}
    st_rm = {r["n"]: r for r in stored_rm[name]["rows"]}
    st_rb = {r["n"]: r for r in stored[name]["random_nulls"]["rows"]}
    for r_new, r in zip(null["rows"], rows):
        n = r["n"]
        assert abs(r["learned_mae_to_target"] - st_single[n]) < 1e-9, (name, n)
        new_rm = np.round(r_new["rank_matched"]["mae_to_target"]["draws"][:20], 4)
        new_rb = np.round(r_new["random_basis_full_rank"]["mae_to_target"]["draws"][:20], 4)
        assert np.allclose(new_rm, st_rm[n]["rank_matched"]["mae_to_target"]["draws"], atol=2e-4), (name, n)
        assert np.allclose(new_rb, st_rb[n]["random_basis"]["mae_to_target"]["draws"], atol=2e-4), (name, n)
        assert r["rank_matched"]["p_first20"] == st_rm[n]["rank_matched"]["empirical_p_to_target"], (name, n)
        assert r["rank_2K"]["p_first20"] == st_rb[n]["random_basis"]["empirical_p_to_target"], (name, n)
    headline[name] = {"K": K, "rank_2K": 2 * K, "basis_file": str(npz[name].relative_to(npz[name].parents[2])),
                      "basis_sha256": sha256_file(npz[name]), "n_grid": null["n_grid"],
                      "table": [r for r in rows if r["n"] in N_REPORT or r["n"] == K],
                      "all_n": rows, "first_n": thresholds(rows),
                      "draws_mae_to_target": {str(r["n"]): {s: r[k]["mae_to_target"]["draws"] for s, k in NULLS.items()}
                                              for r in null["rows"]}}
    print(name, "K", K)
    for r in headline[name]["table"]:
        print(f"  N={r['n']:3d} learned {r['learned_mae_to_target']:6.2f} | rank-2N {r['rank_matched']['mean']:6.2f}"
              f" ± {r['rank_matched']['sd']:5.2f} p={r['rank_matched']['p']:.4f} | rank-2K {r['rank_2K']['mean']:6.2f}"
              f" ± {r['rank_2K']['sd']:5.2f} p={r['rank_2K']['p']:.4f}")
    print(" ", headline[name]["first_n"])

# evaluation-probe variants (ridge basis): p at every N
ridge = bases["ridge"]
gfolds = grouped_folds(groups, 5, 0)
recipe = json.loads((RESULTS / "p1c_direction_evalprobe_recipe.json").read_text())
fits = {"ridge_alpha_1e-3": lambda X, Yv: fit_ridge(X, Yv, 1e-3),
        "adam_c11": lambda X, Yv: fit_adam(X, Yv, ADAM_C11["lr"], ADAM_C11["weight_decay"],
                                           ADAM_C11["epochs"]["circular"], 64, 0)}
variants = {}
for vname, fit in fits.items():
    W, b, rep = in_sample_and_oof(fit, Xte, Y[te], gfolds, kind)
    null = rank_matched_null(Xte, labels, ridge, W, b, kind, SINGLE, N_DRAWS, SEED)
    rows = summarise(null)
    ref = recipe["recipe_results"][vname]
    for r in rows:
        if str(r["n"]) in ref["mae_to_target"]:
            assert abs(r["learned_mae_to_target"] - ref["mae_to_target"][str(r["n"])]) < 1e-6, (vname, r["n"])
            assert r["rank_matched"]["p_first20"] == ref["rank_matched_p"][str(r["n"])], (vname, r["n"])
            assert r["rank_2K"]["p_first20"] == ref["random_basis_p"][str(r["n"])], (vname, r["n"])
    variants[vname] = {"eval_probe": rep, "all_n": rows, "first_n": thresholds(rows),
                       "first_n_first20_draws": {s: first_n([{"n": r["n"], "p": r[s]["p_first20"]} for r in rows], "p",
                                                            0.05) for s in NULLS}}

# split-half: probe fit on one stratified half reads the other half's steered clips; pooled over both directions
in_a = stratified_halves(groups, labels)
halves = {}
for hname, fit_rows, steer_rows in (("A_to_B", in_a, ~in_a), ("B_to_A", ~in_a, in_a)):
    W_h, b_h, rep_h = eval_probe_cv(Xte[fit_rows], Y[te][fit_rows], kind, groups=groups[fit_rows])
    halves[hname] = (rank_matched_null(Xte[steer_rows], labels[steer_rows], ridge, W_h, b_h, kind, SINGLE, N_DRAWS,
                                       SEED), rep_h, int(steer_rows.sum()))
nA, nB = halves["A_to_B"][2], halves["B_to_A"][2]
pooled = []
for ra, rb in zip(halves["A_to_B"][0]["rows"], halves["B_to_A"][0]["rows"]):
    assert ra["n"] == rb["n"]
    lrn = (nA * ra["learned"]["mae_to_target"] + nB * rb["learned"]["mae_to_target"]) / (nA + nB)
    row = {"n": ra["n"], "learned_mae_to_target": lrn}
    for short, key in NULLS.items():
        d = (nA * np.array(ra[key]["mae_to_target"]["draws"]) + nB * np.array(rb[key]["mae_to_target"]["draws"])) / (nA + nB)
        row[short] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "median": float(np.median(d)),
                      "p": empirical_p(lrn, d), "p_first20": empirical_p(lrn, d[:20]),
                      "n_draws_le_learned": int((d <= lrn).sum())}
    pooled.append(row)
variants["split_half"] = {
    "rule": "test split into stratified halves (wm.steer.stratified_halves); the probe fit on one half (alpha by grouped "
            "CV inside it) reads the other half's steered clips; pooled MAE per draw = clip-weighted mean of the two "
            "directions' MAE for the same draw j (both calls draw the same R_j: same seed, same d and rank); pooled "
            "draws use the 4-decimal stored draws",
    "eval_probes": {h: halves[h][1] for h in halves}, "n_steered": {"A_to_B": nA, "B_to_A": nB},
    "all_n": pooled, "first_n": thresholds(pooled),
    "first_n_first20_draws": {s: first_n([{"n": r["n"], "p": r[s]["p_first20"]} for r in pooled], "p", 0.05)
                              for s in NULLS},
    "per_direction": {h: {"all_n": summarise(halves[h][0])} for h in halves}}
for vname, v in variants.items():
    print(vname, v["first_n"], "first-20:", v.get("first_n_first20_draws"))

out = {"dataset": DATASET, "variable": DATASET, "point": POINT, "single_target": SINGLE, "n_test": len(te),
       "all_nan_clips_excluded": n_nan, "n_draws": N_DRAWS, "seed": SEED,
       "protocol": "run_steering's paper protocol (C.12): evaluation probe = ridge, alpha by grouped 5-fold CV inside "
                   "test (alpha = 100), fit on the steered test clips; unit target (sin 90, cos 90) required of every "
                   "probe; wm.steer.rank_matched_null (n_draws 200, seed 0)",
       "nulls": {"rank_matched": "rank 2N: R_j[:, :2N] of R_j = QR(Gaussian [d, 2K]), own least-squares solve",
                 "rank_2K": "rank 2K: the full R_j (identical to random_nulls' random_basis draws, same seed)"},
       "p_rule": "p = (1 + #{draws with MAE-to-target <= learned}) / 201; floor 1/201 = 0.004975; p_first20 = the "
                 "same rule on draws 1-20 (= the stored 20-draw p, asserted for the headline bases and the recipe "
                 "file's N)",
       "first_n_rule": "first_n = first N of the grid with p < threshold; first_n_sustained = first N from which "
                       "every later N of the grid also has p < threshold",
       "reproduces_stored": {"learned_mae_to_target": True, "draws_1_to_20": True, "p_first20_equals_stored": True},
       "eval_probe": eval_report, "bases": headline,
       "evalprobe_variants": {"rule": "ridge basis (K = 37); evaluation probe swapped; p at every N of the grid; "
                                      "ridge_alpha_1e-3 and adam_c11 fit on all 300 test clips as "
                                      "p1c_direction_evalprobe_recipe.json (learned MAE and first-20 p asserted equal "
                                      "at its N)", **variants}}
write_json(RESULTS / "p1c_direction_L9_nulls200.json", out)

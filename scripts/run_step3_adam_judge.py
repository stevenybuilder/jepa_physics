"""QA #475: point-9 direction steering, Adam C.11 basis vs ridge basis, each read by four evaluation probes (judges).

The Part 1 claim compares the ridge basis (5 probes to 10 deg) with the Adam C.11 basis (18 probes), both judged by
our ridge (alpha 100) evaluation probe; results/p1c_direction_evalprobe_recipe.json showed the judge alone moves the
ridge basis from 5 to 3 (ridge alpha 1e-3) or 4 (Adam judge). This scores both bases under every judge:
  ridge_alpha100   wm.steer.eval_probe_cv (alpha by grouped 5-fold CV inside test = 100; the stored judge)
  ridge_alpha_1e-3 closed-form ridge, alpha 1e-3, fit on all test clips (as the recipe file)
  adam_c11         wm.adam_probe.fit_adam, lr 1e-3, weight decay 1e-4 (coupled), 100 epochs, batch 64, init seed 0,
                   targets standardised for training (as the recipe file); the paper gives no batch size
  split_half       wm.steer.stratified_halves: ridge probe (eval_probe_cv) fit on one half reads the other half's
                   steered clips, both directions, per-clip errors pooled
All judges but split_half are fit on the UNSTEERED test clips and read their steered versions (C.12). Single target
90 deg, paper arm (unit target required of each probe), every N from 1 to K. 95% CIs: clip bootstrap (2000 draws,
seed 0) of the per-clip error, and of n_to_10deg (first N whose resampled mean error <= 10 deg). Same-rank null:
wm.steer.rank_matched_null (20 draws, seed 0, n_max 18) under the Adam judge. Stored activations, splits/split_v1.json,
CPU. Writes results/p1c_direction_L9_adam_judge.json.

  python scripts/run_step3_adam_judge.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.adam_probe import fit_adam  # noqa: E402
from wm.inlp import ADAM_C11, INLP_DIR, basis_path, load_basis  # noqa: E402
from wm.probes import (RESULTS, drop_nan_clips, fit_ridge, load_activations, load_table, predict, split_rows,  # noqa: E402
                       standardized_layer, targets, write_json)
from wm.provenance import sha256_file  # noqa: E402
from wm.robustness import in_sample_and_oof  # noqa: E402
from wm.splits import load_split  # noqa: E402
from wm.steer import (build_basis, decode, distance, encode, eval_probe_cv, grouped_folds, rank_matched_null,  # noqa: E402
                      steer, steering_operator, stratified_halves)

DATASET, POINT, SINGLE, SEED, N_BOOT, BAR, N_NULL = "direction", 9, 90.0, 0, 2000, 10.0, 20
N_REPORT = (5, 18)
t0 = time.time()

df = load_table(DATASET)
Y, kind, _ = targets(df, DATASET)
tr, te, folds = split_rows(DATASET, df)
acts = load_activations(DATASET, "meanpool", "vjepa2", None)
tr, te, folds, n_nan = drop_nan_clips(acts, tr, te, folds)
Xtr, Xte = standardized_layer(acts, POINT, tr, te)
Yte = Y[te]
labels = df["theta_degrees"].to_numpy(float)[te]
hashes = load_split(DATASET)["frame_hash"]
groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
gfolds = grouped_folds(groups, 5, SEED)
npz = {"ridge": basis_path(DATASET, DATASET, POINT), "adam": Path(INLP_DIR) / "direction_direction_L9_adam_b64.npz"}
bases = {k: load_basis(p) for k, p in npz.items()}
y_star = encode(SINGLE, kind)

# ---- judges (W, b, report); split_half handled separately
judges = {}
W, b, rep = eval_probe_cv(Xte, Yte, kind, groups=groups)
judges["ridge_alpha100"] = (W, b, rep)
for name, fit in {"ridge_alpha_1e-3": lambda X, Yv: fit_ridge(X, Yv, 1e-3),
                  "adam_c11": lambda X, Yv: fit_adam(X, Yv, ADAM_C11["lr"], ADAM_C11["weight_decay"],
                                                     ADAM_C11["epochs"]["circular"], 64, SEED)}.items():
    judges[name] = in_sample_and_oof(fit, Xte, Yte, gfolds, kind)
in_a = stratified_halves(groups, labels)
halves = {h: eval_probe_cv(Xte[m], Yte[m], kind, groups=groups[m]) for h, m in (("fit_A", in_a), ("fit_B", ~in_a))}


def per_clip_errors(probes):
    """err[judge] = [K + 1, n_test] per-clip |readout - 90| after steering with the first n probes (n = 0..K)."""
    V = build_basis(probes["W"])
    K = len(probes["W"])
    err = {j: np.zeros((K + 1, len(te))) for j in list(judges) + ["split_half"]}
    for n in range(K + 1):
        Xs = steer(Xte, V, probes, y_star, n, steering_operator(V, probes, n) if n else None)
        for j, (Wj, bj, _) in judges.items():
            err[j][n] = distance(decode(predict(Xs, Wj, bj), kind), SINGLE, kind)
        e = np.empty(len(te))
        e[~in_a] = distance(decode(predict(Xs[~in_a], *halves["fit_A"][:2]), kind), SINGLE, kind)
        e[in_a] = distance(decode(predict(Xs[in_a], *halves["fit_B"][:2]), kind), SINGLE, kind)
        err["split_half"][n] = e
    return err


def first_n(curve):
    hit = np.nonzero(curve[1:] <= BAR)[0]
    return int(hit[0] + 1) if len(hit) else None


def first_n_sustained(curve):
    """First N >= 1 from which every later N (to K) also has mean error <= BAR (curves need not be monotone)."""
    above = np.nonzero(curve[1:] > BAR)[0]
    n = int(above[-1] + 2) if len(above) else 1
    return n if n < len(curve) else None


rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(te), size=(N_BOOT, len(te)))
out_bases = {}
for bname, probes in bases.items():
    K = len(probes["W"])
    err = per_clip_errors(probes)
    res = {"K": K, "weights_file": str(npz[bname].relative_to(npz[bname].parents[2])),
           "weights_sha256": sha256_file(npz[bname]), "judges": {}}
    for j, e in err.items():
        curve = e.mean(1)
        boot = e[:, boot_idx].mean(2)                       # [K + 1, N_BOOT]
        bn = [first_n(boot[:, i]) for i in range(N_BOOT)]
        bn_ok = np.array([v for v in bn if v is not None])
        bs = np.array([first_n_sustained(boot[:, i]) or -1 for i in range(N_BOOT)])
        row = {"n_to_10deg": first_n(curve),
               "n_to_10deg_ci95": [int(np.percentile(bn_ok, 2.5)), int(np.percentile(bn_ok, 97.5))],
               "n_to_10deg_boot_never": int(sum(v is None for v in bn)),
               "n_to_10deg_sustained": first_n_sustained(curve),
               "n_to_10deg_sustained_ci95": [int(np.percentile(bs[bs > 0], 2.5)), int(np.percentile(bs[bs > 0], 97.5))],
               "mae_to_target_n0": float(curve[0]),
               "at_n": {str(n): {"mae_to_target": float(curve[n]),
                                 "ci95": [float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))]}
                        for n in N_REPORT if n <= K},
               "curve": [round(float(v), 4) for v in curve]}
        res["judges"][j] = row
    out_bases[bname] = res
    print(bname, {j: (r["n_to_10deg"], r["n_to_10deg_sustained"], r["n_to_10deg_ci95"], {n: round(v["mae_to_target"], 2)
                                                               for n, v in r["at_n"].items()})
                  for j, r in res["judges"].items()})

# ---- reproduce stored numbers
st_ridge = json.loads((RESULTS / "p1c_direction_L9.json").read_text())
st_adam = json.loads((RESULTS / "p1c_direction_L9_adam_basis.json").read_text())
recipe = json.loads((RESULTS / "p1c_direction_evalprobe_recipe.json").read_text())
for bname, st in (("ridge", st_ridge), ("adam", st_adam)):
    assert st["eval_probe"]["alpha"] == judges["ridge_alpha100"][2]["alpha"]
    for r in st["single"]:
        assert abs(out_bases[bname]["judges"]["ridge_alpha100"]["curve"][r["n"]] - r["mae_to_target"]) < 1e-3, (bname, r)
for j in ("ridge_alpha_1e-3", "adam_c11"):
    for n, v in recipe["recipe_results"][j]["mae_to_target"].items():
        assert abs(out_bases["ridge"]["judges"][j]["curve"][int(n)] - v) < 1e-3, (j, n)
    assert out_bases["ridge"]["judges"][j]["n_to_10deg"] == recipe["recipe_results"][j]["n_to_10deg"]

# ---- same-rank (rank 2N) random-basis null under the Adam judge
nulls = {}
for bname, probes in bases.items():
    rm = rank_matched_null(Xte, labels, probes, *judges["adam_c11"][:2], kind, SINGLE, N_NULL, SEED, n_max=max(N_REPORT))
    nulls[bname] = {str(r["n"]): {"learned": r["learned"]["mae_to_target"],
                                  "rank_matched_median": float(np.median(r["rank_matched"]["mae_to_target"]["draws"]))
                                  if "draws" in r["rank_matched"]["mae_to_target"] else None,
                                  "rank_matched_band": r["rank_matched"]["mae_to_target"],
                                  "empirical_p": r["rank_matched"]["empirical_p_to_target"]}
                    for r in rm["rows"] if r["n"] in N_REPORT}
    print("null (adam judge)", bname, {n: (round(v["learned"], 2), v["rank_matched_median"], v["empirical_p"])
                                       for n, v in nulls[bname].items()})

wall = time.time() - t0
table = {b: {j: {"n_to_10deg": r["n_to_10deg"], "n_to_10deg_ci95": r["n_to_10deg_ci95"],
                 "n_to_10deg_sustained": r["n_to_10deg_sustained"],
                 "n_to_10deg_sustained_ci95": r["n_to_10deg_sustained_ci95"],
                 **{f"mae_n{n}": round(v["mae_to_target"], 2) for n, v in r["at_n"].items()},
                 **{f"mae_n{n}_ci95": [round(x, 2) for x in v["ci95"]] for n, v in r["at_n"].items()}}
             for j, r in res["judges"].items()} for b, res in out_bases.items()}
out = {"qa": "#475", "dataset": DATASET, "variable": DATASET, "point": POINT, "single_target": SINGLE,
       "n_test": len(te), "all_nan_clips_excluded": n_nan, "arm": "paper (unit target required of each probe)",
       "bar": f"n_to_10deg = first N >= 1 with mean single-target MAE-to-target <= {BAR} deg (the stored rule); "
              "n_to_10deg_sustained = first N from which every later N up to K stays <= the bar (the Adam judge's "
              "curve on the Adam basis is not monotone: 43.5 deg at N = 1, below 10 at N = 11-15, above at 16-18)",
       "judges": {"ridge_alpha100": {"rule": "wm.steer.eval_probe_cv on all test clips (stored judge)",
                                     "report": judges["ridge_alpha100"][2]},
                  "ridge_alpha_1e-3": {"rule": "fit_ridge alpha 1e-3 on all test clips; OOF by grouped 5 folds",
                                       "report": judges["ridge_alpha_1e-3"][2]},
                  "adam_c11": {"rule": "fit_adam lr 1e-3, wd 1e-4 coupled, 100 epochs, batch 64 (not in paper), "
                                       "init seed 0, MSE on standardised (sin, cos); OOF by grouped 5 folds",
                               "report": judges["adam_c11"][2]},
                  "split_half": {"rule": "stratified halves (seed 0); eval_probe_cv on one half reads the other "
                                         "half's steered clips; per-clip errors pooled over both directions",
                                 "reports": {h: v[2] for h, v in halves.items()}, "n_half_A": int(in_a.sum())}},
       "ci_rule": f"clip bootstrap, {N_BOOT} draws, seed {SEED}, same resamples for every basis x judge; "
                  "n_to_10deg_ci95 from the resampled curves (draws never reaching the bar counted separately)",
       "table": table, "bases": out_bases,
       "adam_judge_rank_matched_null": {"rule": f"wm.steer.rank_matched_null, {N_NULL} draws, seed {SEED}, rank 2N "
                                                 "random subspace of a QR'd Gaussian [d, 2K] (K differs by basis, so "
                                                 "draws differ between bases); p = (1 + #draws <= learned) / 21",
                                        "rows": nulls},
       "reproduces_stored": {"ridge_alpha100_both_bases_vs_p1c_single": True,
                             "ridge_basis_recipe_judges_vs_evalprobe_recipe": True},
       "wall_time_s": round(wall, 1)}
write_json(RESULTS / "p1c_direction_L9_adam_judge.json", out)
print(json.dumps(table, indent=1))
print(f"wall {wall:.1f}s")

"""Sigmoid onset criterion (paper authors' rebuttal) on the stored layer curves, beside the stored 90%-of-max onset.
CPU only, reads results/ only. Writes results/p1a_sigmoid_onset.json.

  python scripts/run_sigmoid_onset.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import RESULTS, write_json  # noqa: E402
from wm.provenance import sha256_file  # noqa: E402
from wm.sigmoid_onset import LAST_BLOCK, criterion, fit_logistic4, onset90  # noqa: E402

N_BOOT, SEED = 2000, 0
PAPER_POINT = 9          # the paper's layer 8 of 24 (~one third depth) = our point 9 (frac 0.375)
files = {}


def load(name):
    files[name] = sha256_file(RESULTS / name)
    return json.loads((RESULTS / name).read_text())


curves = {}
for var in ("direction", "speed", "acceleration"):
    for tag, suf in (("vjepa2", ""), ("random_init", "_random")):
        name = f"p1a_{var}_{var}_meanpool{suf}.json"
        d = load(name)
        L = [r for r in d["layers"] if r["point"] <= LAST_BLOCK]           # blocks 0..24; post-LN point excluded
        curves[f"meanpool/{var}/{tag}"] = {"file": name, "metric": "layers[].cv_mean (5-fold CV R2, train)",
                                           "x": [r["point"] for r in L], "y": [r["cv_mean"] for r in L],
                                           "stored_onset90": d["availability"]["onset"]}
perpatch = {"supplied/vjepa2": "p1a_perpatch_direction_vjepa2.json",
            "supplied_constvel/vjepa2": "p1a_perpatch_direction_vjepa2_constvel.json",
            "hard_seed0/vjepa2": "p1a_perpatch_direction_vjepa2_hard.json",
            "hard_seed1/vjepa2": "p1a_perpatch_direction_vjepa2_hard_seed1.json",
            "hard_seed2/vjepa2": "p1a_perpatch_direction_vjepa2_hard_seed2.json",
            "paper_layout/vjepa2": "p1a_perpatch_direction_vjepa2_paper_layout.json",
            "supplied/random_init": "p1a_perpatch_direction_random.json",
            "hard_seed0/random_init": "p1a_perpatch_direction_random_hard.json",
            "hard_seed1/random_init": "p1a_perpatch_direction_random_hard_seed1.json"}
for key, name in perpatch.items():
    d = load(name)
    for metric in ("perpos_mean_r2", "cross_half_r2", "meanpool_r2"):
        curves[f"perpatch/{key}/{metric}"] = {"file": name, "metric": f"curves.{metric} (held-out test R2)",
                                              "x": d["points"], "y": d["curves"][metric],
                                              "stored_onset90": d["onsets"][metric]["onset"]}

rows = {}
for key, c in curves.items():
    res = criterion(c["x"], c["y"])
    res["onset90_point"] = onset90(c["x"], c["y"])       # recomputed on the sampled points
    res["onset90_frac"] = res["onset90_point"] / LAST_BLOCK
    res["onset90_stored"] = c["stored_onset90"]            # as stored (None = the stored rule found no onset)
    res["within_1_point_of_paper_depth"] = abs(res["inflection_point"] - PAPER_POINT) <= 1.0
    rows[key] = {"file": c["file"], "metric": c["metric"], "points": c["x"], "curve": [round(v, 4) for v in c["y"]],
                 **res}

# fold bootstrap: per-fold 4PL fits, and a 2000-draw bootstrap over folds (resample the 5 fold curves with
# replacement, average, refit)
folds = load("p1a_perpatch_hard_folds.json")
rng = np.random.default_rng(SEED)
boot = {}
for scheme, block in (("random_folds", folds["sets"]), ("start_grouped_folds", folds["folds_start_grouped"])):
    for set_name, s in block.items():
        x = s["points"]
        for metric in ("perpos_mean_r2", "cross_half_r2", "meanpool_r2"):
            F = np.array([f["curves"][metric] for f in s["per_fold"]])
            per_fold = [fit_logistic4(x, y) for y in F]
            idx = np.sort(rng.integers(0, len(F), size=(N_BOOT, len(F))), axis=1)
            cache = {}                                  # a resample's mean curve depends only on its fold multiset
            for ii in map(tuple, idx):
                if ii not in cache:
                    cache[ii] = fit_logistic4(x, F[list(ii)].mean(axis=0))["x0"]
            draws = np.array([cache[ii] for ii in map(tuple, idx)])
            mean_fit = criterion(x, F.mean(axis=0))
            boot[f"{scheme}/{set_name}/{metric}"] = {
                "points": x, "n_folds": len(F),
                "per_fold_inflection_point": [round(f["x0"], 3) for f in per_fold],
                "per_fold_fit_r2": [round(f["r2"], 3) for f in per_fold],
                "per_fold_onset90": [f["onsets"][metric] for f in s["per_fold"]],
                "fold_mean_curve": {k: mean_fit[k] for k in ("inflection_point", "inflection_frac", "fit_r2",
                                                             "rise_fit_pp", "peak_above_chance_pp", "pass",
                                                             "accept_rebuttal_rule")},
                "boot_inflection_point": {"median": float(np.median(draws)),
                                          "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                                          "sd": float(draws.std(ddof=1))},
                "boot_inflection_frac_ci95": [float(np.percentile(draws, 2.5)) / LAST_BLOCK,
                                              float(np.percentile(draws, 97.5)) / LAST_BLOCK]}

out = {"criterion": {
    "source": "sonia_joseph.md l.153-154 (summary of the paper authors' OpenReview rebuttal, "
              "https://openreview.net/forum?id=aijGVmEG9Y; the thread itself was not re-read here)",
    "quote": "Asked to define \"transition point\", the rebuttal introduced a sigmoid criterion: R² > 0.9, inflection "
             "≤ 50% depth, peak ≥ 15 pp above chance",
    "applied_as": "4-parameter logistic y = lo + (hi - lo) / (1 + exp(-k (x - x0))) in layer index x (points 0..24, "
                  "post-LN point excluded), k > 0, x0 bounded to [0, 24], least squares with multi-start; fit R² on "
                  "the sampled points; depth fraction = x0 / 24; chance = R² 0 (every curve here is an R² curve); "
                  "peak = observed max over sampled points. accept_rebuttal_rule uses 'peak >= 15 pp above chance' as "
                  "the source words it; accept_rise_variant replaces it with fitted rise (hi - lo) >= 15 pp",
    "paper_depth": f"the paper's transition at about one third depth = its layer 8 = our point {PAPER_POINT} "
                   f"(frac {PAPER_POINT / LAST_BLOCK:.3f}); within_1_point_of_paper_depth = |x0 - {PAPER_POINT}| <= 1",
    "onset90": "stored rule: first sampled point with the curve >= 90% of its max over sampled points"},
    "curves": rows, "fold_bootstrap": {"rule": f"{N_BOOT} draws, rng seed {SEED}: resample the 5 fold curves with "
                                                "replacement, average, refit the 4PL; per-fold fits beside it. Only "
                                                "126 distinct resamples exist for 5 folds, so the CI is coarse",
                                        "rows": boot},
    "input_sha256": files}
write_json(RESULTS / "p1a_sigmoid_onset.json", out)
print(f"{'curve':58s} {'x0':>6s} {'frac':>5s} {'R2':>6s} {'rise':>6s} {'peak':>6s} {'acc':>4s} {'o90':>4s}")
for k, r in rows.items():
    print(f"{k:58s} {r['inflection_point']:6.2f} {r['inflection_frac']:5.3f} {r['fit_r2']:6.3f} {r['rise_fit_pp']:6.1f} "
          f"{r['peak_above_chance_pp']:6.1f} {'Y' if r['accept_rebuttal_rule'] else 'n':>4s} {r['onset90_point']:4d} {str(r['onset90_stored']):>4s}")
for k, b in boot.items():
    print(k, b["per_fold_inflection_point"], b["fold_mean_curve"]["inflection_point"], b["boot_inflection_point"])

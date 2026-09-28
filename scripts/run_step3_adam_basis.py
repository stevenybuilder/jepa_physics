"""Step 3 with an Adam-built steering basis at the paper's layer (point 9, direction): C.12 l.1243 builds V from C.11's
probe sequence, which is Adam (lr 1e-3, weight decay 1e-4, 100 epochs). Refits exactly the stored
results/p1b_direction_direction_meanpool_L9_adam_b64.json sequence (wm.inlp.adam_sequence, batch 64, init seed
0 + round - 1, patience 3; asserted to reproduce the stored per-round test R²), keeps probes 1..K_first (the paper's
stop rule as stored: first round with test R² < 0.1 or MAE > 80 deg, minus one), saves W_k, b_k to
artifacts/inlp/direction_direction_L9_adam_b64.npz, and runs run_steering with that basis (same split, evaluation
probe, targets, nulls) plus the rank-matched null. Writes results/p1c_direction_L9_adam_basis.json.

  python scripts/run_step3_adam_basis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.inlp import INLP_DIR, adam_sequence, layer_data, save_basis  # noqa: E402
from wm.probes import RESULTS, load_table, standardized_layer  # noqa: E402
from wm.provenance import sha256_file  # noqa: E402
from wm.splits import load_split  # noqa: E402
from wm.steer import build_basis, eval_probe_cv, rank_matched_null, run_steering  # noqa: E402

DATASET, POINT, BATCH, SINGLE = "direction", 9, 64, 90.0
N_REPORT = (1, 2, 3, 5, 10, 20)
stored_seq = json.loads((RESULTS / "p1b_direction_direction_meanpool_L9_adam_b64.json").read_text())
assert stored_seq["recipe"]["batch"] == BATCH and not stored_seq["recipe"]["full_batch"]

Y, kind, score_fn, tr, te, folds, acts, n_nan = layer_data(DATASET, DATASET, "meanpool", "vjepa2", None)
Xtr, Xte = standardized_layer(acts, POINT, tr, te)
summary, W_all, b_all = adam_sequence(Xtr, Y[tr], Xte, Y[te], score_fn, kind, None, BATCH, 0,
                                      stored_seq["patience"], return_bias=True)
r2_new = np.array([r["test_r2"] for r in summary["rounds"]])
r2_old = np.array([r["test_r2"] for r in stored_seq["rounds"]])
assert len(r2_new) == len(r2_old) and summary["K_first"] == stored_seq["K_first"], (len(r2_new), summary["K_first"])
max_dr2 = float(np.abs(r2_new - r2_old).max())
assert max_dr2 < 1e-6, max_dr2
K = summary["K_first"]
W, b = W_all[:K], b_all[:K]
probes = {"Q": build_basis(W), "W": W, "b": b, "alpha": None, "point": POINT, "kind": kind}
npz = Path(INLP_DIR) / f"{DATASET}_{DATASET}_L{POINT}_adam_b{BATCH}.npz"
save_basis(npz, probes["Q"], W, b, np.nan, POINT, kind)

# rank-matched null under run_steering's exact evaluation probe (recomputed identically; checked against its output)
df = load_table(DATASET)
hashes = load_split(DATASET)["frame_hash"]
groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
eval_W, eval_b, eval_report = eval_probe_cv(Xte, Y[te], kind, groups=groups)
labels = df["theta_degrees"].to_numpy(float)[te]
rm_null = rank_matched_null(Xte, labels, probes, eval_W, eval_b, kind, SINGLE)

stored_ridge = json.loads((RESULTS / "p1c_direction_L9.json").read_text())
basis_info = {"recipe": summary["recipe"], "source_sequence": "results/p1b_direction_direction_meanpool_L9_adam_b64.json",
              "stop_rule": "C.11 as stored: K = first round with test R2 < 0.1 or test MAE > 80 deg, minus one "
                           "(K_first); the sequence ran on to patience 3 as stored",
              "K": K, "rank": 2 * K, "rounds_run": len(r2_new), "K_patience": summary["K_patience"],
              "reproduces_stored_sequence": {"max_abs_test_r2_diff": max_dr2, "K_first_equal": True},
              "per_probe_test_r2": [round(float(v), 4) for v in r2_new[:K]],
              "per_probe_train_r2": [round(r["train_r2"], 4) for r in summary["rounds"][:K]],
              "weights_file": str(npz.relative_to(npz.parents[2])), "weights_sha256": sha256_file(npz),
              "ridge_basis_K": stored_ridge["K"]}
out = run_steering(DATASET, DATASET, POINT, layer_role="paper_layer", probes=probes,
                   out_path=RESULTS / "p1c_direction_L9_adam_basis.json",
                   extra={"basis": "Adam C.11 probe sequence (batch 64), not the stored ridge sequence",
                          "basis_provenance": basis_info, "rank_matched_null": rm_null})
assert out["eval_probe"] == eval_report

ridge_rm = json.loads((RESULTS / "p1c_direction_L9_rankmatched.json").read_text())["rank_matched_null"]["rows"]
ridge_single = {r["n"]: r["mae_to_target"] for r in stored_ridge["single"]}
ridge_nulls = {r["n"]: r for r in stored_ridge["random_nulls"]["rows"]}
ridge_rmd = {r["n"]: r for r in ridge_rm}


def first_n(single, bar=10.0):
    return next((r["n"] for r in single if r["n"] > 0 and r["mae_to_target"] <= bar), None)


adam_nulls = {r["n"]: r for r in out["random_nulls"]["rows"]}
adam_rm = {r["n"]: r for r in rm_null["rows"]}
table = []
for n in list(N_REPORT) + [K]:
    row = {"n": n, "adam_mae_to_target": adam_nulls[n]["learned"]["mae_to_target"],
           "adam_rank_matched_mean": adam_rm[n]["rank_matched"]["mae_to_target"]["mean"],
           "adam_p_rank_matched": adam_rm[n]["rank_matched"]["empirical_p_to_target"],
           "adam_rank2K_mean": adam_nulls[n]["random_basis"]["mae_to_target"]["mean"],
           "adam_p_rank2K": adam_nulls[n]["random_basis"]["empirical_p_to_target"]}
    if n in ridge_single:
        row.update({"ridge_mae_to_target": ridge_single[n],
                    "ridge_p_rank_matched": ridge_rmd[n]["rank_matched"]["empirical_p_to_target"],
                    "ridge_p_rank2K": ridge_nulls[n]["random_basis"]["empirical_p_to_target"]})
    table.append(row)
comparison = {"rule": "single target 90 deg, paper arm; beats a null = empirical p at the 1/21 floor",
              "null_dimension_note": f"the Adam-basis rank-2K null (random_nulls.random_basis) is {2 * K}-dimensional "
                                     f"(2 x {K}), so its draws differ from the ridge file's {2 * stored_ridge['K']}-"
                                     f"dimensional null (2 x {stored_ridge['K']}); the rank-matched nulls also draw "
                                     f"R_j from [d, 2K] Gaussians, so their R_j[:, :2N] differ between the files too",
              "n_to_10deg": {"adam": first_n(out["single"]), "ridge_stored": first_n(stored_ridge["single"])},
              "K": {"adam": K, "ridge_stored": stored_ridge["K"]}, "table": table}
obj = json.loads((RESULTS / "p1c_direction_L9_adam_basis.json").read_text())
prov = obj.pop("provenance")
obj["comparison_vs_ridge_basis"] = comparison
obj["provenance"] = prov
(RESULTS / "p1c_direction_L9_adam_basis.json").write_text(json.dumps(obj, indent=1))
for r in table:
    print(r)
print(comparison["n_to_10deg"], comparison["K"])

"""Rank-2 covariance-weighted steering at the paper's layer (point 9, direction), paper §7.1 l.405-412 / C.12.

The only stored rank-2 edit is step 3 at N = 1 (Euclidean minimum norm through the first probe, 78.4 deg from target).
Here x' = x + Σ W (WᵀΣW)⁻¹ (y* − ŷ(x)) (wm.steer.cov_weighted_delta) with W the first stored ridge probe (sin, cos)
and Σ the train-set activation covariance at point 9 (train-standardised coordinates), plus an oracle variant with
W = the held-out evaluation probe itself. Split, standardisation, evaluation probe, off-target speed probe and single
target (90 deg) are exactly run_steering's (results/p1c_direction_L9.json). Null: 20 draws of a random 2-D subspace U
(the rank-matched null's draws R_j[:, :2], seed 0) with the same weighting, x' = x + ΣU (WᵀΣU)⁻¹ (y* − ŷ(x)).
Writes results/p1c_direction_L9_rank2_covweighted.json.

  python scripts/run_step3_rank2_covweighted.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.covariance import ledoit_wolf  # noqa: E402
from wm.inlp import basis_path, load_basis  # noqa: E402
from wm.metrics import readout_radius  # noqa: E402
from wm.probes import (RESULTS, drop_nan_clips, load_activations, load_table, predict, split_rows,  # noqa: E402
                       standardized_layer, targets, write_json)
from wm.splits import load_split  # noqa: E402
from wm.steer import (_band, build_basis, cov_weighted_delta, decode, distance, empirical_p, encode,  # noqa: E402
                      eval_probe_cv, off_target_probe, steer)

DATASET, POINT, SINGLE, N_DRAWS, SEED = "direction", 9, 90.0, 20, 0

df = load_table(DATASET)
Y, kind, _ = targets(df, DATASET)
tr, te, folds = split_rows(DATASET, df)
acts = load_activations(DATASET, "meanpool", "vjepa2", None)
tr, te, folds, n_nan = drop_nan_clips(acts, tr, te, folds)
Xtr, Xte = standardized_layer(acts, POINT, tr, te)
off, off_info = off_target_probe(Xtr, Xte, df, tr, te, folds, kind)
label_all = df["theta_degrees"].to_numpy(float)
labels = label_all[te]
all_targets = np.unique(label_all)
hashes = load_split(DATASET)["frame_hash"]
groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
eval_W, eval_b, eval_report = eval_probe_cv(Xte, Y[te], kind, groups=groups)
probes = load_basis(basis_path(DATASET, DATASET, POINT))
W1, b1 = probes["W"][0], probes["b"][0]          # probe 1 is fit on raw x (nothing removed before it)
K, d = len(probes["W"]), Xte.shape[1]
y_single = encode(SINGLE, kind)
x_norm = np.linalg.norm(Xte, axis=1)
r_off0 = decode(predict(Xte[off["mask"]], off["W"], off["b"]), off["kind"])

Sigma = Xtr.T @ Xtr / (len(Xtr) - 1)             # Xtr is train-standardised: mean 0, unit variance per unit
Sigma_lw, lw_shrink = ledoit_wolf(Xtr, assume_centered=True)


def scores(dx_fn):
    """run_steering's single-target scores (held-out evaluation probe), off-target speed, and the all-targets sweep."""
    Xs = Xte + dx_fn(y_single)
    P = predict(Xs, eval_W, eval_b)
    r = decode(P, kind)
    r_off = decode(predict(Xs[off["mask"]], off["W"], off["b"]), off["kind"])
    dxn = np.linalg.norm(Xs - Xte, axis=1)
    out = {"mae_to_target": float(distance(r, SINGLE, kind).mean()),
           "mae_to_true": float(distance(r, labels, kind).mean()),
           "off_target": {"mae_to_true": float(distance(r_off, off["labels"], off["kind"]).mean()),
                          "mean_abs_change": float(distance(r_off, r_off0, off["kind"]).mean())},
           "norm_ratio_median": float(np.median(np.linalg.norm(Xs, axis=1) / x_norm)),
           "edit_norm_median": float(np.median(dxn)), "radius_median": float(np.median(readout_radius(P)))}
    err = [distance(decode(predict(Xte + dx_fn(encode(t, kind)), eval_W, eval_b), kind), t, kind) for t in all_targets]
    out["all_targets_mae_to_target"] = float(np.mean(err))
    return out


def null(W, b, S, draws):
    rows = [scores(lambda y, U=U: cov_weighted_delta(Xte, W, b, S, y, U)) for U in draws]
    return {"mae_to_target": _band([r["mae_to_target"] for r in rows]),
            "mae_to_true": _band([r["mae_to_true"] for r in rows]),
            "off_target_mean_abs_change": _band([r["off_target"]["mean_abs_change"] for r in rows]),
            "edit_norm_median": _band([r["edit_norm_median"] for r in rows]),
            "all_targets_mae_to_target": _band([r["all_targets_mae_to_target"] for r in rows])}


rng = np.random.default_rng(SEED)                # same stream as rank_matched_null: R_j = QR(G_j [d, 2K]), use R_j[:, :2]
U_draws = [np.linalg.qr(rng.standard_normal((d, 2 * K)))[0][:, :2] for _ in range(N_DRAWS)]
V = build_basis(probes["W"])
euclid = scores(lambda y: steer(Xte, V, probes, y, 1) - Xte)
euclid_null = null(W1, b1, np.eye(d), U_draws)   # Σ = I: exactly rank_matched_null's N = 1 draws

edits = {}
for name, W, b, S, desc in (
        ("euclidean_N1", W1, b1, np.eye(d), "step 3 at N = 1 recomputed (Σ = I): Euclidean minimum-norm edit through "
                                            "stored ridge probe 1; must reproduce p1c_direction_L9.json single[n=1]"),
        ("covweighted_probe1", W1, b1, Sigma, "x' = x + Σ W1 (W1ᵀΣW1)⁻¹ (y* − ŷ1(x)), W1 = stored ridge probe 1, "
                                              "Σ = sample train covariance, no shrinkage"),
        ("covweighted_probe1_ledoitwolf", W1, b1, Sigma_lw, "as covweighted_probe1 with the Ledoit-Wolf shrunk Σ "
                                                            "(sensitivity)"),
        ("covweighted_oracle_evalprobe", eval_W, eval_b, Sigma,
         "ORACLE: W = the held-out evaluation probe itself (fit on these test clips), sample Σ; the evaluation probe "
         "then reads y* exactly by construction, so error to target is 0 up to float error. Upper bound only")):
    learned = euclid if name == "euclidean_N1" else scores(lambda y: cov_weighted_delta(Xte, W, b, S, y))
    nl = euclid_null if name == "euclidean_N1" else null(W, b, S, U_draws)
    edits[name] = {"rule": desc, "WtSigmaW_cond": float(np.linalg.cond(W.T @ S @ W)), **learned,
                   "rank_matched_null": {**nl, "empirical_p_to_target": empirical_p(learned["mae_to_target"],
                                                                                   nl["mae_to_target"]["draws"]),
                                         "floor_1_over_21": empirical_p(learned["mae_to_target"],
                                                                        nl["mae_to_target"]["draws"]) == 1 / (N_DRAWS + 1),
                                         "empirical_p_off_target_change": empirical_p(
                                             learned["off_target"]["mean_abs_change"],
                                             nl["off_target_mean_abs_change"]["draws"]),
                                         "empirical_p_edit_norm": empirical_p(learned["edit_norm_median"],
                                                                              nl["edit_norm_median"]["draws"])}}
    print(f"{name}: to target {learned['mae_to_target']:.2f}, to true {learned['mae_to_true']:.2f}, off-target speed "
          f"{learned['off_target']['mean_abs_change']:.4f}; null {nl['mae_to_target']['mean']:.2f} "
          f"p={edits[name]['rank_matched_null']['empirical_p_to_target']:.3f}")

stored = RESULTS / "p1c_direction_L9.json"
out = {"dataset": DATASET, "variable": DATASET, "kind": kind, "pool": "meanpool", "model": "vjepa2", "point": POINT,
       "n_test": len(te), "single_target": SINGLE, "all_nan_clips_excluded": n_nan, "K_stored_basis": K,
       "space": "train-standardised activations", "eval_probe": eval_report, "off_target_probe": off_info,
       "paper": "§7.1 l.405-412: manipulating only the unit-circle subspace does not effectively steer direction; "
                "C.12 l.1240-1262 steering protocol",
       "protocol": "run_steering's paper protocol (C.12): same split, train standardisation, evaluation probe (ridge, "
                   "alpha by grouped 5-fold CV inside test, fit on the steered test clips), off-target speed probe "
                   "(train, constant-speed clips), single target 90 deg and the 64-value all-targets sweep",
       "sigma": {"estimator": "sample covariance of the train-standardised train activations at point 9, "
                              "Xtrᵀ Xtr / (n - 1)", "n_train": len(tr), "d": d,
                 "rank": int(np.linalg.matrix_rank(Sigma)),
                 "shrinkage": "none for the headline edits: only the 2 x 2 WᵀΣW (or WᵀΣU) is inverted, never Σ "
                              "(its condition number is stored per edit); Ledoit-Wolf (sklearn, assume_centered) "
                              "run as a sensitivity row",
                 "ledoit_wolf_shrinkage": float(lw_shrink)},
       "null_rule": "rank-matched null: 20 draws of a random 2-D subspace U = R_j[:, :2] (R_j = QR of a Gaussian "
                    "[d, 2K], rng seed 0, the stream of rank_matched_null), same Σ as the edit, "
                    "x' = x + ΣU (WᵀΣU)⁻¹ (y* − ŷ(x)) so the steering probe still reads y*; Σ = I gives exactly the "
                    "stored rank_matched_null N = 1 draws. empirical p = (1 + #draws <= learned) / 21 (lower = learned "
                    "better), on error to target (headline), off-target speed change and median edit norm ||x' - x||",
       "stored_reference": {"file": "results/p1c_direction_L9.json",
                            "euclidean_N1_single": next(r for r in json.loads(stored.read_text())["single"]
                                                        if r["n"] == 1)},
       "edits": edits}
write_json(RESULTS / "p1c_direction_L9_rank2_covweighted.json", out)

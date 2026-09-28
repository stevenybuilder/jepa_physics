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
from wm.inlp import INLP_DIR, basis_path, load_basis  # noqa: E402
from wm.metrics import readout_radius  # noqa: E402
from wm.adam_probe import fit_adam  # noqa: E402
from wm.probes import (RESULTS, drop_nan_clips, fit_ridge, load_activations, load_table, predict, split_rows,  # noqa: E402
                       standardized_layer, targets, write_json)
from wm.splits import load_split  # noqa: E402
from wm.steer import (_band, build_basis, cov_weighted_delta, decode, distance, empirical_p, encode,  # noqa: E402
                      eval_probe_cv, off_target_probe, steer, stratified_halves)

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


def stored_read(Xs):
    return predict(Xs, eval_W, eval_b)


def scores(dx_fn, read=stored_read):
    """run_steering's single-target scores (held-out evaluation probe; read(Xs) -> its (sin, cos) readout), off-target
    speed, and the all-targets sweep."""
    Xs = Xte + dx_fn(y_single)
    P = read(Xs)
    r = decode(P, kind)
    r_off = decode(predict(Xs[off["mask"]], off["W"], off["b"]), off["kind"])
    dxn = np.linalg.norm(Xs - Xte, axis=1)
    out = {"mae_to_target": float(distance(r, SINGLE, kind).mean()),
           "mae_to_true": float(distance(r, labels, kind).mean()),
           "off_target": {"mae_to_true": float(distance(r_off, off["labels"], off["kind"]).mean()),
                          "mean_abs_change": float(distance(r_off, r_off0, off["kind"]).mean())},
           "norm_ratio_median": float(np.median(np.linalg.norm(Xs, axis=1) / x_norm)),
           "edit_norm_median": float(np.median(dxn)), "radius_median": float(np.median(readout_radius(P)))}
    err = [distance(decode(read(Xte + dx_fn(encode(t, kind))), kind), t, kind) for t in all_targets]
    out["all_targets_mae_to_target"] = float(np.mean(err))
    return out


def null(W, b, S, draws, read=stored_read):
    rows = [scores(lambda y, U=U: cov_weighted_delta(Xte, W, b, S, y, U), read) for U in draws]
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

null_store = {}                                  # full-precision null rows per (reader, edit), for null_medians
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
    null_store[("ridge_alpha_100_stored", name)] = nl
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

# evaluation-probe variants (as p1c_direction_evalprobe_recipe.json, plus p1c_direction_L9_strict.json's split halves):
# the alpha = 100 probe's shrinkage favours high-variance directions, where Σ-weighted edits push
in_a = stratified_halves(groups, labels)
W_A, b_A, rep_A = eval_probe_cv(Xte[in_a], Y[te][in_a], kind, groups=groups[in_a])
W_B, b_B, rep_B = eval_probe_cv(Xte[~in_a], Y[te][~in_a], kind, groups=groups[~in_a])
W_r, b_r = fit_ridge(Xte, Y[te], 1e-3)
W_ad, b_ad = fit_adam(Xte, Y[te], 1e-3, 1e-4, 100, 64, 0)
readers = {"ridge_alpha_100_stored": ("stored: ridge, alpha by grouped 5-fold CV inside test (alpha = 100)", stored_read),
           "ridge_alpha_1e-3": ("closed-form ridge on the test activations, alpha = 1e-3 (as evalprobe_recipe)",
                                lambda Xs: predict(Xs, W_r, b_r)),
           "adam_c11": ("Adam lr 1e-3, coupled wd 1e-4, 100 epochs, batch 64, init seed 0, standardised targets "
                        "(wm.adam_probe.fit_adam, as evalprobe_recipe)", lambda Xs: predict(Xs, W_ad, b_ad)),
           "split_half": ("test split into stratified halves A/B (wm.steer.stratified_halves, identical clips together, "
                          "as strict_eval); probe fit on one half (alpha by grouped CV inside it) reads the other "
                          f"half's steered clips, both directions pooled over all {len(te)} clips; alpha_A = "
                          f"{rep_A['alpha']}, alpha_B = {rep_B['alpha']}",
                          lambda Xs: np.where(in_a[:, None], predict(Xs, W_B, b_B), predict(Xs, W_A, b_A)))}
variant_edits = {"euclidean_N1": (W1, b1, np.eye(d)), "covweighted_probe1": (W1, b1, Sigma),
                 "covweighted_oracle_evalprobe": (eval_W, eval_b, Sigma)}
KEYS = ("mae_to_target", "mae_to_true", "edit_norm_median")
variants = {}
for vname, (vdesc, read) in readers.items():
    base = decode(read(Xte), kind)
    variants[vname] = {"probe": vdesc, "unsteered": {"mae_to_target": float(distance(base, SINGLE, kind).mean()),
                                                      "mae_to_true": float(distance(base, labels, kind).mean())}}
    for ename, (W, b, S) in variant_edits.items():
        if ename == "euclidean_N1":
            lrn = scores(lambda y: steer(Xte, V, probes, y, 1) - Xte, read)
        else:
            lrn = scores(lambda y, W=W, b=b, S=S: cov_weighted_delta(Xte, W, b, S, y), read)
        nl = null(W, b, S, U_draws, read)
        null_store[(vname, ename)] = nl
        variants[vname][ename] = {
            **{k: lrn[k] for k in KEYS}, "off_target_speed_mean_abs_change": lrn["off_target"]["mean_abs_change"],
            "all_targets_mae_to_target": lrn["all_targets_mae_to_target"],
            "null": {**{k: nl[k]["mean"] for k in KEYS},
                     "off_target_speed_mean_abs_change": nl["off_target_mean_abs_change"]["mean"],
                     "mae_to_target_draws": nl["mae_to_target"]["draws"],
                     "empirical_p_to_target": empirical_p(lrn["mae_to_target"], nl["mae_to_target"]["draws"]),
                     "empirical_p_off_target_change": empirical_p(lrn["off_target"]["mean_abs_change"],
                                                                  nl["off_target_mean_abs_change"]["draws"]),
                     "empirical_p_edit_norm": empirical_p(lrn["edit_norm_median"], nl["edit_norm_median"]["draws"])}}
        e = variants[vname][ename]
        print(f"[{vname}] {ename}: to target {e['mae_to_target']:.2f}, to true {e['mae_to_true']:.2f}, off "
              f"{e['off_target_speed_mean_abs_change']:.4f}, |dx| {e['edit_norm_median']:.2f}; null "
              f"{e['null']['mae_to_target']:.2f} p={e['null']['empirical_p_to_target']:.3f}")
evalprobe_variants = {"rule": "each evaluation probe reads the same steered clips; off-target speed and edit norm do "
                              "not depend on the evaluation probe (repeated per variant for completeness); null = the "
                              "Σ-weighted (Σ = I for euclidean_N1) random 2-D subspace null, same 20 draws; the oracle "
                              "edit is built from the alpha = 100 probe under every variant",
                      "split_half_eval_probes": {"A": rep_A, "B": rep_B, "n_A": int(in_a.sum()),
                                                 "n_B": int((~in_a).sum())},
                      **variants}

# Adam W1: the same edits with the first probe of the Adam C.11 sequence (artifacts/inlp/direction_direction_L9_adam_b64
# .npz, run_step3_adam_basis.py) in place of ridge probe 1; same Σ, evaluation probes, off-target probe and U draws
adam_npz = Path(INLP_DIR) / "direction_direction_L9_adam_b64.npz"
adam_probes = load_basis(adam_npz)
Wa1, ba1 = adam_probes["W"][0], adam_probes["b"][0]
adam_w1 = {"rule": "as edits.covweighted_probe1 / euclidean_N1 and evalprobe_variants, with W1, b1 = probe 1 of the Adam "
                   "C.11 sequence (batch 64) instead of stored ridge probe 1; same sample Σ, evaluation probes, off-target "
                   "speed probe and the same 20 U draws (R_j[:, :2], R_j from [d, 2 x 37] Gaussians, seed 0; the Adam "
                   "basis' own rank-matched stream draws [d, 2 x 84] and would differ)",
           "weights_file": str(adam_npz.relative_to(adam_npz.parents[2])),
           "W1_cosine_to_ridge_W1": [float(Wa1[:, j] @ W1[:, j] / np.linalg.norm(Wa1[:, j]) / np.linalg.norm(W1[:, j]))
                                     for j in range(W1.shape[1])],
           "W1_train_r2_as_readout": float(1 - ((Xtr @ Wa1 + ba1 - Y[tr]) ** 2).sum()
                                           / ((Y[tr] - Y[tr].mean(0)) ** 2).sum()),
           "WtSigmaW_cond": float(np.linalg.cond(Wa1.T @ Sigma @ Wa1))}
for vname, (vdesc, read) in readers.items():
    adam_w1[vname] = {}
    for ename, S in (("euclidean_N1_adamW1", np.eye(d)), ("covweighted_adamW1", Sigma)):
        lrn = scores(lambda y, S=S: cov_weighted_delta(Xte, Wa1, ba1, S, y), read)
        nl = null(Wa1, ba1, S, U_draws, read)
        null_store[(vname, ename)] = nl
        ridge_ref = variants[vname]["euclidean_N1" if ename.startswith("euclid") else "covweighted_probe1"]
        adam_w1[vname][ename] = {
            **{k: lrn[k] for k in KEYS}, "off_target_speed_mean_abs_change": lrn["off_target"]["mean_abs_change"],
            "all_targets_mae_to_target": lrn["all_targets_mae_to_target"],
            "ridge_W1_same_edit": {k: ridge_ref[k] for k in (*KEYS, "off_target_speed_mean_abs_change")},
            "null": {**{k: nl[k]["mean"] for k in KEYS},
                     **{f"{k}_median": float(np.median(nl[k]["draws"])) for k in KEYS},
                     "off_target_speed_mean_abs_change": nl["off_target_mean_abs_change"]["mean"],
                     "off_target_speed_mean_abs_change_median": float(np.median(nl["off_target_mean_abs_change"]["draws"])),
                     "mae_to_target_draws": nl["mae_to_target"]["draws"],
                     "empirical_p_to_target": empirical_p(lrn["mae_to_target"], nl["mae_to_target"]["draws"]),
                     "empirical_p_off_target_change": empirical_p(lrn["off_target"]["mean_abs_change"],
                                                                  nl["off_target_mean_abs_change"]["draws"]),
                     "empirical_p_edit_norm": empirical_p(lrn["edit_norm_median"], nl["edit_norm_median"]["draws"])}}
        e = adam_w1[vname][ename]
        print(f"[adam W1 | {vname}] {ename}: to target {e['mae_to_target']:.2f}, to true {e['mae_to_true']:.2f}, off "
              f"{e['off_target_speed_mean_abs_change']:.4f}, |dx| {e['edit_norm_median']:.2f}; null mean "
              f"{e['null']['mae_to_target']:.2f} median {e['null']['mae_to_target_median']:.2f} "
              f"p={e['null']['empirical_p_to_target']:.3f}")
NKEYS = {"mae_to_target": "mae_to_target", "mae_to_true": "mae_to_true", "edit_norm_median": "edit_norm_median",
         "off_target_speed_mean_abs_change": "off_target_mean_abs_change"}
null_medians = {"rule": "median (and mean, for comparison) over the 20 null draws of each null in edits (read by the "
                        "stored alpha = 100 probe), evalprobe_variants and adam_w1; the means are pulled up by one draw "
                        "(draw index given) whose Σ-weighted edit is far larger than the rest",
                "rows": {}}
for (vname, ename), nl in null_store.items():
    norms = np.array(nl["edit_norm_median"]["draws"])
    null_medians["rows"].setdefault(vname, {})[ename] = {
        **{f"{k}_median": float(np.median(nl[v]["draws"])) for k, v in NKEYS.items()},
        **{f"{k}_mean": nl[v]["mean"] for k, v in NKEYS.items()},
        "largest_edit_norm_draw": {"index": int(norms.argmax()), "edit_norm_median": float(norms.max()),
                                   "mae_to_target": float(nl["mae_to_target"]["draws"][int(norms.argmax())])}}

stored = RESULTS / "p1c_direction_L9.json"
out = {"dataset": DATASET, "variable": DATASET, "kind": kind, "pool": "meanpool", "model": "vjepa2", "point": POINT,
       "n_test": len(te), "single_target": SINGLE, "all_nan_clips_excluded": n_nan, "K_stored_basis": K,
       "space": "train-standardised activations", "eval_probe": eval_report, "off_target_probe": off_info,
       "paper": "§7.1 l.383-385: manipulating only the unit-circle subspace does not effectively steer direction; "
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
       "edits": edits, "evalprobe_variants": evalprobe_variants}
# existing keys are kept as stored (checked to reproduce, floats to 1e-9 relative); only adam_w1 and null_medians are
# added, with their own provenance block
out_path = RESULTS / "p1c_direction_L9_rank2_covweighted.json"
if out_path.exists():
    from wm.provenance import result_provenance
    old = json.loads(out_path.read_text())

    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float) or isinstance(b, float):
            return bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))
        return a == b
    new = json.loads(json.dumps(out))
    bad = [k for k in old if k not in ("provenance", "adam_w1", "null_medians") and not same(old[k], new.get(k))]
    assert not bad, f"stored keys do not reproduce: {bad}"
    merged = {k: v for k, v in old.items() if k not in ("adam_w1", "null_medians")}
    prov = merged.pop("provenance")
    merged["adam_w1"] = adam_w1
    merged["null_medians"] = null_medians
    merged["provenance"] = prov
    merged["provenance_adam_w1_null_medians"] = result_provenance(out, None)
    out_path.write_text(json.dumps(merged, indent=1))
else:
    out["adam_w1"], out["null_medians"] = adam_w1, null_medians
    write_json(out_path, out)

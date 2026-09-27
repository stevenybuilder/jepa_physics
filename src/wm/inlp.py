"""Step 2: orthogonal probe sequence / iterative nullspace probing (paper App. C.11, spec 5.2).

Round k: fit probe P_k on X⁽ᵏ⁾, Q_k = QR(W_k) (2 columns for direction, 1 for scalars), X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ − X⁽ᵏ⁾Q_kQ_kᵀ.
α is fixed (step-1 value). All in the train-standardised coordinates of step 1.

Scoring protocols. A removal is never fit on rows that are later scored with it (the earlier code fit every Q_k on all
train rows and scored the train folds through it, so the scored labels shaped the removal and K came out too small):
  nested  for each of the 5 folds the whole sequence (fit, QR, project, refit) runs on the fold's training part and
          every round is scored on the held-out fold. Per-round fold mean ± SD, K per fold, and a pooled K from the
          fold-mean curve. CV-faithful; this K is the one reported as the dimensionality.
  paper   the sequence fit on all train rows, every round scored on the test split (C.11: 80/20 split, stop when test
          performance approaches chance). Reads test once for step 2 (spec §3), recorded in provenance.
The basis saved for step 3 is the all-train sequence (the paper's, and what steering needs), truncated at the nested
pooled K so that its length is not chosen on test.
α: nested chooses α per fold by CV over that fold's training rows only (alpha_folds); the paper protocol and the
saved basis use the step-1 all-train α, as the paper does. The standardiser (step 1's all-train mean/SD) includes the
held-out fold's activations; it is label-free, so it cannot carry label information into the removals.
"""
import json
from pathlib import Path

import numpy as np

from wm.adam_probe import fit_adam, init_linear
from wm.probes import (ALPHAS, N_POINTS, PROJECT_ROOT, RESULTS, cv_select_alpha, drop_nan_clips, fit_ridge, layer_fraction, load_activations,
                       load_sweep, load_table, predict, result_name, split_rows, standardized_layer,
                       targets)
from wm.provenance import LAYER_NOTE, PAPER_LAYER, PAPER_LAYER_ALT, result_provenance

INLP_DIR = PROJECT_ROOT / "artifacts" / "inlp"
TEST_READ = ("paper protocol: the test split is scored at every round of the all-train probe sequence; this is "
             "step 2's one read of test (spec §3). The nested K and the saved basis length do not use test.")


def project_out(X, Q):
    """Remove span(Q) from the rows of X (Q orthonormal, d × r)."""
    return X - (X @ Q) @ Q.T


def at_chance(s, kind):
    """The paper's stopping rule (C.11), applied to one score dict with r2, mae and base_mae."""
    if kind == "circular":
        return s["r2"] < 0.1 or s["mae"] > 80.0
    return s["r2"] < 0.05 or s["mae"] > 0.9 * s["base_mae"]


def probe_sequence(Xfit, Yfit, fit, score_fn=None, Xeval=None, Yeval=None):
    """Generator over rounds k = 1, 2, … of a sequence fit on (Xfit, Yfit) only.

    Yields (score, W, b, dims_removed, X⁽ᵏ⁾_fit): the round-k probe fit(X⁽ᵏ⁾, Y) (W with any numerical component along
    earlier directions removed), its score on (Xeval, Yeval) through the same accumulated projection (None without
    an eval set; base_mae = MAE of predicting the fit-set mean), then removes span(W) from both sets."""
    Yfit = np.asarray(Yfit, float).reshape(len(Yfit), -1)
    Xf = np.array(Xfit, float)
    Xe = None if Xeval is None else np.array(Xeval, float)
    Ye = None if Yeval is None else np.asarray(Yeval, float).reshape(len(Yeval), -1)
    Q = np.zeros((Xf.shape[1], 0))
    while True:
        W, b = fit(Xf, Yfit)
        W = W - Q @ (Q.T @ W)
        s = None
        if Xe is not None:
            s = score_fn(Ye, predict(Xe, W, b))
            s["base_mae"] = float(np.mean(np.abs(Ye - Yfit.mean(axis=0))))
        yield s, W, b, Q.shape[1], Xf
        Qk = np.linalg.qr(W)[0]
        Qk = Qk - Q @ (Q.T @ Qk)
        Qk = np.linalg.qr(Qk)[0]
        Xf = project_out(Xf, Qk)
        if Xe is not None:
            Xe = project_out(Xe, Qk)
        Q = np.hstack([Q, Qk])


def ridge_fit(alpha):
    return lambda X, Y: fit_ridge(X, Y, alpha)


def protocol_summary(rows, K, prefix, kind, m):
    """K (probes above chance before the first at-chance round), dims = K·m, and the loose-threshold K of Fig. 22
    (direction R² < 0.3, stored also as K_r2_03; scalars R² < 0.1, also K_r2_01) from one protocol's curve."""
    hit_cap = K is None
    K = len(rows) if hit_cap else K
    loose = 0.3 if kind == "circular" else 0.1
    K_loose = next((r["round"] - 1 for r in rows if r[f"{prefix}_r2"] < loose), None)
    if K_loose is None and not hit_cap:   # stopped by the MAE rule before the loose R² threshold
        K_loose = K
    out = {"K": K, "K_probes": K, "dims": K * m, "m": m,
           "dims_note": "dims = 2K for 2-output probes (direction's sin, cos), K for 1-output probes; paper Fig. 22's "
                        "y-axis counts probes (K)",
           "K_loose": K_loose, "loose_threshold": f"R2 < {loose} (paper Fig. 22)",
           ("K_r2_03" if kind == "circular" else "K_r2_01"): K_loose, "hit_round_cap": hit_cap, "rounds": rows}
    if m == 2:
        out["dims_2K"] = 2 * K
    return out


def nested_curve(Xtr, Ytr, folds, alpha, score_fn, kind, max_rounds=None, Yfit=None, alpha_per_fold=True):
    """Protocol (a), nested. For each fold the sequence is fit on the other folds only and every round is scored on
    the held-out fold; the folds advance in lockstep until the fold-mean curve and every fold are at chance (or the
    cap). Yfit (default Ytr) holds the labels the removals are fit on and Ytr the labels scored; they differ only in
    the leakage test (permuting the scored fold's labels in Yfit must not change that fold's curve).
    alpha_per_fold (default): each fold's α is chosen by CV over the fold's training rows (their own folds), so the
    held-out labels do not enter α either; False uses the given α everywhere."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    Yfit = Ytr if Yfit is None else np.asarray(Yfit, float).reshape(len(Yfit), -1)
    m = Ytr.shape[1]
    max_rounds = max_rounds or Xtr.shape[1] // m
    ks = np.unique(folds)
    alphas = [cv_select_alpha(Xtr[folds != k], Yfit[folds != k], folds[folds != k], ALPHAS, score_fn)["alpha"]
              if alpha_per_fold else float(alpha) for k in ks]
    seqs = [probe_sequence(Xtr[folds != k], Yfit[folds != k], ridge_fit(a), score_fn,
                           Xtr[folds == k], Ytr[folds == k]) for k, a in zip(ks, alphas)]
    rows, fold_K, K = [], [None] * len(ks), None
    for k in range(1, max_rounds + 1):
        sc = [next(g)[0] for g in seqs]
        for f, s in enumerate(sc):
            if fold_K[f] is None and at_chance(s, kind):
                fold_K[f] = k - 1
        row = {"round": k, "dims_removed": (k - 1) * m}
        for key in ("r2", "mae", "acc15"):
            if key in sc[0]:
                vals = [s[key] for s in sc]
                row[f"cv_{key}"], row[f"cv_{key}_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
        row["base_mae"] = float(np.mean([s["base_mae"] for s in sc]))
        row["fold_r2"] = [float(s["r2"]) for s in sc]
        rows.append(row)
        if K is None and at_chance({"r2": row["cv_r2"], "mae": row["cv_mae"], "base_mae": row["base_mae"]}, kind):
            K = k - 1
        if K is not None and None not in fold_K:
            break
    out = protocol_summary(rows, K, "cv", kind, m)
    fk = [len(rows) if x is None else x for x in fold_K]
    out.update({"protocol": "nested", "K_source": "pooled: first at-chance round of the fold-mean curve, minus 1",
                "alpha_folds": alphas, "alpha_folds_source": "CV over each fold's training rows" if alpha_per_fold
                else "fixed (given)", "K_folds": fk, "K_fold_mean": float(np.mean(fk)), "K_fold_min": int(min(fk)),
                "K_fold_max": int(max(fk)), "fold_hit_round_cap": [x is None for x in fold_K]})
    return out


def paper_curve_and_basis(Xtr, Ytr, Xte, Yte, alpha, score_fn, kind, n_basis, max_rounds, score_test=True):
    """Protocol (b), paper: the all-train sequence, each round scored on test until test is at chance (when
    score_test), run for at least n_basis rounds. Returns (summary or None, W list, b list) with the first n_basis
    probes (the basis for step 3)."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    m = Ytr.shape[1]
    seq = probe_sequence(Xtr, Ytr, ridge_fit(alpha), score_fn, Xte if score_test else None, Yte)
    rows, Ws, bs, K, k = [], [], [], None, 0
    while k < max_rounds and (len(Ws) < n_basis or (score_test and K is None)):
        k += 1
        s, W, b, dims, _ = next(seq)
        if len(Ws) < n_basis:
            Ws.append(W)
            bs.append(b)
        if score_test:
            row = {"round": k, "dims_removed": dims, "test_r2": s["r2"], "test_mae": s["mae"], "base_mae": s["base_mae"]}
            if "acc15" in s:
                row["test_acc15"] = s["acc15"]
            rows.append(row)
            if K is None and at_chance(s, kind):
                K = k - 1
    if not score_test:
        return None, Ws, bs
    return {**protocol_summary(rows, K, "test", kind, m), "protocol": "paper", "test_read": TEST_READ}, Ws, bs


def inlp(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, kind, max_rounds=None, protocols=("nested",),
         alpha_per_fold=True):
    """The probe sequence under the nested protocol (always) and the paper protocol (if 'paper' in protocols).

    The top level is the nested summary (K = pooled nested K, dims = K·m, rounds = fold-mean curve); 'paper' holds
    the paper protocol's summary (None if not run). Returns (summary, Q [d, K·m], W [K, d, m], b [K, m]) where
    Q, W, b are the first K probes of the all-train sequence (K = nested pooled K)."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    d, m = Xtr.shape[1], Ytr.shape[1]
    max_rounds = max_rounds or d // m
    nested = nested_curve(Xtr, Ytr, folds, alpha, score_fn, kind, max_rounds, alpha_per_fold=alpha_per_fold)
    paper, Ws, bs = paper_curve_and_basis(Xtr, Ytr, Xte, Yte, alpha, score_fn, kind, nested["K"], max_rounds,
                                          "paper" in protocols)
    summary = {**nested, "alpha": float(alpha), "alpha_note": "all-train α: paper protocol and saved basis",
               "standardiser_note": "step-1 all-train standardiser (includes held-out folds' activations; label-free)",
               "paper": paper,
               "basis": f"first {len(Ws)} probes of the all-train sequence (length = nested pooled K)"}
    Q = np.hstack([np.linalg.qr(W)[0] for W in Ws]) if Ws else np.zeros((d, 0))
    W_arr = np.stack(Ws) if Ws else np.zeros((0, d, m))
    b_arr = np.stack(bs) if bs else np.zeros((0, m))
    return summary, Q, W_arr, b_arr


def random_score(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, Q):
    """Score the probe after removing a fixed (label-free) subspace span(Q). cv_*: nested protocol (fit on the other
    folds, score the held-out fold; Q does not depend on labels, so one Q serves every fold). test_*: paper protocol
    (fit on all train, score test), only when Xte is given."""
    Xk = project_out(Xtr, Q)
    sc, base = [], []
    for k in np.unique(folds):
        val = folds == k
        W, b = fit_ridge(Xk[~val], Ytr[~val], alpha)
        sc.append(score_fn(Ytr[val], predict(Xk[val], W, b)))
        base.append(float(np.mean(np.abs(Ytr[val] - Ytr[~val].mean(axis=0)))))
    row = {"dims_removed": Q.shape[1], "base_mae": float(np.mean(base))}
    for key in ("r2", "mae", "acc15"):
        if key in sc[0]:
            row[f"cv_{key}"] = float(np.mean([s[key] for s in sc]))
    if Xte is not None:
        W, b = fit_ridge(Xk, Ytr, alpha)
        test = score_fn(Yte, predict(project_out(Xte, Q), W, b))
        row.update({f"test_{key}": test[key] for key in ("r2", "mae", "acc15") if key in test})
    return row


def random_removal_curve(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, rank_schedule, seeds=10):
    """Control: remove a random orthonormal subspace of each rank in rank_schedule and score the same probe, under
    both protocols (cv_* keys = nested, behind the nested curve; test_* keys = paper, behind the paper curve; test_*
    only when Xte is given). Rows carry means over seeds plus *_seed_sd. Per seed the subspaces are nested (first r
    columns of one random orthonormal basis)."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    d = Xtr.shape[1]
    r_max = max(rank_schedule)
    bases = [np.linalg.qr(np.random.default_rng(s).standard_normal((d, max(r_max, 1))))[0]
             for s in range(seeds)]
    rows = []
    for r in rank_schedule:
        per_seed = [random_score(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, B[:, :r]) for B in bases]
        row = {"dims_removed": int(r)}
        for key in ("cv_r2", "cv_mae", "test_r2", "test_mae", "cv_acc15", "test_acc15"):
            if key not in per_seed[0]:
                continue
            vals = [p[key] for p in per_seed]
            row[key] = float(np.mean(vals))
            row[key + "_seed_sd"] = float(np.std(vals, ddof=1))
        rows.append(row)
    return {"seeds": seeds, "protocols": {"cv_*": "nested", "test_*": "paper"}, "rows": rows}


def isolated_dips(values, drop=0.05):
    """Rounds (1-based) whose score falls more than `drop` below the previous round's and whose next round rises more
    than `drop` above it: Fig. 23's sawtooth teeth (isolated low rounds that recover next round)."""
    v = np.asarray(values, float)
    return [k + 1 for k in range(1, len(v) - 1) if v[k] < v[k - 1] - drop and v[k + 1] > v[k] + drop]


def sawtooth(summary, W, prefix="cv"):
    """Shape of the decay of one protocol's curve (prefix 'cv' = nested fold means, 'test' = paper protocol).

    drops[k] = r2[k] − r2[k+1] (what removing probe k cost). A sawtooth shows as drops that
    alternate big/small: negative lag-1 autocorrelation of the drops, and rounds where R² goes up.
    isolated_dips (per higher-is-better metric): rounds that fall and fully recover next round.
    For 2-output (direction) probes, each W_k (d × 2, columns = sin, cos readouts) has a strongest
    readout axis in the (sin, cos) response plane: the top right-singular vector of W_k. Its angle is
    defined modulo 180°; consecutive-angle ≈ 90° means successive probes alternate between orthogonal
    readouts (paired features). anisotropy = s2/s1 (1: reads sin and cos equally, 0: one axis).
    """
    def decay(values, sign):
        drops = -sign * np.diff(np.asarray(values, float))   # positive = information lost this round
        out = {"drops": drops.tolist(),
               "frac_rises": float(np.mean(drops < 0)) if len(drops) else None,
               "drop_lag1_autocorr": float(np.corrcoef(drops[:-1], drops[1:])[0, 1]) if len(drops) > 2 else None}
        if sign == 1:
            out["isolated_dips"] = isolated_dips(values)
        return out

    rounds = summary["rounds"]
    out = decay([r[f"{prefix}_r2"] for r in rounds], 1)          # R² (top-level keys, as before)
    out["protocol_prefix"] = prefix
    out["by_metric"] = {f"{prefix}_r2": decay([r[f"{prefix}_r2"] for r in rounds], 1),
                        f"{prefix}_mae": decay([r[f"{prefix}_mae"] for r in rounds], -1)}
    if rounds and f"{prefix}_acc15" in rounds[0]:      # the paper's Fig. 23 metric
        out["by_metric"][f"{prefix}_acc15"] = decay([r[f"{prefix}_acc15"] for r in rounds], 1)
    if W.shape[0] and W.shape[2] == 2:
        angles, aniso = [], []
        for Wk in W:
            _, s, vt = np.linalg.svd(Wk, full_matrices=False)
            angles.append(np.degrees(np.arctan2(vt[0, 0], vt[0, 1])) % 180.0)
            aniso.append(s[1] / s[0])
        diff = np.abs(np.diff(angles)) % 180.0
        consecutive = np.minimum(diff, 180.0 - diff)
        out.update({"readout_angle_deg": angles, "consecutive_angle_deg": consecutive.tolist(),
                    "mean_consecutive_angle_deg": float(consecutive.mean()) if len(consecutive) else None,
                    "anisotropy": aniso})
    return out


def save_basis(path, Q, W, b, alpha, point, kind):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, Q=Q, W=W, b=b, alpha=alpha, point=point, kind=kind)


def load_basis(path):
    z = np.load(path)
    return {"Q": z["Q"], "W": z["W"], "b": z["b"], "alpha": float(z["alpha"]),
            "point": int(z["point"]), "kind": str(z["kind"])}


def basis_path(dataset, variable, point, pool="meanpool", inlp_dir=None, model="vjepa2"):
    suffix = ("" if pool == "meanpool" else f"_{pool}") + ("" if model == "vjepa2" else f"_{model}")
    return Path(inlp_dir or INLP_DIR) / f"{dataset}_{variable}_L{point}{suffix}.npz"


def rank_schedule(summary, max_points=16):
    """Ranks at which the random-removal control is scored: the x-values of the nested curve and (if run) the paper
    curve, thinned to ≤ max_points each."""
    out = set()
    for block in (summary, summary.get("paper")):
        if block:
            ranks = [r["dims_removed"] for r in block["rounds"]]
            step = max(1, int(np.ceil(len(ranks) / max_points)))
            out |= set(ranks[::step]) | {ranks[-1]}
    return sorted(out)


def write_result(path, out, **provenance_extra):
    """probes.write_json with extra provenance fields (e.g. the test read)."""
    obj = {**out, "provenance": {**result_provenance(out), **provenance_extra}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def layer_data(dataset, variable, pool, model, act_root):
    """Targets, split rows (all-NaN clips dropped) and activations for one dataset."""
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(dataset, df)
    acts = load_activations(dataset, pool, model, act_root)
    tr, te, folds, n_nan = drop_nan_clips(acts, tr, te, folds)
    return Y, kind, score_fn, tr, te, folds, acts, n_nan


def layer_flags(point, sweep):
    return {"frac": layer_fraction(point), "is_peak": point == sweep["availability"]["peak"],
            "is_onset": point == sweep["availability"]["onset"], "is_paper_layer": point == PAPER_LAYER,
            "is_paper_layer_alt": point == PAPER_LAYER_ALT, "layer_note": LAYER_NOTE}


def run_inlp(dataset, variable, point=None, pool="meanpool", seeds=10, with_random=True,
             act_root=None, results_dir=None, inlp_dir=None, model="vjepa2", protocols=("nested", "paper")):
    """Probe sequence at one layer point (default: the step-1 CV peak), α from step 1, under the nested protocol
    and (if 'paper' in protocols) the paper protocol. Writes results/p1b_{dataset}_{variable}_{pool}_L{point}.json
    and the basis .npz (all-train sequence, nested-K long)."""
    sweep = load_sweep(dataset, variable, pool, results_dir, model)
    point = sweep["availability"]["peak"] if point is None else point
    alpha = sweep["layers"][point]["alpha"]
    Y, kind, score_fn, tr, te, folds, acts, n_nan = layer_data(dataset, variable, pool, model, act_root)
    Xtr, Xte = standardized_layer(acts, point, tr, te)
    summary, Q, W, b = inlp(Xtr, Y[tr], Xte, Y[te], folds, alpha, score_fn, kind, protocols=protocols)
    save_basis(basis_path(dataset, variable, point, pool, inlp_dir, model), Q, W, b, alpha, point, kind)
    paper = summary["paper"]
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "model": model, "point": point,
           **layer_flags(point, sweep), "all_nan_clips_excluded": n_nan, **summary, "sawtooth": sawtooth(summary, W)}
    if paper:
        paper["sawtooth"] = sawtooth(paper, W, "test")
    if with_random:
        out["random"] = random_removal_curve(Xtr, Y[tr], Xte if paper else None, Y[te], folds, alpha, score_fn,
                                             rank_schedule(summary), seeds)
    msg = (f"{dataset}/{variable} point {point}: nested K={summary['K']} (folds {summary['K_fold_min']}-"
           f"{summary['K_fold_max']}) dims={summary['dims']} K({summary['loose_threshold']})={summary['K_loose']}")
    print(msg + (f"; paper K={paper['K']} dims={paper['dims']}" if paper else ""))
    write_result(Path(results_dir or RESULTS) / (result_name("p1b", dataset, variable, pool, model) + f"_L{point}.json"),
                 out, **({"test_read": TEST_READ} if paper else {"test_read": "none (nested protocol only)"}))
    return out


def run_dims_vs_layer(dataset, variable, pool="meanpool", act_root=None, results_dir=None, inlp_dir=None,
                      model="vjepa2", protocols=("nested", "paper")):
    """Probe sequence at every layer point (no random control); writes ..._dims.json for Fig. 22."""
    sweep = load_sweep(dataset, variable, pool, results_dir, model)
    Y, kind, score_fn, tr, te, folds, acts, n_nan = layer_data(dataset, variable, pool, model, act_root)
    rows = []
    for point in range(N_POINTS):
        alpha = sweep["layers"][point]["alpha"]
        Xtr, Xte = standardized_layer(acts, point, tr, te)
        summary, Q, W, b = inlp(Xtr, Y[tr], Xte, Y[te], folds, alpha, score_fn, kind, protocols=protocols)
        save_basis(basis_path(dataset, variable, point, pool, inlp_dir, model), Q, W, b, alpha, point, kind)
        row = {"point": point, "frac": layer_fraction(point), "post_ln": point == N_POINTS - 1, "alpha": alpha,
               **{k: summary[k] for k in ("K", "K_probes", "dims", "K_loose", "hit_round_cap", "K_fold_mean",
                                          "K_fold_min", "K_fold_max")}, "dims_2K": summary.get("dims_2K")}
        if summary["paper"]:
            row["paper"] = {k: summary["paper"].get(k) for k in ("K", "K_probes", "dims", "dims_2K", "K_loose",
                                                                  "hit_round_cap")}
        rows.append(row)
        print(f"{dataset}/{variable} point {point:2d}: nested K={summary['K']} dims={summary['dims']}"
              + (f"; paper K={summary['paper']['K']}" if summary["paper"] else ""))
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "model": model,
           "protocols": {"top level": "nested (CV-faithful)", "paper": "all-train sequence scored on test"},
           "caveat": ("K depends on each layer's ridge alpha (fixed at that layer's step-1 CV value; smaller alpha "
                      "admits more probes), so compare K across layers with the per-layer alpha in view"),
           "layers": rows}
    write_result(Path(results_dir or RESULTS) / (result_name("p1b", dataset, variable, pool, model) + "_dims.json"),
                 out, test_read=TEST_READ if "paper" in protocols else "none (nested protocol only)")
    return out


# ---------------------------------------------------------------- the paper's Adam recipe (C.11)

ADAM_C11 = {"lr": 1e-3, "weight_decay": 1e-4, "epochs": {"circular": 100, "scalar": 50, "vector": 50},
            "source": "App. C.11: Adam, lr 1e-3, weight decay 1e-4, 100 epochs (direction) / 50 (speed, IntPhys), "
                      "MSE loss, 80/20 split; batch size not given"}


def adam_sequence(Xtr, Ytr, Xte, Yte, score_fn, kind, max_rounds=None, batch=None, seed=0, patience=3,
                  fail_move=0.25, decoupled=False):
    """The probe sequence with the paper's C.11 Adam probe at every round, paper protocol (fit on all train, score
    on test). Adam is coupled L2 (torch.optim.Adam weight_decay) unless decoupled; batch=None is full batch; round k
    uses init seed seed + k − 1 (torch.nn.Linear init).

    Per round: test R², MAE (and acc15 for direction); train MSE, R² (and acc15) at the end of training; ‖W_k‖ and
    ‖W_k − W_init‖ (how far the probe moved from its random init; an untrained probe keeps its random init, and QR
    of a random W removes a random direction, i.e. almost nothing). failed_to_train: movement only,
    ‖W_k − W_init‖ < fail_move · ‖W_1 − W_init,1‖; at_chance_on_train (train R² below the stop threshold) is reported
    separately, since a genuine information dip is at chance on train too. Runs until `patience` consecutive rounds are at chance
    on test, or max_rounds. K_first = first at-chance round − 1 (the paper's rule); K_patience = start of the
    at-chance run − 1."""
    Ytr = np.asarray(Ytr, float).reshape(len(Ytr), -1)
    Yte = np.asarray(Yte, float).reshape(len(Yte), -1)
    (n, d), m = Xtr.shape, Ytr.shape[1]
    max_rounds = max_rounds or d // m
    lr, wd, epochs = ADAM_C11["lr"], ADAM_C11["weight_decay"], ADAM_C11["epochs"][kind]
    batch = batch or n
    state = {"k": 0}

    def fit(X, Y):
        state["k"] += 1
        return fit_adam(X, Y, lr, wd, epochs, batch, seed + state["k"] - 1, decoupled)

    stop_r2 = 0.1 if kind == "circular" else 0.05
    seq = probe_sequence(Xtr, Ytr, fit, score_fn, Xte, Yte)
    rows, Ws, K_first, run_start, move1 = [], [], None, None, None
    for k in range(1, max_rounds + 1):
        s, W, b, dims, Xk = next(seq)
        P = predict(Xk, W, b)
        tr_s = score_fn(Ytr, P)
        move = float(np.linalg.norm(W - init_linear(d, m, seed + k - 1)[0]))
        move1 = move if move1 is None else move1
        row = {"round": k, "dims_removed": dims, "test_r2": s["r2"], "test_mae": s["mae"], "base_mae": s["base_mae"],
               "train_mse": float(np.mean((P - Ytr) ** 2)), "train_r2": tr_s["r2"],
               "W_norm": float(np.linalg.norm(W)), "W_move_from_init": move}
        if "acc15" in s:
            row.update({"test_acc15": s["acc15"], "train_acc15": tr_s["acc15"]})
        row["failed_to_train"] = bool(move < fail_move * move1)
        row["at_chance_on_train"] = bool(tr_s["r2"] < stop_r2)
        rows.append(row)
        Ws.append(W)
        chance = at_chance(s, kind)
        if chance and K_first is None:
            K_first = k - 1
        run_start = (run_start or k) if chance else None
        if run_start is not None and k - run_start + 1 >= patience:
            break
    hit_cap = run_start is None or k - run_start + 1 < patience
    W_arr = np.stack(Ws)
    summary = {"recipe": {**ADAM_C11, "epochs_used": epochs, "batch": batch, "full_batch": batch >= n,
                          "weight_decay_type": "decoupled (AdamW)" if decoupled else "coupled L2 (Adam)",
                          "init_seed": f"{seed} + round - 1"},
               "protocol": "paper (fit on all train, score on test)", "m": m,
               "K_first": len(rows) if K_first is None else K_first,
               "K_patience": len(rows) if hit_cap else run_start - 1, "patience": patience, "hit_round_cap": hit_cap,
               "failed_rule": f"|W - W_init| < {fail_move} x round 1's (movement only)",
               "at_chance_on_train_rule": f"train R2 < {stop_r2}",
               "rounds": rows}
    summary["dims_first"] = summary["K_first"] * m
    st = sawtooth(summary, W_arr, "test")
    metric = "test_acc15" if "test_acc15" in rows[0] else "test_r2"
    dips = st["by_metric"][metric]["isolated_dips"]
    body = rows[:summary["K_patience"]] if not hit_cap else rows   # before the final at-chance run (info exhausted)
    failed = [r["round"] for r in body if r["failed_to_train"]]
    ok = len(body) - len(failed)
    st["dips_vs_failed"] = {
        "metric": metric, "rounds_counted": f"1..{len(body)} (before the final at-chance run)",
        "isolated_dips": dips, "failed_rounds": failed,
        "dips_that_failed": sorted(set(dips) & set(failed)),
        "dips_at_chance_on_train": sorted(set(dips) & {r["round"] for r in body if r["at_chance_on_train"]}),
        "p_dip_given_failed": len(set(dips) & set(failed)) / len(failed) if failed else None,
        "p_dip_given_trained": len(set(dips) - set(failed)) / ok if ok else None,
        "hypothesis": "sawtooth teeth are rounds whose Adam probe barely trained (removes ~nothing, next round "
                      "recovers); supported if dips are mostly failed rounds and failed rounds are mostly dips"}
    summary["sawtooth"] = st
    return summary, W_arr


def run_adam_sequence(dataset, variable, point, pool="meanpool", act_root=None, results_dir=None, model="vjepa2",
                      batch=64, max_rounds=None, patience=3, seed=0):
    """Adam-recipe probe sequence at one layer. batch 64 (default, as the App. B parity check) or 0/None = full batch
    (only 100 / 50 Adam steps at lr 1e-3). Writes results/p1b_{dataset}_{variable}_{pool}_L{point}_adam_b{batch}.json
    or ..._adam_full.json."""
    batch = batch or None
    sweep = load_sweep(dataset, variable, pool, results_dir, model)
    Y, kind, score_fn, tr, te, folds, acts, n_nan = layer_data(dataset, variable, pool, model, act_root)
    Xtr, Xte = standardized_layer(acts, point, tr, te)
    summary, _ = adam_sequence(Xtr, Y[tr], Xte, Y[te], score_fn, kind, max_rounds, batch, seed, patience)
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "model": model, "point": point,
           **layer_flags(point, sweep), "all_nan_clips_excluded": n_nan, **summary}
    dv = summary["sawtooth"]["dips_vs_failed"]
    tag = "full" if summary["recipe"]["full_batch"] else f"b{batch}"
    print(f"{dataset}/{variable} point {point} adam ({tag}): K_first={summary['K_first']} K_patience={summary['K_patience']} "
          f"dips={dv['isolated_dips']} failed={dv['failed_rounds']}")
    write_result(Path(results_dir or RESULTS) / (result_name("p1b", dataset, variable, pool, model)
                                                 + f"_L{point}_adam_{tag}.json"), out, test_read=TEST_READ)
    return out


# ---------------------------------------------------------------- subspace angles (spec §6 item 7, paper C.4)

def orthonormal(M):
    """Orthonormal basis of span(M) (columns), dropping numerically null directions."""
    U, s, _ = np.linalg.svd(np.asarray(M, float), full_matrices=False)
    return U[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else U[:, :0]


def subspace_overlap(QA, QB, n_random=20, seed=0):
    """Principal angles between span(QA) and span(QB) (paper C.4), with a random reference.

    Paper metrics (C.4, Table 3): mean principal angle, Grassmann distance sqrt(Σθ²) (Eq. 2, radians), and
    the projection overlap in both directions, A ← B = ‖Q_AᵀQ_B‖²_F / k_B (Eq. 1, random expectation k_A/d)
    and B ← A = ‖Q_AᵀQ_B‖²_F / k_A (expectation k_B/d). Extras below.

    angles_deg: all min(k_A, k_B) principal angles, ascending (0° = shared direction, 90° = orthogonal).
    overlap = ‖Q_AᵀQ_B‖²_F / k_B, the mean squared cosine: the fraction of B's subspace inside A's.
    Its expectation for a uniformly random B is k_A / d. The random band redraws B (same k_B) n_random
    times and reports the 5-95% range of the overlap and of the smallest angle.
    """
    QA, QB = orthonormal(QA), orthonormal(QB)
    d, kA, kB = QA.shape[0], QA.shape[1], QB.shape[1]
    cos = np.clip(np.linalg.svd(QA.T @ QB, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    rng = np.random.default_rng(seed)
    r_over, r_min = [], []
    for _ in range(n_random):
        R = np.linalg.qr(rng.standard_normal((d, kB)))[0]
        c = np.clip(np.linalg.svd(QA.T @ R, compute_uv=False), 0.0, 1.0)
        r_over.append(float(np.sum(c ** 2) / kB))
        r_min.append(float(np.degrees(np.arccos(c.max()))))
    ang_rad = np.radians(angles)
    return {"k_A": int(kA), "k_B": int(kB), "d": int(d), "angles_deg": np.sort(angles).tolist(),
            # the paper's C.4 metrics (Table 3): mean angle, Eq. 1 both ways with Eq. 3 expectations, Eq. 2
            "mean_angle_deg": float(angles.mean()),
            "grassmann_distance_rad": float(np.sqrt(np.sum(ang_rad ** 2))),
            "overlap_A_from_B": float(np.sum(cos ** 2) / kB), "random_expectation_A_from_B": kA / d,
            "overlap_B_from_A": float(np.sum(cos ** 2) / kA), "random_expectation_B_from_A": kB / d,
            # extras
            "min_angle_deg": float(angles.min()), "median_angle_deg": float(np.median(angles)),
            "overlap": float(np.sum(cos ** 2) / kB), "random_expectation": kA / d,
            "random_overlap_p05_p95": [float(np.percentile(r_over, 5)), float(np.percentile(r_over, 95))],
            "random_min_angle_p05_p95": [float(np.percentile(r_min, 5)), float(np.percentile(r_min, 95))]}


def covectors_into(Q_B, std_B, std_A):
    """Map readout directions (covectors) from dataset B's standardised coordinates into dataset A's.

    A readout w·z_B with z_B = (x − μ_B)/σ_B is the raw covector w/σ_B, which in A's coordinates is
    w·σ_A/σ_B. This is the map under which the overlap answers the off-target question: a steering edit
    Δz_A changes B's readout by (w ⊙ σ_A/σ_B)·Δz_A. Returns an orthonormal basis of the mapped span."""
    return orthonormal(Q_B * (np.asarray(std_A) / np.asarray(std_B))[:, None])

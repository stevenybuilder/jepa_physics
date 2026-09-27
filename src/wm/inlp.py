"""Step 2: orthogonal probe sequence / iterative nullspace probing (paper App. C.11, spec 5.2).

Round k: fit probe P_k on X⁽ᵏ⁾ (train), Q_k = QR(W_k) (2 columns for direction, 1 for scalars),
X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾ − X⁽ᵏ⁾Q_kQ_kᵀ, applied to train, folds and test alike. α is fixed (step-1 value).
All in the train-standardised coordinates of step 1.
"""
from pathlib import Path

import numpy as np

from wm.probes import (N_POINTS, PROJECT_ROOT, RESULTS, fit_ridge, layer_fraction, load_activations,
                       load_sweep, load_table, predict, result_name, split_rows, standardized_layer,
                       targets, write_json)

INLP_DIR = PROJECT_ROOT / "artifacts" / "inlp"


def project_out(X, Q):
    """Remove span(Q) from the rows of X (Q orthonormal, d × r)."""
    return X - (X @ Q) @ Q.T


def fold_scores(X, Y, folds, alpha, score_fn):
    """Fold mean of R² and MAE (fit fold-out, score fold-in), plus the predict-the-fold-out-mean MAE."""
    r2s, maes, base, accs = [], [], [], []
    for k in np.unique(folds):
        val = folds == k
        W, b = fit_ridge(X[~val], Y[~val], alpha)
        s = score_fn(Y[val], predict(X[val], W, b))
        r2s.append(s["r2"])
        maes.append(s["mae"])
        accs.append(s.get("acc15"))
        base.append(float(np.mean(np.abs(Y[val] - Y[~val].mean(axis=0)))))
    out = {"r2": float(np.mean(r2s)), "r2_sd": float(np.std(r2s, ddof=1)),
           "mae": float(np.mean(maes)), "base_mae": float(np.mean(base))}
    if accs[0] is not None:   # direction: accuracy within 15° (paper Fig. 4c, Fig. 23)
        out["acc15"] = float(np.mean(accs))
    return out


def at_chance(s, kind):
    """The paper's stopping rule, applied to the fold-mean scores."""
    if kind == "circular":
        return s["r2"] < 0.1 or s["mae"] > 80.0
    return s["r2"] < 0.05 or s["mae"] > 0.9 * s["base_mae"]


def score_round(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, Q):
    """Score one probe after removing span(Q): fold mean on train and one test score."""
    Xk_tr, Xk_te = project_out(Xtr, Q), project_out(Xte, Q)
    cv = fold_scores(Xk_tr, Ytr, folds, alpha, score_fn)
    W, b = fit_ridge(Xk_tr, Ytr, alpha)
    test = score_fn(Yte, predict(Xk_te, W, b))
    row = {"dims_removed": Q.shape[1], "cv_r2": cv["r2"], "cv_r2_sd": cv["r2_sd"], "cv_mae": cv["mae"],
           "base_mae": cv["base_mae"], "test_r2": test["r2"], "test_mae": test["mae"]}
    if "acc15" in cv:
        row.update({"cv_acc15": cv["acc15"], "test_acc15": test["acc15"]})
    return row, cv, W, b


def inlp(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, kind, max_rounds=None):
    """Run the probe sequence until the fold-mean score is at chance.

    K counts the probes that were above chance; the first at-chance probe is scored and recorded but
    not removed. Dimensionality = K·m (m = 2 for direction, 1 for scalars). K_loose is the K at the
    looser thresholds of Fig. 22 (direction R² < 0.3, stored also as K_r2_03; speed and acceleration
    R² < 0.1, stored also as K_r2_01).

    Returns (summary dict, Q [d, K·m], W [K, d, m], b [K, m]). W_k is stored after removing any
    numerical component along earlier directions; in exact arithmetic the ridge solution already lies
    in the row space of X⁽ᵏ⁾, which is orthogonal to Q_1..Q_{k-1}.
    """
    Ytr = Ytr.reshape(len(Ytr), -1)
    d, m = Xtr.shape[1], Ytr.shape[1]
    max_rounds = max_rounds or d // m
    Q = np.zeros((d, 0))
    Ws, bs, rounds = [], [], []
    K, K03 = None, None
    for k in range(1, max_rounds + 1):
        row, cv, W, b = score_round(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, Q)
        rounds.append({"round": k, **row})
        if K03 is None and cv["r2"] < (0.3 if kind == "circular" else 0.1):
            K03 = k - 1
        if at_chance(cv, kind):
            K = k - 1
            break
        W = W - Q @ (Q.T @ W)
        Qk, _ = np.linalg.qr(W)
        Q = np.hstack([Q, Qk])
        Ws.append(W)
        bs.append(b)
    hit_cap = K is None
    K = len(Ws)
    if K03 is None and not hit_cap:   # stopped by the MAE rule before the loose R² threshold
        K03 = K
    loose = "R2 < 0.3" if kind == "circular" else "R2 < 0.1"
    summary = {"K": K, "dims": K * m, "m": m, "K_loose": K03, "loose_threshold": f"{loose} (paper Fig. 22)",
               ("K_r2_03" if kind == "circular" else "K_r2_01"): K03, "hit_round_cap": hit_cap,
               "alpha": float(alpha), "rounds": rounds}
    W_arr = np.stack(Ws) if Ws else np.zeros((0, d, m))
    b_arr = np.stack(bs) if bs else np.zeros((0, m))
    return summary, Q, W_arr, b_arr


def random_removal_curve(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, rank_schedule, seeds=10):
    """Control: remove a random orthonormal subspace of each rank in rank_schedule and score the same
    probe. Rows have the INLP rounds' keys (means over seeds) plus *_seed_sd, so the curves overlay.
    Per seed the subspaces are nested (first r columns of one random orthonormal basis)."""
    Ytr = Ytr.reshape(len(Ytr), -1)
    d = Xtr.shape[1]
    r_max = max(rank_schedule)
    bases = [np.linalg.qr(np.random.default_rng(s).standard_normal((d, max(r_max, 1))))[0]
             for s in range(seeds)]
    rows = []
    for r in rank_schedule:
        per_seed = [score_round(Xtr, Ytr, Xte, Yte, folds, alpha, score_fn, B[:, :r])[0] for B in bases]
        row = {"dims_removed": int(r)}
        for key in ("cv_r2", "cv_mae", "test_r2", "test_mae", "cv_acc15", "test_acc15"):
            if key not in per_seed[0]:
                continue
            vals = [p[key] for p in per_seed]
            row[key] = float(np.mean(vals))
            row[key + "_seed_sd"] = float(np.std(vals, ddof=1))
        rows.append(row)
    return {"seeds": seeds, "rows": rows}


def sawtooth(summary, W):
    """Shape of the decay.

    drops[k] = cv_r2[k] − cv_r2[k+1] (what removing probe k cost). A sawtooth shows as drops that
    alternate big/small: negative lag-1 autocorrelation of the drops, and rounds where R² goes up.
    For 2-output (direction) probes, each W_k (d × 2, columns = sin, cos readouts) has a strongest
    readout axis in the (sin, cos) response plane: the top right-singular vector of W_k. Its angle is
    defined modulo 180°; consecutive-angle ≈ 90° means successive probes alternate between orthogonal
    readouts (paired features). anisotropy = s2/s1 (1: reads sin and cos equally, 0: one axis).
    """
    def decay(values, sign):
        drops = -sign * np.diff(np.asarray(values, float))   # positive = information lost this round
        return {"drops": drops.tolist(),
                "frac_rises": float(np.mean(drops < 0)) if len(drops) else None,
                "drop_lag1_autocorr": float(np.corrcoef(drops[:-1], drops[1:])[0, 1]) if len(drops) > 2 else None}

    out = decay([r["cv_r2"] for r in summary["rounds"]], 1)          # R² (top-level keys, as before)
    out["by_metric"] = {"cv_r2": decay([r["cv_r2"] for r in summary["rounds"]], 1),
                        "cv_mae": decay([r["cv_mae"] for r in summary["rounds"]], -1)}
    if summary["rounds"] and "cv_acc15" in summary["rounds"][0]:      # the paper's Fig. 23 metric
        out["by_metric"]["cv_acc15"] = decay([r["cv_acc15"] for r in summary["rounds"]], 1)
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


def basis_path(dataset, variable, point, pool="meanpool", inlp_dir=None):
    suffix = "" if pool == "meanpool" else f"_{pool}"
    return Path(inlp_dir or INLP_DIR) / f"{dataset}_{variable}_L{point}{suffix}.npz"


def rank_schedule(summary, max_points=16):
    """Ranks at which the random-removal control is scored: the INLP x-values, thinned to ≤ max_points."""
    ranks = [r["dims_removed"] for r in summary["rounds"]]
    step = max(1, int(np.ceil(len(ranks) / max_points)))
    return sorted(set(ranks[::step]) | {ranks[-1]})


def run_inlp(dataset, variable, point=None, pool="meanpool", seeds=10, with_random=True,
             act_root=None, results_dir=None, inlp_dir=None):
    """Probe sequence at one layer point (default: the step-1 CV peak), α from step 1.
    Writes results/p1b_{dataset}_{variable}_{pool}_L{point}.json and the basis .npz."""
    sweep = load_sweep(dataset, variable, pool, results_dir)
    point = sweep["availability"]["peak"] if point is None else point
    alpha = sweep["layers"][point]["alpha"]
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(dataset, df)
    Xtr, Xte = standardized_layer(load_activations(dataset, pool, act_root=act_root), point, tr, te)
    summary, Q, W, b = inlp(Xtr, Y[tr], Xte, Y[te], folds, alpha, score_fn, kind)
    save_basis(basis_path(dataset, variable, point, pool, inlp_dir), Q, W, b, alpha, point, kind)
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "point": point,
           "frac": layer_fraction(point), "is_peak": point == sweep["availability"]["peak"],
           "is_onset": point == sweep["availability"]["onset"],
           **summary, "sawtooth": sawtooth(summary, W)}
    if with_random:
        out["random"] = random_removal_curve(Xtr, Y[tr], Xte, Y[te], folds, alpha, score_fn,
                                             rank_schedule(summary), seeds)
    print(f"{dataset}/{variable} point {point}: K={summary['K']} dims={summary['dims']} K({summary['loose_threshold']})={summary['K_loose']}")
    write_json(Path(results_dir or RESULTS) / (result_name("p1b", dataset, variable, pool) + f"_L{point}.json"), out)
    return out


def run_dims_vs_layer(dataset, variable, pool="meanpool", act_root=None, results_dir=None, inlp_dir=None):
    """Probe sequence at every layer point (no random control); writes ..._dims.json for Fig. 22."""
    sweep = load_sweep(dataset, variable, pool, results_dir)
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(dataset, df)
    acts = load_activations(dataset, pool, act_root=act_root)
    rows = []
    for point in range(N_POINTS):
        alpha = sweep["layers"][point]["alpha"]
        Xtr, Xte = standardized_layer(acts, point, tr, te)
        summary, Q, W, b = inlp(Xtr, Y[tr], Xte, Y[te], folds, alpha, score_fn, kind)
        save_basis(basis_path(dataset, variable, point, pool, inlp_dir), Q, W, b, alpha, point, kind)
        rows.append({"point": point, "frac": layer_fraction(point), "post_ln": point == N_POINTS - 1,
                     "alpha": alpha, "K": summary["K"], "dims": summary["dims"], "K_loose": summary["K_loose"],
                     "hit_round_cap": summary["hit_round_cap"]})
        print(f"{dataset}/{variable} point {point:2d}: K={summary['K']} dims={summary['dims']}")
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "layers": rows}
    write_json(Path(results_dir or RESULTS) / (result_name("p1b", dataset, variable, pool) + "_dims.json"), out)
    return out


# ---------------------------------------------------------------- subspace angles (spec §6 item 7, paper C.4)

def orthonormal(M):
    """Orthonormal basis of span(M) (columns), dropping numerically null directions."""
    U, s, _ = np.linalg.svd(np.asarray(M, float), full_matrices=False)
    return U[:, s > s.max() * 1e-10] if s.size and s.max() > 0 else U[:, :0]


def subspace_overlap(QA, QB, n_random=20, seed=0):
    """Principal angles between span(QA) and span(QB) (paper C.4), with a random reference.

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
    return {"k_A": int(kA), "k_B": int(kB), "d": int(d), "angles_deg": np.sort(angles).tolist(),
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

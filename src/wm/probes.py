"""Step 1: linear probes at every layer (spec 5.1, paper App. B).

Everything is in one coordinate system: features standardised with train-row statistics.
A probe is closed-form ridge f(x) = xW + b; alpha is picked on the 5 train folds of the split.
"""
import json
import warnings
from functools import partial
from pathlib import Path

import numpy as np

from wm.data import PROJECT_ROOT, load_table
from wm.metrics import circular_mae_deg, r2, radius_summary, readout_radius
from wm.splits import load_split

ALPHAS = np.logspace(-2, 4, 13)
N_POINTS = 26          # embedding, blocks 1..24, final post-LayerNorm
LAST_BLOCK = 24        # point 25 is the post-LN copy of block 24; kept out of the availability rule
ACT_ROOT = PROJECT_ROOT / "artifacts" / "activations"
RESULTS = PROJECT_ROOT / "results"
VARIABLES = {"direction": ["direction"],
             "speed": ["speed", "vxvy"],
             "acceleration": ["acceleration", "axay"]}


def layer_fraction(point):
    """x-axis of every layer plot: block index / 24; the post-LN point also sits at 1.0."""
    return min(point, LAST_BLOCK) / LAST_BLOCK


# ---------------------------------------------------------------- probe

class Standardizer:
    """Per-feature z-score with train statistics. NaN (disk-pool clips with the disk never in
    frame) are ignored in the fit and set to the train mean (0) on transform."""

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.maximum(np.nanstd(X, axis=0), 1e-6)
        return self

    def transform(self, X):
        return np.nan_to_num((X - self.mean) / self.std, nan=0.0)


def fit_ridge(X, Y, alpha):
    """Closed-form ridge with an unpenalised intercept.

    X [n, d]; Y [n] or [n, m]. Centre both, solve (XcᵀXc + αI) W = XcᵀYc, then b = ȳ − x̄W.
    Returns W [d, m] and b [m].
    """
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return W, y_mean - x_mean @ W


def ridge_path(X, Y, alphas):
    """fit_ridge for many alphas at the cost of one: with XcᵀXc = U diag(λ) Uᵀ,
    W_α = U diag(1/(λ + α)) Uᵀ XcᵀYc. Returns [(W, b) per alpha]."""
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    lam, U = np.linalg.eigh(Xc.T @ Xc)
    R = U.T @ (Xc.T @ Yc)
    fits = []
    for alpha in alphas:
        W = U @ (R / (lam + alpha)[:, None])
        fits.append((W, y_mean - x_mean @ W))
    return fits


def predict(X, W, b):
    return X @ W + b


# ---------------------------------------------------------------- targets and scores

def score(Y, P, kind):
    """{'r2', 'mae'} for true Y and predictions P, both [n, m].

    r2: mean of the per-column R². mae: circular MAE in degrees for 'circular' (Y = [sin, cos]),
    else mean absolute error over all columns, in physical units. acc15 (circular only): fraction of
    clips decoded within 15° of the truth, the accuracy the paper plots in Fig. 4c and Fig. 23.
    """
    Y, P = np.asarray(Y, float).reshape(len(Y), -1), np.asarray(P, float).reshape(len(P), -1)
    out = {"r2": float(np.mean([r2(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))}
    if kind == "circular":
        theta = np.degrees(np.arctan2(Y[:, 0], Y[:, 1]))
        out["mae"] = circular_mae_deg(theta, P[:, 0], P[:, 1])
        err = (np.degrees(np.arctan2(P[:, 0], P[:, 1])) - theta + 180.0) % 360.0 - 180.0
        out["acc15"] = float(np.mean(np.abs(err) <= 15.0))
    else:
        out["mae"] = float(np.mean(np.abs(Y - P)))
    return out


def targets(df, variable):
    """Target matrix Y [n, m], its kind, and the score function for one variable.

    direction    -> [sin θ, cos θ]            (circular: circular MAE + mean R²)
    speed, accel -> [magnitude]               (scalar: MAE + R²)
    vxvy, axay   -> magnitude · [cos θ, sin θ] (vector: MAE + mean R²; Fig. 2b's Cartesian pair)
    """
    theta = np.radians(df["theta_degrees"].to_numpy(float))
    magnitude = {"speed": "speed_mps", "vxvy": "speed_mps",
                 "acceleration": "acceleration_mps2", "axay": "acceleration_mps2"}
    if variable == "direction":
        Y, kind = np.stack([np.sin(theta), np.cos(theta)], axis=1), "circular"
    elif variable in ("speed", "acceleration"):
        Y, kind = df[magnitude[variable]].to_numpy(float)[:, None], "scalar"
    elif variable in ("vxvy", "axay"):
        m = df[magnitude[variable]].to_numpy(float)
        Y, kind = np.stack([m * np.cos(theta), m * np.sin(theta)], axis=1), "vector"
    else:
        raise ValueError(variable)
    return Y, kind, partial(score, kind=kind)


# ---------------------------------------------------------------- cross-validation

def cv_select_alpha(X, Y, folds, alphas=ALPHAS, score_fn=None):
    """Pick α by the fold-mean R² over the split's train folds (fit on fold-out, score fold-in).

    Returns the best alpha, its per-fold scores, fold mean ± SD (ddof=1) of R² and MAE, the mean-R²
    curve over all alphas, and the out-of-fold predictions at the best alpha (used for the bootstrap).
    """
    score_fn = score_fn or partial(score, kind="scalar")
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    fold_ids = np.unique(folds)
    paths = {k: ridge_path(X[folds != k], Y[folds != k], alphas) for k in fold_ids}   # one eigh per fold
    best = None
    curve = []
    for i, alpha in enumerate(alphas):
        oof = np.zeros_like(Y)
        fold_scores = []
        for k in fold_ids:
            val = folds == k
            W, b = paths[k][i]
            oof[val] = predict(X[val], W, b)
            fold_scores.append(score_fn(Y[val], oof[val]))
        mean_r2 = float(np.mean([s["r2"] for s in fold_scores]))
        curve.append(mean_r2)
        if best is None or mean_r2 > best[0]:
            best = (mean_r2, alpha, fold_scores, oof)
    _, alpha, fold_scores, oof = best
    r2s = [s["r2"] for s in fold_scores]
    maes = [s["mae"] for s in fold_scores]
    return {"alpha": float(alpha), "fold_scores": fold_scores,
            "cv_mean": float(np.mean(r2s)), "cv_sd": float(np.std(r2s, ddof=1)),
            "cv_mae_mean": float(np.mean(maes)), "cv_mae_sd": float(np.std(maes, ddof=1)),
            "curve": curve, "oof": oof}


# ---------------------------------------------------------------- availability rule

def availability(curve):
    """Onset / peak / decline of a CV curve over points 0..24 (the post-LN point is excluded).

    onset = first point whose score reaches 90% of the curve's maximum (None if the max is <= 0);
    peak = argmax; decline = peak score − final-block score.
    """
    curve = np.asarray(curve, float)
    peak = int(np.argmax(curve))
    onset = int(np.argmax(curve >= 0.9 * curve[peak])) if curve[peak] > 0 else None
    return {"onset": onset, "peak": peak, "peak_score": float(curve[peak]),
            "final_score": float(curve[-1]), "decline": float(curve[peak] - curve[-1])}


def bootstrap_onset(Y, oofs, score_fn, n_boot=200, seed=0):
    """95% CI of the onset layer: resample train clips with replacement and recompute each layer's
    score from its fixed out-of-fold predictions (no refitting, so 200 draws are cheap).
    Note: this uses the pooled out-of-fold R², not the fold mean, for each draw.
    Returns (CI or None, number of draws with no onset, i.e. max R² <= 0); those draws are excluded
    from the CI and the count is reported beside it."""
    rng = np.random.default_rng(seed)
    onsets = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Y), len(Y))
        curve = [score_fn(Y[idx], oof[idx])["r2"] for oof in oofs]
        onsets.append(availability(curve)["onset"])
    n_none = sum(o is None for o in onsets)
    onsets = np.array([o for o in onsets if o is not None])
    if len(onsets) == 0:
        return None, n_none
    return [int(np.percentile(onsets, 2.5)), int(np.percentile(onsets, 97.5))], n_none


# ---------------------------------------------------------------- data plumbing

def load_activations(dataset, pool="meanpool", model="vjepa2", act_root=None):
    """Memory-mapped activations [N, 26, d] (meanpool) or [N, 26, 8, d] (time/disk pools), in
    manifest id order (checked against ids.json when extract.merge wrote one)."""
    folder = Path(act_root or ACT_ROOT) / dataset / model
    acts = np.load(folder / f"{pool}.npy", mmap_mode="r")
    ids = load_table(dataset)["id"].tolist()
    if (folder / "ids.json").exists():
        assert json.loads((folder / "ids.json").read_text()) == ids, "activations not in manifest order"
    assert acts.shape[0] == len(ids) and acts.shape[1] == N_POINTS, acts.shape
    return acts


def layer_matrix(acts, point):
    """float64 [N, d] at one layer point. 8-step pools are averaged over time (nanmean: the disk
    pool is NaN where the disk is out of frame)."""
    X = np.asarray(acts[:, point], dtype=np.float32)
    if X.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN clips stay NaN
            X = np.nanmean(X, axis=1)
    return X.astype(np.float64)


def split_rows(dataset, df=None):
    """Row positions of train and test clips and the fold of each train row."""
    df = load_table(dataset) if df is None else df
    split = load_split(dataset)
    pos = {int(i): r for r, i in enumerate(df["id"])}
    tr = np.array([pos[i] for i in split["train_ids"]])
    te = np.array([pos[i] for i in split["test_ids"]])
    folds = np.array([split["fold"][str(i)] for i in split["train_ids"]])
    return tr, te, folds


def standardized_layer(acts, point, tr, te):
    """Train-standardised X_train, X_test at one layer point (the one coordinate system)."""
    X = layer_matrix(acts, point)
    st = Standardizer().fit(X[tr])
    return st.transform(X[tr]), st.transform(X[te])


def result_name(step, dataset, variable, pool, model="vjepa2", shuffled=False):
    name = f"{step}_{dataset}_{variable}_{pool}"
    return name + ("" if model == "vjepa2" else f"_{model}") + ("_shuffled" if shuffled else "")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------- the sweep

def layer_sweep(dataset, variable, pool="meanpool", model="vjepa2", shuffled=False,
                act_root=None, results_dir=None, n_boot=200, seed=0):
    """Probe every layer point; write results/p1a_{dataset}_{variable}_{pool}[_shuffled].json.

    Per point: standardise on train, pick α on the 5 folds, report fold mean ± SD, then refit on all
    of train and score test once. shuffled=True permutes the train labels (seed 0) and runs the
    identical pipeline; test labels stay true, so its test score is the chance level.
    """
    df = load_table(dataset)
    Y, kind, score_fn = targets(df, variable)
    tr, te, folds = split_rows(dataset, df)
    Ytr, Yte = Y[tr], Y[te]
    if shuffled:
        Ytr = np.random.default_rng(seed).permutation(Ytr)
    motion_tr, motion_te = df["motion"].to_numpy()[tr], df["motion"].to_numpy()[te]
    motions = sorted(set(motion_tr)) if len(set(motion_tr)) > 1 else []
    acts = load_activations(dataset, pool, model, act_root)

    layers, oofs, test_preds = [], [], []
    for point in range(N_POINTS):
        Xtr, Xte = standardized_layer(acts, point, tr, te)
        cv = cv_select_alpha(Xtr, Ytr, folds, ALPHAS, score_fn)
        W, b = fit_ridge(Xtr, Ytr, cv["alpha"])
        Pte = predict(Xte, W, b)
        test = score_fn(Yte, Pte)
        row = {"point": point, "frac": layer_fraction(point), "post_ln": point == N_POINTS - 1,
               "alpha": cv["alpha"], "cv_mean": cv["cv_mean"], "cv_sd": cv["cv_sd"],
               "cv_mae_mean": cv["cv_mae_mean"], "cv_mae_sd": cv["cv_mae_sd"],
               "test_r2": test["r2"], "test_mae": test["mae"],
               "n_train": len(tr), "n_test": len(te)}
        if kind == "circular":   # readout radius ‖(ŝ, ĉ)‖ of the test predictions (spec §4); accuracy within 15°
            row["test_radius"] = radius_summary(Pte)
            row["cv_acc15_mean"] = float(np.mean([f["acc15"] for f in cv["fold_scores"]]))
            row["test_acc15"] = test["acc15"]
        if motions:   # direction set: score the same predictions within each motion type
            row["by_motion"] = {}
            for m in motions:
                fs = [score_fn(Ytr[(folds == k) & (motion_tr == m)], cv["oof"][(folds == k) & (motion_tr == m)])
                      for k in np.unique(folds)]
                r2s = [s["r2"] for s in fs]
                t = score_fn(Yte[motion_te == m], Pte[motion_te == m])
                row["by_motion"][m] = {"cv_mean": float(np.mean(r2s)), "cv_sd": float(np.std(r2s, ddof=1)),
                                       "cv_mae_mean": float(np.mean([s["mae"] for s in fs])),
                                       "test_r2": t["r2"], "test_mae": t["mae"]}
        layers.append(row)
        oofs.append(cv["oof"])
        test_preds.append(Pte)
        print(f"{dataset}/{variable} point {point:2d}: alpha={cv['alpha']:.3g} "
              f"cv R2={cv['cv_mean']:.3f}±{cv['cv_sd']:.3f} test R2={test['r2']:.3f} MAE={test['mae']:.3g}")

    curve = [r["cv_mean"] for r in layers[:LAST_BLOCK + 1]]
    avail = availability(curve)
    avail["onset_frac"] = None if avail["onset"] is None else layer_fraction(avail["onset"])
    avail["peak_frac"] = layer_fraction(avail["peak"])
    avail["post_ln_score"] = layers[-1]["cv_mean"]
    avail["onset_ci"], avail["onset_boot_draws_without_onset"] = bootstrap_onset(
        Ytr, oofs[:LAST_BLOCK + 1], score_fn, n_boot, seed)
    avail["rule"] = "onset = first point with CV R2 >= 90% of max over points 0..24; CI: 200-draw clip bootstrap of out-of-fold predictions"

    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "model": model,
           "shuffled": shuffled, "alphas": ALPHAS.tolist(), "n_train": len(tr), "n_test": len(te),
           "layers": layers, "availability": avail}
    if kind == "circular":   # (ŝ, ĉ) on test at onset, peak and final block, for fig1b
        shown = sorted({p for p in (avail["onset"], avail["peak"], LAST_BLOCK) if p is not None})
        out["test_theta"] = df["theta_degrees"].to_numpy()[te].tolist()
        out["test_pred"] = {str(p): test_preds[p].round(4).tolist() for p in shown}
        out["test_radius"] = {str(p): readout_radius(test_preds[p]).round(4).tolist() for p in shown}
    path = Path(results_dir or RESULTS) / (result_name("p1a", dataset, variable, pool, model, shuffled) + ".json")
    write_json(path, out)
    return out


def chosen_layers(sweep, role="both"):
    """Layer points for steps 2-3 from the step-1 availability rule: {point: role}. role 'peak' (best CV
    score), 'onset' (first point at 90% of max; the paper steers at its emergence layer, layer 8) or
    'both'. If onset and peak coincide the point is run once, labelled 'peak+onset'."""
    av = sweep["availability"]
    picks = {"peak": [av["peak"]], "onset": [av["onset"]], "both": [av["peak"], av["onset"]]}[role]
    out = {}
    for name, point in zip(["peak", "onset"] if role == "both" else [role], picks):
        if point is not None:
            out[point] = f"{out[point]}+{name}" if point in out else name
    return out


def load_sweep(dataset, variable, pool="meanpool", results_dir=None, model="vjepa2"):
    """The step-1 results file of the same model; steps 2 and 3 take α and the chosen layer from it."""
    path = Path(results_dir or RESULTS) / (result_name("p1a", dataset, variable, pool, model) + ".json")
    return json.loads(path.read_text())

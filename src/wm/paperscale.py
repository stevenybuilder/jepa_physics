"""Step 1 at the paper's scale (Part 1 extra, spec §6 item 2 / §7 outcomes): does the block-1 onset survive
the paper's design (8 directions, a few hundred clips, "grouped" CV)?

Same recipe as wm.probes: standardise on the rows entering CV (wm.probes.Standardizer), closed-form ridge with an
unpenalised intercept, α picked over wm.probes.ALPHAS on 5 folds, onset = 90%-of-max rule (wm.probes.availability).
The only change is the solver: with n ≈ 200 clips and d = 1024 features the ridge path is solved in its dual form
(eigh of the n×n Gram matrix instead of the d×d one); tests check it equals wm.probes.cv_select_alpha to float error.
"""
import warnings

import numpy as np

from wm.probes import ALPHAS, LAST_BLOCK, Standardizer, availability

PAPER_DIRECTIONS = np.arange(0, 360, 45)   # the paper's 8 directions, degrees


def eight_direction_rows(theta_deg, step=45.0):
    """Row positions whose direction is a multiple of `step` degrees (all clips at those angles)."""
    th = np.mod(np.asarray(theta_deg, float), 360.0)
    r = np.mod(th, step)
    return np.where(np.isclose(r, 0.0, atol=1e-6) | np.isclose(r, step, atol=1e-6))[0]


def random_folds(n, n_folds=5, seed=0, groups=None):
    """Fold id per row: units (rows sharing a `groups` value, e.g. wm.splits frame_hash = byte-identical clips; each
    row its own unit if None) are shuffled and dealt round-robin to n_folds folds, so a unit never spans two folds."""
    groups = np.arange(n) if groups is None else np.asarray(groups)
    uniq, unit = np.unique(groups, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_unit = np.empty(len(uniq), int)
    fold_of_unit[perm] = np.arange(len(uniq)) % n_folds
    return fold_of_unit[unit]


def direction_grouped_folds(theta_deg, n_folds=5, seed=0):
    """Fold id per row such that every fold holds out whole directions: the distinct angles are shuffled and
    dealt round-robin to the folds (8 directions, 5 folds -> 2, 2, 2, 1, 1 directions per fold)."""
    th = np.round(np.mod(np.asarray(theta_deg, float), 360.0), 6)
    groups = np.unique(th)
    assert len(groups) >= n_folds, f"{len(groups)} directions cannot fill {n_folds} folds"
    order = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {groups[g]: i % n_folds for i, g in enumerate(order)}
    return np.array([fold_of[t] for t in th])


def dual_cv(X, Y, folds, score_fn, alphas=ALPHAS, select="fold_mean"):
    """wm.probes.cv_select_alpha, dual solver. X is already standardised.

    Per fold: centre on the fold-out rows, K = XcXcᵀ = V diag(μ) Vᵀ, W_α = Xcᵀ V diag(1/(μ+α)) Vᵀ Yc (identical to
    (XcᵀXc + αI)⁻¹ XcᵀYc). select='fold_mean' picks α by the fold-mean R² (as wm.probes); 'pooled' by the R² of all
    out-of-fold predictions together, the only defined score when a fold holds out whole directions (a fold with a
    single direction has zero target variance). Returns alpha, cv_mean / cv_sd (fold mean ± SD, None if undefined),
    pooled (OOF R²), the pooled and fold-mean curves over α, and the OOF predictions at the chosen α.
    """
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    fold_ids = np.unique(folds)
    preds = np.zeros((len(alphas),) + Y.shape)
    for k in fold_ids:
        fit, val = folds != k, folds == k
        xm, ym = X[fit].mean(axis=0), Y[fit].mean(axis=0)
        Xc = X[fit] - xm
        mu, V = np.linalg.eigh(Xc @ Xc.T)
        A = (X[val] - xm) @ Xc.T @ V
        R = V.T @ (Y[fit] - ym)
        for i, a in enumerate(alphas):
            preds[i][val] = A @ (R / (mu + a)[:, None]) + ym
    pooled, fold_mean, fold_sd = [], [], []
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(len(alphas)):
            pooled.append(score_fn(Y, preds[i])["r2"])
            fs = [score_fn(Y[folds == k], preds[i][folds == k])["r2"] for k in fold_ids]
            ok = all(np.isfinite(fs))
            fold_mean.append(float(np.mean(fs)) if ok else None)
            fold_sd.append(float(np.std(fs, ddof=1)) if ok else None)
    if select == "fold_mean":
        assert all(v is not None for v in fold_mean), "fold-mean R² undefined; use select='pooled'"
        best = int(np.argmax(fold_mean))
    elif select == "pooled":
        best = int(np.argmax(pooled))
    else:
        raise ValueError(select)
    return {"alpha": float(alphas[best]), "cv_mean": fold_mean[best], "cv_sd": fold_sd[best],
            "pooled": float(pooled[best]), "curve_pooled": [float(v) for v in pooled], "curve_fold_mean": fold_mean,
            "oof": preds[best]}


def probe_rows(X_all, rows, Y_all, folds, score_fn, select="fold_mean"):
    """One CV fit on a subset: standardise on the subset (as wm.probes does on its train split), then dual_cv."""
    X = Standardizer().fit(X_all[rows]).transform(X_all[rows])
    return dual_cv(X, Y_all[rows], folds, score_fn, ALPHAS, select)


def onset_draws(Y, oofs, score_fn, n_boot=200, seed=0):
    """Onset (90%-of-max rule over points 0..24) per clip-bootstrap draw of the fixed out-of-fold predictions, as in
    wm.probes.bootstrap_onset but returning the draws (None where the curve's max is <= 0)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Y), len(Y))
        curve = [score_fn(Y[idx], o[idx])["r2"] for o in oofs[:LAST_BLOCK + 1]]
        out.append(availability(curve)["onset"])
    return out


def ci_from_draws(draws):
    """[2.5th, 97.5th] percentile of the non-None onsets, and the count of None draws."""
    d = np.array([o for o in draws if o is not None])
    n_none = int(sum(o is None for o in draws))
    if len(d) == 0:
        return None, n_none
    return [int(np.percentile(d, 2.5)), int(np.percentile(d, 97.5))], n_none


def summarise_curve(curve, oofs, Y, score_fn, n_boot=200, seed=0):
    """Onset / peak of a curve over points 0..24 plus the bootstrap onset CI from its OOF predictions."""
    av = availability(curve[:LAST_BLOCK + 1])
    draws = onset_draws(Y, oofs, score_fn, n_boot, seed)
    ci, n_none = ci_from_draws(draws)
    return {"onset": av["onset"], "peak": av["peak"], "peak_score": av["peak_score"],
            "onset_ci": ci, "onset_boot_draws_without_onset": n_none}


def motion_rows(rows, motion, keep="velocity"):
    """The entries of `rows` whose clip has motion == keep (motion: per-clip labels over all rows of the table),
    order preserved. The paper's velocity set is constant-speed only (App. A), so paper-scale rows use keep='velocity'."""
    rows = np.asarray(rows)
    return rows[np.asarray(motion)[rows] == keep]

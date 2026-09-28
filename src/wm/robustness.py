"""Audit robustness checks (paper-vs-code procedural deviations, Part 1): helpers for scripts/run_audit_robustness.py.

Grouped CV folds (App. B "5-fold grouped cross-validation", key unstated), a binned-accuracy score beside R² for the
sawtooth test (Fig. 4c "probe accuracy"), the per-round decay statistics of wm.inlp.sawtooth for any metric, and a
probe-sequence runner that records every key of the score dict. Everything else is reused from wm.probes / wm.inlp /
wm.steer / wm.paperscale.
"""
import numpy as np

from wm.inlp import isolated_dips, probe_sequence
from wm.paperscale import random_folds
from wm.probes import predict, score

N_BINS = 8


def start_cells(x, y, cell=1.0, lo=-2.0):
    """Integer cell id of each start position on a square grid of side `cell` metres anchored at `lo`."""
    ix = np.floor((np.asarray(x, float) - lo) / cell).astype(int)
    iy = np.floor((np.asarray(y, float) - lo) / cell).astype(int)
    return ix * 1000 + iy


def value_grouped_folds(values, n_folds=5, seed=0):
    """Fold per row with every distinct value (a label value or a start cell) held out whole: distinct values are
    shuffled and dealt round-robin to the folds (wm.paperscale.random_folds with groups = values)."""
    values = np.round(np.asarray(values, float), 6)
    assert len(np.unique(values)) >= n_folds, "fewer groups than folds"
    return random_folds(len(values), n_folds, seed, groups=values)


DIR_STEP = 360.0 / 64                  # label spacing of the 64-direction sets (5.625 deg)
DIR_EDGE0 = 22.5 - DIR_STEP / 2        # 19.6875: bin edges at 19.6875 + 45 k, the label midpoints nearest 22.5 + 45 k


def direction_bin(theta_deg, edge0=DIR_EDGE0):
    """8-way 45 deg bin with edges at edge0 + 45 k. Default edges sit at label midpoints (half a 5.625 deg step from
    22.5 + 45 k, which are themselves labels), so no label lies on an edge; bin 0 = [-25.3125, 19.6875) holds the 8
    labels -22.5 ... 16.875, centred 2.8 deg from 0 (an even count per bin cannot centre exactly on a label)."""
    return (np.floor(np.mod(np.asarray(theta_deg, float) - edge0 + 45.0, 360.0) / 45.0).astype(int)) % N_BINS


def equal_count_edges(values, n_bins=N_BINS):
    """Inner edges (n_bins − 1) of equal-count bins of `values` (label quantiles; label-only, no features), each
    moved to the midpoint between the two adjacent distinct label values around it, so no label sits on an edge."""
    values = np.asarray(values, float)
    q = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    u = np.unique(values)
    mids = (u[:-1] + u[1:]) / 2
    return mids[np.clip(np.searchsorted(u, q, side="right") - 1, 0, len(mids) - 1)]


def theta_lookup(Y, theta_deg):
    """Map rows (sin, cos) of Y back to their theta_degrees label (keys rounded to 9 d.p.), so the true bin comes from
    the label itself, not from atan2(sin, cos)."""
    table = {tuple(np.round(y, 9)): float(t) for y, t in zip(np.asarray(Y, float), np.asarray(theta_deg, float))}
    return lambda Yq: np.array([table[tuple(np.round(y, 9))] for y in np.asarray(Yq, float)])


def binned_score(kind, edges=None, theta_of=None):
    """score_fn: wm.probes.score plus 'bacc8', the fraction of clips whose prediction lands in the true value's bin.
    circular: direction_bin (edges at label midpoints) of the true theta_degrees label (theta_of, from theta_lookup;
    atan2 of Y only if None) vs of atan2(sin, cos) of the prediction; scalar: bins cut at `edges` (equal_count_edges)."""
    def fn(Y, P):
        Y, P = np.asarray(Y, float).reshape(len(Y), -1), np.asarray(P, float).reshape(len(P), -1)
        out = score(Y, P, kind)
        if kind == "circular":
            bt = direction_bin(theta_of(Y) if theta_of is not None else np.degrees(np.arctan2(Y[:, 0], Y[:, 1])))
            bp = direction_bin(np.degrees(np.arctan2(P[:, 0], P[:, 1])))
        else:
            bt, bp = np.digitize(Y[:, 0], edges), np.digitize(P[:, 0], edges)
        out["bacc8"] = float(np.mean(bt == bp))
        return out
    return fn


def sector_folds(theta_deg, n_folds=5, seed=0):
    """Fold per row with whole contiguous 45 deg direction sectors held out (the 8 direction_bin sectors, dealt
    round-robin to the folds: 2, 2, 2, 1, 1 sectors per fold)."""
    return value_grouped_folds(direction_bin(theta_deg), n_folds, seed)


def decay_stats(values):
    """wm.inlp.sawtooth's per-metric statistics for a higher-is-better curve: per-round drops, fraction of rises,
    lag-1 autocorrelation of the drops (negative = alternating big/small = sawtooth) and isolated dips."""
    drops = -np.diff(np.asarray(values, float))
    return {"drop_lag1_autocorr": float(np.corrcoef(drops[:-1], drops[1:])[0, 1]) if len(drops) > 2 else None,
            "frac_rises": float(np.mean(drops < 0)) if len(drops) else None,
            "isolated_dips": isolated_dips(values), "n_isolated_dips": len(isolated_dips(values)),
            "n_rounds": len(values)}


def run_rounds(Xfit, Yfit, fit, score_fn, Xeval, Yeval, n_rounds):
    """n_rounds of wm.inlp.probe_sequence; every key of score_fn's dict per round (fit on Xfit, scored on Xeval)."""
    seq = probe_sequence(Xfit, Yfit, fit, score_fn, Xeval, Yeval)
    return [{"round": k + 1, **next(seq)[0]} for k in range(n_rounds)]


def in_sample_and_oof(fit, X, Y, folds, kind):
    """In-sample score of fit(X, Y) and fold-mean out-of-fold R² / MAE under `folds` (for an evaluation probe)."""
    W, b = fit(X, Y)
    ins = score(Y, predict(X, W, b), kind)
    fs = []
    for k in np.unique(folds):
        Wk, bk = fit(X[folds != k], Y[folds != k])
        fs.append(score(Y[folds == k], predict(X[folds == k], Wk, bk), kind))
    return W, b, {"in_sample": {"r2": ins["r2"], "mae": ins["mae"]},
                  "out_of_fold": {"r2_fold_mean": float(np.mean([s["r2"] for s in fs])),
                                  "r2_fold_sd": float(np.std([s["r2"] for s in fs], ddof=1)),
                                  "mae_fold_mean": float(np.mean([s["mae"] for s in fs]))}}


def k_below(values, thr):
    """Probes kept before the first round whose R² falls below thr (first such round − 1, rounds 1-based); None if
    the curve never falls below thr (censored)."""
    return next((k for k, v in enumerate(values) if v < thr), None)


def lockstep_until(seqs, thr, max_rounds):
    """Advance wm.inlp.probe_sequence generators (one per fold, or one for the paper protocol) in lockstep until the
    mean R² over them falls below thr (that round included) or max_rounds. Per round: mean r2, mae, base_mae (and
    acc15 when scored) over the generators, plus each generator's r2."""
    rows = []
    for k in range(1, max_rounds + 1):
        sc = [next(g)[0] for g in seqs]
        row = {"round": k, **{key: float(np.mean([s[key] for s in sc])) for key in ("r2", "mae", "base_mae", "acc15")
                              if key in sc[0]}, "each_r2": [float(s["r2"]) for s in sc]}
        rows.append(row)
        if row["r2"] < thr:
            break
    return rows

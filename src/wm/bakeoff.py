"""Steering bake-off at matched delivered norm (project spec section 6 item 9; the "fewer probes" question).

Every arm moves test clips toward a held-out target value; each clip's edit is rescaled to the same ||delta||
(the spline arm's), so arms differ only in where they put that norm. Arms, with their edit rank:
  spline             PCA-k spline walk, additive (rank <= k)
  chord              straight line between PCA-k chord points (rank <= k); the chord runs through curve.points
                     (the smoothed knots for a smoothing spline)
  chord_raw          as chord, through the RAW kept centroids (Goodfire's chord between raw centroids; equals chord
                     for an interpolating spline) (rank <= k)
  centroid_transport x + mu(target) - mu(source), full-space centroids of the knot clips (a held-out target has no
                     centroid: its full-space chord point is used) (rank: full)
  ring_rotation      rotate the clip's coordinates in the circular-chart plane (X ~ mu + A[cos, sin], k = 1 ring)
                     by the angle change: rank 2, norm-preserving in the chart (direction only)
  probe_qr           Part 1's multi-probe subspace steer (wm.steer.steer, basis QR of the step-2 probe sequence),
                     in train-standardised coordinates (rank = basis columns)
  snap               replace the activation with the target's full-space centroid / chord point (the floor)
Evaluators: the held-out ridge probe and a small MLP, both fit on probe-fold clips (disjoint from the knots and from
the steered clips). A linear evaluator shares filter geometry with the probe-QR arm; the MLP does not.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wm import geometry_checks as gc
from wm import manifold as mf


class MLPReadout:
    """Small MLP evaluator (one hidden layer of 64, early stopping). Same outputs and scoring as ProbeReadout."""
    name = "mlp"

    def __init__(self, X, values, periodic, seed=0):
        self.periodic = periodic
        y = np.asarray(values, dtype=float)
        if periodic:
            y = np.stack([np.sin(np.radians(y)), np.cos(np.radians(y))], axis=1)
        else:
            self.mu, self.sd = y.mean(), y.std() + 1e-12
            y = (y - self.mu) / self.sd
        self.model = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True, max_iter=2000, random_state=seed)).fit(X, y)

    def predict(self, X):
        out = self.model.predict(np.asarray(X).reshape(-1, np.shape(X)[-1]))
        if self.periodic:
            return np.degrees(np.arctan2(out[:, 0], out[:, 1])) % 360.0
        return out.ravel() * self.sd + self.mu


def effective_rank(deltas):
    """Participation ratio of the edits across clips (uncentred): how many directions the arm actually uses."""
    D = np.asarray(deltas, dtype=float).reshape(len(deltas), -1)
    lam = np.linalg.svd(D, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def ring_rotation_delta(x, chart, dtheta_deg):
    """Rotate each clip's chart coordinates by dtheta (degrees): delta = A (Rot c - c), c = pinv(A)(x - mu)."""
    P = np.linalg.pinv(chart["A"])
    c = (x - chart["mu"]) @ P.T                                          # [n, 2] (cos, sin) coordinates
    a = np.radians(np.asarray(dtheta_deg, dtype=float))
    rot = np.stack([c[:, 0] * np.cos(a) - c[:, 1] * np.sin(a), c[:, 0] * np.sin(a) + c[:, 1] * np.cos(a)], 1)
    return (rot - c) @ chart["A"].T


def probe_qr_delta(x, basis, target, periodic, mu, sd):
    """Part 1's steer in train-standardised coordinates, mapped back to raw activations."""
    from wm.steer import build_basis, encode, steer
    V = build_basis(list(basis["W"]))
    Xs = (x - mu) / sd
    t = encode(np.full(len(x), target), "circular" if periodic else "scalar")
    return (steer(Xs, V, basis, t, len(basis["W"])) - Xs) * sd, V.shape[1]


def refit_inlp_basis(d, rows, variable, max_rounds=None, rows_note="knot folds at kept values only"):
    """Part 1's probe sequence (wm.inlp.inlp) refit on `rows` only, standardised on those rows, alpha by CV over
    their folds. Returns (basis dict {Q, W, b}, (mu, sd), note). Q is in the rows' standardised coordinates; Q / sd
    gives raw-space directions."""
    from wm.inlp import inlp
    from wm.probes import cv_select_alpha, targets
    Y, kind, score_fn = targets(d["df"], variable)
    X = d["X"].astype(np.float64)
    mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-8
    Xs = (X - mu) / sd
    folds = d["fold"][rows]
    alpha = cv_select_alpha(Xs[rows], Y[rows], folds, score_fn=score_fn)["alpha"]
    summary, Q, W, b = inlp(Xs[rows], Y[rows], None, None, folds, alpha, score_fn, kind, max_rounds)   # nested only: no test read
    note = {"rows": rows_note, "n_rows": int(rows.sum()), "alpha": alpha,
            "n_probes": int(len(W)), "folds": sorted(int(f) for f in np.unique(folds))}
    return {"Q": Q, "W": W, "b": b}, (mu, sd), note


def refit_probe_basis(d, m, variable, max_rounds=None):
    """The probe-QR arm's basis refit on the knot rows at kept values only, so it never sees the probe folds (where
    both evaluators are fit), the test clips, or the held-out values."""
    return refit_inlp_basis(d, m["knot"], variable, max_rounds)


def arm_deltas(x, src, tgt, m, chart, basis, std, periodic):
    """Endpoint edits [n, D] of every available arm, before norm matching, and each arm's nominal rank."""
    pca = m["pca"]
    src_c = m["curve"].coord_of_value(src)
    tgt_c = m["curve"].coord_of_value(np.full(len(src), tgt))
    dZ_s = m["curve"](tgt_c) - m["curve"](src_c)
    dZ_l = mf.piecewise_linear_point(m["curve"], np.full(len(src), tgt)) - mf.piecewise_linear_point(m["curve"], src)
    full = m["full_curve"]
    mu_t = np.broadcast_to(mf.piecewise_linear_point(full, tgt), x.shape)
    mu_s = mf.piecewise_linear_point(full, src)
    k = pca.components.shape[0]
    out = {"spline": (pca.lift_delta(dZ_s), k), "chord": (pca.lift_delta(dZ_l), k),
           "centroid_transport": (mu_t - mu_s, x.shape[1]), "snap": (mu_t - x, x.shape[1])}
    if periodic and chart is not None:
        dtheta = (tgt - src + 180.0) % 360.0 - 180.0
        out["ring_rotation"] = (ring_rotation_delta(x, chart, dtheta), 2)
    if basis is not None:
        out["probe_qr"] = probe_qr_delta(x, basis, tgt, periodic, *std)
    if "raw_curve" in m:
        raw = m["raw_curve"]
        dZ_r = mf.piecewise_linear_point(raw, np.full(len(src), tgt)) - mf.piecewise_linear_point(raw, src)
        out["chord_raw"] = (pca.lift_delta(dZ_r), k)
    return out


def match_norm(delta, ref):
    """Rescale each row of delta to the norm of the same row of ref (rows with zero norm stay zero)."""
    n, r = np.linalg.norm(delta, axis=1, keepdims=True), np.linalg.norm(ref, axis=1, keepdims=True)
    return delta * np.where(n > 0, r / np.where(n > 0, n, 1.0), 0.0)


def run_bakeoff(d, m, picks, periodic, basis=None, seed=0, basis_std=None):
    """Steer every picked clip with every arm at the spline arm's ||delta||, read with both evaluators.
    Returns dict(arms={arm: summary}, n_clips, evaluators)."""
    probe_rows = d["role"] == "probe"
    evals = {"probe": mf.ProbeReadout(d["X"][probe_rows], d["y"][probe_rows], periodic),
             "mlp": MLPReadout(d["X"][probe_rows], d["y"][probe_rows], periodic, seed)}
    knot = m["knot"]
    chart = gc.fit_circular_chart(d["X"][knot], d["y"][knot]) if periodic else None
    tr = d["is_train"]
    std = basis_std or (d["X"][tr].mean(0), d["X"][tr].std(0) + 1e-8)
    acc = {}
    for tgt, pick in picks.items():
        x, src = d["X"][pick].astype(float), d["y"][pick]
        deltas = arm_deltas(x, src, tgt, m, chart, basis, std, periodic)
        ref = deltas["spline"][0]
        for arm, (delta, rank) in deltas.items():
            a = acc.setdefault(arm, {"rank": rank, "matched": {e: [] for e in evals}, "unmatched": {e: [] for e in evals},
                                     "deltas": [], "norm_unmatched": []})
            dm = match_norm(delta, ref)
            a["deltas"].append(dm)
            a["norm_unmatched"].append(np.linalg.norm(delta, axis=1))
            for e, ev in evals.items():
                a["matched"][e].append(mf.value_error(ev.predict(x + dm), tgt, periodic))
                a["unmatched"][e].append(mf.value_error(ev.predict(x + delta), tgt, periodic))
    out = {}
    for arm, a in acc.items():
        out[arm] = {"nominal_rank": int(a["rank"]), "effective_rank": effective_rank(np.concatenate(a["deltas"])),
                    "mean_norm_unmatched": float(np.concatenate(a["norm_unmatched"]).mean()),
                    **{f"err_{e}_matched": float(np.concatenate(v).mean()) for e, v in a["matched"].items()},
                    **{f"err_{e}_unmatched": float(np.concatenate(v).mean()) for e, v in a["unmatched"].items()}}
    base = {e: float(np.mean([mf.value_error(ev.predict(d["X"][p]), t, periodic).mean() for t, p in picks.items()]))
            for e, ev in evals.items()}
    return {"arms": out, "unsteered_err": base, "n_steered": int(sum(len(p) for p in picks.values()))}

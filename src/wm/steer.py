"""Step 3: multi-probe subspace steering (paper App. C.12, spec 5.3).

Basis V = QR([W_1ᵀ … W_Kᵀ]) from the step-2 probe sequence (train only). Steer: c = Vᵀx, x⊥ = x − Vc,
solve least squares for c* so the first n probes read the target, x* = Vc* + x⊥. The evaluation probe
is fit on the test activations (the paper's protocol) and never used to build V.
All vectors live in the train-standardised coordinates of steps 1 and 2.
"""
from pathlib import Path

import numpy as np

from wm.inlp import basis_path, load_basis
from wm.probes import (RESULTS, fit_ridge, layer_fraction, load_activations, load_sweep, load_table,
                       predict, score, split_rows, standardized_layer, targets, write_json)


def build_basis(W_list):
    """V [d, K·m]: orthonormal basis of the span of all probe weight columns."""
    V, _ = np.linalg.qr(np.hstack(list(W_list)))
    return V


def readout_weights(probes, n):
    """Weights and biases of the first n probes as functions of the raw (un-residualised) x.

    Probe k was trained on X⁽ᵏ⁾ = x − Q_<k Q_<kᵀ x, where Q_<k are the directions removed before
    round k. Its readout on x is therefore (x − Q_<k Q_<kᵀ x) W_k + b_k = x (W_k − Q_<k Q_<kᵀ W_k) + b_k,
    which is linear in x with effective weight W̃_k = P_<k W_k. We apply P_<k explicitly so each probe
    is evaluated on exactly the input it was trained on. (For our INLP output W_k is already orthogonal
    to Q_<k, so W̃_k = W_k; the projection only matters for probes from elsewhere.) Every W̃_k lies in
    span(W_1..W_k) ⊆ span(V), so the probes cannot see x⊥ and only c needs to change.
    Returns W̃ [d, n·m] and b̃ [n·m].
    """
    W, b, Q = probes["W"], probes["b"], probes["Q"]
    m = W.shape[2]
    cols = []
    for k in range(n):
        Q_prev = Q[:, :k * m]
        cols.append(W[k] - Q_prev @ (Q_prev.T @ W[k]))
    return np.hstack(cols), np.concatenate(list(b[:n]))


def steering_operator(V, probes, n):
    """(W̃, b̃, A⁺) for n probes, where A = W̃ᵀV maps subspace coordinates c to probe readouts."""
    Wt, bt = readout_weights(probes, n)
    return Wt, bt, np.linalg.pinv(Wt.T @ V)


def steer(X, V, probes, target, n_probes, operator=None):
    """Move X [N, d] (or [d]) inside span(V) so the first n_probes probes read `target`.

    target: the per-probe output, [m] or [N, m] ((sin θ*, cos θ*) for direction, the value for
    scalars); it is required of every one of the n probes. The equations Σ probes: W̃ᵀ(Vc* + x⊥) + b̃ = y*
    are linear in c* and, with n·m ≤ K·m, underdetermined; we take the least-squares solution closest
    to the clip's own coordinates: c* = c + A⁺(y* − ŷ(x)), where ŷ(x) is the probes' current readout.
    Then x* = Vc* + x⊥ = x + V(c* − c). Zero dose: n_probes = 0, or a target equal to the current
    readout, returns x unchanged.
    """
    X = np.asarray(X, dtype=float)
    if n_probes == 0:
        return X.copy()
    Wt, bt, A_pinv = operator or steering_operator(V, probes, n_probes)
    y_star = np.tile(np.asarray(target, float), n_probes)       # same target for each probe
    dc = (y_star - (X @ Wt + bt)) @ A_pinv.T
    return X + dc @ V.T


# ---------------------------------------------------------------- evaluation

def encode(values, kind):
    """Label values -> probe output space."""
    values = np.asarray(values, float)
    if kind == "circular":
        r = np.radians(values)
        return np.stack([np.sin(r), np.cos(r)], axis=-1)
    return values[..., None]


def decode(P, kind):
    return np.degrees(np.arctan2(P[:, 0], P[:, 1])) % 360.0 if kind == "circular" else P[:, 0]


def distance(a, b, kind):
    """Per-clip |a − b|; wrapped to [0, 180] for angles."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return np.minimum(d % 360.0, 360.0 - d % 360.0) if kind == "circular" else d


def evaluate(Xte, labels, probes, eval_W, eval_b, kind, single_target, all_targets, n_bins=8):
    """Steer every test clip and read it with the held-out evaluation probe, for n = 0..K.

    single_target: every clip steered to one value (θ* = 90° for direction).
    all_targets: every clip steered to each value in turn (the 64 label values); errors are also
    binned by the requested shift |label − target| (wrapped for angles).
    Norm ratio ‖x*‖/‖x‖ is per clip, in standardised coordinates.
    """
    V = build_basis(probes["W"])
    K = len(probes["W"])
    x_norm = np.linalg.norm(Xte, axis=1)
    read = lambda Xs: decode(predict(Xs, eval_W, eval_b), kind)
    shifts = np.stack([distance(labels, t, kind) for t in all_targets])            # [T, N]
    top = 180.0 if kind == "circular" else float(shifts.max())
    edges = np.linspace(0.0, top, n_bins + 1)
    bin_of = np.clip(np.digitize(shifts, edges[1:-1]), 0, n_bins - 1)

    single, sweep, heat = [], [], []
    ratios_at_K = None
    for n in range(K + 1):
        op = steering_operator(V, probes, n) if n else None
        Xs = steer(Xte, V, probes, encode(single_target, kind), n, op)
        r = read(Xs)
        ratio = np.linalg.norm(Xs, axis=1) / x_norm
        single.append({"n": n, "mae_to_target": float(distance(r, single_target, kind).mean()),
                       "mae_to_true": float(distance(r, labels, kind).mean()),
                       "norm_ratio_median": float(np.median(ratio)),
                       "norm_ratio_p05": float(np.percentile(ratio, 5)),
                       "norm_ratio_p95": float(np.percentile(ratio, 95))})
        if n == K:
            ratios_at_K = ratio
        err_t, err_true = [], []
        for t in all_targets:
            r = read(steer(Xte, V, probes, encode(t, kind), n, op))
            err_t.append(distance(r, t, kind))
            err_true.append(distance(r, labels, kind))
        err_t, err_true = np.stack(err_t), np.stack(err_true)
        sweep.append({"n": n, "mae_to_target": float(err_t.mean()), "mae_to_true": float(err_true.mean())})
        heat.append([float(err_t[bin_of == j].mean()) if (bin_of == j).any() else None for j in range(n_bins)])

    return {"K": K, "single_target": float(single_target), "single": single, "all_targets": sweep,
            "shift_bins": {"edges": edges.tolist(),
                           "counts": [int((bin_of == j).sum()) for j in range(n_bins)],
                           "mae_to_target": heat},
            "per_clip_at_K": {"shift": distance(labels, single_target, kind).tolist(),
                              "norm_ratio": ratios_at_K.tolist()}}


def run_steering(dataset, variable=None, point=None, pool="meanpool", act_root=None, results_dir=None,
                 inlp_dir=None):
    """Paper protocol: steering basis from the train probe sequence (step 2), evaluation probe fit on
    test activations with the step-1 α, test clips steered to θ* = 90° (median value for scalars) and
    to every label value. Writes results/p1c_{dataset}_L{point}.json."""
    variable = variable or dataset
    sweep = load_sweep(dataset, variable, pool, results_dir)
    point = sweep["availability"]["peak"] if point is None else point
    probes = load_basis(basis_path(dataset, variable, point, pool, inlp_dir))
    assert len(probes["W"]) > 0, "step 2 found no probe above chance at this layer"
    df = load_table(dataset)
    Y, kind, _ = targets(df, variable)
    assert kind in ("circular", "scalar"), "steering is defined for direction, speed and acceleration"
    tr, te, _ = split_rows(dataset, df)
    Xtr, Xte = standardized_layer(load_activations(dataset, pool, act_root=act_root), point, tr, te)
    labels = df["theta_degrees"].to_numpy(float)[te] if kind == "circular" else Y[te, 0]
    all_targets = np.unique(df["theta_degrees"] if kind == "circular" else Y[:, 0])
    single = 90.0 if kind == "circular" else float(np.median(all_targets))
    eval_W, eval_b = fit_ridge(Xte, Y[te], probes["alpha"])
    eval_fit = score(Y[te], predict(Xte, eval_W, eval_b), kind)
    res = evaluate(Xte, labels, probes, eval_W, eval_b, kind, single, all_targets)
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "point": point,
           "frac": layer_fraction(point), "alpha": probes["alpha"], "n_test": len(te),
           "eval_probe_in_sample": eval_fit, "space": "train-standardised activations", **res}
    s = res["single"]
    print(f"{dataset}/{variable} point {point}: K={res['K']} MAE-to-target {s[0]['mae_to_target']:.3g} -> "
          f"{s[-1]['mae_to_target']:.3g}, MAE-to-true {s[0]['mae_to_true']:.3g} -> {s[-1]['mae_to_true']:.3g}")
    suffix = ("" if variable == dataset else f"_{variable}") + ("" if pool == "meanpool" else f"_{pool}")
    write_json(Path(results_dir or RESULTS) / f"p1c_{dataset}_L{point}{suffix}.json", out)
    return out

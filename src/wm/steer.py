"""Step 3: multi-probe subspace steering (paper App. C.12, spec 5.3).

Basis V = QR([W_1ᵀ … W_Kᵀ]) from the step-2 probe sequence (train only). Steer: c = Vᵀx, x⊥ = x − Vc,
solve least squares for c* so the first n probes read the target, x* = Vc* + x⊥. The evaluation probe
is fit on the test activations (the paper's protocol) and never used to build V.
All vectors live in the train-standardised coordinates of steps 1 and 2.
"""
from functools import partial
from pathlib import Path

import numpy as np

from wm.inlp import basis_path, load_basis
from wm.metrics import radius_summary, readout_radius
from wm.splits import load_split
from wm.probes import (ALPHAS, RESULTS, cv_select_alpha, drop_nan_clips, fit_ridge, layer_fraction, load_activations,
                       load_sweep, load_table, predict, score, split_rows, standardized_layer, targets,
                       write_json)


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


def steer_delta(X, V, probes, target, n_probes, operator=None, per_probe=False):
    """The coordinate change dc = c* − c [N, K·m] that steer() applies through V (x* = x + V dc).

    per_probe=False: target is one probe output, [m] or [N, m], required of each of the n probes.
    per_probe=True: target holds a separate output for every probe of the sequence, [K·m] or [N, K·m]
    (probe-major); the first n·m entries are used (the radius-matched arm, see matched_targets).
    """
    X = np.asarray(X, dtype=float)
    Wt, bt, A_pinv = operator or steering_operator(V, probes, n_probes)
    target = np.asarray(target, float)
    if per_probe:
        y_star = target[..., :Wt.shape[1]]
    else:
        y_star = np.tile(target, n_probes)                      # same target for each probe
    return (y_star - (X @ Wt + bt)) @ A_pinv.T


def steer(X, V, probes, target, n_probes, operator=None, per_probe=False):
    """Move X [N, d] (or [d]) inside span(V) so the first n_probes probes read `target`.

    target: the per-probe output, [m] or [N, m] ((sin θ*, cos θ*) for direction, the value for
    scalars); it is required of every one of the n probes (per_probe=True: one output per probe, see
    steer_delta). The equations Σ probes: W̃ᵀ(Vc* + x⊥) + b̃ = y*
    are linear in c* and, with n·m ≤ K·m, underdetermined; we take the least-squares solution closest
    to the clip's own coordinates: c* = c + A⁺(y* − ŷ(x)), where ŷ(x) is the probes' current readout.
    Then x* = Vc* + x⊥ = x + V(c* − c). Zero dose: n_probes = 0, or a target equal to the current
    readout, returns x unchanged. V may be any orthonormal basis containing enough of the probes'
    span to solve (a random basis gives the calibrated-random null, random_nulls).
    """
    X = np.asarray(X, dtype=float)
    if n_probes == 0:
        return X.copy()
    return X + steer_delta(X, V, probes, target, n_probes, operator, per_probe) @ V.T


def matched_targets(Xtr, labels_tr, probes, values):
    """Radius-matched steering targets: for each value, every probe's mean readout over the real train
    clips with that label. Returns [T, K·m] (probe-major, for per_probe=True).

    Ridge readouts shrink (radius < 1 for direction, towards the mean for scalars), so asking every
    probe for the unit vector (sin θ*, cos θ*) pushes activations beyond the data. Here each probe is
    asked for what it reads on real clips at the target, its own shrinkage included (later probes of
    the sequence are weaker and read smaller radii)."""
    Wt, bt = readout_weights(probes, len(probes["W"]))
    readout = Xtr @ Wt + bt
    labels_tr = np.asarray(labels_tr, float)
    out = []
    for v in np.atleast_1d(values):
        rows = np.isclose(labels_tr, v)
        assert rows.any(), f"no train clip with label {v}"
        out.append(readout[rows].mean(axis=0))
    return np.stack(out)


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


def evaluate(Xte, labels, probes, eval_W, eval_b, kind, single_target, all_targets, n_bins=8,
             single_full=None, all_full=None, off=None):
    """Steer every test clip and read it with the held-out evaluation probe, for n = 0..K.

    single_target: every clip steered to one value (θ* = 90° for direction).
    all_targets: every clip steered to each value in turn (the 64 label values); errors are also
    binned by the requested shift |label − target| (wrapped for angles).
    single_full [K·m], all_full [T, K·m]: per-probe targets (the radius-matched arm, matched_targets);
    None = the paper's target, encode(value) required of every probe.
    Norm ratio ‖x*‖/‖x‖ is per clip, in standardised coordinates. For direction the evaluation
    probe's readout radius ‖(ŝ, ĉ)‖ is logged beside the angle (per n, and per clip at n = 0 and K).
    off: optional off-target probe (off_target_probe) read on the single-target steered clips at every n.
    """
    V = build_basis(probes["W"])
    K = len(probes["W"])
    per_probe = single_full is not None
    y_single = single_full if per_probe else encode(single_target, kind)
    y_all = all_full if per_probe else [encode(t, kind) for t in all_targets]
    x_norm = np.linalg.norm(Xte, axis=1)
    read = lambda Xs: predict(Xs, eval_W, eval_b)
    shifts = np.stack([distance(labels, t, kind) for t in all_targets])            # [T, N]
    top = 180.0 if kind == "circular" else float(shifts.max())
    edges = np.linspace(0.0, top, n_bins + 1)
    bin_of = np.clip(np.digitize(shifts, edges[1:-1]), 0, n_bins - 1)

    single, sweep, heat = [], [], []
    ratios_at_K = radius_at_K = radius_at_0 = None
    for n in range(K + 1):
        op = steering_operator(V, probes, n) if n else None
        Xs = steer(Xte, V, probes, y_single, n, op, per_probe)
        P = read(Xs)
        r = decode(P, kind)
        ratio = np.linalg.norm(Xs, axis=1) / x_norm
        row = {"n": n, "mae_to_target": float(distance(r, single_target, kind).mean()),
               "mae_to_true": float(distance(r, labels, kind).mean()),
               "norm_ratio_median": float(np.median(ratio)),
               "norm_ratio_p05": float(np.percentile(ratio, 5)),
               "norm_ratio_p95": float(np.percentile(ratio, 95))}
        if kind == "circular":
            row["radius"] = radius_summary(P)
            if n == 0:
                radius_at_0 = readout_radius(P)
        if off is not None:
            r_off = decode(predict(Xs[off["mask"]], off["W"], off["b"]), off["kind"])
            if n == 0:
                off_before = r_off
            row["off_target"] = {"mae_to_true": float(distance(r_off, off["labels"], off["kind"]).mean()),
                                 "mean_abs_change": float(distance(r_off, off_before, off["kind"]).mean())}
        single.append(row)
        if n == K:
            ratios_at_K = ratio
            radius_at_K = readout_radius(P) if kind == "circular" else None
        err_t, err_true, radii = [], [], []
        for t, y in zip(all_targets, y_all):
            P = read(steer(Xte, V, probes, y, n, op, per_probe))
            r = decode(P, kind)
            err_t.append(distance(r, t, kind))
            err_true.append(distance(r, labels, kind))
            if kind == "circular":
                radii.append(readout_radius(P))
        err_t, err_true = np.stack(err_t), np.stack(err_true)
        row = {"n": n, "mae_to_target": float(err_t.mean()), "mae_to_true": float(err_true.mean())}
        if kind == "circular":
            row["radius_median"] = float(np.median(radii))
        sweep.append(row)
        heat.append([float(err_t[bin_of == j].mean()) if (bin_of == j).any() else None for j in range(n_bins)])

    per_clip = {"shift": distance(labels, single_target, kind).tolist(), "norm_ratio": ratios_at_K.tolist()}
    if kind == "circular":
        per_clip["radius"] = radius_at_K.round(4).tolist()
        per_clip["radius_unsteered"] = radius_at_0.round(4).tolist()
    return {"K": K, "single_target": float(single_target), "single": single, "all_targets": sweep,
            "shift_bins": {"edges": edges.tolist(),
                           "counts": [int((bin_of == j).sum()) for j in range(n_bins)],
                           "mae_to_target": heat},
            "per_clip_at_K": per_clip}


def _band(vals):
    vals = np.asarray(vals, float)
    return {"mean": float(vals.mean()), "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)), "draws": vals.round(4).tolist()}


def empirical_p(learned, draws):
    """Rank of the learned MAE-to-target among the null draws, as (1 + #{draws ≤ learned}) / (m + 1).
    Lower MAE is better, so 1/(m+1) (the resolution) means the learned steer beats every draw and
    1 means it is worse than all of them."""
    draws = np.asarray(draws, float)
    return float((1 + np.sum(draws <= learned)) / (len(draws) + 1))


def random_nulls(Xte, labels, probes, eval_W, eval_b, kind, single_target, n_draws=20, seed=0,
                 target_full=None):
    """Two random nulls for the single-target steer, n_draws draws each, at every n = 1..K.

    random_basis (calibrated random): a random orthonormal R [d, K·m], the rank of V, with its own
      least-squares solve, x* = x + R (W̃ᵀR)⁺ (y* − ŷ(x)). The steering probes still read the target
      (W̃ᵀR has full row rank almost surely), but the edit lives in a random subspace.
    random_orientation: the learned coordinate change dc (steer_delta) applied through a random
      orthonormal Q [d, K·m], x* = x + Q dc, so ‖Q dc‖ = ‖V dc‖ per clip: same per-clip dose, random
      direction.
    Draws are fixed across n (draw j uses the same basis at every n). Each null reports the band of
    MAE-to-target and MAE-to-true over draws (mean, 5th and 95th percentile, and the draws), the band
    of the median norm ratio, and the empirical rank of the learned MAE-to-target (empirical_p).
    target_full: per-probe targets [K·m] (radius-matched arm); None = the paper's unit target.
    """
    V = build_basis(probes["W"])
    K, d, r = len(probes["W"]), Xte.shape[1], V.shape[1]
    per_probe = target_full is not None
    y = target_full if per_probe else encode(single_target, kind)
    rng = np.random.default_rng(seed)
    R_draws = [np.linalg.qr(rng.standard_normal((d, r)))[0] for _ in range(n_draws)]
    Q_draws = [np.linalg.qr(rng.standard_normal((d, r)))[0] for _ in range(n_draws)]
    x_norm = np.linalg.norm(Xte, axis=1)

    def scores(Xs):
        P = predict(Xs, eval_W, eval_b)
        ang = decode(P, kind)
        out = {"mae_to_target": float(distance(ang, single_target, kind).mean()),
               "mae_to_true": float(distance(ang, labels, kind).mean()),
               "norm_ratio_median": float(np.median(np.linalg.norm(Xs, axis=1) / x_norm))}
        if kind == "circular":
            out["radius_median"] = float(np.median(readout_radius(P)))
        return out

    rows = []
    for n in range(1, K + 1):
        dc = steer_delta(Xte, V, probes, y, n, None, per_probe)
        learned = scores(Xte + dc @ V.T)
        row = {"n": n, "learned": learned}
        for name, draws in (("random_basis", [scores(steer(Xte, R, probes, y, n, None, per_probe)) for R in R_draws]),
                            ("random_orientation", [scores(Xte + dc @ Q.T) for Q in Q_draws])):
            row[name] = {key: _band([s[key] for s in draws]) for key in draws[0]}
            row[name]["empirical_p_to_target"] = empirical_p(learned["mae_to_target"],
                                                             [s["mae_to_target"] for s in draws])
        rows.append(row)
    return {"n_draws": n_draws, "rank": r, "seed": seed,
            "rank_rule": "empirical_p_to_target = (1 + #{null draws with MAE-to-target <= learned}) / (n_draws + 1); "
                         "resolution 1/(n_draws + 1), small = learned beats the null",
            "rows": rows}


def grouped_folds(groups, k=5, seed=0):
    """Fold per row with identical clips (same group, e.g. frame hash) kept in one fold."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    return (np.random.default_rng(seed).permutation(len(uniq)) % k)[inv]


def stratified_halves(groups, labels, seed=0):
    """Bool mask of half A: identical-clip units dealt alternately to A and B within each label value."""
    rng = np.random.default_rng(seed)
    groups, labels = np.asarray(groups), np.asarray(labels)
    in_a = np.zeros(len(groups), bool)
    turn = 0
    for v in np.unique(labels):
        units = rng.permutation(np.unique(groups[labels == v]))
        for u in units:
            in_a[groups == u] = turn % 2 == 0
            turn += 1
    return in_a


def eval_probe_cv(Xte, Yte, kind, seed=0, groups=None):
    """Evaluation probe on the test activations (the paper's protocol), with α chosen by 5-fold CV
    inside test rather than taken from step 1; folds keep identical clips (groups = frame hash) together.
    Returns (W, b, report): the in-sample fit (the paper reports this: 0.99 from 103 clips in d = 1024)
    and the out-of-fold score at the chosen α (n < d here too, so in-sample R² is optimistic)."""
    groups = np.arange(len(Yte)) if groups is None else groups
    folds = grouped_folds(groups, 5, seed)
    cv = cv_select_alpha(Xte, Yte, folds, ALPHAS, partial(score, kind=kind))
    W, b = fit_ridge(Xte, Yte, cv["alpha"])
    fit = score(Yte, predict(Xte, W, b), kind)
    report = {"alpha": cv["alpha"], "alpha_rule": "5-fold CV inside test, folds grouped by frame hash (identical clips together), seed 0, 13 log-spaced values",
              "in_sample": {"r2": fit["r2"], "mae": fit["mae"]},
              "out_of_fold": {"r2_fold_mean": cv["cv_mean"], "r2_fold_sd": cv["cv_sd"],
                              "mae_fold_mean": cv["cv_mae_mean"], "mae_fold_sd": cv["cv_mae_sd"]}}
    return W, b, report


def off_target_probe(Xtr, Xte, df, tr, te, folds, kind):
    """The off-target readout (spec 5.3, §6 item 4): steering direction should leave speed alone and vice
    versa. A ridge probe for the other variable, fit on train clips of the same dataset (α by CV on the
    train folds, same coordinates as the steer) and read on the steered test clips. Direction set: speed
    probe on constant-speed clips only (the accelerating half has no single speed). Speed and
    acceleration sets: direction probe on all clips (64 directions, fully crossed)."""
    if kind == "circular":
        name, off_kind = "speed", "scalar"
        y = df["speed_mps"].to_numpy(float)
        motion = df["motion"].to_numpy()
        m_tr, m_te = motion[tr] == "velocity", motion[te] == "velocity"
        Y = y[:, None]
        labels = y[te][m_te]
    else:
        name, off_kind = "direction", "circular"
        th = np.radians(df["theta_degrees"].to_numpy(float))
        Y = np.stack([np.sin(th), np.cos(th)], axis=1)
        m_tr, m_te = np.ones(len(tr), bool), np.ones(len(te), bool)
        labels = df["theta_degrees"].to_numpy(float)[te]
    cv = cv_select_alpha(Xtr[m_tr], Y[tr][m_tr], folds[m_tr], ALPHAS, partial(score, kind=off_kind))
    W, b = fit_ridge(Xtr[m_tr], Y[tr][m_tr], cv["alpha"])
    info = {"variable": name, "alpha": cv["alpha"], "cv_r2_train": cv["cv_mean"], "n_test_clips": int(m_te.sum()),
            "fit": "train clips of this dataset" + (" with motion == velocity" if kind == "circular" else "")}
    return {"W": W, "b": b, "kind": off_kind, "mask": m_te, "labels": labels}, info


def run_steering(dataset, variable=None, point=None, pool="meanpool", act_root=None, results_dir=None,
                 inlp_dir=None, n_draws=20, layer_role=None, model="vjepa2", strict_eval=False):
    """Paper protocol: steering basis from the train probe sequence (step 2), evaluation probe fit on
    test activations (α by CV inside test), test clips steered to θ* = 90° (for scalars, the upper
    median label value) and to every label value, with the paper's unit target. Extras, labelled in
    the output: a radius-matched arm (matched_targets) and two random nulls per n (random_nulls).
    Writes results/p1c_{dataset}_L{point}.json."""
    variable = variable or dataset
    sweep = load_sweep(dataset, variable, pool, results_dir, model)
    point = sweep["availability"]["peak"] if point is None else point
    probes = load_basis(basis_path(dataset, variable, point, pool, inlp_dir, model))
    assert len(probes["W"]) > 0, "step 2 found no probe above chance at this layer"
    df = load_table(dataset)
    Y, kind, _ = targets(df, variable)
    assert kind in ("circular", "scalar"), "steering is defined for direction, speed and acceleration"
    tr, te, folds = split_rows(dataset, df)
    acts = load_activations(dataset, pool, model, act_root)
    tr, te, folds, n_nan = drop_nan_clips(acts, tr, te, folds)
    Xtr, Xte = standardized_layer(acts, point, tr, te)
    off, off_info = off_target_probe(Xtr, Xte, df, tr, te, folds, kind)
    label_all = df["theta_degrees"].to_numpy(float) if kind == "circular" else Y[:, 0]
    labels = label_all[te]
    all_targets = np.unique(label_all)
    single = 90.0 if kind == "circular" else float(all_targets[len(all_targets) // 2])
    hashes = load_split(dataset)["frame_hash"]
    groups = np.array([hashes[str(i)] for i in df["id"].to_numpy()[te]])
    eval_W, eval_b, eval_report = eval_probe_cv(Xte, Y[te], kind, groups=groups)
    res = evaluate(Xte, labels, probes, eval_W, eval_b, kind, single, all_targets, off=off)
    nulls = random_nulls(Xte, labels, probes, eval_W, eval_b, kind, single, n_draws)
    m_all = matched_targets(Xtr, label_all[tr], probes, all_targets)
    m_single = m_all[np.flatnonzero(np.isclose(all_targets, single))[0]]
    res_m = evaluate(Xte, labels, probes, eval_W, eval_b, kind, single, all_targets,
                     single_full=m_single, all_full=m_all, off=off)
    res_m["random_nulls"] = random_nulls(Xte, labels, probes, eval_W, eval_b, kind, single, n_draws,
                                         target_full=m_single)
    m = probes["W"].shape[2]
    res_m["target_single_per_probe"] = m_single.reshape(-1, m).round(4).tolist()
    if kind == "circular":
        res_m["target_single_radius_per_probe"] = np.linalg.norm(m_single.reshape(-1, m), axis=1).round(4).tolist()
    res_m["target_rule"] = ("extra arm: probe k's target is its mean readout over train clips whose label equals "
                            "the target value (its own shrinkage included); the paper's arm asks every probe for "
                            "the unit/true value")
    if strict_eval:   # extra: probe fit on one stratified half of test, steering evaluated on the other
        in_a = stratified_halves(groups, labels)
        strict = {"rule": "test split into two halves stratified by label, identical clips kept together; evaluation "
                          "probe fit on one half (alpha by grouped CV inside it), the other half steered and read",
                  "n_A": int(in_a.sum()), "n_B": int((~in_a).sum())}
        for name, fit_rows, steer_rows in (("A_to_B", in_a, ~in_a), ("B_to_A", ~in_a, in_a)):
            W_h, b_h, rep_h = eval_probe_cv(Xte[fit_rows], Y[te][fit_rows], kind, groups=groups[fit_rows])
            r = evaluate(Xte[steer_rows], labels[steer_rows], probes, W_h, b_h, kind, single, all_targets)
            strict[name] = {"eval_probe": rep_h, "single": r["single"], "all_targets": r["all_targets"]}
        res["strict_eval"] = strict
    out = {"dataset": dataset, "variable": variable, "kind": kind, "pool": pool, "model": model, "point": point,
           "frac": layer_fraction(point), "alpha": probes["alpha"], "n_test": len(te),
           "layer_role": layer_role, "is_peak": point == sweep["availability"]["peak"],
           "is_onset": point == sweep["availability"]["onset"], "eval_probe": eval_report, "eval_probe_in_sample": eval_report["in_sample"],
           "space": "train-standardised activations", "all_nan_clips_excluded": n_nan,
           "protocol": "paper protocol (C.12): the evaluation probe is fit on the same test clips it steers; "
                       "strict_eval (if run) is the extra split-half version",
           "arm": "paper: unit target (sin θ*, cos θ*) / true value required of every probe", **res,
           "random_nulls": nulls, "radius_matched": res_m, "off_target_probe": off_info}
    s = res["single"]
    print(f"{dataset}/{variable} point {point}: K={res['K']} MAE-to-target {s[0]['mae_to_target']:.3g} -> "
          f"{s[-1]['mae_to_target']:.3g}, MAE-to-true {s[0]['mae_to_true']:.3g} -> {s[-1]['mae_to_true']:.3g}; "
          f"eval probe R2 in-sample {eval_report['in_sample']['r2']:.3f}, out-of-fold "
          f"{eval_report['out_of_fold']['r2_fold_mean']:.3f}")
    suffix = (("" if variable == dataset else f"_{variable}") + ("" if pool == "meanpool" else f"_{pool}")
              + ("" if model == "vjepa2" else f"_{model}"))
    write_json(Path(results_dir or RESULTS) / f"p1c_{dataset}_L{point}{suffix}.json", out)
    return out

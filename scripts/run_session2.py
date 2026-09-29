"""GPU session 2 (spec §6 items 2 and 5, PRIORITIES Tier 2 S2/S6/S8): propagation, predictor readout, time-reversed
extraction, and the stimulus-difficulty extraction. One script, several stages:

  plan             CPU, local. From stored activations only: carriers, targets, and every per-clip edit delta (raw
                   activation space, d = 1024) for each (layer, arm, carrier, target). -> artifacts/session2/plan.npz
  parity           box. Re-encode direction ids 0..15 at batch 16 and compare with the stored meanpool (Δ = 0 check at
                   the extraction's own batch composition). -> artifacts/session2/forward/parity.json
  forward          box (or CPU smoke). Source clip full + context-only encodings, twins rendered and encoded, every
                   edit propagated to the final LN (meanpool at every later point) and through the predictor
                   (context frames 1-8 only). Resumable per carrier group. -> artifacts/session2/forward/group_*.npz
  timerev          box. All 1500 direction clips with frame order reversed -> artifacts/activations/direction/vjepa2_timerev/
  extract-stimuli  box. paper_layout + hard stimulus sets (scripts/render_hard_stimuli.py), vjepa2 and random models,
                   batch 16, extract.py pooling -> artifacts/activations/stimuli_{set}/{model}/
  score            CPU, local. Readouts (a) probes refit per read layer, (b) agreement with the rendered twin and with
                   real target-value clips, (c) predictor recovery R, plus controls. -> results/session2_*.json

Arms (edits at layer L, all added to every token): probe_qr (Part 1 C.12 least squares at N = K, unit target),
radius_matched (per-probe target = that probe's mean readout radius on basis clips), spline (Part 2 periodic spline
endpoint, shift mode, residual kept; interpolating, Goodfire A.3), spline_smooth (the count-weighted smoothing spline,
Goodfire B.1: at the full plan the interpolating spline's endpoint delta is 1.25-10.7x the natural centroid change
across the held-out arc, 10.7x at L12; the smoothing spline's is 0.6-0.9x), chord (straight line between the polyline points, endpoint), random_matched
(probe_qr's coordinates through a random orthonormal basis of the same rank, rescaled to probe_qr's raw-space norm);
zero (the unedited clip) and shuffled_target (each edit scored against a different target's twin / value of the same
carrier; no extra forward pass) are computed at score time.

Design (defaults): targets are the 8 values of the contiguous 45-degree held-out arc (manifold.heldout_design, seed
0); the spline, the chord and the Part 1 basis are all built from knot clips (train folds 0-2) at the kept values
(--part1-basis design; 'paper' uses all train clips and the stored step-2 basis where it exists); read probes are fit
on probe clips (train folds 3-4, all 64 values); carriers are test clips. Edits made in train-standardised
coordinates are mapped to raw space as delta_raw = delta_std * sd (sd of the rows that built the basis).

  python scripts/run_session2.py plan  [--n-carriers 200 --n-targets 4 --layers 2 8 12 22]
  python scripts/run_session2.py forward [--group 8 --batch-size 16]
  python scripts/run_session2.py score
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm import manifold as mf  # noqa: E402
from wm.data import PROJECT_ROOT, decode, disk_mask, frame_hash, load_table  # noqa: E402
from wm.inlp import basis_path, inlp, load_basis, save_basis  # noqa: E402
from wm.p2_data import load_inputs  # noqa: E402
from wm.probes import (Standardizer, cv_select_alpha, fit_ridge, load_sweep, predict, score,  # noqa: E402
                       targets)
from wm.provenance import provenance  # noqa: E402
from wm.steer import build_basis, encode as encode_target, readout_weights, steer_delta, steering_operator  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
RESULTS = PROJECT_ROOT / "results"
ACT_ROOT = PROJECT_ROOT / "artifacts" / "activations"
STIMULI = PROJECT_ROOT / "artifacts" / "stimuli"
DATASET = "direction"
EDIT_ARMS = ("probe_qr", "radius_matched", "spline", "spline_smooth", "chord", "random_matched")
GPU_S_PER_CLIP = 0.16          # results/gpu_session1.json: RTX 4080 SUPER, batch 16, full 16-frame forward
D = 1024


def wrap(d):
    d = np.abs(np.asarray(d, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def angle_of(P):
    return np.degrees(np.arctan2(P[..., 0], P[..., 1])) % 360.0


def write(path, obj, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {**obj, "provenance": provenance(seeds=fields.pop("seeds", None), **fields)}
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ plan (CPU, stored activations)

def default_layers():
    av = load_sweep(DATASET, DATASET)["availability"]
    return sorted({av["onset"], 8, 12, av["peak"]})


def inlp_code_hash():
    """sha256 (first 12 hex) of the code that produces a probe sequence (wm/inlp.py + wm/probes.py as on disk), so a
    cached basis is never reused after either file changes."""
    h = hashlib.sha256()
    for name in ("inlp.py", "probes.py"):
        h.update((PROJECT_ROOT / "src" / "wm" / name).read_bytes())
    return h.hexdigest()[:12]


def basis_cache_path(out_dir, L, mode, holdout, code, alpha):
    """Refit cache file; keyed by α too, since α is read from the step-1 sweep JSON, which can be regenerated."""
    return Path(out_dir) / f"inlp_{DATASET}_L{L}_{mode}_{holdout}_{code}_a{float(alpha):.6g}.npz"


def check_stored_basis(probes, Xtr_std, Ytr, L, alpha):
    """The stored step-2 basis must come from exactly these standardised rows: same point and alpha, and its first
    probe must equal a fresh ridge fit on them (round 1 removes nothing, so W_1 = fit_ridge(X_train_std, Y, alpha))."""
    assert probes["point"] == L, (probes["point"], L)
    assert np.isclose(probes["alpha"], alpha), (probes["alpha"], alpha)
    W1, _ = fit_ridge(Xtr_std, Ytr, alpha)
    err = float(np.abs(W1 - probes["W"][0]).max() / np.abs(W1).max())
    assert err < 1e-6, f"stored basis at L{L} was not fit on these train rows / this standardiser (rel {err:.2e})"
    return err


def part1_basis(d, rows, L, mode, holdout, out_dir):
    """Standardiser + INLP probe sequence for the Part 1 arms at layer L.

    mode 'design': fit on `rows` (knot clips at kept values), standardised on those same rows.
    mode 'paper': the Part 1 protocol, all train clips standardised on all train clips (as probes.standardized_layer);
      reuses artifacts/inlp/direction_direction_L{L}.npz when present, after check_stored_basis confirms it was fit on
      exactly that frame, else refits. Held-out target values are then inside the basis (flagged in the output).
    Refits are cached in out_dir keyed by mode, holdout and inlp_code_hash()."""
    df = d["df"]
    Y, kind, score_fn = targets(df, DATASET)
    alpha = load_sweep(DATASET, DATASET)["layers"][L]["alpha"]
    if mode == "paper":
        rows = d["is_train"]
    st = Standardizer().fit(d["X"][rows].astype(np.float64))
    Xb = st.transform(d["X"][rows].astype(np.float64))
    if mode == "paper" and basis_path(DATASET, DATASET, L).exists():
        probes = load_basis(basis_path(DATASET, DATASET, L))
        err = check_stored_basis(probes, Xb, Y[rows], L, alpha)
        return st, probes, f"stored step-2 basis (all train clips; first-probe refit rel diff {err:.1e})", rows
    code = inlp_code_hash()
    cache = basis_cache_path(out_dir, L, mode, holdout, code, alpha)
    if cache.exists():
        return st, load_basis(cache), f"cached {cache.name}", rows
    probe = d["role"] == "probe"
    Xp = st.transform(d["X"][probe].astype(np.float64))
    t0 = time.time()
    summary, Q, W, b = inlp(Xb, Y[rows], Xp, Y[probe], d["fold"][rows], alpha, score_fn, kind)
    save_basis(cache, Q, W, b, alpha, L, kind)
    print(f"  INLP L{L} ({mode}): K={summary['K']} in {time.time() - t0:.0f} s")
    return st, load_basis(cache), f"INLP refit on {int(rows.sum())} clips ({mode}), alpha {alpha}, code {code}", rows


def layer_deltas(d, L, y, carriers, tg, held, arms, out, holdout="contiguous", part1_mode="design", k=64,
                 spline="interp", waypoints=0, angle=None):
    """Every arm's per-clip edit delta at layer L (raw space) for carriers x targets tg -> [C, A, T, D], plus meta
    (X, K, V, basis source, basis rows, per-probe radius, angle choice). angle: None = the session-2 rule
    (mf.choose_angle_source); "labels" forces the labels angle in the activation plane (choose_angle_source's own
    labels branch), i.e. monotone knot order."""
    C, T = tg.shape
    X = d["X"].astype(np.float64)
    knot = (d["role"] == "knot") & ~np.isin(y, held if holdout != "none" else [])
    st, probes, bsrc, rows_b = part1_basis(d, knot, L, part1_mode, holdout, out)
    V = build_basis(probes["W"])
    K = len(probes["W"])
    op = steering_operator(V, probes, K)
    Xc = X[carriers]
    Zs = st.transform(Xc)
    deltas = np.zeros((C, len(arms), T, D), np.float32)
    # Part 1: probe-QR at N = K (unit target) and radius-matched targets
    Wt, bt = readout_weights(probes, K)
    rb = np.linalg.norm((st.transform(X[rows_b]) @ Wt + bt).reshape(-1, K, 2), axis=-1).mean(0)           # [K]
    Rr = np.linalg.qr(np.random.default_rng(1000 + L).standard_normal((D, V.shape[1])))[0]
    for j in range(T):
        yt = encode_target(tg[:, j], "circular")                                                       # [C, 2]
        dc = steer_delta(Zs, V, probes, yt, K, op)
        full_t = (rb[None, :, None] * yt[:, None, :]).reshape(C, -1)
        dc_m = steer_delta(Zs, V, probes, full_t, K, op, per_probe=True)
        d_qr = (dc @ V.T) * st.std
        deltas[:, arms.index("probe_qr"), j] = d_qr
        deltas[:, arms.index("radius_matched"), j] = (dc_m @ V.T) * st.std
        d_r = (dc @ Rr.T) * st.std
        scale = np.linalg.norm(d_qr, axis=1) / np.maximum(np.linalg.norm(d_r, axis=1), 1e-12)
        deltas[:, arms.index("random_matched"), j] = d_r * scale[:, None]
    # Part 2: periodic spline and chord in raw PCA-k space, residual kept (additive, shift mode)
    pca = mf.fit_pca(X[knot], k)
    cent = mf.centroids(pca.project(X[knot]), y[knot])
    choice = mf.choose_angle_source(cent["C"], cent["values"])
    if angle == "labels":
        choice = {"angle": "labels", "plane": "activation"}
    curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=spline)
    curve_s = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline="smooth")
    Z = pca.project(Xc)
    src = y[carriers]
    Kw = max(2, waypoints)
    for j in range(T):
        ta, tb = curve.coord_of_value(src), curve.coord_of_value(tg[:, j])
        Zsp = mf.manifold_coords(Z, curve, ta, tb, Kw)
        Zch = mf.linear_coords(Z, mf.piecewise_linear_point(curve, src), mf.piecewise_linear_point(curve, tg[:, j]), Kw)
        deltas[:, arms.index("spline"), j] = pca.lift_delta(Zsp[:, -1] - Z)
        deltas[:, arms.index("spline_smooth"), j] = pca.lift_delta(
            mf.manifold_coords(Z, curve_s, curve_s.coord_of_value(src), curve_s.coord_of_value(tg[:, j]), 2)[:, -1] - Z)
        deltas[:, arms.index("chord"), j] = pca.lift_delta(Zch[:, -1] - Z)
        for w in range(1, Kw - 1):
            if f"spline_wp{w}" in arms:
                deltas[:, arms.index(f"spline_wp{w}"), j] = pca.lift_delta(Zsp[:, w] - Z)
                deltas[:, arms.index(f"chord_wp{w}"), j] = pca.lift_delta(Zch[:, w] - Z)
    return deltas, {"X": X, "K": K, "V": V, "bsrc": bsrc, "rows_b": rows_b, "rb": rb, "choice": choice}


def plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = args.layers or default_layers()
    rng = np.random.default_rng(args.seed)
    d0 = load_inputs(DATASET, layers[0])
    df, y = d0["df"], d0["y"]
    values = np.unique(y)
    if args.holdout == "none":
        held, hinfo = values, {"design": "none", "note": "no held-out values; spline and chord endpoints coincide at knots"}
    else:
        mask, hinfo = mf.heldout_design(values, args.holdout, True, seed=args.seed)
        held = values[mask]
    test = np.flatnonzero(d0["role"] == "test")
    carriers = np.sort(rng.choice(test, size=min(args.n_carriers, len(test)), replace=False))
    tg = np.zeros((len(carriers), args.n_targets))
    for i, c in enumerate(carriers):
        pool = held[~np.isclose(held, y[c])]
        tg[i] = rng.choice(pool, size=args.n_targets, replace=False)
    arms = list(EDIT_ARMS)
    if args.waypoints > 2:
        arms += [f"{a}_wp{k}" for a in ("spline", "chord") for k in range(1, args.waypoints - 1)]
    info = {"layers": layers, "carrier_rows": carriers.tolist(), "carrier_ids": df["id"].to_numpy()[carriers].tolist(),
            "targets": tg.tolist(), "arms": arms, "edit_arms": list(EDIT_ARMS), "holdout": hinfo,
            "held_values": held.tolist(), "n_targets": args.n_targets, "part1_basis": args.part1_basis,
            "spline": args.spline, "k": args.k, "waypoints": args.waypoints, "seed": args.seed,
            "inlp_code_sha256_12": inlp_code_hash(), "per_layer": {}}
    if args.part1_basis == "paper" and args.holdout != "none":
        info["part1_basis_note"] = "paper basis: all train clips, so the held-out target values are inside the Part 1 basis"
    arrays = {"carrier_rows": carriers, "carrier_ids": df["id"].to_numpy()[carriers], "targets": tg}
    full = np.load(ACT_ROOT / DATASET / "vjepa2" / "meanpool.npy", mmap_mode="r")
    arrays["carrier_stored_meanpool"] = np.asarray(full[carriers], np.float32)
    arrays["parity_ids"] = np.arange(16)
    arrays["parity_stored_meanpool"] = np.asarray(full[:16], np.float32)
    C, T = tg.shape
    # twins rendered here (local PyAV/FFmpeg build) and shipped as MP4: the box decodes them, so its PyAV version
    # (17.1.0 on py3.10) cannot change the encoder output
    from wm.render_twin import load_meta, write_twin
    tw_hash, tw_vis = [], []
    for i, c in enumerate(carriers):
        meta = load_meta(DATASET, int(df["id"].iloc[c]))
        for j in range(T):
            fr = write_twin(meta, float(tg[i, j]), out / "twins" / f"twin_{i:04d}_{j}.mp4")
            tw_hash.append(frame_hash(fr))
            tw_vis.append(float((disk_mask(fr)[1] > 0).mean()))
    arrays["twin_frame_hash"] = np.array(tw_hash).reshape(C, T)
    info["twins"] = {"dir": "twins/twin_{carrier_index:04d}_{target_index}.mp4",
                     "visible_frac_mean": float(np.mean(tw_vis)),
                     "frac_twins_disk_always_visible": float(np.mean(np.array(tw_vis) == 1.0))}
    for L in layers:
        t0 = time.time()
        d = load_inputs(DATASET, L)
        deltas, m = layer_deltas(d, L, y, carriers, tg, held, arms, out, args.holdout, args.part1_basis, args.k,
                                 args.spline, args.waypoints)
        X, K, V, bsrc, rows_b, rb, choice = (m[q] for q in ("X", "K", "V", "bsrc", "rows_b", "rb", "choice"))
        Xc = X[carriers]
        src = y[carriers]
        arrays[f"deltas_L{L}"] = deltas
        # same-layer readout before any GPU work: a probe fit on probe clips (never built an edit) reads x + delta
        probe = d["role"] == "probe"
        sp = Standardizer().fit(X[probe])
        Yp = targets(df, DATASET)[0][probe]
        alpha = load_sweep(DATASET, DATASET)["layers"][L]["alpha"]
        W, b = fit_ridge(sp.transform(X[probe]), Yp, alpha)
        same = {}
        for a_i, a in enumerate(arms):
            P = predict(sp.transform(Xc[:, None] + deltas[:, a_i]), W, b)                               # [C, T, 2]
            same[a] = {"err_to_target_mean": float(wrap(angle_of(P) - tg).mean()),
                       "err_to_true_mean": float(wrap(angle_of(P) - src[:, None]).mean()),
                       "delta_norm_raw_median": float(np.median(np.linalg.norm(deltas[:, a_i], axis=-1)))}
        P0 = predict(sp.transform(Xc), W, b)
        # natural-change reference at L: distance between raw centroids (probe clips) at source and target value
        mu = {v: X[probe & np.isclose(y, v)].mean(0) for v in values}
        nat = np.array([[np.linalg.norm(mu[tg[i, j]] - mu[src[i]]) for j in range(T)] for i in range(C)])
        info["per_layer"][str(L)] = {
            "K": K, "rank": int(V.shape[1]), "basis_source": bsrc, "basis_rows": int(rows_b.sum()),
            "angle_choice": {"angle": choice["angle"], "plane": choice["plane"]},
            "radius_per_probe": rb.round(4).tolist(),
            "unedited_err_to_target_mean": float(wrap(angle_of(P0)[:, None] - tg).mean()),
            "same_layer_readout": same,
            "natural_change_centroid_median": float(np.median(nat)),
            "delta_over_centroid_change_median": {a: float(np.median(np.linalg.norm(deltas[:, i], axis=-1) / nat))
                                                  for i, a in enumerate(arms)},
            "seconds": round(time.time() - t0, 1)}
        print(f"L{L}: K={K}  same-layer err-to-target " +
              ", ".join(f"{a} {same[a]['err_to_target_mean']:.1f}" for a in EDIT_ARMS) +
              f"  (unedited {info['per_layer'][str(L)]['unedited_err_to_target_mean']:.1f})  {time.time() - t0:.0f} s")
    np.savez(out / "plan.npz", **arrays)
    (out / "plan.json").write_text(json.dumps(info, indent=1))
    write(Path(args.results_dir) / f"session2_plan{args.tag}.json", info, seeds={"plan": args.seed},
          stage="plan", plan_npz=str(out / "plan.npz"))


def load_plan(out):
    z = np.load(Path(out) / "plan.npz")
    info = json.loads((Path(out) / "plan.json").read_text())
    return {k: z[k] for k in z.files}, info


# ================================================================ GPU stages

def model_and_device(args):
    import torch
    from wm.extract import load_model, pick_device, set_precision
    set_precision()
    device = torch.device(args.device) if args.device else pick_device()
    if device.type == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    return load_model("vjepa2", device), device


def parity(args):
    import torch
    from wm.extract import preprocess
    from wm.propagate import prefix
    arrays, _ = load_plan(args.out)
    model, device = model_and_device(args)
    df = load_table(DATASET).set_index("id")
    ids = arrays["parity_ids"].tolist()
    frames = [decode(df.loc[i, "video"]) for i in ids]
    t0 = time.time()
    _, _, mp = prefix(model, preprocess(frames).to(device), [])
    ref = arrays["parity_stored_meanpool"]
    rel = (np.abs(mp - ref).max(axis=(0, 2)) / np.abs(ref).max(axis=(0, 2))).tolist()
    rep = {"ids": ids, "batch": len(ids), "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "per_layer_rel_maxabs_over_maxabs": rel, "max_rel": max(rel), "max_abs": float(np.abs(mp - ref).max()),
           "rule": "spec §2 revised: per-layer max|d|/max|x| < 1e-3 (hard); target <= 1e-4 at the extraction's own batch",
           "pass_hard": max(rel) < 1e-3, "pass_1e-4": max(rel) <= 1e-4, "seconds": time.time() - t0}
    p = Path(args.out) / "forward" / "parity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("max_rel", "max_abs", "pass_hard", "pass_1e-4")}))
    if not rep["pass_hard"]:
        raise SystemExit("parity FAILED the 1e-3 rule: stop and inspect before the forward stage")


def forward(args):
    import torch
    from wm.extract import preprocess
    from wm.predictor_readout import (N_CTX, edited_prediction, pool_steps, predict_future, real_future)
    from wm.propagate import prefix, suffix
    from wm.render_twin import load_meta, render
    arrays, info = load_plan(args.out)
    model, device = model_and_device(args)
    fdir = Path(args.out) / "forward"
    fdir.mkdir(parents=True, exist_ok=True)
    df = load_table(DATASET)
    layers, arms = info["layers"], info["arms"]
    pred_arms = [a for a in arms if a in EDIT_ARMS]
    rows, tg = arrays["carrier_rows"], arrays["targets"]
    C, T = tg.shape
    groups = [list(range(s, min(s + args.group, C))) for s in range(0, C, args.group)]
    if args.max_groups:
        groups = groups[:args.max_groups]
    bs = args.batch_size
    timing = {"n_full_src": 0, "n_full_twin": 0, "n_ctx": 0, "n_suffix": {str(L): 0 for L in layers},
              "n_pred": {str(L): 0 for L in layers}, "s_full_src": 0.0, "s_ctx_src": 0.0, "s_twin_render": 0.0,
              "s_twin_encode": 0.0, "s_suffix": {str(L): 0.0 for L in layers}, "s_pred": {str(L): 0.0 for L in layers}}
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)

    def tic():
        sync()
        return time.time()

    for g, idx in enumerate(groups):
        path = fdir / f"group_{g:03d}.npz"
        if path.exists() and not args.overwrite:
            print(f"group {g}: exists, skipping")
            continue
        G = len(idx)
        vids = [df["video"].iloc[rows[i]] for i in idx]
        frames = [decode(v) for v in vids]
        pv = preprocess(frames).to(device)
        t = tic()
        saved, final, mp_src = prefix(model, pv, layers)
        timing["s_full_src"] += tic() - t
        timing["n_full_src"] += G
        t = tic()
        ctx_saved, ctx_final, ctx_mp_src = prefix(model, pv[:, :8], layers)
        z_src = predict_future(model, ctx_final, args.mask_index)
        timing["s_ctx_src"] += tic() - t
        timing["n_ctx"] += G
        rf_src = real_future(final)
        src_fut_mask = np.stack([disk_mask(f)[0][4:] for f in frames])                  # [G, 4, 16, 16]
        # twins
        t = tic()
        twin_frames = []
        for i in idx:
            for j in range(T):
                f = Path(args.out) / "twins" / f"twin_{i:04d}_{j}.mp4"
                twin_frames.append(decode(f) if f.exists() else render(load_meta(DATASET, int(df["id"].iloc[rows[i]])),
                                                                        float(tg[i, j])))
        timing["s_twin_render"] += tic() - t
        twin_hash = [frame_hash(f) for f in twin_frames]
        if "twin_frame_hash" in arrays:
            assert twin_hash == [str(h) for h in arrays["twin_frame_hash"][idx].reshape(-1)], "twin frames differ from plan"
        twin_masks = np.stack([disk_mask(f)[0] for f in twin_frames]).reshape(G, T, 8, 16, 16)
        twin_vis = np.stack([(disk_mask(f)[1] > 0).mean() for f in twin_frames]).reshape(G, T)
        t = tic()
        tw_mp, tw_rf, tw_z = [], [], []
        for s in range(0, len(twin_frames), bs):
            tpv = preprocess(twin_frames[s:s + bs]).to(device)
            _, tfin, tmp = prefix(model, tpv, [])
            _, tcf, _ = prefix(model, tpv[:, :8], [])
            tw_mp.append(tmp)
            tw_rf.append(real_future(tfin))
            tw_z.append(predict_future(model, tcf, args.mask_index))
        tw_rf = torch.cat(tw_rf).view(G, T, N_CTX, -1)
        tw_z = torch.cat(tw_z).view(G, T, N_CTX, -1)
        tw_mp = np.concatenate(tw_mp).reshape(G, T, 26, D)
        timing["s_twin_encode"] += tic() - t
        timing["n_full_twin"] += G * T
        # predictor ceiling: how far the predictor's own twin forecast moves along the real twin change
        disk_tok = torch.as_tensor(np.logical_or(src_fut_mask[:, None], twin_masks[:, :, 4:]).reshape(G, T, -1),
                                   device=device)                                                   # [G, T, 1024]
        rchg = tw_rf - rf_src[:, None]                                                             # [G, T, 1024, D]

        def proj(a, b, m=None):
            if m is not None:
                a, b = a * m[..., None], b * m[..., None]
            return ((a * b).sum((-1, -2)) / (b * b).sum((-1, -2)).clamp_min(1e-12))

        def cosine(a, b):
            a, b = a.flatten(-2), b.flatten(-2)
            return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)

        out = {"idx": np.array(idx), "carrier_ids": df["id"].to_numpy()[rows[idx]],
               "src_meanpool": mp_src, "src_ctx_meanpool": ctx_mp_src, "twin_meanpool": tw_mp,
               "src_pred_pooled": pool_steps(z_src).cpu().numpy(), "src_real_future_pooled": pool_steps(rf_src).cpu().numpy(),
               "twin_pred_pooled": pool_steps(tw_z.flatten(0, 1)).view(G, T, 4, -1).cpu().numpy(),
               "twin_real_future_pooled": pool_steps(tw_rf.flatten(0, 1)).view(G, T, 4, -1).cpu().numpy(),
               "twin_visible_frac": twin_vis, "twin_frame_hash": np.array(twin_hash),
               "src_pred_vs_real_mse": ((z_src - rf_src) ** 2).mean((-1, -2)).cpu().numpy(),
               "R_ceiling_realdiff": proj(tw_z - z_src[:, None], rchg).cpu().numpy(),
               "R_ceiling_realdiff_disk": proj(tw_z - z_src[:, None], rchg, disk_tok).cpu().numpy()}
        for L in layers:
            erows = [(a, i, j) for a in range(G) for i in range(len(arms)) for j in range(T)]
            emp = np.zeros((G, len(arms), T, 26 - L, D), np.float32)
            Ap = len(pred_arms)
            epp = np.zeros((G, Ap, T, 4, D), np.float32)
            R = {k: np.zeros((G, Ap, T, T), np.float32) for k in
                 ("R_pred", "R_real", "R_realdiff", "R_realdiff_disk", "R_pred_disk", "cos_realdiff")}
            dL = torch.as_tensor(arrays[f"deltas_L{L}"][idx], device=device)                     # [G, A, T, D]
            for s in range(0, len(erows), bs):
                chunk = erows[s:s + bs]
                ca = torch.as_tensor([c[0] for c in chunk], device=device)
                dl = torch.stack([dL[a, i, j] for a, i, j in chunk])
                t = tic()
                r = suffix(model, saved[L][ca], L, dl)
                timing["s_suffix"][str(L)] += tic() - t
                timing["n_suffix"][str(L)] += len(chunk)
                for k, (a, i, j) in enumerate(chunk):
                    emp[a, i, j] = r["meanpool"][k]
                pc = [k for k, (a, i, j) in enumerate(chunk) if arms[i] in pred_arms]
                if not pc or args.no_predictor:
                    continue
                t = tic()
                z_e, _ = edited_prediction(model, ctx_saved[L][ca[pc]], L, dl[pc], args.mask_index)
                timing["s_pred"][str(L)] += tic() - t
                timing["n_pred"][str(L)] += len(pc)
                pooled = pool_steps(z_e).cpu().numpy()
                for q, k in enumerate(pc):
                    a, i, j = chunk[k]
                    ip = pred_arms.index(arms[i])
                    epp[a, ip, j] = pooled[q]
                    de = (z_e[q] - z_src[a])[None]                                                  # [1, 1024, D]
                    R["R_pred"][a, ip, j] = proj(de, tw_z[a] - z_src[a][None]).cpu().numpy()
                    R["R_pred_disk"][a, ip, j] = proj(de, tw_z[a] - z_src[a][None], disk_tok[a]).cpu().numpy()
                    R["R_real"][a, ip, j] = proj(de, tw_rf[a] - z_src[a][None]).cpu().numpy()
                    R["R_realdiff"][a, ip, j] = proj(de, rchg[a]).cpu().numpy()
                    R["R_realdiff_disk"][a, ip, j] = proj(de, rchg[a], disk_tok[a]).cpu().numpy()
                    R["cos_realdiff"][a, ip, j] = cosine(de, rchg[a]).cpu().numpy()
            out[f"edit_meanpool_L{L}"] = emp
            out[f"edit_pred_pooled_L{L}"] = epp
            for k, v in R.items():
                out[f"{k}_L{L}"] = v
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        tmp.rename(path)
        print(f"group {g + 1}/{len(groups)} ({G} carriers) done; suffix s " +
              ", ".join(f"L{L} {timing['s_suffix'][str(L)]:.0f}" for L in layers))
    timing["device"] = str(device)
    timing["gpu"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    timing["torch"] = torch.__version__
    timing["mask_index"] = args.mask_index
    timing["batch_size"] = bs
    timing["group"] = args.group
    (fdir / "timing.json").write_text(json.dumps(timing, indent=1))
    est = project_cost(timing, info, C_full=args.project_carriers, T_full=args.project_targets,
                       layers_full=args.project_layers or default_layers())
    (fdir / "cost_estimate.json").write_text(json.dumps(est, indent=1))
    print(json.dumps(est, indent=1))


def project_cost(timing, info, C_full=200, T_full=4, layers_full=None, gpu_s_per_clip=GPU_S_PER_CLIP,
                 n_timerev=1500, n_stim=4 * 392):
    """Scale measured per-unit times to the full run in units of one full 16-frame forward on the same device, then
    to GPU seconds at gpu_s_per_clip (gpu_session1: 0.16 s/clip on an RTX 4080 SUPER at batch 16).
    Suffix and predictor costs are fit as a + b * (24 - L) blocks over the measured layers (least squares; one
    layer: proportional), so a smoke on layers 12 and 22 projects the full layer set."""
    layers_full = layers_full or info["layers"]
    full = timing["s_full_src"] / max(timing["n_full_src"], 1)
    ctx = timing["s_ctx_src"] / max(timing["n_ctx"], 1) / full
    meas = [L for L in info["layers"] if timing["n_suffix"][str(L)]]
    blocks = np.array([24 - L for L in meas], float)

    def fit(kind):
        y = np.array([timing[f"s_{kind}"][str(L)] / max(timing[f"n_{kind}"][str(L)], 1) / full for L in meas])
        if len(meas) >= 2:
            b, a = np.polyfit(blocks, y, 1)
        else:
            a, b = 0.0, y[0] / blocks[0]
        return float(a), float(b), y.tolist()

    sa, sb, s_meas = fit("suffix")
    pa, pb, p_meas = fit("pred") if any(timing["n_pred"][str(L)] for L in meas) else (0.0, 0.0, [])
    A, Ap = len(info["arms"]), len([a for a in info["arms"] if a in EDIT_ARMS])
    units_src = C_full * (1 + ctx)
    units_twin = C_full * T_full * (1 + ctx)
    units_edit = sum(C_full * T_full * (A * (sa + sb * (24 - L)) + Ap * (pa + pb * (24 - L))) for L in layers_full)
    total = units_src + units_twin + units_edit + n_timerev + n_stim
    return {"per_unit_over_full_forward": {"full_forward_s_this_device": full, "ctx_encode_plus_predictor": ctx,
                                           "suffix_fit_a_b_per_block": [sa, sb], "suffix_measured": dict(zip(map(str, meas), s_meas)),
                                           "edit_ctx_plus_predictor_fit_a_b_per_block": [pa, pb],
                                           "pred_measured": dict(zip(map(str, meas), p_meas))},
            "full_forward_equivalents": {"sources": units_src, "twins": units_twin, "edits": units_edit,
                                         "timerev": n_timerev, "stimuli": n_stim, "total": total},
            "gpu_minutes_at_0.16s": total * gpu_s_per_clip / 60.0,
            "assumes": f"C={C_full} carriers x T={T_full} targets x {A} arms ({Ap} through the predictor) x layers "
                       f"{layers_full}; GPU time = full-forward equivalents x 0.16 s (gpu_session1 batch-16 rate, "
                       "CPU cost ratios assumed to carry over); scoring runs locally"}


def timerev(args):
    """Frame-reversed extraction of every direction clip (spec §6.2): same pooling and chunk format as extract.py."""
    import torch
    from wm.extract import CHUNK, chunk_arrays, chunk_ok, encode, merge, pool, preprocess, save_npy, write_index
    model, device = model_and_device(args)
    df = load_table(DATASET)
    if args.limit:
        df = df.head(args.limit)
    out_dir = ACT_ROOT / DATASET / "vjepa2_timerev" if not args.act_out else Path(args.act_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(i) for i in df["id"]]
    videos = dict(zip(ids, df["video"]))
    chunks = [ids[s:s + CHUNK] for s in range(0, len(ids), CHUNK)]
    t_run = time.time()
    for c, cids in enumerate(chunks):
        if chunk_ok(out_dir, c, cids, "vjepa2"):
            continue
        store = {n: [] for n in chunk_arrays("vjepa2")}
        hashes, secs = {}, {}
        for s in range(0, len(cids), args.batch_size):
            b = cids[s:s + args.batch_size]
            t0 = time.time()
            frames = [np.ascontiguousarray(decode(videos[i])[::-1]) for i in b]
            masks = [disk_mask(f)[0] for f in frames]
            for i, f in zip(b, frames):
                hashes[i] = frame_hash(f)
            points, _ = encode(model, preprocess(frames).to(device))
            mean, tpool, dpool = pool(points, torch.from_numpy(np.stack(masks)).to(device))
            store["meanpool"].append(mean.astype(np.float32))
            store["timepool"].append(tpool.astype(np.float16))
            store["diskpool"].append(dpool.astype(np.float16))
            store["diskmask"].append(np.stack(masks))
            for i in b:
                secs[i] = (time.time() - t0) / len(b)
        for name, (shape, dtype) in chunk_arrays("vjepa2").items():
            save_npy(out_dir / f"chunk_{c:04d}_{name}.npy", np.concatenate(store[name]))
        (out_dir / f"chunk_{c:04d}.json").write_text(json.dumps(
            {"ids": cids, "frame_hash": {str(i): hashes[i] for i in cids},
             "seconds_per_clip": {str(i): secs[i] for i in cids}}))
        print(f"timerev chunk {c + 1}/{len(chunks)}")
    write_index(out_dir, ids, len(chunks), "vjepa2", device, time.time() - t_run)
    merge(out_dir, ids)
    idx = json.loads((out_dir / "index.json").read_text())
    idx["frame_order"] = "reversed: frame 16 first (decode(video)[::-1]); frame_hash is of the reversed stack"
    idx["labels"] = "use the forward metadata; the reversed clip moves at theta + 180 (accelerating clips decelerate to rest)"
    (out_dir / "index.json").write_text(json.dumps(idx, indent=1))


def extract_stimuli(args):
    from wm.extract import merge, pick_device, run_extraction, set_precision
    import torch
    set_precision()
    device = torch.device(args.device) if args.device else pick_device()
    for name in args.stimulus_sets:
        root = STIMULI / name
        df = load_table(DATASET, root=root)
        if args.limit:
            df = df.head(args.limit)
        for kind in ("vjepa2", "random"):
            out_dir = (Path(args.act_out) if args.act_out else ACT_ROOT) / f"stimuli_{name}" / kind
            run_extraction(df, kind, out_dir, batch_size=args.batch_size, device=device)
            merge(out_dir, df["id"].tolist())
            print(f"stimuli {name}/{kind}: done")


# ================================================================ score (CPU)

def boot_ci(per_carrier, n=1000, seed=0):
    v = np.asarray(per_carrier, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    idx = np.random.default_rng(seed).integers(0, len(v), (n, len(v)))
    m = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def load_groups(fdir, keys=None):
    files = sorted(Path(fdir).glob("group_*.npz"))
    assert files, f"no forward outputs in {fdir}"
    out = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if keys is None or k in keys:
                out.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def shuffle_index(T):
    return [(j + 1) % T for j in range(T)]


def score_stage(args):
    arrays, info = load_plan(args.out)
    fdir = Path(args.out) / "forward"
    F = load_groups(fdir)
    idx = F["idx"]
    layers, arms = info["layers"], info["arms"]
    pred_arms = [a for a in arms if a in EDIT_ARMS]
    tg = arrays["targets"][idx]
    rows = arrays["carrier_rows"][idx]
    C, T = tg.shape
    shuf = shuffle_index(T)
    df = load_table(DATASET)
    y = df["theta_degrees"].to_numpy(float)
    src = y[rows]
    full = np.load(ACT_ROOT / DATASET / "vjepa2" / "meanpool.npy", mmap_mode="r")
    d0 = load_inputs(DATASET, 0)
    probe, test = d0["role"] == "probe", d0["role"] == "test"
    Y = targets(df, DATASET)[0]
    sweep = load_sweep(DATASET, DATASET)
    # zero-edit reproduction on the forward device vs the stored (GPU batch-16) meanpool, per layer
    ref = arrays["carrier_stored_meanpool"][idx]
    repro = (np.abs(F["src_meanpool"] - ref).max(axis=(0, 2)) / np.abs(ref).max(axis=(0, 2)))
    # read probes per point (probe clips, all 64 values), alpha from the step-1 sweep
    readers = {}
    for p in range(26):
        Xp = np.asarray(full[:, p], np.float64)
        st = Standardizer().fit(Xp[probe])
        W, b = fit_ridge(st.transform(Xp[probe]), Y[probe], sweep["layers"][p]["alpha"])
        readers[p] = (st, W, b)

    def read(X, p):
        st, W, b = readers[p]
        sh = X.shape
        P = predict(st.transform(X.reshape(-1, sh[-1]).astype(np.float64)), W, b).reshape(sh[:-1] + (2,))
        return angle_of(P), np.linalg.norm(P, axis=-1)

    tmean = {}
    def target_mean(v, p):
        key = (float(v), p)
        if key not in tmean:
            tmean[key] = np.asarray(full[test & np.isclose(y, v), p], np.float64).mean(0)
        return tmean[key]

    prop = {"arms": arms + ["zero", "shuffled_target(probe_qr)", "shuffled_target(spline)"], "layers": layers,
            "read_points": list(range(26)), "n_carriers": int(C), "n_targets": int(T),
            "zero_edit_reproduction_per_layer_rel": repro.tolist(),
            "readout_a": {}, "readout_b": {}, "delta_over_natural_twin_change": {}}
    src_mp, twin_mp = F["src_meanpool"].astype(np.float64), F["twin_meanpool"].astype(np.float64)
    ang0 = np.stack([read(src_mp[:, p], p)[0] for p in range(26)], 1)                  # [C, 26]
    for L in layers:
        E = F[f"edit_meanpool_L{L}"].astype(np.float64)                                 # [C, A, T, 26-L, D]
        a_rows, b_rows = {}, {}
        for i, arm in enumerate(arms):
            err_t, err_true, rad, err_sh, Ract, cosv, agr, near, Rsh = ([] for _ in range(9))
            for q, p in enumerate(range(L, 26)):
                ang, r = read(E[:, i, :, q], p)                                          # [C, T]
                err_t.append(wrap(ang - tg).mean(1))
                err_true.append(wrap(ang - src[:, None]).mean(1))
                err_sh.append(wrap(ang - tg[:, shuf]).mean(1))
                rad.append(r.mean(1))
                de = E[:, i, :, q] - src_mp[:, None, p]
                dt = twin_mp[:, :, p] - src_mp[:, None, p]
                Ract.append(((de * dt).sum(-1) / np.maximum((dt * dt).sum(-1), 1e-12)).mean(1))
                Rsh.append(((de * dt[:, shuf]).sum(-1) / np.maximum((dt[:, shuf] ** 2).sum(-1), 1e-12)).mean(1))
                cosv.append(((de * dt).sum(-1) / np.maximum(np.linalg.norm(de, axis=-1) * np.linalg.norm(dt, axis=-1), 1e-12)).mean(1))
                agr.append((1 - np.linalg.norm(E[:, i, :, q] - twin_mp[:, :, p], axis=-1)
                            / np.maximum(np.linalg.norm(dt, axis=-1), 1e-12)).mean(1))
                nr = np.stack([mf.nearest_real_agreement(E[:, i, j, q], src_mp[:, p],
                                                         np.stack([target_mean(v, p) for v in tg[:, j]]))
                               for j in range(T)], 1)
                near.append(nr.mean(1))
            pts = list(range(L, 26))
            summ = lambda arr: [boot_ci(v) for v in arr]                                 # noqa: E731
            a_rows[arm] = {"read_points": pts, "err_to_target": summ(err_t), "err_to_true": summ(err_true),
                           "err_to_shuffled_target": summ(err_sh), "readout_radius": summ(rad)}
            b_rows[arm] = {"read_points": pts, "R_act_twin": summ(Ract), "R_act_shuffled_twin": summ(Rsh),
                           "cos_to_twin_change": summ(cosv), "agreement_twin": summ(agr),
                           "nearest_real_agreement": summ(near)}
            if arm in EDIT_ARMS:
                dn = np.linalg.norm(arrays[f"deltas_L{L}"][idx][:, i], axis=-1)
                nat = np.linalg.norm(twin_mp[:, :, L] - src_mp[:, None, L], axis=-1)
                prop["delta_over_natural_twin_change"].setdefault(str(L), {})[arm] = {
                    "median_ratio": float(np.median(dn / np.maximum(nat, 1e-12))),
                    "delta_norm_median": float(np.median(dn)), "twin_change_norm_median": float(np.median(nat))}
        a_rows["zero"] = {"read_points": list(range(L, 26)),
                          "err_to_target": [boot_ci(wrap(ang0[:, p, None] - tg).mean(1)) for p in range(L, 26)],
                          "err_to_true": [boot_ci(wrap(ang0[:, p] - src)) for p in range(L, 26)]}
        prop["readout_a"][str(L)] = a_rows
        prop["readout_b"][str(L)] = b_rows
        print(f"L{L}: err-to-target at final LN " + ", ".join(
            f"{a} {a_rows[a]['err_to_target'][-1]['mean']:.1f}" for a in list(EDIT_ARMS) + ["zero"]))
    prop["readout_notes"] = {
        "a": "direction probe refit at every read point on probe clips (train folds 3-4, all 64 values), alpha from the "
             "step-1 sweep; read on the edited clip's meanpool at points L..25. At read point = L the Part 1 arms are "
             "read by a probe of the same family that built them (not independent evidence).",
        "b": "R_act_twin = <x_edit - x_src, x_twin - x_src> / ||x_twin - x_src||^2 on meanpool at each later point, "
             "against the rendered counterfactual twin (same start and speed profile, direction theta*); "
             "agreement_twin = 1 - ||x_edit - x_twin|| / ||x_src - x_twin||; nearest_real_agreement = "
             "manifold.nearest_real_agreement against the mean of real test clips at theta*. x_src is the forward "
             "stage's own zero-edit pass (same device), so device noise cancels.",
        "shuffled_target": "edit toward target j scored against target (j+1) mod T of the same carrier (value and "
                           "twin); no extra forward pass",
        "behaviour_rung": "readouts a and b are not behaviour (Part 2 design note §7); the predictor file is"}
    write(Path(args.results_dir) / f"session2_propagation{args.tag}.json", prop, stage="score",
          forward_dir=str(fdir), mask_index=None)

    # ------------------------------------------------ predictor readout (c)
    tp = np.asarray(np.load(ACT_ROOT / DATASET / "vjepa2" / "timepool.npy", mmap_mode="r")[:, 25, 4:], np.float32)
    Xf = tp.mean(1).astype(np.float64)                                                    # real future, pooled
    stf = Standardizer().fit(Xf[probe])
    folds = d0["fold"][probe]
    cv = cv_select_alpha(stf.transform(Xf[probe]), Y[probe], folds, score_fn=lambda a, b: score(a, b, "circular"))
    Wf, bf = fit_ridge(stf.transform(Xf[probe]), Y[probe], cv["alpha"])
    fut = lambda Z: angle_of(predict(stf.transform(Z.reshape(-1, D)), Wf, bf)).reshape(Z.shape[:-1])   # noqa: E731
    test_ang = fut(Xf[test])
    pred = {"mask_index": json.loads((fdir / "timing.json").read_text()).get("mask_index") if (fdir / "timing.json").exists() else None,
            "future_probe": {"fit": "ridge on stored timepool[final LN, tubelets 4-7] averaged over steps, probe clips "
                                    "(folds 3-4), alpha by 2-fold CV on those folds", "alpha": cv["alpha"],
                             "cv_r2": cv["cv_mean"],
                             "test_err_on_real_future_mean": float(wrap(test_ang - y[test]).mean())},
            "controls": {}, "per_layer": {}}
    zs = F["src_pred_pooled"].mean(1)
    zt = F["twin_pred_pooled"].mean(2)
    pred["controls"]["unsteered_predictor_preserves_direction"] = {
        "future_err_pred_src_vs_true": boot_ci(wrap(fut(zs) - src)),
        "future_err_real_src_vs_true": boot_ci(wrap(fut(F["src_real_future_pooled"].mean(1)) - src)),
        "future_err_pred_twin_vs_target": boot_ci(wrap(fut(zt) - tg).mean(1)),
        "future_err_real_twin_vs_target": boot_ci(wrap(fut(F["twin_real_future_pooled"].mean(2)) - tg).mean(1)),
        "src_pred_vs_real_token_mse": boot_ci(F["src_pred_vs_real_mse"])}
    pred["controls"]["predictor_ceiling"] = {
        "R_ceiling_realdiff": boot_ci(F["R_ceiling_realdiff"].mean(1)),
        "R_ceiling_realdiff_disk": boot_ci(F["R_ceiling_realdiff_disk"].mean(1)),
        "note": "the predictor's own forecast for the real twin context, projected on the real twin-minus-source "
                "future change: the most an edit of the context can be expected to recover through this predictor"}
    pred["controls"]["twin_visible_frac_mean"] = float(F["twin_visible_frac"].mean())
    for L in layers:
        row = {}
        for ip, arm in enumerate(pred_arms):
            if f"R_pred_L{L}" not in F:
                continue
            r = {}
            for key in ("R_pred", "R_real", "R_realdiff", "R_realdiff_disk", "R_pred_disk", "cos_realdiff"):
                M = F[f"{key}_L{L}"][:, ip]                                                  # [C, T(edit), T(twin)]
                diag = np.stack([M[:, j, j] for j in range(T)], 1).mean(1)
                off = np.stack([M[:, j, shuf[j]] for j in range(T)], 1).mean(1)
                r[key] = boot_ci(diag)
                r[key + "_shuffled_target"] = boot_ci(off)
            ze = F[f"edit_pred_pooled_L{L}"][:, ip]                                          # [C, T, 4, D]
            de, dt = ze - F["src_pred_pooled"][:, None], F["twin_pred_pooled"] - F["src_pred_pooled"][:, None]
            r["R_pooled_pred"] = boot_ci(((de * dt).sum((-1, -2)) / np.maximum((dt * dt).sum((-1, -2)), 1e-12)).mean(1))
            fa = fut(ze.mean(2))
            r["future_err_to_target"] = boot_ci(wrap(fa - tg).mean(1))
            r["future_err_to_true"] = boot_ci(wrap(fa - src[:, None]).mean(1))
            r["future_err_to_shuffled_target"] = boot_ci(wrap(fa - tg[:, shuf]).mean(1))
            row[arm] = r
        row["zero"] = {"future_err_to_target": boot_ci(wrap(fut(zs)[:, None] - tg).mean(1)), "R": 0.0}
        pred["per_layer"][str(L)] = row
        if row and "probe_qr" in row:
            print(f"L{L}: R_realdiff " + ", ".join(f"{a} {row[a]['R_realdiff']['mean']:.3f}" for a in pred_arms if a in row)
                  + f" | ceiling {pred['controls']['predictor_ceiling']['R_ceiling_realdiff']['mean']:.3f}")
    pred["definitions"] = {
        "R_pred": "<z_edit - z_src, z_twin_pred - z_src> / ||.||^2 on all 1024 predicted future tokens; z_twin_pred = "
                  "the predictor's forecast from the rendered twin's context (like for like)",
        "R_real": "spec §6.5 literal: target = the twin's real future tokens (encoder, full 16-frame twin, post-LN); "
                  "includes the predictor's own error in the denominator",
        "R_realdiff": "<z_edit - z_src, f_twin - f_src> / ||f_twin - f_src||^2 with f = real future tokens: predicted "
                      "change against the real change (predictor bias cancels)",
        "*_disk": "same, restricted to future tokens where the source or the twin disk is (disk mask union)",
        "shuffled_target": "edit toward target j scored against the twin of target (j+1) mod T",
        "context": "frames 1-8 encoded alone (tokens 0..1023), edit added to every context token at layer L, "
                   "predictor forecasts tubelets 4..7; edits were computed from the full-clip meanpool at L"}
    write(Path(args.results_dir) / f"session2_predictor{args.tag}.json", pred, stage="score", forward_dir=str(fdir))

    # ------------------------------------------------ time-reversed control (if extracted)
    trdir = Path(args.timerev_dir) if args.timerev_dir else ACT_ROOT / DATASET / "vjepa2_timerev"
    if (trdir / "meanpool.npy").exists():
        R_ = np.load(trdir / "meanpool.npy", mmap_mode="r")
        ids_tr = json.loads((trdir / "ids.json").read_text())
        sel = np.isin(df["id"].to_numpy(), ids_tr) & test
        pos = {int(i): k for k, i in enumerate(ids_tr)}
        rr = np.array([pos[int(i)] for i in df["id"].to_numpy()[sel]])
        motion = df["motion"].to_numpy()[sel]
        rows_tr = []
        for p in range(26):
            ang, rad = read(np.asarray(R_[rr, p], np.float64), p)
            e_same, e_flip = wrap(ang - y[sel]), wrap(ang - (y[sel] + 180.0))
            rows_tr.append({"point": p, "err_vs_theta": float(e_same.mean()), "err_vs_theta_plus_180": float(e_flip.mean()),
                            "frac_closer_to_flipped": float((e_flip < e_same).mean()),
                            "by_motion": {m: {"err_vs_theta": float(e_same[motion == m].mean()),
                                              "err_vs_theta_plus_180": float(e_flip[motion == m].mean())}
                                          for m in np.unique(motion)}})
        write(Path(args.results_dir) / f"session2_timerev{args.tag}.json",
              {"n_test_clips": int(sel.sum()), "probe": "forward-clip direction probe per point (probe clips)",
               "reading": "a probe that tracks motion direction reads theta + 180 on the reversed clip; one that tracks "
                          "position/occupancy reads theta (a reversed velocity clip has the same frame set)",
               "rows": rows_tr}, stage="score", timerev_dir=str(trdir))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("plan", "parity", "forward", "timerev", "extract-stimuli", "score"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--tag", default="", help="suffix for results files (e.g. _smoke)")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="default: onset, 8, 12, peak from step 1")
    p.add_argument("--n-carriers", type=int, default=200)
    p.add_argument("--n-targets", type=int, default=4)
    p.add_argument("--holdout", default="contiguous", choices=("contiguous", "scattered", "none"))
    p.add_argument("--part1-basis", default="design", choices=("design", "paper"))
    p.add_argument("--spline", default="interp", choices=("interp", "smooth"))
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--waypoints", type=int, default=0, help=">2 adds interior spline/chord waypoints as extra arms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--group", type=int, default=8, help="carriers per resumable forward group")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-groups", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-predictor", action="store_true")
    p.add_argument("--mask-index", type=int, default=0, help="predictor mask token (0 = the only trained one)")
    p.add_argument("--project-carriers", type=int, default=200)
    p.add_argument("--project-targets", type=int, default=4)
    p.add_argument("--project-layers", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0, help="timerev / extract-stimuli: first N clips only (smoke)")
    p.add_argument("--act-out", default=None, help="timerev / extract-stimuli output root override (smoke)")
    p.add_argument("--timerev-dir", default=None)
    p.add_argument("--stimulus-sets", nargs="*", default=["paper_layout", "hard"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    {"plan": plan, "parity": parity, "forward": forward, "timerev": timerev,
     "extract-stimuli": extract_stimuli, "score": score_stage}[args.stage](args)


if __name__ == "__main__":
    main()

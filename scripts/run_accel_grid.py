"""Acceleration with mean speed decorrelated: is acceleration read beyond a velocity intermediate?

Every supplied accelerating clip starts at rest, so acceleration a, mean speed and displacement are one number
(mean speed = 0.3125 s x a). This renders a factorial grid of MEAN SPEED m x ACCELERATION a (decelerations included),
so that a is uncorrelated with mean speed and with displacement (= 0.625 s x m for every clip) by construction:
initial speed v0 = m - 0.3125 a, final speed v(0.625 s) = m + 0.3125 a, both > 0 (no reversal).

  render  (Mac, CPU)   GRID_M x GRID_A cells x --per-cell clips; per cell the directions are 360/k apart with a random
                       offset, starts uniform in [-1.2, 1.2]^2 (the speed/acceleration sets' start box), redrawn until
                       the disk centre stays within |x|, |y| <= 3.4 m (disk fully on screen) for all 16 frames.
                       Frames via wm.render_twin.write_twin (the validated twin renderer: 256 x 256, 16 frames, 24 fps,
                       mp4v codec) -> artifacts/accel_grid/videos/clip_XXXX.mp4 + plan.json (per-clip metadata).
  extract (box, GPU)   V-JEPA 2 and the random-init copy (torch.manual_seed(0), wm.extract.load_model('random')):
                       meanpool [n, 26, D] f32 and timepool [n, 26, 8, D] f16 of the grid -> feat_{model}.npz; plus the
                       random-init timepool of the supplied speed set at POINTS (for the transfer speed probe; the
                       V-JEPA 2 one is artifacts/activations/speed/vjepa2/timepool.npy) -> speed_random_timepool.npz.
                       GPU under flock /tmp/wm_gpu.lock.
  score   (Mac, CPU)   per model and point: ridge a with mean speed + displacement partialled out, joint (m, a) ridge,
                       per-step speed decode (nested in-set probe and transfer probe from the speed set), the acceleration
                       readout with the decoded speed sequence partialled out, and MLP(activations) vs MLP(decoded
                       speed sequence) -> results/p5_accel_decorrelated.json, figures/fig_accel_decorrelated.png.
  magnitude (Mac, CPU) QA #358: |a| (the paper's target) beside signed a, ridge plain and with mean speed +
                       displacement partialled, same folds / bootstrap -> results/p5_accel_decorrelated_magnitude.json.
  cartloco (Mac, CPU)  QA #383/#384: Cartesian (ax, ay), signed a and |a|, partialled ridge, under the per-clip folds
                       and leave-one-cell-out -> results/p5_accel_decorrelated_cartesian_loco.json.

  python scripts/run_accel_grid.py render|extract|score|magnitude|cartloco [--out artifacts/accel_grid] [--n-smoke 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, disk_pixels, frame_hash  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "accel_grid"
GRID_M = (1.0, 1.75, 2.5, 3.25)            # mean speed over the clip (m/s)
GRID_A = (-3.0, -1.5, 0.0, 1.5, 3.0)       # acceleration (m/s^2)
PER_CELL = 12
FPS, N_FRAMES = 24.0, 16
T_CLIP = (N_FRAMES - 1) / FPS              # 0.625 s from frame 0 to frame 15
T_MEAN = T_CLIP / 2                        # 0.3125 s: mean over the frames (and over the 8 tubelet midpoints)
TAU = (2 * np.arange(8) + 0.5) / FPS       # tubelet midpoints (wm.timeman.step_times)
LIM = 3.4                                  # |x|, |y| bound for the disk centre (disk radius 0.33 m; screen +-4 m)
POINTS = (1, 4, 8, 9, 12, 16, 22)
MODELS = ("vjepa2", "random")
N_FOLDS = 5
NB = 2000
MLP_SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- design (pure; tests/test_accel_grid.py)

def design(per_cell=PER_CELL, seed=0, grid_m=GRID_M, grid_a=GRID_A):
    """Per-clip metadata list for the m x a factorial. Each dict follows the supplied metadata.json schema plus
    mean_speed_mps, displacement_m, final_speed_mps, cell."""
    rng = np.random.default_rng(seed)
    out = []
    for i, m in enumerate(grid_m):
        for j, a in enumerate(grid_a):
            v0 = m - a * T_MEAN
            assert v0 > 0 and m + a * T_MEAN > 0, (m, a)
            off = rng.uniform(0, 360.0 / per_cell)
            for k in range(per_cell):
                th = (off + k * 360.0 / per_cell) % 360.0
                for _ in range(1000):
                    start = rng.uniform(-1.2, 1.2, 2)
                    if np.abs(trajectory(start, th, v0, a)).max() <= LIM:
                        break
                else:
                    raise RuntimeError("no on-screen start")
                out.append({"id": len(out), "primary_label": "acceleration", "magnitude": float(a),
                            "theta_degrees": float(th), "motion": "acceleration" if a != 0 else "velocity",
                            "speed_mps": float(v0), "acceleration_mps2": float(a),
                            "start_position_xy_m": [float(start[0]), float(start[1])], "fps": int(FPS),
                            "frames": N_FRAMES, "mean_speed_mps": float(m), "displacement_m": float(m * T_CLIP),
                            "final_speed_mps": float(m + a * T_MEAN), "cell": [i, j]})
    return out


def trajectory(start, theta_deg, v0, a):
    """Disk centre (m) [16, 2]; same kinematics as wm.render_twin.trajectory_m."""
    t = np.arange(N_FRAMES) / FPS
    s = v0 * t + 0.5 * a * t ** 2
    th = np.radians(theta_deg)
    return np.asarray(start, float)[None] + s[:, None] * np.array([np.cos(th), np.sin(th)])[None]


def step_speeds(v0, a):
    """Instantaneous speed at each tubelet midpoint [n, 8]."""
    return np.asarray(v0, float)[:, None] + np.asarray(a, float)[:, None] * TAU[None]


def folds_by_cell(cells, k=N_FOLDS, seed=0):
    """Fold id per clip, each clip its own group; clips of a cell spread round-robin over folds in a random order."""
    rng = np.random.default_rng(seed)
    cells = [tuple(c) for c in cells]
    f = np.zeros(len(cells), int)
    for c in sorted(set(cells)):
        idx = rng.permutation([i for i, x in enumerate(cells) if x == c])
        f[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return f


# ---------------------------------------------------------------- probes (pure)

def _ridge(X, Y, per_target=False):
    """Standardise + ridge, alpha from 15 log-spaced values in [1e-2, 1e5] (wm.timeman.ALPHAS). With more rows than
    features (the per-step decoders: clip x step rows) alpha is chosen by efficient LOO (GCV); with n <= p (clip-level
    readouts, ~190 clips x 1,024 / 8,192 features) by 5-fold CV over the training clips, because the LOO shortcut is
    numerically unstable near interpolation (it picked alpha = 0.01 on one fold, held-out R^2 -3)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X)
    cv = KFold(5, shuffle=True, random_state=0) if X.shape[0] <= X.shape[1] else None
    kw = {} if cv is not None else {"alpha_per_target": per_target}
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15), cv=cv, **kw)).fit(X, Y)


def _lstsq_resid(Ztr, Atr, Zte, Ate):
    """Residualise A on [1, Z] with coefficients fit on the train rows; returns (train resid, test resid)."""
    D = lambda Z: np.column_stack([np.ones(len(Z)), Z])                            # noqa: E731
    B = np.linalg.lstsq(D(Ztr), Atr, rcond=None)[0]
    return Atr - D(Ztr) @ B, Ate - D(Zte) @ B


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def partial_ridge_oof(X, y, Z, folds, Ztr_by_fold=None):
    """Held-out prediction of y from X with nuisance Z [n, q] regressed out of both sides inside each training fold
    (test rows residualised with the train coefficients). Ztr_by_fold[f] optionally replaces Z for the TRAIN rows of
    fold f (used when Z is itself an out-of-fold estimate). Returns (y residual, prediction), both pooled OOF."""
    X, y, Z = np.asarray(X, float), np.asarray(y, float), np.asarray(Z, float).reshape(len(y), -1)
    res, pred = np.zeros(len(y)), np.zeros(len(y))
    for f in np.unique(folds):
        te = folds == f
        Ztr = Z[~te] if Ztr_by_fold is None else Ztr_by_fold[f]
        ytr, yte = _lstsq_resid(Ztr, y[~te], Z[te], y[te])
        Xtr, Xte = _lstsq_resid(Ztr, X[~te], Z[te], X[te])
        pred[te] = _ridge(Xtr, ytr).predict(Xte)
        res[te] = yte
    return res, pred


def oof(fit_predict, n, folds):
    out = None
    for f in np.unique(folds):
        te = folds == f
        p = np.asarray(fit_predict(~te, te), float)
        if out is None:
            out = np.zeros((n,) + p.shape[1:])
        out[te] = p
    return out


def nested_step_decode(T, S, folds):
    """Per-step speed decode from per-tubelet features T [n, 8, D] with labels S [n, 8], nested inside the clip folds.
    Returns (test: [n, 8] outer out-of-fold decode, train: {f: [n_train_f, 8]} inner out-of-fold decode of the outer
    training clips of fold f, so the downstream reader is trained on decodes with realistic held-out noise)."""
    n, st, D = T.shape
    test, train = np.zeros((n, st)), {}
    for f in np.unique(folds):
        te = folds == f
        tr_idx = np.flatnonzero(~te)
        m = _ridge(T[tr_idx].reshape(-1, D), S[tr_idx].reshape(-1))
        test[te] = m.predict(T[te].reshape(-1, D)).reshape(-1, st)
        inner = folds[tr_idx]
        tr = np.zeros((len(tr_idx), st))
        for g in np.unique(inner):
            ite = inner == g
            mi = _ridge(T[tr_idx[~ite]].reshape(-1, D), S[tr_idx[~ite]].reshape(-1))
            tr[ite] = mi.predict(T[tr_idx[ite]].reshape(-1, D)).reshape(-1, st)
        train[f] = tr
    return test, train


def mlp_oof(X, y, folds, Xtr_by_fold=None, seeds=MLP_SEEDS):
    """One-hidden-layer MLP (wm.bakeoff.MLPReadout: 64 units, alpha 1e-2, early stopping), seed-averaged, OOF."""
    from wm.bakeoff import MLPReadout
    X = np.asarray(X, float).reshape(len(y), -1)

    def fp(tr, te):
        f = int(np.unique(folds[te])[0])
        Xtr = X[tr] if Xtr_by_fold is None else np.asarray(Xtr_by_fold[f]).reshape(tr.sum(), -1)
        return np.mean([MLPReadout(Xtr, y[tr], False, s).predict(X[te]) for s in seeds], axis=0)
    return oof(fp, len(y), folds)


def boot_r2(y, p, nb=NB, seed=0, p2=None):
    """Point R^2 and 95% clip-bootstrap CI (and, with p2, of R^2(p) - R^2(p2) on the same draws)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    draws, diffs = [], []
    for _ in range(nb):
        i = rng.integers(0, len(y), len(y))
        draws.append(r2(y[i], p[i]))
        if p2 is not None:
            diffs.append(draws[-1] - r2(y[i], np.asarray(p2)[i]))
    out = {"r2": r2(y, p), "ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}
    if p2 is not None:
        out["minus_other"] = {"diff": r2(y, p) - r2(y, p2), "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
                              "frac_draws_le_0": float(np.mean(np.array(diffs) <= 0))}
    return out


def boot_r2_pair(y1, p1, y2, p2, nb=NB, seed=0):
    """Paired clip bootstrap of R^2(y1, p1) - R^2(y2, p2) (two targets scored on the same clips and draws)."""
    rng = np.random.default_rng(seed)
    y1, p1, y2, p2 = (np.asarray(v, float) for v in (y1, p1, y2, p2))
    d = []
    for _ in range(nb):
        i = rng.integers(0, len(y1), len(y1))
        d.append(r2(y1[i], p1[i]) - r2(y2[i], p2[i]))
    return {"diff": r2(y1, p1) - r2(y2, p2), "ci95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
            "frac_draws_le_0": float(np.mean(np.array(d) <= 0))}


def magnitude_scores(X, a, nuis, folds, nb=NB, seed=0):
    """Signed a and |a| (the paper's target, Joseph et al. 2026 sec. 5) under the signed run's protocol: plain OOF
    ridge, and ridge with the nuisance columns (mean speed, displacement) OLS-regressed out of features and target
    inside each training fold; R^2 with a 95% clip bootstrap. abs_minus_signed: paired bootstrap of the partialled
    R^2(|a|) - R^2(a). An order-blind code (e.g. a time-pooled mean) cannot see the sign of a, since a decelerating
    clip is a time-reversed accelerating one heading the other way; |a| does not need the order."""
    X, a = np.asarray(X, float), np.asarray(a, float)
    n, out, part = len(a), {}, {}
    for name, y in (("signed_a", a), ("abs_a", np.abs(a))):
        plain = oof(lambda tr, te: _ridge(X[tr], y[tr]).predict(X[te]), n, folds)
        yr, pr = partial_ridge_oof(X, y, nuis, folds)
        part[name] = (yr, pr)
        out[name] = {"ridge": boot_r2(y, plain, nb, seed),
                     "ridge_partial_mean_speed_displacement": boot_r2(yr, pr, nb, seed)}
    out["abs_minus_signed_partial"] = boot_r2_pair(*part["abs_a"], *part["signed_a"], nb=nb, seed=seed)
    y = np.abs(a)
    p = abs_of_signed_oof(X, a, folds)
    out["abs_a_from_abs_signed_ridge"] = {**boot_r2(y, p, nb, seed),
                                          "minus_abs_ridge": boot_r2_pair(y, p, *part["abs_a"], nb=nb, seed=seed)}
    return out


def abs_of_signed_oof(X, a, folds):
    """|a| read as |signed ridge prediction|, calibrated per outer fold by OLS |a| ~ 1 + |a_hat| fit on inner
    out-of-fold signed predictions of the training clips (so the calibration never sees its own training fit and
    the test fold never enters). Tests whether |a| is available through the signed code, a nonlinearity a linear
    |a| ridge cannot express."""
    X, a = np.asarray(X, float), np.asarray(a, float)
    out = np.zeros(len(a))
    for f in np.unique(folds):
        te = folds == f
        tr = np.flatnonzero(~te)
        inner = np.zeros(len(tr))
        for g in np.unique(folds[tr]):
            ite = folds[tr] == g
            inner[ite] = _ridge(X[tr[~ite]], a[tr[~ite]]).predict(X[tr[ite]])
        c = np.polyfit(np.abs(inner), np.abs(a[tr]), 1)
        out[te] = np.polyval(c, np.abs(_ridge(X[tr], a[tr]).predict(X[te])))
    return out


def loco_folds(cells):
    """Leave-one-cell-out: fold id = design-cell index, so every clip of a (mean speed, a) cell is held out
    together and the test cell's (m, a) combination is never seen in training (QA #384)."""
    keys = sorted({tuple(c) for c in cells})
    return np.array([keys.index(tuple(c)) for c in cells])


def cartesian_targets(a, theta_deg):
    """Cartesian acceleration (ax, ay) = a (cos th, sin th): the paper's Table 1 acceleration target (QA #383)."""
    th = np.radians(np.asarray(theta_deg, float))
    return np.asarray(a, float) * np.cos(th), np.asarray(a, float) * np.sin(th)


def boot_r2_multi(ys, ps, nb=NB, seed=0):
    """Per-component R^2 and their mean, with 95% CIs from shared clip-bootstrap draws."""
    rng = np.random.default_rng(seed)
    ys, ps = [np.asarray(y, float) for y in ys], [np.asarray(q, float) for q in ps]
    n, draws = len(ys[0]), []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        draws.append([r2(y[i], q[i]) for y, q in zip(ys, ps)])
    draws = np.array(draws)
    point = [r2(y, q) for y, q in zip(ys, ps)]
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]                     # noqa: E731
    return {"components": [{"r2": point[k], "ci95": ci(draws[:, k])} for k in range(len(ys))],
            "mean": {"r2": float(np.mean(point)), "ci95": ci(draws.mean(1))}}


NB_CELL = 1000


def cellblock_ci(ys, ps, cells, nb=NB_CELL, seed=0):
    """95% CI of R^2 (per component and their mean) under a cell-block bootstrap: resample the design cells with
    replacement and take all clips of each drawn cell (the clips of a cell are not independent). ys/ps: lists."""
    rng = np.random.default_rng(seed)
    keys = sorted({tuple(c) for c in cells})
    idx = [np.flatnonzero([tuple(c) == k for c in cells]) for k in keys]
    ys, ps = [np.asarray(y, float) for y in ys], [np.asarray(q, float) for q in ps]
    draws = []
    for _ in range(nb):
        i = np.concatenate([idx[j] for j in rng.integers(0, len(keys), len(keys))])
        draws.append([r2(y[i], q[i]) for y, q in zip(ys, ps)])
    draws = np.array(draws)
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]                     # noqa: E731
    return {"components": [ci(draws[:, k]) for k in range(len(ys))], "mean": ci(draws.mean(1))}


def partial_targets_scores(X, a, theta_deg, nuis, folds, nb=NB, seed=0, cells=None, keep=None):
    """Partialled ridge (mean speed + displacement out of features and target inside each training fold) for signed
    a, |a| and Cartesian (ax, ay) under a given fold assignment."""
    ax, ay = cartesian_targets(a, theta_deg)
    out, cart = {}, []
    for name, y in (("signed_a", a), ("abs_a", np.abs(a)), ("ax", ax), ("ay", ay)):
        yr, pr = partial_ridge_oof(X, y, nuis, folds)
        if keep is not None:
            keep[name] = np.stack([yr, pr])
        if name in ("ax", "ay"):
            cart.append((yr, pr))
        else:
            out[name] = boot_r2(yr, pr, nb, seed)
            if cells is not None:
                out[name]["ci95_cellblock"] = cellblock_ci([yr], [pr], cells)["components"][0]
    out["cartesian"] = boot_r2_multi([c[0] for c in cart], [c[1] for c in cart], nb, seed)
    if cells is not None:
        cb = cellblock_ci([c[0] for c in cart], [c[1] for c in cart], cells)
        for k in range(2):
            out["cartesian"]["components"][k]["ci95_cellblock"] = cb["components"][k]
        out["cartesian"]["mean"]["ci95_cellblock"] = cb["mean"]
    return out


# ---------------------------------------------------------------- stages

def render(args):
    from wm.render_twin import validate, write_twin
    out = Path(args.out)
    plan = design(args.per_cell, 0)
    if args.n_smoke:
        plan = plan[::max(1, len(plan) // args.n_smoke)][:args.n_smoke]
    hashes, counts = [], []
    for k, meta in enumerate(plan):
        fr = write_twin(meta, None, out / "videos" / f"clip_{k:04d}.mp4")
        hashes.append(frame_hash(fr))
        counts.append(disk_pixels(fr).reshape(N_FRAMES, -1).sum(1))
    counts = np.array(counts)
    val = validate("acceleration", n=args.n_validate, seed=0) if args.n_validate else {}
    info = {"n": len(plan), "grid_mean_speed_mps": list(GRID_M), "grid_acceleration_mps2": list(GRID_A),
            "per_cell": args.per_cell, "design_seed": 0, "clips": plan, "frame_hash": hashes,
            "disk_pixels_per_frame": {"min": int(counts.min()), "median": float(np.median(counts)),
                                      "max": int(counts.max())},
            "renderer_validation_acceleration": {k: val[k] for k in ("codec", "mask_twin_iou_mean", "verdict")
                                                 if k in val}}
    (out / "plan.json").write_text(json.dumps(info, indent=1))
    print("rendered", len(plan), info["disk_pixels_per_frame"], info["renderer_validation_acceleration"].get("verdict"))


def extract(args):
    import torch
    from wm.data import load_table
    from wm.extract import disk_mask, encode, load_model, pick_device, pool, preprocess, set_precision
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from time_predictor import gpu_lock
    out = Path(args.out)
    info = json.loads((out / "plan.json").read_text())
    set_precision()
    dev = pick_device()
    frames = [decode(out / "videos" / f"clip_{i:04d}.mp4") for i in range(info["n"])]
    assert [frame_hash(f) for f in frames] == info["frame_hash"], "decoded frames differ from the Mac render"
    sp = None
    if args.speed_root:
        sp = load_table("speed", root=args.speed_root)
    for model_kind in MODELS:
        model = load_model(model_kind, dev)
        with gpu_lock() as waited, torch.no_grad():
            t0 = time.time()
            mp, tp = [], []
            for s in range(0, len(frames), 8):
                fb = frames[s:s + 8]
                masks = torch.from_numpy(np.stack([disk_mask(f)[0] for f in fb])).to(dev)
                points, _ = encode(model, preprocess(fb).to(dev))
                m, t, _ = pool(points, masks)
                mp.append(m.astype(np.float32))
                tp.append(t.astype(np.float16))
            secs = time.time() - t0
            np.savez(out / f"feat_{model_kind}.npz", meanpool=np.concatenate(mp), timepool=np.concatenate(tp),
                     seconds=secs, lock_wait_s=waited, device=str(dev),
                     gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
            print(model_kind, "grid", round(secs, 1), "s", flush=True)
            if model_kind == "random" and sp is not None:
                t0 = time.time()
                tp = []
                vids = list(sp["video"])
                for s in range(0, len(vids), 8):
                    fb = [decode(v) for v in vids[s:s + 8]]
                    masks = torch.from_numpy(np.stack([disk_mask(f)[0] for f in fb])).to(dev)
                    points, _ = encode(model, preprocess(fb).to(dev))
                    tp.append(pool(points, masks)[1][:, list(POINTS)].astype(np.float16))
                    if s % 256 == 0:
                        print("speed random", s, round(time.time() - t0, 1), flush=True)
                secs_sp = time.time() - t0
                np.savez(out / "speed_random_timepool.npz", timepool=np.concatenate(tp), points=np.array(POINTS),
                         ids=sp["id"].to_numpy(), speed=sp["speed_mps"].to_numpy(), seconds=secs_sp)
                print("random speed set", round(secs_sp, 1), "s", flush=True)
        del model
        torch.cuda.empty_cache()
    (out / "EXTRACT_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z"))


def _load_plan(out):
    info = json.loads((Path(out) / "plan.json").read_text())
    clips = info["clips"]
    a = np.array([c["acceleration_mps2"] for c in clips])
    m = np.array([c["mean_speed_mps"] for c in clips])
    v0 = np.array([c["speed_mps"] for c in clips])
    disp = np.array([c["displacement_m"] for c in clips])
    S_true = step_speeds(v0, a)
    folds = folds_by_cell([c["cell"] for c in clips])
    return info, clips, a, m, v0, disp, S_true, folds


def score(args):
    """Score the models in --models (default both); each model's block goes to <out>/score_<model>.json, then merge
    unless --no-merge (so the two models can run as separate processes)."""
    pts = [int(p) for p in args.points.split(",")] if args.points else list(POINTS)
    for model in args.models.split(","):
        r = score_model(args, model, pts)
        for L in pts:
            (Path(args.out) / f"score_{model}_L{L}.json").write_text(json.dumps(
                {**{k: v for k, v in r.items() if k != "points"}, "point": r["points"][str(L)]}, indent=1))
    if not args.no_merge:
        merge(args)


def score_model(args, model, points=POINTS):
    from wm.data import load_table
    out = Path(args.out)
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    n = len(a)
    nuis = np.column_stack([m, disp])
    sp_df = load_table("speed")
    sp_v = sp_df["speed_mps"].to_numpy(float)
    res = {"models": {}}
    feat = np.load(out / f"feat_{model}.npz")
    res["models"][model] = {"gpu_seconds_grid": float(feat["seconds"]), "device": str(feat["device"]),
                            "gpu": str(feat["gpu"]), "points": {}}
    if model == "vjepa2":
        sp_tp = np.load(Path(args.act_root) / "speed" / "vjepa2" / "timepool.npy", mmap_mode="r")
        sp_idx = {L: L for L in POINTS}
    else:
        z = np.load(out / "speed_random_timepool.npz")
        assert (z["ids"] == sp_df["id"].to_numpy()).all()
        sp_tp = z["timepool"]
        sp_idx = {L: i for i, L in enumerate(z["points"].tolist())}
        res["models"][model]["gpu_seconds_speed_set"] = float(z["seconds"])
    for L in points:
        t0 = time.time()
        Xm = np.asarray(feat["meanpool"][:, L], np.float64)
        T = np.asarray(feat["timepool"][:, L], np.float64)
        Xt = T.reshape(n, -1)
        r = {}
        # 1. plain and partialled ridge for a
        for name, X in (("meanpool", Xm), ("timepool", Xt)):
            plain = oof(lambda tr, te: _ridge(X[tr], a[tr]).predict(X[te]), n, folds)
            ya, pa = partial_ridge_oof(X, a, nuis, folds)
            ym, pm = partial_ridge_oof(X, m, a[:, None], folds)
            J = np.column_stack([oof(lambda tr, te: _ridge(X[tr], y[tr]).predict(X[te]), n, folds)
                                 for y in (m, a)])          # multi-output ridge, alpha per target = one fit per column
            r[name] = {"ridge_a": boot_r2(a, plain),
                       "ridge_a_partial_mean_speed_displacement": boot_r2(ya, pa),
                       "ridge_mean_speed_partial_a": boot_r2(ym, pm),
                       "joint_ridge": {"mean_speed": boot_r2(m, J[:, 0]), "a": boot_r2(a, J[:, 1])}}
        # 2. per-step speed decode: nested in-set probe and transfer probe from the supplied speed set
        S_in, S_in_tr = nested_step_decode(T, S_true, folds)
        Fs = np.nan_to_num(np.asarray(sp_tp[:, sp_idx[L]], np.float64))
        tprobe = _ridge(Fs.reshape(-1, Fs.shape[-1]), np.repeat(sp_v, Fs.shape[1]))
        S_tr = tprobe.predict(T.reshape(-1, T.shape[-1])).reshape(n, -1)
        slope = lambda S: (S - S.mean(1, keepdims=True)) @ (TAU - TAU.mean()) / ((TAU - TAU.mean()) ** 2).sum()  # noqa: E731
        dec = {}
        for nm, S in (("inset_nested", S_in), ("transfer_speed_set", S_tr)):
            sl = slope(S)
            dec[nm] = {"step_speed_r2": r2(S_true.ravel(), S.ravel()),
                       "mean_speed_r2": r2(m, S.mean(1)),
                       "a_from_decoded_slope": {"slope_on_a": float(np.polyfit(a, sl, 1)[0]),
                                                "corr": float(np.corrcoef(a, sl)[0, 1]),
                                                "r2_raw": r2(a, sl)}}
        # the acceleration readout with the decoded speed sequence partialled out (in-set, nested)
        for name, X in (("meanpool", Xm), ("timepool", Xt)):
            ys, ps = partial_ridge_oof(X, a, S_in, folds, S_in_tr)
            r[name]["ridge_a_partial_decoded_speed_seq"] = boot_r2(ys, ps)
            yt, pt = partial_ridge_oof(X, a, S_tr, folds)
            r[name]["ridge_a_partial_transfer_speed_seq"] = boot_r2(yt, pt)
        r["decoded_speed"] = dec
        r["linear_a_from_decoded_speed_seq"] = {
            "inset_nested": boot_r2(a, oof(lambda tr, te: _lstsq_pred(S_in_tr[int(folds[te][0])], a[tr], S_in[te]),
                                           n, folds)),
            "transfer_speed_set": boot_r2(a, oof(lambda tr, te: _lstsq_pred(S_tr[tr], a[tr], S_tr[te]), n, folds))}
        # 3. MLP on activations vs MLP on the decoded per-step speeds
        p_sin = mlp_oof(S_in, a, folds, S_in_tr)
        p_str = mlp_oof(S_tr, a, folds)
        p_am = mlp_oof(Xm, a, folds)
        p_at = mlp_oof(Xt, a, folds)
        r["mlp"] = {"act_meanpool": boot_r2(a, p_am, p2=p_sin), "act_timepool": boot_r2(a, p_at, p2=p_sin),
                    "speedseq_inset_nested": boot_r2(a, p_sin), "speedseq_transfer": boot_r2(a, p_str),
                    "act_meanpool_vs_transfer": boot_r2(a, p_am, p2=p_str)["minus_other"],
                    "act_timepool_vs_transfer": boot_r2(a, p_at, p2=p_str)["minus_other"]}
        res["models"][model]["points"][str(L)] = r
        print(model, L, f"{time.time() - t0:.0f}s",
              "partial", round(r["meanpool"]["ridge_a_partial_mean_speed_displacement"]["r2"], 3),
              round(r["timepool"]["ridge_a_partial_mean_speed_displacement"]["r2"], 3),
              "mlp act", round(r["mlp"]["act_meanpool"]["r2"], 3), round(r["mlp"]["act_timepool"]["r2"], 3),
              "speedseq", round(r["mlp"]["speedseq_inset_nested"]["r2"], 3),
              round(r["mlp"]["speedseq_transfer"]["r2"], 3), flush=True)
    return res["models"][model]


def merge(args):
    import os
    from wm.provenance import provenance, sha256_file
    out = Path(args.out)
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    n = len(a)
    corr = lambda x: float(np.corrcoef(a, x)[0, 1])                                  # noqa: E731
    res = {"n_clips": n, "grid": {"mean_speed_mps": info["grid_mean_speed_mps"],
                                  "acceleration_mps2": info["grid_acceleration_mps2"], "per_cell": info["per_cell"],
                                  "initial_speed_range_mps": [float(v0.min()), float(v0.max())],
                                  "final_speed_range_mps": [float(S_true.min()), float((m + a * T_MEAN).max())]},
           "design_correlations_with_a": {"mean_speed": corr(m), "displacement": corr(disp), "initial_speed": corr(v0),
                                          "final_speed": corr(m + a * T_MEAN)},
           "render": {k: info[k] for k in ("disk_pixels_per_frame", "renderer_validation_acceleration")},
           "folds": {"k": N_FOLDS, "rule": "each clip its own group; clips of each (m, a) cell spread over folds"},
           "models": {}}
    # floors / oracle (point-independent)
    res["reference_mlps"] = {
        "true_step_speeds_oracle": boot_r2(a, mlp_oof(S_true, a, folds)),
        "true_mean_speed_only": boot_r2(a, mlp_oof(m[:, None], a, folds))}
    for model in args.models.split(","):
        parts = {L: json.loads((out / f"score_{model}_L{L}.json").read_text()) for L in POINTS}
        res["models"][model] = {**{k: v for k, v in parts[POINTS[0]].items() if k != "point"},
                                "points": {str(L): parts[L]["point"] for L in POINTS}}
    res["keys"] = {
        "design_correlations_with_a": "Pearson r of a with each kinematic variable over the rendered clips",
        "models[m].points[L].{meanpool,timepool}": "features at point L: meanpool = mean over 2,048 tokens [D]; "
            "timepool = the 8 per-tubelet means concatenated [8D]. ridge_a: OOF ridge R^2 for a. "
            "ridge_a_partial_mean_speed_displacement: mean speed and displacement regressed (OLS, train fold) out of "
            "features and a, ridge on the residuals, R^2 of the held-out residual a. ridge_mean_speed_partial_a: the "
            "same for mean speed with a partialled. joint_ridge: ridge to (m, a) with alpha per target (the outputs "
            "decouple, so one fit per column), R^2 of each column. ridge_a_partial_decoded_speed_seq: a and features residualised on the 8 decoded per-step speeds "
            "(nested in-set probe; train rows use inner out-of-fold decodes); _transfer_: on the speed-set probe's.",
        "models[m].points[L].decoded_speed": "inset_nested: per-step ridge (tubelet features -> instantaneous speed "
            "at the tubelet midpoint) fit on outer-train clips only; transfer_speed_set: per-step ridge fit on every "
            "supplied speed-set clip x step (constant speed), read on the grid. a_from_decoded_slope: OLS slope of "
            "decoded speed on tubelet time, compared with a without any fit to a.",
        "models[m].points[L].mlp": "one-hidden-layer MLP (wm.bakeoff.MLPReadout, 64 units, alpha 1e-2, early "
            "stopping), 5 seeds averaged, OOF over 5 clip folds, predicting a. act_meanpool / act_timepool: on "
            "activations; speedseq_*: on the 8 decoded per-step speeds. r2 with 95% clip bootstrap; minus_other = "
            "paired bootstrap of R^2(act) - R^2(speed sequence) (inset nested unless named _vs_transfer).",
        "reference_mlps": "the same MLP on the TRUE per-step speeds (oracle ceiling) and on true mean speed only "
                          "(floor, ~0 by design)"}
    res["provenance"] = provenance(
        seeds={"design": 0, "folds": 0, "bootstrap": 0, "mlp": list(MLP_SEEDS), "random_init_torch_manual_seed": 0},
        points=list(POINTS), script="scripts/run_accel_grid.py",
        script_sha256=sha256_file(Path(__file__)), plan_sha256=sha256_file(out / "plan.json"),
        feature_sha256={p.name: sha256_file(p) for p in sorted(out.glob("*.npz"))},
        box=args.box, act_root=args.act_root, threads=os.environ.get("OMP_NUM_THREADS"))
    res["gpu_seconds_total"] = float(sum(v["gpu_seconds_grid"] + v.get("gpu_seconds_speed_set", 0.0)
                                         for v in res["models"].values()))
    Path(args.result).write_text(json.dumps(res, indent=1))
    figure(res, args.fig)
    print("wrote", args.result, "gpu s", res["gpu_seconds_total"])


def _magnitude_cell(out, model, L, mlp=True):
    """One (model, point) cell of the magnitude stage: meanpool and timepool features."""
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    feat = np.load(Path(out) / f"feat_{model}.npz")
    n = len(a)
    t0 = time.time()
    X = {"meanpool": np.asarray(feat["meanpool"][:, L], np.float64),
         "timepool": np.asarray(feat["timepool"][:, L], np.float64).reshape(n, -1)}
    r = {k: magnitude_scores(v, a, np.column_stack([m, disp]), folds) for k, v in X.items()}
    if mlp:
        for k, v in X.items():
            # |a| is a nonlinear function of a signed code, so a linear ridge can miss it; the signed run's MLP
            # (wm.bakeoff.MLPReadout, 5 seeds, same folds; mean speed needs no partial, r = 0 by design)
            pa, pm = mlp_oof(v, np.abs(a), folds), mlp_oof(v, a, folds)
            r[k]["mlp"] = {"abs_a": boot_r2(np.abs(a), pa), "signed_a": boot_r2(a, pm),
                           "abs_minus_signed": boot_r2_pair(np.abs(a), pa, a, pm)}
    print(model, L, f"{time.time() - t0:.0f}s", {k: (round(v["signed_a"]["ridge_partial_mean_speed_displacement"]["r2"], 3),
                                                      round(v["abs_a"]["ridge_partial_mean_speed_displacement"]["r2"], 3))
                                                  for k, v in r.items()}, flush=True)
    return model, L, r


def magnitude(args):
    """QA #358: |a| (the paper's target) beside signed a, same features, folds, nuisance partial-out and bootstrap
    as the signed run -> results/p5_accel_decorrelated_magnitude.json."""
    import os
    from joblib import Parallel, delayed
    from wm.provenance import provenance, sha256_file
    out = Path(args.out)
    pts = [int(p) for p in args.points.split(",")] if args.points else list(POINTS)
    models = args.models.split(",")
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    cells = Parallel(n_jobs=args.jobs)(delayed(_magnitude_cell)(out, mo, L, not args.no_mlp) for mo in models for L in pts)
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])                                   # noqa: E731
    res = {"n_clips": len(a), "qa": "#358",
           "abs_a_levels": {str(v): int(c) for v, c in zip(*np.unique(np.abs(a), return_counts=True))},
           "design_correlations": {"abs_a~mean_speed": corr(np.abs(a), m), "abs_a~displacement": corr(np.abs(a), disp),
                                   "abs_a~signed_a": corr(np.abs(a), a), "signed_a~mean_speed": corr(a, m)},
           "folds": {"k": N_FOLDS, "rule": "folds_by_cell(seed 0), as the signed run"},
           "models": {mo: {"points": {}} for mo in models}}
    for mo, L, r in cells:
        res["models"][mo]["points"][str(L)] = r
    res["keys"] = {
        "models[m].points[L].{meanpool,timepool}.{signed_a,abs_a}": "ridge: OOF ridge R^2 (standardise + RidgeCV, "
            "alpha by 5-fold CV over training clips, as run_accel_grid._ridge). ridge_partial_mean_speed_displacement: "
            "mean speed and displacement OLS-regressed out of features and target on the training fold, test rows "
            "residualised with train coefficients, R^2 of the held-out residual (the signed run's "
            "ridge_a_partial_mean_speed_displacement). r2 with 95% clip bootstrap (2,000 draws, seed 0).",
        "abs_minus_signed_partial": "paired clip bootstrap of partialled R^2(|a|) - R^2(signed a)",
        "abs_a_from_abs_signed_ridge": "|a| read as |OOF signed-a ridge prediction|, OLS-calibrated (|a| ~ 1 + |a_hat|) "
            "on nested inner out-of-fold predictions of the training clips; R^2 for |a| (no partial: |a| has r = 0 with "
            "mean speed and displacement by design); minus_abs_ridge = paired bootstrap vs the plain linear |a| ridge.",
        "models[m].points[L].{meanpool,timepool}.mlp": "one-hidden-layer MLP on activations (wm.bakeoff.MLPReadout, "
            "64 units, alpha 1e-2, early stopping), seeds 0-4 averaged, OOF over the same 5 folds, no partial (mean "
            "speed and displacement have r = 0 with a and |a| by design); signed_a is the signed run's mlp.act_*.",
        "meanpool / timepool": "mean over all 2,048 tokens [D] (order-blind) / 8 per-tubelet means concatenated [8D]",
        "random": "untrained copy of V-JEPA 2 (torch.manual_seed(0) random init), same clips"}
    res["provenance"] = provenance(
        seeds={"design": 0, "folds": 0, "bootstrap": 0, "ridge_inner_cv_kfold": 0, "random_init_torch_manual_seed": 0,
               "mlp": list(MLP_SEEDS) if not args.no_mlp else None},
        points=pts, script="scripts/run_accel_grid.py (stage magnitude)",
        script_sha256=sha256_file(Path(__file__)), plan_sha256=sha256_file(out / "plan.json"),
        feature_sha256={f"feat_{mo}.npz": sha256_file(out / f"feat_{mo}.npz") for mo in models},
        signed_run_result="results/p5_accel_decorrelated.json", threads=os.environ.get("OMP_NUM_THREADS"),
        compute="Mac CPU, features already on disk (extracted on vast 53255833 for the signed run); no GPU")
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)


def _cartloco_cell(out, model, L, oof_dir=None):
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    th = np.array([c["theta_degrees"] for c in clips])
    lf = loco_folds([c["cell"] for c in clips])
    feat = np.load(Path(out) / f"feat_{model}.npz")
    n, t0 = len(a), time.time()
    X = {"meanpool": np.asarray(feat["meanpool"][:, L], np.float64),
         "timepool": np.asarray(feat["timepool"][:, L], np.float64).reshape(n, -1)}
    nuis = np.column_stack([m, disp])
    cells = [c["cell"] for c in clips]
    r, keep = {}, {}
    for k, v in X.items():
        r[k] = {}
        for cv, fo in (("per_clip_folds", folds), ("leave_one_cell_out", lf)):
            kp = {}
            r[k][cv] = partial_targets_scores(v, a, th, nuis, fo, cells=cells, keep=kp)
            keep.update({f"{k}__{cv}__{t}": arr for t, arr in kp.items()})
    if oof_dir:
        np.savez(Path(oof_dir) / f"oof_{model}_L{L}.npz", **keep)
    print(model, L, f"{time.time() - t0:.0f}s", {k: {cv: (round(w["signed_a"]["r2"], 3), round(w["abs_a"]["r2"], 3),
                                                          round(w["cartesian"]["mean"]["r2"], 3)) for cv, w in v.items()}
                                                  for k, v in r.items()}, flush=True)
    return model, L, r


def cartloco(args):
    """QA #383 (Cartesian acceleration, the paper's Table 1 target) and #384 (leave-one-cell-out CV) ->
    results/p5_accel_decorrelated_cartesian_loco.json."""
    import os
    from joblib import Parallel, delayed
    from wm.provenance import provenance, sha256_file
    out = Path(args.out)
    pts = [int(p) for p in args.points.split(",")] if args.points else list(POINTS)
    models = args.models.split(",")
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    th = np.array([c["theta_degrees"] for c in clips])
    ax, ay = cartesian_targets(a, th)
    lf = loco_folds([c["cell"] for c in clips])
    cells = Parallel(n_jobs=args.jobs)(delayed(_cartloco_cell)(out, mo, L, args.oof_dir) for mo in models for L in pts)
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])                                   # noqa: E731
    res = {"n_clips": len(a), "qa": ["#383", "#384"],
           "design_correlations": {"ax~mean_speed": corr(ax, m), "ay~mean_speed": corr(ay, m), "ax~ay": corr(ax, ay),
                                   "ax~signed_a": corr(ax, a), "ay~signed_a": corr(ay, a)},
           "folds": {"per_clip_folds": {"k": N_FOLDS, "rule": "folds_by_cell(seed 0), the signed run's folds: each "
                                        "clip its own group, each cell's clips spread over the 5 folds"},
                     "leave_one_cell_out": {"k": int(lf.max() + 1), "rule": "loco_folds: hold out all 12 clips of one "
                                            "(mean speed, a) cell; R^2 on the pooled out-of-fold predictions"}},
           "models": {mo: {"points": {}} for mo in models}}
    for mo, L, r in cells:
        res["models"][mo]["points"][str(L)] = r
    res["keys"] = {
        "models[m].points[L].{meanpool,timepool}.{per_clip_folds,leave_one_cell_out}": "ridge with mean speed and "
            "displacement OLS-regressed out of features and target on the training folds (test rows residualised with "
            "train coefficients), R^2 of the held-out residual target, 95% clip bootstrap (2,000 draws, seed 0). "
            "signed_a reproduces the signed run's ridge_a_partial_mean_speed_displacement under per_clip_folds.",
        "cartesian": "ax = a cos(theta), ay = a sin(theta), each ridge-read separately; components [ax, ay] and their "
                     "mean, CIs from shared bootstrap draws",
        "inner_alpha_cv": "ridge alpha chosen by 5-fold KFold(shuffle, seed 0) over the training clips in both schemes "
                          "(not grouped by cell), as run_accel_grid._ridge"}
    res["provenance"] = provenance(
        seeds={"design": 0, "folds": 0, "bootstrap": 0, "ridge_inner_cv_kfold": 0, "random_init_torch_manual_seed": 0},
        points=pts, script="scripts/run_accel_grid.py (stage cartloco)",
        script_sha256=sha256_file(Path(__file__)), plan_sha256=sha256_file(out / "plan.json"),
        feature_sha256={f"feat_{mo}.npz": sha256_file(out / f"feat_{mo}.npz") for mo in models},
        threads=os.environ.get("OMP_NUM_THREADS"), compute="Mac CPU, features already on disk; no GPU")
    res["keys"]["ci95_cellblock"] = ("95% CI from a cell-block bootstrap: 1,000 draws (seed 0) resampling the 20 "
                                     "(mean speed, a) cells with replacement, all 12 clips of each drawn cell, over "
                                     "the same pooled out-of-fold predictions as ci95 (clip bootstrap)")
    res["provenance"]["seeds"]["cellblock_bootstrap"] = 0
    res["provenance"]["oof_predictions_dir"] = args.oof_dir
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)
    if args.oof_dir:
        cellboot_supplement(args, pts, models)


def cellboot_supplement(args, pts, models):
    """Cell-block CIs for the magnitude stage's partialled signed a and |a| scores, from the saved per-clip-fold
    OOF predictions (the magnitude stage's ridge_partial_mean_speed_displacement; identical fits) ->
    results/p5_accel_decorrelated_cellboot.json."""
    from wm.provenance import provenance, sha256_file
    out = Path(args.out)
    info, clips, a, m, v0, disp, S_true, folds = _load_plan(out)
    cells = [c["cell"] for c in clips]
    mag = json.loads((PROJECT_ROOT / "results" / "p5_accel_decorrelated_magnitude.json").read_text())
    res = {"n_clips": len(a), "n_cells": len({tuple(c) for c in cells}), "qa": "acceleration audit: cell-block CIs",
           "models": {mo: {"points": {}} for mo in models}}
    for mo in models:
        for L in pts:
            z = np.load(Path(args.oof_dir) / f"oof_{mo}_L{L}.npz")
            res["models"][mo]["points"][str(L)] = cell = {}
            for pool in ("meanpool", "timepool"):
                cell[pool] = {}
                for t in ("signed_a", "abs_a"):
                    yr, pr = z[f"{pool}__per_clip_folds__{t}"]
                    ref = mag["models"][mo]["points"][str(L)][pool][t]["ridge_partial_mean_speed_displacement"]
                    assert abs(r2(yr, pr) - ref["r2"]) < 1e-9, (mo, L, pool, t)
                    cell[pool][t] = {"ridge_partial_mean_speed_displacement": {
                        "r2": ref["r2"], "ci95_clip": ref["ci95"],
                        "ci95_cellblock": cellblock_ci([yr], [pr], cells)["components"][0]}}
    res["keys"] = {"ridge_partial_mean_speed_displacement": "the magnitude stage's (and, for signed_a, the signed "
                   "run's) partialled ridge under the per-clip folds; r2 and ci95_clip copied from "
                   "p5_accel_decorrelated_magnitude.json after asserting the refit OOF predictions reproduce r2; "
                   "ci95_cellblock: 1,000 draws (seed 0) over the 20 cells",
                   "not_covered": "plain ridge, abs_a_from_abs_signed_ridge, MLP readouts and the signed run's "
                                  "decoded-speed readouts: their OOF predictions were not saved and need a refit"}
    res["provenance"] = provenance(
        seeds={"design": 0, "folds": 0, "cellblock_bootstrap": 0}, points=pts,
        script="scripts/run_accel_grid.py (stage cartloco, cellboot_supplement)",
        script_sha256=sha256_file(Path(__file__)), plan_sha256=sha256_file(out / "plan.json"),
        feature_sha256={f"feat_{mo}.npz": sha256_file(out / f"feat_{mo}.npz") for mo in models},
        source_result_sha256={"p5_accel_decorrelated_magnitude.json": sha256_file(
            PROJECT_ROOT / "results" / "p5_accel_decorrelated_magnitude.json")},
        oof_predictions_dir=args.oof_dir)
    dst = PROJECT_ROOT / "results" / "p5_accel_decorrelated_cellboot.json"
    dst.write_text(json.dumps(res, indent=1))
    print("wrote", dst)


def _lstsq_pred(Ztr, ytr, Zte):
    D = lambda Z: np.column_stack([np.ones(len(Z)), Z])                            # noqa: E731
    return D(Zte) @ np.linalg.lstsq(D(Ztr), ytr, rcond=None)[0]


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    col = {"vjepa2": "C0", "random": "C3"}
    lab = {"vjepa2": "V-JEPA 2", "random": "untrained copy"}
    for model, d in res["models"].items():
        P = [int(p) for p in d["points"]]
        g = lambda f: np.array([f(d["points"][str(p)]) for p in P])                      # noqa: E731
        for ax, feat, ls in ((axes[0], "meanpool", "-"), (axes[0], "timepool", "--")):
            y = g(lambda r: r[feat]["ridge_a_partial_mean_speed_displacement"]["r2"])
            lo = g(lambda r: r[feat]["ridge_a_partial_mean_speed_displacement"]["ci95"][0])
            hi = g(lambda r: r[feat]["ridge_a_partial_mean_speed_displacement"]["ci95"][1])
            ax.plot(P, y, ls, marker="o", color=col[model], label=f"{lab[model]}, {feat}")
            ax.fill_between(P, lo, hi, color=col[model], alpha=0.12)
        y = g(lambda r: r["timepool"]["ridge_a_partial_decoded_speed_seq"]["r2"])
        axes[1].plot(P, y, "-o", color=col[model], label=f"{lab[model]}, timepool | decoded speeds")
        y = g(lambda r: r["meanpool"]["ridge_a_partial_decoded_speed_seq"]["r2"])
        axes[1].plot(P, y, ":o", color=col[model], label=f"{lab[model]}, meanpool | decoded speeds")
        for key, ls, nm in (("act_timepool", "-", "MLP(timepool)"), ("speedseq_inset_nested", "--", "MLP(decoded speeds)")):
            y = g(lambda r: r["mlp"][key]["r2"])
            e = np.abs(np.stack([g(lambda r: r["mlp"][key]["ci95"][0]), g(lambda r: r["mlp"][key]["ci95"][1])]) - y)
            axes[2].errorbar(np.array(P) + (0.2 if model == "random" else -0.2), y, yerr=e, fmt=ls + "o",
                             color=col[model], capsize=2, label=f"{lab[model]}, {nm}")
    axes[0].set_title("ridge R² for a, mean speed + displacement partialled", fontsize=9)
    axes[1].set_title("ridge R² for a, decoded per-step speeds partialled (nested)", fontsize=9)
    orc = res["reference_mlps"]["true_step_speeds_oracle"]["r2"]
    axes[2].axhline(orc, color="gray", lw=0.8, ls=":", label=f"MLP(true speeds) {orc:.2f}")
    axes[2].set_title("MLP for a: activations vs decoded speed sequence (95% CI)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("point (block output)")
        ax.set_ylabel("held-out R²")
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=6.5)
    c = res["design_correlations_with_a"]["mean_speed"]
    fig.suptitle(f"Acceleration with mean speed decorrelated: {res['n_clips']} rendered clips, "
                 f"{len(res['grid']['mean_speed_mps'])} mean speeds x {len(res['grid']['acceleration_mps2'])} "
                 f"accelerations, r(a, mean speed) = {c:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "extract", "score", "merge", "magnitude", "cartloco"])
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--n-smoke", type=int, default=0)
    ap.add_argument("--n-validate", type=int, default=20)
    ap.add_argument("--speed-root", default=None, help="extract: supplied speed-set dir (manifest.jsonl + videos)")
    ap.add_argument("--act-root", default=str(PROJECT_ROOT / "artifacts" / "activations"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--points", default="", help="score: comma-separated subset of POINTS (default all)")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--no-mlp", action="store_true", help="magnitude: skip the MLP readouts")
    ap.add_argument("--oof-dir", default=None, help="cartloco: save OOF predictions here and write the cell-block "
                    "supplement results/p5_accel_decorrelated_cellboot.json")
    ap.add_argument("--jobs", type=int, default=7, help="magnitude: parallel (model, point) cells")
    ap.add_argument("--box", default="vast 53255833 (box 5, RTX 4060 Ti 16 GB)")
    ap.add_argument("--result", default=str(PROJECT_ROOT / "results" / "p5_accel_decorrelated.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_accel_decorrelated.png"))
    args = ap.parse_args()
    if args.stage == "magnitude" and args.result == ap.get_default("result"):
        args.result = str(PROJECT_ROOT / "results" / "p5_accel_decorrelated_magnitude.json")
    if args.stage == "cartloco" and args.result == ap.get_default("result"):
        args.result = str(PROJECT_ROOT / "results" / "p5_accel_decorrelated_cartesian_loco.json")
    {"render": render, "extract": extract, "score": score, "merge": merge, "magnitude": magnitude, "cartloco": cartloco}[args.stage](args)


if __name__ == "__main__":
    main()

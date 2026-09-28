"""The closed-form four-number direction edit read by the predictor, and half-ring extrapolation.

Direction is linear in a 4-D polar-Fourier frame: Z ~ a + B [cos t, sin t, cos 2t, sin 2t] (OLS on knot rows at kept
values in the session-2 PCA-64; run_motion_geometry exp3 'fourier2_4d', run_bakeoff_unified_16arc 'fourier2_4d').
This asks whether that frame is a CONTROL (an edit built from it moves the predictor's forecast to the steered twin's)
or only a DESCRIPTION, reusing session2_direction_natural_norm unchanged (carriers, twins, natural norms, disk-token
set, forward, predictor-native probe, bootstrap).

  carriers / targets  plan.npz: 200 test clips x 4 held-out targets in the contiguous arc 303.75-343.125 deg (seed 0)
  TASK 1 arms (full kept ring, 56 values)
         fourier  pca.lift_delta((F(t*) - F(src)) B), B fit by OLS on the knot rows at kept values in the session-2 PCA
         spline / chord / null   the session-2 deltas stored in artifacts/session2/natural_norm/deltas.npz (the paper's
                  interpolating periodic spline, the chord through the kept centroids, the spline to the far end 143.4 deg)
         (no FITPACK arm: session2_direction_natural_norm does not compute one, and none is added here)
  TASK 2 arms (half-ring fits; the 45-deg target arc lies outside the fitted span)
         folds    A: knot values 123.75..298.125 (32 values; targets 5.6-45 deg past its upper end 298.125)
                  B: knot values 348.75..163.125 (32 values, wrapping through 0; targets 5.6-45 deg before its lower end
                  348.75). They are the two 180-deg halves of the kept 315-deg ring adjacent to the held arc on either
                  side (overlap 123.75..163.125, 8 values). PCA-64, centroids, Fourier map, spline and chord are all
                  refit on the fold's knot rows only. Carriers enter a fold when their source value is inside its span.
         fourier_half     (F(t*) - F(src)) B_fold
         spline_tangent   interpolating cubic spline with NATURAL boundary conditions (the authors' non-periodic
                          CubicSpline1D; parameterised by the label value) through the fold centroids, continued
                          linearly along its derivative at the end knot v_e nearest the target (= its own extrapolation):
                          [S(v_e) + S'(v_e)(t* - v_e)] - S(src)
         chord_nearest    straight line to the nearest fitted knot's centroid: C(v_e) - PL(src) (PL = polyline through
                          the fold centroids; target independent within a fold)
  sites / norms
         P22  pooled at point 22: delta added to every context token; own norm, and natural = rescaled per
              (carrier, target) to ||twin_meanpool_22 - src_meanpool_22|| (session2_direction_natural_norm 'natural')
         D12  point 12, object tokens only (run_token_patching.token_sets obj, context tubelets 0-3); natural = each obj
              token gets unit(delta) x ||h_twin[i] - h_src[i]|| at point 12 (session2_direction_natural_norm disk_forward);
              own = each obj token gets the arm's delta at its own norm
  readouts
         err / R      predictor-native direction probe on the step-mean pooled forecast (as the natural-norm script)
         forced choice   probe-free: is the edited forecast closer, in L2 over the POOLED forecast (token-mean per
                      predicted step, 4 steps x 1024 concatenated), to the steered twin's native forecast (session-2
                      twin_pred_pooled cache) than to the carrier's own unedited native forecast? fraction of
                      (carrier, target) pairs; CI = bootstrap over carriers of the per-carrier fraction (boot_ci, seed 0)
         encoder-level (no predictor)   point-L ridge probe (mf.ProbeReadout on the probe clips) on x + delta for the
                      pooled deltas at points 12 and 22, x = the carrier's full-clip meanpool at L

  plan     Mac CPU  -> artifacts/session2/fourier4/plan.npz + plan_meta.json
  forward  box GPU  -> <root>/out/pred.npz + forward_info.json     (python ... forward <root> [n_carriers])
  score    Mac CPU  -> results/session2_fourier4_predictor.json + figures/fig_fourier4_predictor.png
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
NN = OUT / "natural_norm"
F4 = OUT / "fourier4"
RES = PROJECT_ROOT / "results"
T1_ARMS = ("fourier", "spline", "chord", "null")
EXT_ARMS = ("fourier_half", "spline_tangent", "chord_nearest")
EXT_SITES = ("P22_own", "P22_natural", "D12_own", "D12_natural")
FOLDS = {"A": (123.75, 298.125), "B": (348.75, 163.125)}        # (first, last) kept value, walking up the ring
# task-1 GPU jobs: (name, site, arm, cond); parity jobs reproduce cached session-2 forecasts
T1_JOBS = (("P22_fourier_own", 22, "fourier", "own"), ("P22_fourier_natural", 22, "fourier", "natural"),
           ("P22_chord_own_parity", 22, "chord", "own"),
           ("D12_fourier_natural", 12, "fourier", "natural"), ("D12_chord_natural_parity", 12, "chord", "natural"),
           ("D12_fourier_own", 12, "fourier", "own"), ("D12_spline_own", 12, "spline", "own"),
           ("D12_chord_own", 12, "chord", "own"), ("D12_null_own", 12, "null", "own"))


# ------------------------------------------------------------------------------------------------ pure helpers
def fold_values(first, last, step=5.625):
    """Kept values from `first` walking up (mod 360) to `last`, and their unwrapped copies (monotone increasing)."""
    n = int(round(((last - first) % 360.0) / step)) + 1
    u = first + step * np.arange(n)
    return u % 360.0, u


def unwrap_to(v, first):
    """Value(s) mapped to [first - 180, first + 180) around the fold start's unwrapped frame (first itself is kept)."""
    v = np.asarray(v, float)
    return first + ((v - first + 180.0) % 360.0) - 180.0


def in_span(v, first, last):
    return ((np.asarray(v, float) - first) % 360.0) <= ((last - first) % 360.0) + 1e-9


def ext_deltas_pca(Z, src, tgt, vals_u, cent, B, first):
    """Task-2 displacements in the fold PCA frame. Z [n, k] carriers, src [n], tgt [n, T], vals_u [m] unwrapped fold
    values (increasing), cent [m, k] fold centroids, B [4, k] fold Fourier map. Returns dict arm -> [n, T, k] and the
    end-knot value per target [n, T] (unwrapped) and distance past it (deg)."""
    from scipy.interpolate import CubicSpline
    from wm.motiongeom import fourier_features as ff
    n, T = tgt.shape
    s_u = unwrap_to(src, first)
    t_rel = (np.asarray(tgt, float) - vals_u[-1]) % 360.0            # degrees past the upper end
    b_rel = (vals_u[0] - np.asarray(tgt, float)) % 360.0              # degrees before the lower end
    up = t_rel <= b_rel
    v_e = np.where(up, vals_u[-1], vals_u[0])
    t_u = np.where(up, vals_u[-1] + t_rel, vals_u[0] - b_rel)
    # the authors' non-periodic spline (refs/causalab .../spline/cubic.py CubicSpline1D bc="natural", l.7-8, l.341-349;
    # wm.manifold l.128-132): natural BCs (f'' = 0 at the end knots), linear extrapolation along the end slope.
    # QA #350: the first plan used not-a-knot (kept as plan_notaknot.npz; forecasts of that arm are not scored).
    cs = CubicSpline(vals_u, cent, axis=0, bc_type="natural")
    d1 = cs.derivative(1)
    pl = lambda v: np.stack([np.interp(v, vals_u, cent[:, j]) for j in range(cent.shape[1])], -1)  # noqa
    S_src, PL_src = cs(s_u), pl(s_u)                                     # [n, k]
    end_pt = np.where(up[..., None], cs(vals_u[-1]), cs(vals_u[0]))      # [n, T, k]
    end_d = np.where(up[..., None], d1(vals_u[-1]), d1(vals_u[0]))
    end_c = np.where(up[..., None], cent[-1], cent[0])
    out = {"fourier_half": (ff(tgt.reshape(-1), 2) - np.repeat(ff(src, 2), T, 0)).reshape(n, T, 4) @ B,
           "spline_tangent": end_pt + end_d * (t_u - v_e)[..., None] - S_src[:, None],
           "chord_nearest": end_c - PL_src[:, None]}
    return out, v_e, np.where(up, t_rel, b_rel)


def fourier_delta_pca(src, tgt, B):
    from wm.motiongeom import fourier_features as ff
    n, T = tgt.shape
    return (ff(tgt.reshape(-1), 2) - np.repeat(ff(src, 2), T, 0)).reshape(n, T, 4) @ B


def forced_choice(e, tw, src):
    """1 where the edited forecast e is closer (L2 over all trailing dims) to the twin's than to the source's.
    e, tw: [..., *F]; src broadcastable. Returns bool [...]."""
    ax = tuple(range(-2, 0))
    de = np.sqrt(((e - tw) ** 2).sum(ax))
    ds = np.sqrt(((e - src) ** 2).sum(ax))
    return de < ds


def recovery_full(e, tw, src):
    """Probe-free recovery <e - src, tw - src> / ||tw - src||^2 over the last two dims (wm.predictor_readout.recovery on
    the pooled forecast). forced_choice == (recovery_full > 0.5) exactly: closer to tw than to src iff the projection
    passes the midpoint."""
    a, b = e - src, tw - src
    return (a * b).sum((-2, -1)) / np.maximum((b * b).sum((-2, -1)), 1e-12)


def twin_identify(e, tw, src):
    """Target-specific probe-free readout: e [C, T, ...] edited forecasts for the carrier's T targets, tw [C, T, ...]
    the rendered twins' forecasts, src [C, 1, ...] the unedited forecast. 1 where the edit's forecast change e - src
    has its highest cosine with tw_j - src among the carrier's T twins (T-way identification; chance 1/T). A
    target-independent edit (the null) picks the same twin for every target and scores 1/T by construction."""
    a = (e - src).reshape(e.shape[0], e.shape[1], -1)
    b = (tw - src).reshape(tw.shape[0], tw.shape[1], -1)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    cos = np.einsum("ctf,ckf->ctk", a, b)
    return cos.argmax(-1) == np.arange(e.shape[1])[None]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ plan (CPU)
def plan():
    from wm import manifold as mf
    from wm import motiongeom as mg
    from wm.p2_data import load_inputs
    from run_session2 import load_plan
    arrays, info = load_plan(OUT)
    dz = np.load(NN / "deltas.npz")
    idx, carriers, tg = dz["idx"], dz["carrier_rows"], dz["targets"]
    C, T = tg.shape
    held = np.array(info["held_values"])
    out = {"idx": idx, "carrier_rows": carriers, "targets": tg,
           "natural_norm_L22": dz["natural_norm_L22"], "natural_norm_L12": dz["natural_norm_L12"]}
    meta = {"k": info["k"], "held_values": held.tolist(), "folds": {}, "t1": {}}
    ext_in = np.zeros((C, 2), bool)
    ext_gap = np.zeros((C, 2, T))
    for L in (12, 22):
        d = load_inputs("direction", L)
        y, X = d["y"], d["X"].astype(np.float64)
        knot = (d["role"] == "knot") & ~np.isin(y, held)
        pca = mf.fit_pca(X[knot], info["k"])                          # the session-2 / natural-norm PCA
        Zk = pca.project(X[knot])
        _, B = mg.fit_linear_encoding(Zk, mg.fourier_features(y[knot], 2))
        Z, src = pca.project(X[carriers]), y[carriers]
        dfour = pca.lift_delta(fourier_delta_pca(src, tg, B))           # [C, T, D]
        own = dz[f"deltas_L{L}"][:, 0].astype(np.float64)               # [C, A(spline, chord, null), T, D]
        t1 = np.concatenate([dfour[:, None], own], 1)                   # [C, 4, T, D] in T1_ARMS order
        out[f"t1_L{L}"] = t1.astype(np.float32)
        # fit quality of the 4-D map on the knot rows (R^2 in PCA-64)
        a0, _ = mg.fit_linear_encoding(Zk, mg.fourier_features(y[knot], 2))
        r2 = 1 - ((Zk - a0 - mg.fourier_features(y[knot], 2) @ B) ** 2).sum() / ((Zk - Zk.mean(0)) ** 2).sum()
        meta["t1"][str(L)] = {"own_norm_median": {a: float(np.median(np.linalg.norm(t1[:, i], axis=-1)))
                                                  for i, a in enumerate(T1_ARMS)},
                              "natural_norm_median": float(np.median(dz[f"natural_norm_L{L}"])),
                              "fourier_fit_r2_pca64_knot_rows": float(r2),
                              "cos_fourier_vs_chord_median": float(np.median(
                                  (t1[:, 0] * t1[:, 2]).sum(-1) / np.linalg.norm(t1[:, 0], axis=-1)
                                  / np.linalg.norm(t1[:, 2], axis=-1))),
                              "cos_fourier_vs_spline_median": float(np.median(
                                  (t1[:, 0] * t1[:, 1]).sum(-1) / np.linalg.norm(t1[:, 0], axis=-1)
                                  / np.linalg.norm(t1[:, 1], axis=-1)))}
        ext = np.zeros((C, 2, len(EXT_ARMS), T, X.shape[1]))
        for fi, (fname, (first, last)) in enumerate(FOLDS.items()):
            vals, vals_u = fold_values(first, last)
            assert not np.isin(vals, held).any() and len(vals) == 32
            rows = knot & np.isin(y, vals)
            fp = mf.fit_pca(X[rows], info["k"])
            Zf = fp.project(X[rows])
            cent = np.stack([Zf[np.isclose(y[rows], v)].mean(0) for v in vals])
            _, Bf = mg.fit_linear_encoding(Zf, mg.fourier_features(y[rows], 2))
            ins = in_span(src, first, last)
            ext_in[:, fi] = ins
            dd, v_e, gap = ext_deltas_pca(fp.project(X[carriers]), src, tg, vals_u, cent, Bf, first)
            ext_gap[:, fi] = gap
            for ai, a in enumerate(EXT_ARMS):
                ext[:, fi, ai] = fp.lift_delta(dd[a])
            ext[~ins, fi] = 0.0
            meta["folds"].setdefault(fname, {"values": [float(vals[0]), float(vals[-1])], "n_values": len(vals),
                                             "n_knot_rows": int(rows.sum()), "n_carriers_in_span": int(ins.sum()),
                                             "gap_deg_range": [float(gap[ins].min()), float(gap[ins].max())]})
            meta["folds"][fname][f"own_norm_median_L{L}"] = {
                a: float(np.median(np.linalg.norm(ext[ins, fi, ai], axis=-1))) for ai, a in enumerate(EXT_ARMS)}
        out[f"ext_L{L}"] = ext.astype(np.float32)
        print(L, json.dumps(meta["t1"][str(L)]), flush=True)
    out["ext_in"], out["ext_gap"] = ext_in, ext_gap
    meta["n_carriers_in_any_fold"] = int(ext_in.any(1).sum())
    meta["n_carriers_in_both_folds"] = int(ext_in.all(1).sum())
    F4.mkdir(parents=True, exist_ok=True)
    np.savez(F4 / "plan.npz", **out)
    meta["plan_sha256"] = sha256_file(F4 / "plan.npz")
    (F4 / "plan_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["folds"], indent=1))


# ------------------------------------------------------------------------------------------------ forward (GPU)
def forward(root, n_carriers=200, batch_size=12):
    import torch
    torch.set_num_threads(4)
    from wm.data import disk_mask
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    from run_token_patching import token_sets
    root = Path(root)
    (root / "out").mkdir(exist_ok=True)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    enc = model.encoder
    z = np.load(root / "plan.npz")
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    C, T = len(rows), z["targets"].shape[1]
    D = 1024
    t1d = {L: z[f"t1_L{L}"][:C].astype(np.float64) for L in (12, 22)}
    exd = {L: z[f"ext_L{L}"][:C].astype(np.float64) for L in (12, 22)}
    nat22 = z["natural_norm_L22"][:C].astype(np.float64)
    ext_in = z["ext_in"][:C]
    df = load_table("direction")
    t1 = np.zeros((C, len(T1_JOBS), T, 4, D), np.float32)
    ext = np.full((C, 2, len(EXT_ARMS), len(EXT_SITES), T, 4, D), np.nan, np.float32)
    src_pp = np.zeros((C, 4, D), np.float32)
    twin_pp = np.zeros((C, T, 4, D), np.float32)
    ntok = np.zeros((C, T), np.int32)
    t0 = time.time()
    n_jobs = 0

    @torch.no_grad()
    def run(h, L):
        for layer in enc.layer[L:]:
            h = layer(h, None, None, False)[0]
        return pool_steps(predict_future(model, enc.layernorm(h), 0))

    unit = lambda v: v / max(np.linalg.norm(v), 1e-12)  # noqa
    for c in range(C):
        fs = decode(df["video"].iloc[rows[c]])
        ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
        ms = disk_mask(fs)[0]
        pv = preprocess([fs] + ft).to(device)
        saved, fin, _ = prefix(model, pv[:, :8], [12, 22])
        pp = pool_steps(predict_future(model, fin, 0)).cpu().numpy()
        src_pp[c], twin_pp[c] = pp[0], pp[1:]
        jobs = {12: [], 22: []}                                                # (key, E [1024, D] tensor)
        h12 = saved[12]
        for j in range(T):
            obj = torch.as_tensor(token_sets(ms, disk_mask(ft[j])[0])["obj"], device=device)
            ntok[c, j] = int(obj.sum())
            n_i = (h12[1 + j] - h12[0]).norm(dim=-1) * obj                        # twin per-token change, 0 off obj
            objf = obj.to(h12.dtype)

            def mk(L, vec, cond):
                v = torch.as_tensor(vec, dtype=h12.dtype, device=device)
                if L == 22:                                                        # pooled: every context token
                    s = 1.0 if cond == "own" else float(nat22[c, j] / max(np.linalg.norm(vec), 1e-12))
                    return (s * v)[None, :].expand(1024, D)
                if cond == "own":                                                  # obj tokens, own delta each
                    return objf[:, None] * v[None, :]
                return n_i[:, None] * torch.as_tensor(unit(vec), dtype=h12.dtype, device=device)[None, :]
            for q, (_, L, arm, cond) in enumerate(T1_JOBS):
                jobs[L].append((("t1", q, j), mk(L, t1d[L][c, T1_ARMS.index(arm), j], cond)))
            for fi in range(2):
                if not ext_in[c, fi]:
                    continue
                for ai in range(len(EXT_ARMS)):
                    for si, site in enumerate(EXT_SITES):
                        L = int(site[1:3])
                        jobs[L].append((("ext", fi, ai, si, j), mk(L, exd[L][c, fi, ai, j], site.split("_")[1])))
        for L in (12, 22):
            jl = jobs[L]
            for b in range(0, len(jl), batch_size):
                ch = jl[b:b + batch_size]
                p = run(saved[L][:1] + torch.stack([e for _, e in ch]), L).cpu().numpy()
                for q, (key, _) in enumerate(ch):
                    if key[0] == "t1":
                        t1[c, key[1], key[2]] = p[q]
                    else:
                        ext[c, key[1], key[2], key[3], key[4]] = p[q]
            n_jobs += len(jl)
        if c % 10 == 0 or c == C - 1:
            el = time.time() - t0
            print(f"{c + 1}/{C} carriers {el:.0f}s  eta {el / (c + 1) * (C - c - 1):.0f}s  jobs {n_jobs}", flush=True)
    secs = time.time() - t0
    np.savez(root / "out" / "pred.npz", idx=idx, t1=t1, ext=ext, src_pred_pooled=src_pp, twin_pred_pooled=twin_pp,
             ntok=ntok)
    info = {"seconds": secs, "n_carriers": C, "n_jobs": n_jobs, "batch_size": batch_size, "t1_jobs": [j[0] for j in T1_JOBS],
            "ext_arms": list(EXT_ARMS), "ext_sites": list(EXT_SITES), "folds": list(FOLDS), "mask_index": 0,
            "plan_sha256": sha256_file(root / "plan.npz"), "script_sha256": sha256_file(Path(__file__)),
            "box": os.environ.get("WM_BOX"),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "out" / "forward_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def forward_spline(root, plan_name, n_carriers=200, batch_size=12):
    """QA #350 re-run of the spline_tangent extrapolation arm only (natural-BC plan), same sites / norms / forward."""
    import torch
    torch.set_num_threads(4)
    from wm.data import disk_mask
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    from run_token_patching import token_sets
    root = Path(root)
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    enc = model.encoder
    z = np.load(root / plan_name)
    idx, rows = z["idx"][:n_carriers], z["carrier_rows"][:n_carriers]
    C, T, D = len(rows), z["targets"].shape[1], 1024
    ai = EXT_ARMS.index("spline_tangent")
    exd = {L: z[f"ext_L{L}"][:C, :, ai].astype(np.float64) for L in (12, 22)}
    nat22 = z["natural_norm_L22"][:C].astype(np.float64)
    ext_in = z["ext_in"][:C]
    df = load_table("direction")
    out = np.full((C, 2, len(EXT_SITES), T, 4, D), np.nan, np.float32)
    src_pp = np.zeros((C, 4, D), np.float32)
    t0, n_jobs = time.time(), 0

    @torch.no_grad()
    def run(h, L):
        for layer in enc.layer[L:]:
            h = layer(h, None, None, False)[0]
        return pool_steps(predict_future(model, enc.layernorm(h), 0))

    for c in range(C):
        if not ext_in[c].any():
            continue
        fs = decode(df["video"].iloc[rows[c]])
        ft = [decode(root / "twins" / f"twin_{int(idx[c]):04d}_{j}.mp4") for j in range(T)]
        ms = disk_mask(fs)[0]
        pv = preprocess([fs] + ft).to(device)
        saved, fin, _ = prefix(model, pv[:, :8], [12, 22])
        src_pp[c] = pool_steps(predict_future(model, fin[:1], 0)).cpu().numpy()[0]
        h12 = saved[12]
        jobs = {12: [], 22: []}
        for j in range(T):
            obj = torch.as_tensor(token_sets(ms, disk_mask(ft[j])[0])["obj"], device=device)
            n_i = (h12[1 + j] - h12[0]).norm(dim=-1) * obj
            objf = obj.to(h12.dtype)
            for fi in range(2):
                if not ext_in[c, fi]:
                    continue
                for si, site in enumerate(EXT_SITES):
                    L, cond = int(site[1:3]), site.split("_")[1]
                    vec = exd[L][c, fi, j]
                    v = torch.as_tensor(vec, dtype=h12.dtype, device=device)
                    if L == 22:
                        sc = 1.0 if cond == "own" else float(nat22[c, j] / max(np.linalg.norm(vec), 1e-12))
                        E = (sc * v)[None, :].expand(1024, D)
                    elif cond == "own":
                        E = objf[:, None] * v[None, :]
                    else:
                        E = n_i[:, None] * (v / max(float(v.norm()), 1e-12))[None, :]
                    jobs[L].append(((fi, si, j), E))
        for L in (12, 22):
            jl = jobs[L]
            for b in range(0, len(jl), batch_size):
                ch = jl[b:b + batch_size]
                p = run(saved[L][:1] + torch.stack([e for _, e in ch]), L).cpu().numpy()
                for q, ((fi, si, j), _) in enumerate(ch):
                    out[c, fi, si, j] = p[q]
            n_jobs += len(jl)
        if c % 20 == 0:
            print(f"{c + 1}/{C} carriers {time.time() - t0:.0f}s jobs {n_jobs}", flush=True)
    secs = time.time() - t0
    (root / "out").mkdir(exist_ok=True)
    np.savez(root / "out" / "pred_spline_natural.npz", idx=idx, ext_spline=out, src_pred_pooled=src_pp)
    info = {"seconds": secs, "n_carriers": C, "n_jobs": n_jobs, "plan": plan_name,
            "plan_sha256": sha256_file(root / plan_name), "script_sha256": sha256_file(Path(__file__)),
            "box": os.environ.get("WM_BOX"),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__}
    (root / "out" / "forward_spline_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


# ------------------------------------------------------------------------------------------------ score (CPU)
def score():
    from wm import manifold as mf
    from run_session2 import angle_of, boot_ci, load_groups, load_plan, wrap, write
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    nat_json = json.loads((RES / "session2_predictor_native_readout.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    arrays, info = load_plan(OUT)
    F = load_groups(OUT / "forward", keys={"idx", "src_pred_pooled", "twin_pred_pooled"})
    fo = Path(os.environ.get("F4_OUT", F4 / "out"))
    z = np.load(fo / "pred.npz")
    pz = np.load(F4 / "plan.npz")
    finfo = json.loads((fo / "forward_info.json").read_text())
    meta = json.loads((F4 / "plan_meta.json").read_text())
    C = finfo["n_carriers"]
    assert (z["idx"] == F["idx"][:C]).all() and (pz["idx"][:C] == z["idx"]).all()
    rows, tg = pz["carrier_rows"][:C], pz["targets"][:C]
    T = tg.shape[1]
    # QA #350: reference forecasts = this pass's own native forecasts (same box, same batch path as the edits);
    # the session-2 cache is the sensitivity check and, for the twins, the positive control
    SRCc = F["src_pred_pooled"][:C].astype(np.float64)
    TWc = F["twin_pred_pooled"][:C].astype(np.float64)
    SRC = z["src_pred_pooled"].astype(np.float64)                       # [C, 4, D] unedited native forecast (rerun)
    TW = z["twin_pred_pooled"].astype(np.float64)                       # [C, T, 4, D] rendered twins' forecasts (rerun)
    u = lambda th: np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], -1)  # noqa
    d_true = u(tg) - u(y[rows])[:, None]
    proj = lambda a, bb: (a * bb).sum(-1) / np.maximum((bb * bb).sum(-1), 1e-9)  # noqa
    Ps = readP(SRC.mean(1))

    def metrics(Zp, mask=None):
        """Zp [C, T, 4, D] edited pooled forecasts; mask [C] carriers to use. Per-(c, t) err, R, forced choice."""
        Zp = Zp.astype(np.float64)
        Pe = readP(Zp.mean(2))
        return {"err": wrap(angle_of(Pe) - tg), "R": proj(Pe - Ps[:, None], d_true),
                "fc": forced_choice(Zp, TW, SRC[:, None]).astype(float), "rec": recovery_full(Zp, TW, SRC[:, None]),
                "id": twin_identify(Zp, TW, SRC[:, None]).astype(float)}

    def summ(m, mask=None):
        k = np.ones(C, bool) if mask is None else mask
        return {"err_to_target": boot_ci(m["err"][k].mean(1)), "R": boot_ci(m["R"][k].mean(1)),
                "forced_choice_frac": boot_ci(m["fc"][k].mean(1)), "recovery_full": boot_ci(m["rec"][k].mean(1)),
                "twin_id_acc": boot_ci(m["id"][k].mean(1)), "n_pairs": int(k.sum() * T)}

    nrm = lambda x: np.linalg.norm(x.reshape(x.shape[0], x.shape[1], -1), axis=-1)  # noqa
    tw_rel = nrm(TW - TWc) / nrm(TWc)                                    # [C, T] relative L2 rerun vs cache
    tw_move = nrm(TWc - SRCc[:, None]) / nrm(TWc)
    xm = np.linalg.norm((TW[:, :, None] - TWc[:, None]).reshape(C, T, T, -1), axis=-1).argmin(-1)
    par = {"src_forecast_rerun_vs_cache_rel_maxabs": float(np.abs(SRC - SRCc).max() / np.abs(SRCc).max()),
           "twin_forecast_rerun_vs_cache_rel_maxabs": float(np.abs(TW - TWc).max() / np.abs(TWc).max()),
           "twin_forecast_rerun_vs_cache_rel_L2": {"median": float(np.median(tw_rel)), "p90": float(np.percentile(tw_rel, 90)),
                                                   "max": float(tw_rel.max()), "frac_below_1e-3": float((tw_rel < 1e-3).mean()),
                                                   "frac_carriers_all_below_1e-3": float((tw_rel < 1e-3).all(1).mean()),
                                                   "carriers_above_1e-3": np.flatnonzero((tw_rel >= 1e-3).any(1)).tolist()},
           "twin_minus_src_rel_L2_median_for_scale": float(np.median(tw_move)),
           "twin_rerun_nearest_cache_twin_is_same_target_frac": float((xm == np.arange(T)[None]).mean()),
           "note": "UNRESOLVED. Pixels are identical: all 800 decoded twin clips hash-match plan twin_frame_hash "
                   "(artifacts/session2/fourier4/out/twin_hash_check.json); source forecasts match the cache (rel max-abs "
                   "above). The on-box twin forecasts reproduce themselves across paths (the disk-token chord at the "
                   "twin per-token dose matches session2_direction_natural_norm disk_pred exactly, and that run encoded "
                   "the same [source + 4 twins] batch), but differ from the session-2 twin_pred_pooled cache by the "
                   "rel L2 shown (median / p90 / max). The session-2 cache encoded twins in multi-carrier batches "
                   "(run_session2 forward, bs); fp32/TF32-off batch effects are normally ~1e-5, so batching alone "
                   "is not a sufficient explanation. Primary reference = this pass's forecasts (same pass as the edits); "
                   "cache-referenced numbers are in task1.sensitivity_reference_session2_cache. Consumers of the "
                   "session-2 twin cache (twin ceilings in the natural-norm files) should be rechecked."}
    nn = np.load(NN / "pred.npz")
    dk = np.load(NN / "disk_pred.npz")
    t1 = z["t1"]
    J = {j[0]: i for i, j in enumerate(T1_JOBS)}
    par["P22_chord_own_rerun_vs_cache_rel_maxabs"] = float(np.abs(t1[:, J["P22_chord_own_parity"]] - nn["pred_L22"][:C, 0, 1]).max()
                                                           / np.abs(nn["pred_L22"]).max())
    par["D12_chord_natural_rerun_vs_cache_rel_maxabs"] = float(np.abs(t1[:, J["D12_chord_natural_parity"]] - dk["pred"][:C, 1]).max()
                                                               / np.abs(dk["pred"]).max())
    # --- task 1 table: site x cond x arm -> forecasts [C, T, 4, D]
    NNA = ("spline", "chord", "null")
    fore = {("P22", "own", "fourier"): t1[:, J["P22_fourier_own"]], ("P22", "natural", "fourier"): t1[:, J["P22_fourier_natural"]],
            ("D12", "natural", "fourier"): t1[:, J["D12_fourier_natural"]], ("D12", "own", "fourier"): t1[:, J["D12_fourier_own"]]}
    for k, a in enumerate(NNA):
        fore[("P22", "own", a)] = nn["pred_L22"][:C, 0, k]
        fore[("P22", "natural", a)] = nn["pred_L22"][:C, 1, k]
        fore[("D12", "natural", a)] = dk["pred"][:C, k]
        fore[("D12", "own", a)] = t1[:, J[f"D12_{a}_own"]]
    M1 = {key: metrics(v) for key, v in fore.items()}
    twm = metrics(TW)
    task1 = {"table": {}, "paired": {}}
    for (site, cond, arm), m in M1.items():
        task1["table"].setdefault(site, {}).setdefault(cond, {})[arm] = summ(m)
    for site in ("P22", "D12"):
        for cond in ("own", "natural"):
            pr = {}
            for o in ("chord", "spline", "null"):
                a, bb = M1[(site, cond, "fourier")], M1[(site, cond, o)]
                pr[f"fourier_minus_{o}"] = {q: boot_ci((a[q] - bb[q]).mean(1)) for q in ("err", "R", "fc", "rec", "id")}
            for o in ("chord", "spline"):
                a, bb = M1[(site, cond, o)], M1[(site, cond, "null")]
                pr[f"{o}_minus_null"] = {q: boot_ci((a[q] - bb[q]).mean(1)) for q in ("err", "fc", "rec", "id")}
            task1["paired"].setdefault(site, {})[cond] = pr
    task1["sensitivity_reference_session2_cache"] = {
        f"{site}_{cond}": {arm: {"forced_choice_frac": boot_ci(forced_choice(fore[(site, cond, arm)].astype(np.float64), TWc, SRCc[:, None]).mean(1)),
                                 "recovery_full": boot_ci(recovery_full(fore[(site, cond, arm)].astype(np.float64), TWc, SRCc[:, None]).mean(1)),
                                 "twin_id_acc": boot_ci(twin_identify(fore[(site, cond, arm)].astype(np.float64), TWc, SRCc[:, None]).mean(1))}
                           for arm in T1_ARMS} for site in ("P22", "D12") for cond in ("own", "natural")}
    pos = metrics(TWc)
    task1["positive_control_twin_forecast_session2_cache_vs_rerun_reference"] = {
        "forced_choice_frac": boot_ci(pos["fc"].mean(1)), "recovery_full": boot_ci(pos["rec"].mean(1)),
        "twin_id_acc": boot_ci(pos["id"].mean(1)), "err_to_target": boot_ci(pos["err"].mean(1)),
        "note": "the rendered twin's native forecast (independent session-2 computation) scored against this pass's twin "
                "vs unedited forecasts: the readouts' ceiling; the far-end null (arm 'null') is the negative control"}
    task1["positive_control_self_rerun_twin"] = {"forced_choice_frac": boot_ci(metrics(TW)["fc"].mean(1)),
                                                  "note": "trivially 1 (distance 0 to itself); shown for completeness"}
    task1["twin_ceiling"] = {"err_to_target": boot_ci(twm["err"].mean(1)), "R": boot_ci(twm["R"].mean(1))}
    task1["unedited_err_to_target"] = boot_ci(wrap(angle_of(Ps)[:, None] - tg).mean(1))
    task1["obj_token_fraction_mean"] = float(z["ntok"].mean() / 1024)
    # --- encoder-level readouts of the pooled deltas (no predictor)
    enc = {}
    for L in (12, 22):
        d = load_inputs("direction", L)
        pr_ = mf.ProbeReadout(d["X"][d["role"] == "probe"], d["y"][d["role"] == "probe"], True)
        x = d["X"][rows].astype(np.float64)
        px = pr_.raw(x)
        nat = pz[f"natural_norm_L{L}"][:C].astype(np.float64)

        def enc_m(delta, cond, pr_=pr_, x=x, px=px, nat=nat):
            dl = delta if cond == "own" else delta * (nat / np.maximum(np.linalg.norm(delta, axis=-1), 1e-12))[..., None]
            P = pr_.raw((x[:, None] + dl).reshape(-1, x.shape[1])).reshape(C, T, 2)
            return {"err": wrap(angle_of(P) - tg), "R": proj(P - px[:, None], d_true)}
        e1 = {}
        for i, a in enumerate(T1_ARMS):
            for cond in ("own", "natural"):
                m = enc_m(pz[f"t1_L{L}"][:C, i].astype(np.float64), cond)
                e1.setdefault(cond, {})[a] = {"err_to_target": boot_ci(m["err"].mean(1)), "R": boot_ci(m["R"].mean(1))}
        e2 = {}
        for cond in ("own", "natural"):
            ms_ = {}
            for ai, a in enumerate(EXT_ARMS):
                errs, Rs = np.full((C, 2, T), np.nan), np.full((C, 2, T), np.nan)
                for fi in range(2):
                    m = enc_m(pz[f"ext_L{L}"][:C, fi, ai].astype(np.float64), cond)
                    ins = pz["ext_in"][:C, fi]
                    errs[ins, fi], Rs[ins, fi] = m["err"][ins], m["R"][ins]
                ms_[a] = (errs, Rs)
                keep = pz["ext_in"][:C].any(1)
                e2.setdefault(cond, {})[a] = {"err_to_target": boot_ci(np.nanmean(errs[keep], (1, 2))),
                                              "R": boot_ci(np.nanmean(Rs[keep], (1, 2)))}
            keep = pz["ext_in"][:C].any(1)
            for o in ("spline_tangent", "chord_nearest"):
                e2[cond][f"fourier_half_minus_{o}_err"] = boot_ci(np.nanmean(ms_["fourier_half"][0][keep] - ms_[o][0][keep], (1, 2)))
        enc[str(L)] = {"task1_pooled_delta": e1, "task2_pooled_delta": e2}
    # --- task 2
    ext = z["ext"].copy()
    spf = fo / "pred_spline_natural.npz"
    if spf.exists():                                                   # QA #350: natural-BC spline arm replaces not-a-knot
        zs = np.load(spf)
        assert (zs["idx"] == z["idx"]).all()
        ext[:, :, EXT_ARMS.index("spline_tangent")] = zs["ext_spline"]
        finfo_spline = json.loads((fo / "forward_spline_info.json").read_text())
        assert finfo_spline["plan_sha256"] == sha256_file(F4 / "plan.npz")
        par["spline_rerun_src_forecast_vs_rerun_rel_maxabs"] = float(
            np.abs(zs["src_pred_pooled"][ins_any := pz["ext_in"][:C].any(1)] - SRC[ins_any]).max() / np.abs(SRC).max())
    else:
        finfo_spline = None
    ins = pz["ext_in"][:C]
    gap = pz["ext_gap"][:C]
    keep = ins.any(1)
    task2 = {"table": {}, "per_fold": {}, "paired": {}, "err_by_gap": {}, "full_ring_fourier_same_carriers": {}}
    Mx = {}
    for si, site in enumerate(EXT_SITES):
        for ai, a in enumerate(EXT_ARMS):
            accs = {q: np.full((C, 2, T), np.nan) for q in ("err", "R", "fc", "rec", "id")}
            for fi in range(2):
                sel = ins[:, fi]
                Zp = np.where(sel[:, None, None, None], ext[:, fi, ai, si], 0.0)
                m = metrics(Zp)
                for q in accs:
                    accs[q][sel, fi] = m[q][sel]
                task2["per_fold"].setdefault(list(FOLDS)[fi], {}).setdefault(site, {})[a] = {
                    "err_to_target": boot_ci(m["err"][sel].mean(1)), "R": boot_ci(m["R"][sel].mean(1)),
                    "forced_choice_frac": boot_ci(m["fc"][sel].mean(1)), "recovery_full": boot_ci(m["rec"][sel].mean(1)),
                    "twin_id_acc": boot_ci(m["id"][sel].mean(1)), "n_carriers": int(sel.sum())}
            Mx[(site, a)] = accs
            task2["table"].setdefault(site, {})[a] = {
                "err_to_target": boot_ci(np.nanmean(accs["err"][keep], (1, 2))),
                "R": boot_ci(np.nanmean(accs["R"][keep], (1, 2))),
                "forced_choice_frac": boot_ci(np.nanmean(accs["fc"][keep], (1, 2))),
                "recovery_full": boot_ci(np.nanmean(accs["rec"][keep], (1, 2))),
                "twin_id_acc": boot_ci(np.nanmean(accs["id"][keep], (1, 2))),
                "n_carriers": int(keep.sum()), "n_carrier_fold_target_triples": int(ins.sum() * T)}
            bins = [(5, 12), (12, 23), (23, 34), (34, 46)]
            task2["err_by_gap"].setdefault(site, {})[a] = {
                f"{lo}-{hi}deg": {"err": float(np.nanmean(np.where((gap >= lo) & (gap < hi) & ins[..., None], accs["err"], np.nan))),
                                  "fc": float(np.nanmean(np.where((gap >= lo) & (gap < hi) & ins[..., None], accs["fc"], np.nan))),
                                  "n": int(((gap >= lo) & (gap < hi) & ins[..., None]).sum())} for lo, hi in bins}
        pr = {}
        for o in ("spline_tangent", "chord_nearest"):
            a_, b_ = Mx[(site, "fourier_half")], Mx[(site, o)]
            pr[f"fourier_half_minus_{o}"] = {q: boot_ci(np.nanmean(a_[q][keep] - b_[q][keep], (1, 2))) for q in ("err", "R", "fc", "rec", "id")}
        task2["paired"][site] = pr
        full = {"P22_own": ("P22", "own"), "P22_natural": ("P22", "natural"), "D12_own": ("D12", "own"),
                "D12_natural": ("D12", "natural")}[site]
        task2["full_ring_fourier_same_carriers"][site] = summ(M1[(*full, "fourier")], keep)
    task2["twin_ceiling_same_carriers"] = {"err_to_target": boot_ci(twm["err"][keep].mean(1)), "R": boot_ci(twm["R"][keep].mean(1))}
    task2["null_same_carriers"] = {s: summ(M1[(s[:3], s[4:], "null")], keep) for s in EXT_SITES}
    res = {"question": "is the 4-D polar-Fourier frame a control (a closed-form edit built from it moves the predictor's "
                       "forecast to the steered twin's) or only a description; and does it extrapolate beyond a half-ring fit?",
           "n_carriers": int(C), "n_targets": int(T), "folds": meta["folds"], "plan_meta": meta,
           "forward": finfo, "forward_spline_natural": finfo_spline,
           "gpu_seconds_total": finfo["seconds"] + (finfo_spline["seconds"] if finfo_spline else 0.0), "parity": par, "task1": task1, "task2": task2, "encoder_level": enc,
           "fitpack_arm": "not computed: session2_direction_natural_norm has no FITPACK arm; none added",
           "definitions": {
               "err_to_target": "circular error (deg) of the predictor-native direction probe (session-2, 480 probe clips, "
                                "stored alpha) on the step-mean pooled forecast, mean over the carrier's targets",
               "R": "<P_edit - P_src, u(t*) - u(src)> / ||u(t*) - u(src)||^2 in probe output space",
               "forced_choice_frac": "fraction of (carrier, target) pairs whose edited forecast is closer in L2 over the pooled "
                                     "forecast (token mean per predicted step; 4 x 1024 concatenated) to the rendered steered "
                                     "twin's native forecast than to the carrier's unedited native forecast (both recomputed "
                                     "in this forward pass; session-2 cache as sensitivity); per-carrier fraction, bootstrap "
                                     "over carriers",
               "recovery_full": "probe-free <F_edit - F_src, F_twin - F_src> / ||F_twin - F_src||^2 over the same pooled "
                                "forecast (1 = all the way to the twin's forecast in projection); forced choice is exactly "
                                "recovery_full > 0.5, so the fraction is a thresholded version of this graded number",
               "twin_id_acc": "target-specific probe-free readout: fraction of (carrier, target) pairs whose edited forecast "
                              "change (F_edit - F_src, same pooled forecast) has its highest cosine with that target's twin "
                              "change (F_twin_j - F_src) among the carrier's 4 twins; chance 0.25; a target-independent edit scores 0.25",
               "CI": "boot_ci: 1000 bootstrap resamples of carriers (clips), seed 0, percentile 95%",
               "sites": {"P22": "pooled: delta on every context token at point 22",
                         "D12": "object tokens only at point 12 (obj = disk(src)|disk(twin) + 1-patch ring, tubelets 0-3)"},
               "conds": {"own": "P22: the delta as built; D12: the delta added to each obj token",
                         "natural": "P22: rescaled to ||twin_meanpool_22 - src_meanpool_22||; D12: unit(delta) x the twin's "
                                    "per-token change at point 12 on each obj token"},
               "task2_units": "carriers whose source lies in a fold's span; per carrier, mean over its (fold, target) entries"}}
    write(Path(os.environ.get("F4_RESULT", RES / "session2_fourier4_predictor.json")), res, stage="fourier4_predictor_score", seeds={"bootstrap": 0, "plan": 0},
          cache=str(fo / "pred.npz"), cache_sha256=sha256_file(fo / "pred.npz"), script_sha256=sha256_file(Path(__file__)),
          gpu_seconds=finfo["seconds"] + (finfo_spline["seconds"] if finfo_spline else 0.0), box=finfo.get("box"),
          smoke_run={"n_carriers": 4, "gpu_seconds": 26.0, "file": "results/session2_fourier4_predictor_smoke.json"})
    for site in ("P22", "D12"):
        for cond in ("own", "natural"):
            print(site, cond, {a: (round(v["err_to_target"]["mean"], 1), round(v["R"]["mean"], 3),
                                   round(v["forced_choice_frac"]["mean"], 3), round(v["recovery_full"]["mean"], 3), round(v["twin_id_acc"]["mean"], 3)) for a, v in task1["table"][site][cond].items()})
    for site in EXT_SITES:
        print("ext", site, {a: (round(v["err_to_target"]["mean"], 1), round(v["R"]["mean"], 3),
                                round(v["forced_choice_frac"]["mean"], 3), round(v["recovery_full"]["mean"], 3), round(v["twin_id_acc"]["mean"], 3)) for a, v in task2["table"][site].items()})
    print(json.dumps(par))
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"fourier": "#2a9d8f", "spline": "#3b6fb6", "chord": "#d9822b", "null": "#999999",
            "fourier_half": "#2a9d8f", "spline_tangent": "#3b6fb6", "chord_nearest": "#d9822b"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    groups = [("P22", "own"), ("P22", "natural"), ("D12", "own"), ("D12", "natural")]
    for r, q, lab in ((0, "forced_choice_frac", "forced choice: closer to twin than to source"),
                      (1, "err_to_target", "forecast direction error to target (deg)")):
        a = ax[r, 0]
        for gi, (site, cond) in enumerate(groups):
            for k, arm in enumerate(T1_ARMS):
                v = res["task1"]["table"][site][cond][arm][q]
                x = gi * 5 + k
                a.bar(x, v["mean"], color=cols[arm], width=0.9, label=arm if gi == 0 else None)
                a.errorbar(x, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], color="k", lw=1)
        a.set_xticks([gi * 5 + 1.5 for gi in range(4)])
        a.set_xticklabels([f"{s} {c}" for s, c in groups])
        a.set_ylabel(lab)
        if q == "err_to_target":
            a.axhline(res["task1"]["unedited_err_to_target"]["mean"], ls="--", color="k", lw=1, label="unedited")
            a.axhline(res["task1"]["twin_ceiling"]["err_to_target"]["mean"], ls=":", color="g", lw=1.5, label="twin")
        else:
            a.axhline(0.5, ls=":", color="k", lw=0.8)
            pc_ = res["task1"]["positive_control_twin_forecast_session2_cache_vs_rerun_reference"]["forced_choice_frac"]["mean"]
            a.axhline(pc_, ls="-", color="g", lw=1.2, label=f"twin forecast (positive control) {pc_:.2f}")
        a.set_title(f"Task 1 (full-ring fit): {q}")
        a.legend(fontsize=8)
    a = ax[0, 1]
    for gi, site in enumerate(EXT_SITES):
        for k, arm in enumerate(EXT_ARMS):
            v = res["task2"]["table"][site][arm]["forced_choice_frac"]
            x = gi * 4 + k
            a.bar(x, v["mean"], color=cols[arm], width=0.9, label=arm if gi == 0 else None)
            a.errorbar(x, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], color="k", lw=1)
    a.set_xticks([gi * 4 + 1 for gi in range(4)])
    a.set_xticklabels(EXT_SITES)
    a.axhline(0.5, ls=":", color="k", lw=0.8)
    a.set_ylabel("forced choice: closer to twin than to source")
    a.set_title("Task 2 (half-ring fit, targets outside the fitted span)")
    a.legend(fontsize=8)
    a = ax[1, 1]
    for site, ls in (("P22_natural", "-"), ("D12_natural", "--")):
        for arm in EXT_ARMS:
            g = res["task2"]["err_by_gap"][site][arm]
            ks = list(g)
            a.plot([np.mean([float(t) for t in k[:-3].split("-")]) for k in ks], [g[k]["err"] for k in ks], ls,
                   marker="o", color=cols[arm], label=f"{arm} {site}")
    a.set_xlabel("target distance past the fitted end knot (deg)")
    a.set_ylabel("forecast direction error to target (deg)")
    a.set_title("Task 2: error vs extrapolation distance (natural norm)")
    a.legend(fontsize=7)
    fig.suptitle(f"Four-number (4-D polar-Fourier) direction edit read by the predictor "
                 f"({res['n_carriers']} carriers x {res['n_targets']} targets)")
    fig.tight_layout()
    out = Path(os.environ.get("F4_FIG", PROJECT_ROOT / "figures" / "fig_fourier4_predictor.png"))
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        plan()
    elif cmd == "forward":
        forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "forward_spline":
        forward_spline(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    elif cmd == "score":
        score()

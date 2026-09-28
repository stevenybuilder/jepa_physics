"""Is the third (cos 2θ, saddle) axis of the direction ring causally used by the predictor? (REPORT §4.2 follow-up)

The direction centroids form an ellipse bent out of its plane along a cos 2θ axis (p2_ellipse_direction.json: 0.20 /
0.24 of centroid variance at points 12 / 22). Test at point 22 (positive control: point-22 edits reach the predictor)
and point 12, with the session-2 predictor readout (context frames 1-8 encoded alone, edit added to every context
token at point L, blocks L+1..24, final LN, predictor with mask token 0, per-step pooled forecast), bf16 autocast
forward, fp32 scoring. Carriers = the 16 headline-arc carriers (direction, contiguous held-out block seed 0, target
320.625 deg; the same 16 as run_repair_attribution.py).

Axis (fit on TRAIN centroids only, all 5 train folds, PCA-64 of the train rows as run_ellipse.py):
  C(θ) ≈ mu + A1 [cos θ, sin θ] + A2 [cos 2θ, sin 2θ];  ring plane = span(A1);  saddle axis u = top left singular
  vector of A2 with the ring plane projected out. Primary: the full-clip (16-frame, stored meanpool) axis, the object
  of REPORT §4.2. Secondary: the same fit on the context-only (frames 1-8) train meanpool, which is what the edit acts
  on. The carrier's coordinate is read on its own context-only meanpool at L, centred at the context-only ring centre.
Arms (edit = uniform shift of every context token, as all steering here):
  saddle_x{0,2,m1}     coordinate along u scaled by 0 (removed = projected onto the ring plane), 2, -1
  saddlectx_x{...}     the same with the context-only-fitted axis
  rand{j}_x{...}       20 random axes in the PCA-64 complement of (ring plane, u), each with context-only train
                       variance equal to u's (bisection on a spectral tilt), same centring and scales
  bent_w{k} / flat_w{k}  the headline smoothing-spline path (run_part2 manifold arm, K = 50) at 8 waypoints
                       t = 1/8..1, and the same waypoints with the component along u removed (path flattened into
                       the ring plane)
Readouts (score): probes fit on the predictor's own unedited fp32 forecasts of the 480 probe clips (native cache):
direction (session2_native_readout step-mean probe), speed (step-mean, ridge), position (per-step pixel centroid,
as session2_native_readout), forecast step (time; per-step forecasts -> step index). Plus diagnostics: does the saddle
coordinate track speed / start position / visibility within a direction value?

  python scripts/run_saddle_axis.py plan                          (local CPU)
  python scripts/run_saddle_axis.py forward --root <out> [--smoke] (box GPU; takes flock /tmp/wm_gpu.lock itself)
  python scripts/run_saddle_axis.py score --root <out>             (local CPU; results + figure)
"""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_saddle"
RES = Path(os.environ.get("SADDLE_RES", PROJECT_ROOT / "results"))        # override for scoring a --smoke forward
FIG = Path(os.environ.get("SADDLE_FIG", PROJECT_ROOT / "figures"))
D = 1024
TARGET = 320.625
N_CAR = 16
POINTS = (12, 22)
K_ARC = 50
WP = np.round(np.linspace(0, 1, 9)[1:] * (K_ARC - 1)).astype(int)       # waypoint indices, t = 1/8 .. 1
SCALES = {"x0": 0.0, "x2": 2.0, "xm1": -1.0}
N_RAND = 20
LOCK = "/tmp/wm_gpu.lock"
RADIAL = {"r025": 0.25, "r05": 0.5, "r15": 1.5, "r2": 2.0}   # ring-plane radius scales (point 22)
RADIAL_POINTS = (22,)


# ================================================================ pure helpers (tests/test_saddle_axis.py)

def fit_saddle(C, theta):
    """C [m, k] centroids at angles theta [m] (radians). Fits C ≈ mu + A1 [cos, sin] + A2 [cos 2θ, sin 2θ].
    Returns mu [k], R [k, 2] (orthonormal ring plane = span A1), u [k] (unit saddle axis: top left singular vector of
    A2 with R projected out, sign so its cos 2θ loading is >= 0), phase_deg (s(θ) ∝ cos 2(θ - phase)), and shares
    of the centroid variance (about mu) along R and u, plus the second/first singular value of the projected A2
    (0 = a pure one-axis saddle)."""
    C, th = np.asarray(C, float), np.asarray(theta, float)
    M = np.stack([np.ones_like(th), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th)], 1)
    coef = np.linalg.lstsq(M, C, rcond=None)[0]
    mu, A1, A2 = coef[0], coef[1:3].T, coef[3:5].T
    R = np.linalg.svd(A1, full_matrices=False)[0][:, :2]
    A2p = A2 - R @ (R.T @ A2)
    U, S, _ = np.linalg.svd(A2p, full_matrices=False)
    u = U[:, 0]
    w = u @ A2p
    if w[0] < 0:
        u, w = -u, -w
    Cc = C - mu
    tot = (Cc ** 2).sum(1).mean()
    s = Cc @ u
    fs = np.abs(np.exp(-1j * np.outer(np.arange(1, 5), th)) @ s) ** 2
    return {"mu": mu, "R": R, "u": u, "phase_deg": float(np.degrees(0.5 * np.arctan2(w[1], w[0]))),
            "amp": float(np.linalg.norm(w)), "share_u": float((s ** 2).mean() / tot),
            "share_ring": float(((Cc @ R) ** 2).sum(1).mean() / tot),
            "rank_ratio_A2": float(S[1] / S[0]) if len(S) > 1 else 0.0,
            "u_harmonic_share_k2": float(fs[1] / fs.sum()), "a1_axis_ratio": float(np.linalg.svd(A1, compute_uv=False)[1]
                                                                                      / np.linalg.svd(A1, compute_uv=False)[0])}


def rescale_axis(x, mu, u, scale):
    """Scale the coordinate of x [..., k] along unit u about mu by `scale` (0 = project onto u's complement).
    Returns the shift to add: (scale - 1) ((x - mu) · u) u."""
    c = (np.asarray(x, float) - mu) @ u
    return (scale - 1.0) * c[..., None] * u, c


def complement_basis(B, k):
    """Orthonormal basis [k, k - q] of the complement of the columns of B [k, q] (B need not be orthonormal)."""
    Q = np.linalg.qr(B)[0]
    P = np.eye(k) - Q @ Q.T
    U, S, _ = np.linalg.svd(P)
    return U[:, : k - B.shape[1]]


def variance_matched_axes(Sigma, Q, target, n, seed=0, iters=200):
    """n random unit axes in span(Q) (Q [k, q] orthonormal) whose variance under Sigma [k, k] equals `target`.
    Each draw: z ~ N(0, I_q); axis(p) ∝ Q V diag(l^(p/2)) z with M = Q' Sigma Q = V diag(l) V'. The Rayleigh
    quotient is monotone in p (a tilted weighted mean of l), so p is found by bisection. Returns [n, k], p [n]."""
    lam, V = np.linalg.eigh(Q.T @ Sigma @ Q)
    lam = np.clip(lam, lam.max() * 1e-12, None)
    if not lam.min() < target < lam.max():
        raise ValueError(f"target variance {target:.4g} outside the complement's range [{lam.min():.4g}, {lam.max():.4g}]")
    rng = np.random.default_rng(seed)
    out, ps = [], []
    ll = np.log(lam)
    for _ in range(n):
        z2 = rng.standard_normal(len(lam)) ** 2

        def tilt(p):
            e = p * ll
            return np.exp(e - e.max())

        def rq(p):
            w = tilt(p) * z2
            return (w * lam).sum() / w.sum()
        lo, hi = -200.0, 200.0
        if not rq(lo) < target < rq(hi):
            raise ValueError("bisection bracket does not contain the target")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rq(mid) < target else (lo, mid)
        p = 0.5 * (lo + hi)
        g = np.sqrt(tilt(p)) * np.sqrt(z2) * np.sign(rng.standard_normal(len(lam)))
        a = Q @ (V @ g)
        out.append(a / np.linalg.norm(a))
        ps.append(p)
    return np.array(out), np.array(ps)


def rescale_plane(x, mu, B, scale):
    """Scale the component of x [..., k] in the plane of orthonormal B [k, 2] about mu by `scale` (the angle in the
    plane is kept). Returns the shift (scale - 1) B B' (x - mu) and the in-plane radius ||B' (x - mu)||."""
    c = (np.asarray(x, float) - mu) @ B
    return (scale - 1.0) * c @ B.T, np.linalg.norm(c, axis=-1)


def variance_matched_planes(Sig, E, targets, n, seed=0):
    """n random orthonormal 2-planes [n, k, 2] in the complement of the columns of E [k, q], axis i having variance
    targets[i] under Sig (the second axis is drawn in the complement of E and the first axis)."""
    k = Sig.shape[0]
    out = []
    for j in range(n):
        a1 = variance_matched_axes(Sig, complement_basis(E, k), targets[0], 1, seed=seed * 1000 + 2 * j)[0][0]
        a2 = variance_matched_axes(Sig, complement_basis(np.c_[E, a1], k), targets[1], 1, seed=seed * 1000 + 2 * j + 1)[0][0]
        out.append(np.c_[a1, a2])
    return np.array(out)


def flatten_path(W, u):
    """Remove the component along unit u [D] from every waypoint delta W [..., D]."""
    return W - (W @ u)[..., None] * u


def summarize(v, n=1000, seed=0):
    """Mean and carrier-bootstrap 95% CI (NaNs dropped)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(len(v))}


# ================================================================ plan (CPU, local)

def plan():
    from run_part2 import build, compose, pick_clips, subspace_arms
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    from wm.provenance import git_commit, sha256_file
    out, meta = {}, {"target": TARGET, "n_carriers": N_CAR, "points": {}, "waypoint_index_of_K50": WP.tolist()}
    rep = np.load(PROJECT_ROOT / "artifacts" / "p5_repair" / "plan.npz")
    ids = None
    for L in POINTS:
        d = load_inputs("direction", L)
        X = d["X"].astype(np.float64)
        tr = d["is_train"]
        pca = mf.fit_pca(X[tr], 64)
        cent = mf.centroids(pca.project(X[tr]), d["y"][tr])
        f = fit_saddle(cent["C"], np.radians(cent["values"]))
        # 3rd centroid PC, for parity with run_ellipse's "third centroid axis"
        Vt = np.linalg.svd(cent["C"] - cent["C"].mean(0), full_matrices=False)[2]
        cos_pc = np.abs(Vt[:4] @ f["u"])
        # headline arc
        m = build(d, 64, "unsupervised", "contiguous", 0, n_controls=0, spline="smooth")
        pick = pick_clips(d, m["held"], 48, 0)[TARGET][:N_CAR]
        cid = d["df"]["id"].to_numpy()[pick]
        ids = cid if ids is None else ids
        assert (ids == cid).all()
        x = X[pick]
        Z, resid = m["pca"].project(x), m["pca"].complement(x)
        Za = subspace_arms(Z, d["y"][pick], TARGET, m, K_ARC)
        W = compose(m["pca"], Za["manifold"][:, WP], resid) - x[:, None]           # [C, 8, D]
        u_raw = f["u"] @ pca.components
        Wf = flatten_path(W, u_raw)
        assert (rep["carrier_ids"] == cid).all()
        par = float(np.abs(W[:, -1] - rep[f"delta_spline{L}"]).max() / np.abs(rep[f"delta_spline{L}"]).max())
        bend = np.abs(W @ u_raw) / np.linalg.norm(W, axis=-1)
        meta["points"][str(L)] = {
            "n_train": int(tr.sum()), "angle_source": m["curve"].coord_source,
            "full_clip_axis": {k: v for k, v in f.items() if k not in ("mu", "R", "u")},
            "abs_cos_u_with_centroid_PC1_4": cos_pc.tolist(),
            "endpoint_delta_vs_repair_plan_rel_maxabs": par,
            "path_bend_fraction_abs_delta_dot_u_over_norm_by_t": bend.mean(0).tolist(),
            "path_delta_norm_by_t": np.linalg.norm(W, axis=-1).mean(0).tolist()}
        out.update({f"pca_mean_L{L}": pca.mean, f"pca_comp_L{L}": pca.components, f"u64_L{L}": f["u"],
                    f"R64_L{L}": f["R"], f"mu64_L{L}": f["mu"], f"path_bent_L{L}": W.astype(np.float32),
                    f"path_flat_L{L}": Wf.astype(np.float32), f"clean_mp_L{L}": x.astype(np.float32)})
        if L == POINTS[0]:
            out["src"] = d["y"][pick]
            out["train_ids"] = d["df"]["id"].to_numpy()[tr]
            out["train_y"] = d["y"][tr]
    out["carrier_ids"] = ids
    meta["git"] = git_commit()
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", **out)
    meta["plan_sha256"] = sha256_file(ART / "plan.npz")
    (ART / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


# ================================================================ forward (GPU, box)

def conditions(L=None):
    c = ["unedited"] + [f"{a}_{s}" for a in ("saddle", "saddlectx") for s in SCALES]
    c += [f"rand{j}_{s}" for j in range(N_RAND) for s in SCALES]
    c += [f"{p}_w{k}" for p in ("bent", "flat") for k in range(len(WP))]
    if L in RADIAL_POINTS:
        c += [f"radial_{r}" for r in RADIAL] + [f"radrand{j}_{r}" for j in range(N_RAND) for r in RADIAL]
    return c


def ctx_geometry(P, ctx_train, y, L):
    """Context-only geometry at L in the full-clip PCA-64 frame: ring centre, ctx-fitted axis, variance-matched
    random axes. Returns dict of 64-d quantities."""
    from wm import manifold as mf
    mean, comp = P[f"pca_mean_L{L}"], P[f"pca_comp_L{L}"]
    Zc = (ctx_train - mean) @ comp.T
    cent = mf.centroids(Zc, y)
    fc = fit_saddle(cent["C"], np.radians(cent["values"]))
    u, R = P[f"u64_L{L}"], P[f"R64_L{L}"]
    Sig = np.cov(Zc.T)
    target = float(u @ Sig @ u)
    Q = complement_basis(np.c_[R, u], 64)
    rax, ps = variance_matched_axes(Sig, Q, target, N_RAND, seed=L)
    tR = [float(R[:, i] @ Sig @ R[:, i]) for i in range(2)]
    rpl = variance_matched_planes(Sig, np.c_[R, u], tR, N_RAND, seed=L) if L in RADIAL_POINTS else None
    Cc = cent["C"] - fc["mu"]
    tot = (Cc ** 2).sum(1).mean()
    return {"mu_ctx": fc["mu"], "u_ctx": fc["u"] * np.sign(fc["u"] @ u), "rand": rax, "tilt_p": ps, "rand_planes": rpl,
            "var_ring_axes_ctx_clips": tR,
            "var_u_ctx_clips": target, "var_rand_ctx_clips": np.einsum("nk,kl,nl->n", rax, Sig, rax),
            "ctx_centroid_share_along_full_u": float(((Cc @ u) ** 2).mean() / tot),
            "ctx_fit": {k: v for k, v in fc.items() if k not in ("mu", "R", "u")},
            "cos_u_ctx_vs_full": float(abs(fc["u"] @ u)),
            "ring_plane_principal_cos": np.linalg.svd(fc["R"].T @ R, compute_uv=False).tolist()}


def forward(root, batch=8, smoke=False):
    import torch
    from wm.data import decode, load_table
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import _mean, prefix, suffix

    set_precision()
    dev = pick_device()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    P = dict(np.load(ART / "plan.npz"))
    df = load_table("direction")
    vid = dict(zip(df["id"], df["video"]))
    tids, cids = P["train_ids"], P["carrier_ids"]
    if smoke:
        tids, cids = tids[::20], cids[:batch]
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)           # noqa: E731
    with open(LOCK, "a") as lk:
        t_wait = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - t_wait
        print(f"lock acquired after {waited:.0f}s", flush=True)
        t0 = time.time()
        model = load_model("vjepa2", dev)
        # ---- A: context-only meanpool of train clips at points 12, 22
        ctx = np.zeros((len(tids), 2, D), np.float32)
        for s in range(0, len(tids), 16):
            pv = preprocess([decode(vid[i]) for i in tids[s:s + 16]]).to(dev)[:, :8]
            with torch.no_grad(), ac():
                _, _, pooled = _prefix_f32(prefix, model, pv)
            ctx[s:s + 16] = pooled[:, list(POINTS)]
            if s % 320 == 0:
                print(f"train ctx {s + len(pv)}/{len(tids)} {time.time() - t0:.0f}s", flush=True)
        t_ctx = time.time() - t0
        ytr = P["train_y"] if not smoke else P["train_y"][::20]
        geo = {L: ctx_geometry(P, ctx[:, i].astype(np.float64), ytr, L) for i, L in enumerate(POINTS)}
        # ---- B: edits on the 16 carriers
        conds = {L: conditions(L) for L in POINTS}
        out = {f"ctx_train_mp_L{L}": ctx[:, i] for i, L in enumerate(POINTS)}
        store = {}
        for s in range(0, len(cids), batch):
            pv = preprocess([decode(vid[i]) for i in cids[s:s + batch]]).to(dev)[:, :8]
            rows = slice(s, s + batch)
            with torch.no_grad(), ac():
                saved, ctx_final, pooled = _prefix_f32(prefix, model, pv, list(POINTS))
            for L in POINTS:
                g = geo[L]
                mean, comp = P[f"pca_mean_L{L}"], P[f"pca_comp_L{L}"]
                xmp = pooled[:, L].astype(np.float64)
                z = (xmp - mean) @ comp.T
                store.setdefault(f"ctx_mp_L{L}", []).append(xmp.astype(np.float32))
                for name in conds[L]:
                    if name == "unedited":
                        delta64 = None
                    elif name.startswith(("bent", "flat")):
                        k = int(name.split("_w")[1])
                        delta = P[f"path_{name.split('_')[0]}_L{L}"][rows, k]
                    elif name.startswith(("radial", "radrand")):
                        ax, sc = name.rsplit("_", 1)
                        B = P[f"R64_L{L}"] if ax == "radial" else g["rand_planes"][int(ax[7:])]
                        delta64, rad = rescale_plane(z, g["mu_ctx"], B, RADIAL[sc])
                        delta = delta64 @ comp
                        store.setdefault(f"radius_{ax}_L{L}", {})[s] = rad
                    else:
                        ax, sc = name.rsplit("_", 1)
                        u = (P[f"u64_L{L}"] if ax == "saddle" else g["u_ctx"] if ax == "saddlectx"
                             else g["rand"][int(ax[4:])])
                        delta64, coef = rescale_axis(z, g["mu_ctx"], u, SCALES[sc])
                        delta = delta64 @ comp
                        store.setdefault(f"coef_{ax}_L{L}", {})[s] = coef
                    if name == "unedited":
                        delta = np.zeros((xmp.shape[0], D))
                    dt = torch.as_tensor(np.asarray(delta, np.float32), device=dev)
                    with torch.no_grad(), ac():
                        r = suffix(model, saved[L], L, dt, return_final=True)
                        pr = pool_steps(predict_future(model, r["final"])).float().cpu().numpy()
                    store.setdefault(f"pred/{L}/{name}", []).append(pr)
                    store.setdefault(f"delta_norm/{L}/{name}", []).append(np.linalg.norm(np.asarray(delta, float), axis=-1))
            del saved
            torch.cuda.empty_cache()
            print(f"carriers {s + batch}/{len(cids)} {time.time() - t0:.0f}s", flush=True)
        seconds = time.time() - t0
    for k, v in store.items():
        out[k] = np.concatenate([v[q] for q in sorted(v)]) if isinstance(v, dict) else np.concatenate(v)
    for L in POINTS:
        g = geo[L]
        out.update({f"mu_ctx_L{L}": g["mu_ctx"], f"u_ctx_L{L}": g["u_ctx"], f"rand_L{L}": g["rand"],
                    f"var_rand_ctx_L{L}": g["var_rand_ctx_clips"]})
        if g["rand_planes"] is not None:
            out[f"rand_planes_L{L}"] = g["rand_planes"]
    info = {"seconds_gpu_total": seconds, "seconds_train_ctx": t_ctx, "lock_wait_s": waited,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "bf16 autocast forward",
            "smoke": smoke, "batch": batch, "conditions": conds, "n_train_ctx": int(len(tids)),
            "geometry": {str(L): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in geo[L].items()
                                  if k not in ("mu_ctx", "u_ctx", "rand", "rand_planes")} for L in POINTS}}
    np.savez(root / "forward.npz", **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1, default=float))
    print("done", seconds, flush=True)


def _prefix_f32(prefix, model, pv, points=()):
    """wm.propagate.prefix under autocast with the residual stream kept fp32 (the conv patch embed returns bf16
    under autocast); pooled in fp32."""
    import torch
    enc = model.encoder
    h = enc.embeddings(pv).float()                     # fp32 residual stream, bf16 matmuls (autocast)
    pts, saved, pooled = set(points), {}, [h.float().reshape(h.shape[0], -1, 256, D).mean((1, 2))]
    for i, layer in enumerate(enc.layer):
        h = layer(h, None, None, False)[0]
        if i + 1 in pts:
            saved[i + 1] = h
        pooled.append(h.float().reshape(h.shape[0], -1, 256, D).mean((1, 2)))
    final = enc.layernorm(h)
    pooled.append(final.float().reshape(h.shape[0], -1, 256, D).mean((1, 2)))
    return saved, final, torch.stack(pooled, 1).cpu().numpy()


# ================================================================ score (CPU, local)

def score(root):
    from run_session2 import angle_of, wrap, write
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, load_sweep, predict, score as pscore, targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    finfo = json.loads((root / "forward_info.json").read_text())
    P = dict(np.load(ART / "plan.npz"))
    meta = json.loads((ART / "plan.json").read_text())
    df = load_table("direction")
    y = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    d0 = load_inputs("direction", 0)
    probe = d0["role"] == "probe"
    folds5 = np.random.default_rng(0).permutation(np.arange(probe.sum()) % 5)
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    cen = np.load(PROJECT_ROOT / "artifacts/session2/centroids_direction.npy")               # [1500, 8, 2]
    speed = df["speed_mps"].to_numpy(float)

    def fit(X, Yt, fl, kind):
        st = Standardizer().fit(X)
        cv = cv_select_alpha(st.transform(X), Yt, fl, score_fn=lambda a, b: pscore(a, b, kind))
        W, b = fit_ridge(st.transform(X), Yt, cv["alpha"])
        return (st, W, b), {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"]}

    def apply(pr, X):
        st, W, b = pr
        sh = X.shape
        return predict(st.transform(X.reshape(-1, sh[-1])), W, b).reshape(sh[:-1] + (-1,))

    pdir, idir = fit(Zall[probe].mean(1), Y[probe], folds5, "circular")
    pspd, ispd = fit(Zall[probe].mean(1), speed[probe][:, None], folds5, "linear")
    ppos, ipos = {}, {}
    for k, t in enumerate(range(4, 8)):
        ok = np.isfinite(cen[:, t]).all(1)
        ppos[k], ipos[k] = fit(Zall[probe & ok, k], cen[probe & ok, t], folds5[ok[probe]], "linear")
    Xs = Zall[probe].reshape(-1, D)
    ys = np.tile(np.arange(4.0), probe.sum())[:, None]
    fs = np.repeat(folds5, 4)
    pstep, istep = fit(Xs, ys, fs, "linear")
    train = d0["is_train"]
    Y2 = np.c_[np.cos(2 * np.radians(y)), np.sin(2 * np.radians(y))]
    p2t, i2t = fit(Zall[train].mean(1), Y2[train], d0["fold"][train], "linear")   # cos 2θ / sin 2θ, train clips
    rows = np.searchsorted(df["id"].to_numpy(), P["carrier_ids"])
    NC = len(F[f"pred/{POINTS[0]}/unedited"])                            # 16 (8 in a --smoke forward)
    rows = rows[:NC]
    src = y[rows]
    pw = Zall[probe].reshape(probe.sum(), -1)
    ii = np.random.default_rng(1).integers(0, len(pw), (4000, 2))
    ii = ii[ii[:, 0] != ii[:, 1]]
    fc_scale = float(np.median(np.linalg.norm(pw[ii[:, 0]] - pw[ii[:, 1]], axis=1)))

    def read(Zp):
        Pd = apply(pdir, Zp.mean(-2))
        return {"angle": angle_of(Pd), "radius": np.linalg.norm(Pd, axis=-1),
                "speed": apply(pspd, Zp.mean(-2))[..., 0],
                "pos": np.stack([apply(ppos[k], Zp[..., k, :]) for k in range(4)], -2),
                "step": np.stack([apply(pstep, Zp[..., k, :])[..., 0] for k in range(4)], -1),
                "c2": apply(p2t, Zp.mean(-2))}

    res = {"question": "is the cos 2θ (saddle) axis of the direction ring causally used by the predictor?",
           "config": {"target_deg": TARGET, "carrier_ids": P["carrier_ids"].tolist(), "carrier_true_deg": src.tolist(),
                      "points": POINTS, "scales": SCALES, "n_random_axes": N_RAND, "waypoint_t": (np.arange(1, 9) / 8).tolist(),
                      "plan": meta, "forward": finfo},
           "readers": {"direction_stepmean": idir, "speed_stepmean": ispd,
                       "position_per_step": {str(4 + k): v for k, v in ipos.items()}, "forecast_step": istep,
                       "cos2theta_sin2theta_stepmean_train_clips": i2t,
                       "forecast_change_unit": {"median_pairwise_distance_probe_forecasts": fc_scale}},
           "points": {}}
    per_L = {}
    for L in POINTS:
        U = read(F[f"pred/{L}/unedited"].astype(np.float64))
        base_err = wrap(U["angle"] - src)
        phi = np.radians(meta["points"][str(L)]["full_clip_axis"]["phase_deg"])
        ph2 = np.array([np.cos(2 * phi), np.sin(2 * phi)])
        tr2 = np.c_[np.cos(2 * np.radians(src)), np.sin(2 * np.radians(src))]
        csad = F[f"coef_saddle_L{L}"]
        parity = {"bf16_unedited_vs_fp32_native_cache_rel_maxabs":
                  float(np.abs(F[f"pred/{L}/unedited"] - Zall[rows]).max() / np.abs(Zall[rows]).max()),
                  "bf16_vs_fp32_direction_readout_abs_deg": summarize(wrap(U["angle"] - read(Zall[rows])["angle"]))}
        R = {"parity": parity, "unedited": {"forecast_dir_err_to_true": summarize(base_err),
                                            "forecast_dir_err_to_target": summarize(wrap(U["angle"] - TARGET)),
                                            "forecast_radius": summarize(U["radius"]),
                                            "speed_pred": summarize(U["speed"]),
                                            "speed_abs_err": summarize(np.abs(U["speed"] - speed[rows])),
                                            "pos_px_err": summarize(np.nanmean(np.linalg.norm(U["pos"] - cen[rows, 4:8], axis=-1), -1)),
                                            "c2_readout_err_deg_2theta_over_2": summarize(wrap(np.degrees(np.arctan2(U["c2"][:, 1], U["c2"][:, 0])) - 2 * src) / 2),
                                            "c2_readout_projection_on_true": summarize((U["c2"] * tr2).sum(1)),
                                            "carrier_saddle_coord_ctx": csad.tolist()}}

        def effects(name):
            E = read(F[f"pred/{L}/{name}"].astype(np.float64))
            dz = F[f"pred/{L}/{name}"].astype(np.float64) - F[f"pred/{L}/unedited"]
            return {"dir_err": wrap(E["angle"] - src), "d_dir_err": wrap(E["angle"] - src) - base_err,
                    "dir_shift": wrap(E["angle"] - U["angle"]), "d_speed": E["speed"] - U["speed"],
                    "pos_shift_px": np.linalg.norm(E["pos"] - U["pos"], axis=-1).mean(-1),
                    "d_step": (E["step"] - U["step"]).mean(-1),
                    "forecast_change_rel": np.linalg.norm(dz.reshape(len(dz), -1), axis=1) / fc_scale,
                    "radius": E["radius"], "err_target": wrap(E["angle"] - TARGET),
                    "delta_norm": F[f"delta_norm/{L}/{name}"],
                    "d_c2_phase": (E["c2"] - U["c2"]) @ ph2, "d_c2_true": ((E["c2"] - U["c2"]) * tr2).sum(1),
                    "d_c2_abs": np.linalg.norm(E["c2"] - U["c2"], axis=1),
                    "d_c2_aligned": ((E["c2"] - U["c2"]) @ ph2) * np.sign((SCALES.get(name.rsplit("_", 1)[-1], np.nan) - 1) * csad),
                    "E": E}

        keys = ("dir_err", "d_dir_err", "dir_shift", "d_speed", "pos_shift_px", "d_step", "forecast_change_rel", "delta_norm",
                "d_c2_phase", "d_c2_aligned", "d_c2_true", "d_c2_abs")
        ed = {}
        for ax in ("saddle", "saddlectx"):
            for sc in SCALES:
                e = effects(f"{ax}_{sc}")
                ed[(ax, sc)] = e
                R.setdefault(ax, {})[sc] = {k: summarize(e[k]) for k in keys}
        rnd = {sc: [effects(f"rand{j}_{sc}") for j in range(N_RAND)] for sc in SCALES}
        for sc in SCALES:
            blk = {}
            for k in keys:
                M = np.array([r[k] for r in rnd[sc]])                        # [20, C]
                dm = M.mean(1)
                sv = ed[("saddle", sc)][k].mean()
                blk[k] = {"per_carrier_mean_over_draws": summarize(M.mean(0)),
                          "draw_means_pct2.5_50_97.5": np.percentile(dm, [2.5, 50, 97.5]).tolist(),
                          "draw_means_min_max": [float(dm.min()), float(dm.max())],
                          "frac_draws_abs_ge_saddle": float((np.abs(dm) >= abs(sv)).mean()),
                          "paired_saddle_minus_random": summarize(ed[("saddle", sc)][k] - M.mean(0))}
            R.setdefault("random_axes", {})[sc] = blk
        # removed coordinate sizes (saddle vs random): |coef| per carrier on the context-only meanpool
        cs = np.abs(F[f"coef_saddle_L{L}"])
        cr = np.abs(np.array([F[f"coef_rand{j}_L{L}"] for j in range(N_RAND)]))
        R["removed_coordinate_abs"] = {"saddle": summarize(cs), "random_mean_over_draws": summarize(cr.mean(0)),
                                       "saddlectx": summarize(np.abs(F[f"coef_saddlectx_L{L}"])),
                                       "random_ctx_train_variance_over_saddle": [float(v) for v in np.percentile(
                                           F[f"var_rand_ctx_L{L}"] / finfo["geometry"][str(L)]["var_u_ctx_clips"], [0, 50, 100])]}
        # ---- path: bent vs flattened
        path = {"bent": [effects(f"bent_w{k}") for k in range(len(WP))], "flat": [effects(f"flat_w{k}") for k in range(len(WP))]}
        pr = {}
        for p in ("bent", "flat"):
            pr[p] = {"endpoint_err_to_target": summarize(path[p][-1]["err_target"]),
                     "forecast_radius_by_t": [summarize(e["radius"]) for e in path[p]],
                     "min_forecast_radius_over_path": summarize(np.min([e["radius"] for e in path[p]], 0)),
                     "mean_forecast_radius_over_path": summarize(np.mean([e["radius"] for e in path[p]], 0)),
                     "err_to_target_by_t": [summarize(e["err_target"]) for e in path[p]],
                     "endpoint_d_speed": summarize(path[p][-1]["d_speed"]),
                     "endpoint_pos_shift_px": summarize(path[p][-1]["pos_shift_px"])}
        pr["flat_minus_bent"] = {
            "endpoint_err_to_target": summarize(path["flat"][-1]["err_target"] - path["bent"][-1]["err_target"]),
            "min_forecast_radius": summarize(np.min([e["radius"] for e in path["flat"]], 0)
                                             - np.min([e["radius"] for e in path["bent"]], 0)),
            "mean_forecast_radius": summarize(np.mean([e["radius"] for e in path["flat"]], 0)
                                              - np.mean([e["radius"] for e in path["bent"]], 0)),
            "radius_by_t": [summarize(path["flat"][k]["radius"] - path["bent"][k]["radius"]) for k in range(len(WP))],
            "err_to_target_by_t": [summarize(path["flat"][k]["err_target"] - path["bent"][k]["err_target"])
                                   for k in range(len(WP))],
            "endpoint_forecast_dir_difference_deg": summarize(wrap(read(F[f"pred/{L}/flat_w7"].astype(np.float64))["angle"]
                                                                   - read(F[f"pred/{L}/bent_w7"].astype(np.float64))["angle"]))}
        # edit-point readout of the path with the point-L probe on the stored full-clip meanpool (run_part2 radius)
        dL = load_inputs("direction", L)
        sw = load_sweep("direction", "direction")
        stL = Standardizer().fit(dL["X"][probe].astype(np.float64))
        WL, bL = fit_ridge(stL.transform(dL["X"][probe].astype(np.float64)), Y[probe], sw["layers"][L]["alpha"])
        x0 = P[f"clean_mp_L{L}"][:NC].astype(np.float64)
        ep = {}
        for p in ("bent", "flat"):
            Wp = P[f"path_{p}_L{L}"][:NC].astype(np.float64)
            Pr = predict(stL.transform((x0[:, None] + Wp).reshape(-1, D)), WL, bL).reshape(len(x0), len(WP), 2)
            ep[p] = {"radius": np.linalg.norm(Pr, axis=-1), "err": wrap(angle_of(Pr) - TARGET)}
        u_raw = P[f"u64_L{L}"] @ P[f"pca_comp_L{L}"]
        pr["edit_point_probe"] = {p: {"min_radius_over_path": summarize(ep[p]["radius"].min(1)),
                                      "endpoint_err_to_target": summarize(ep[p]["err"][:, -1])} for p in ep}
        pr["edit_point_probe"]["flat_minus_bent_min_radius"] = summarize(ep["flat"]["radius"].min(1) - ep["bent"]["radius"].min(1))
        pr["edit_point_probe"]["flat_minus_bent_endpoint_err"] = summarize(ep["flat"]["err"][:, -1] - ep["bent"]["err"][:, -1])
        Wb = P[f"path_bent_L{L}"][:NC].astype(np.float64)
        pr["bend_component"] = {"abs_delta_dot_u_by_t": [summarize(np.abs(Wb[:, k] @ u_raw)) for k in range(len(WP))],
                                "fraction_of_delta_norm_by_t": [summarize(np.abs(Wb[:, k] @ u_raw) / np.linalg.norm(Wb[:, k], axis=1))
                                                                for k in range(len(WP))]}
        R["path_bent_vs_flat"] = pr
        R["byproduct_diagnostics"] = byproduct(L, P, df, cen)
        res["points"][str(L)] = R
        per_L[L] = {"U": base_err, "ed": ed, "rnd": rnd, "path": path}
        if L in RADIAL_POINTS:
            radial(L, F, P, df, U, src, speed, effects, finfo, res, root, write, sha256_file)
        write(RES / f"p5_saddle_axis_L{L}.json", {**{k: v for k, v in res.items() if k != "points"}, "point": L,
                                                   "result": R}, stage="p5_saddle_axis",
              seeds={"native_probe_folds": 0, "random_axes": L, "bootstrap": 0},
              forward_npz_sha256=sha256_file(root / "forward.npz"), plan_npz_sha256=sha256_file(ART / "plan.npz"))
    figure(per_L)
    return res


def radial(L, F, P, df, U, src, speed, effects, finfo, res, root, write, sha256_file):
    """Ring-plane radius steering at L: forecast speed, direction, direction spread across carriers; random
    variance-matched planes as the null; implied speed from the context-only train radius-by-speed curve."""
    def signed(a):
        return (np.asarray(a) + 180.0) % 360.0 - 180.0

    def circ_sd(e):
        Rr = np.abs(np.exp(1j * np.radians(e)).mean())
        return float(np.degrees(np.sqrt(-2 * np.log(max(Rr, 1e-12)))))

    def boot_stat(e, fn, n=1000, seed=0):
        idx = np.random.default_rng(seed).integers(0, len(e), (n, len(e)))
        b = np.array([fn(e[i]) for i in idx])
        return {"value": fn(e), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": int(len(e))}

    mean, comp = P[f"pca_mean_L{L}"], P[f"pca_comp_L{L}"]
    Rg = P[f"R64_L{L}"]
    mu = F[f"mu_ctx_L{L}"]
    tr_ids = P["train_ids"]
    if len(F[f"ctx_train_mp_L{L}"]) != len(tr_ids):          # --smoke forward: every 20th train clip
        tr_ids = tr_ids[::20]
    trow = np.searchsorted(df["id"].to_numpy(), tr_ids)
    zt = (F[f"ctx_train_mp_L{L}"].astype(np.float64) - mean) @ comp.T
    rad_t = np.linalg.norm((zt - mu) @ Rg, axis=1)
    sp_t = speed[trow]
    sv = np.unique(sp_t)
    rmean = np.array([rad_t[sp_t == v].mean() for v in sv])
    order = np.argsort(rmean)

    def implied_speed(r):
        return np.interp(r, rmean[order], sv[order])
    rad0 = F[f"radius_radial_L{L}"]
    e0 = signed(U["angle"] - src)
    out = {"question": "does the predictor read the ring-plane radius (which grows with speed) as speed or as "
                       "direction confidence? radius scaled at fixed angle about the context-only ring centre",
           "config": {**res["config"], "radial_scales": RADIAL, "plane": "full-clip train-centroid ring plane "
                      "(span of the k = 1 chart), centre = context-only train ring centre"},
           "readers": res["readers"],
           "ring_radius_by_speed_ctx_train": {"speed": sv.tolist(), "mean_radius": rmean.tolist(),
                                              "corr_radius_speed_clips": float(np.corrcoef(rad_t, sp_t)[0, 1])},
           "carriers": {"ring_radius_ctx": summarize(rad0), "true_speed": speed[np.searchsorted(df["id"].to_numpy(), P["carrier_ids"])][:len(rad0)].tolist(),
                        "implied_speed_from_radius": summarize(implied_speed(rad0))},
           "unedited": {"speed_pred": summarize(U["speed"]), "forecast_radius": summarize(U["radius"]),
                        "dir_err_to_true": summarize(wrap_deg(e0)), "dir_spread_circ_sd_deg": boot_stat(e0, circ_sd)},
           "radial": {}, "random_planes": {}, "radial_minus_random": {}}
    for sc, f in RADIAL.items():
        e = effects(f"radial_{sc}")
        er = signed(e["E"]["angle"] - src)
        rnd = [effects(f"radrand{j}_{sc}") for j in range(N_RAND)]
        blk = {"speed_pred": summarize(e["E"]["speed"]), "d_speed": summarize(e["d_speed"]),
               "implied_speed_change_from_radius": summarize(implied_speed(f * rad0) - implied_speed(rad0)),
               "dir_err_to_true": summarize(e["dir_err"]), "d_dir_err": summarize(e["d_dir_err"]),
               "dir_shift": summarize(e["dir_shift"]), "forecast_radius": summarize(e["radius"]),
               "d_forecast_radius": summarize(e["radius"] - U["radius"]),
               "dir_spread_circ_sd_deg": boot_stat(er, circ_sd),
               "pos_shift_px": summarize(e["pos_shift_px"]), "d_step": summarize(e["d_step"]),
               "d_c2_abs": summarize(e["d_c2_abs"]), "forecast_change_rel": summarize(e["forecast_change_rel"]),
               "delta_norm": summarize(e["delta_norm"])}
        out["radial"][sc] = blk
        rb = {}
        for k, fn in (("d_speed", lambda r: r["d_speed"]), ("d_dir_err", lambda r: r["d_dir_err"]),
                      ("dir_shift", lambda r: r["dir_shift"]), ("d_forecast_radius", lambda r: r["radius"] - U["radius"]),
                      ("forecast_change_rel", lambda r: r["forecast_change_rel"]), ("delta_norm", lambda r: r["delta_norm"]),
                      ("pos_shift_px", lambda r: r["pos_shift_px"])):
            M = np.array([fn(r) for r in rnd])
            dm = M.mean(1)
            rb[k] = {"per_carrier_mean_over_draws": summarize(M.mean(0)),
                     "draw_means_pct2.5_50_97.5": np.percentile(dm, [2.5, 50, 97.5]).tolist(),
                     "frac_draws_abs_ge_radial": float((np.abs(dm) >= abs(fn(e).mean())).mean())}
            out["radial_minus_random"].setdefault(sc, {})[k] = summarize(fn(e) - M.mean(0))
        sds = [circ_sd(signed(r["E"]["angle"] - src)) for r in rnd]
        rb["dir_spread_circ_sd_deg"] = {"draw_pct2.5_50_97.5": np.percentile(sds, [2.5, 50, 97.5]).tolist()}
        out["random_planes"][sc] = rb
    out["random_plane_removed_radius"] = summarize(np.mean([F[f"radius_radrand{j}_L{L}"] for j in range(N_RAND)], 0))
    write(RES / f"p5_radial_steering_L{L}.json", out, stage="p5_radial_steering",
          seeds={"native_probe_folds": 0, "random_planes": L, "bootstrap": 0},
          forward_npz_sha256=sha256_file(root / "forward.npz"), plan_npz_sha256=sha256_file(ART / "plan.npz"))


def wrap_deg(d):
    d = np.abs(np.asarray(d, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def byproduct(L, P, df, cen):
    """Is the saddle coordinate a byproduct of another clip variable? Full-clip train clips at L, coordinate along u
    about the ring centre. (i) R² of s on [cos 2θ, sin 2θ] vs on the 64 direction values (all direction-driven
    variance); (ii) within-value (residual after the value mean) correlation of s with speed, start x / y, visible
    tubelets, and |mean pixel x - 128|, |mean pixel y - 128|; (iii) across the 64 values, correlation of the value
    mean of s with the value means of those nuisances."""
    from wm.p2_data import load_inputs
    d = load_inputs("direction", L)
    tr = d["is_train"]
    X = d["X"][tr].astype(np.float64)
    th = np.radians(d["y"][tr])
    s = (X - P[f"pca_mean_L{L}"]) @ P[f"pca_comp_L{L}"].T @ P[f"u64_L{L}"] - P[f"mu64_L{L}"] @ P[f"u64_L{L}"]
    c = cen[tr]
    nuis = {"speed": df["speed_mps"].to_numpy(float)[tr], "start_x": df["start_x"].to_numpy(float)[tr],
            "start_y": df["start_y"].to_numpy(float)[tr], "visible_tubelets": np.isfinite(c).all(-1).sum(1).astype(float),
            "abs_mean_px_x_from_centre": np.abs(np.nanmean(c[..., 0], 1) - 128),
            "abs_mean_px_y_from_centre": np.abs(np.nanmean(c[..., 1], 1) - 128)}

    def r2(Xd, v):
        Xd = np.c_[np.ones(len(v)), Xd]
        b = np.linalg.lstsq(Xd, v, rcond=None)[0]
        return float(1 - ((v - Xd @ b) ** 2).sum() / ((v - v.mean()) ** 2).sum())
    vals, inv = np.unique(d["y"][tr], return_inverse=True)
    onehot = np.eye(len(vals))[inv][:, 1:]
    out = {"r2_cos2theta_sin2theta": r2(np.c_[np.cos(2 * th), np.sin(2 * th)], s), "r2_direction_value": r2(onehot, s)}
    sm = np.array([s[inv == i].mean() for i in range(len(vals))])
    res = s - sm[inv]
    w, a = {}, {}
    for k, v in nuis.items():
        ok = np.isfinite(v)
        vm = np.array([np.nanmean(v[inv == i]) for i in range(len(vals))])
        vr = v - vm[inv]
        w[k] = float(np.corrcoef(res[ok], vr[ok])[0, 1]) if vr[ok].std() > 0 else None
        a[k] = float(np.corrcoef(sm, vm)[0, 1]) if vm.std() > 0 else None
    out["within_value_corr_with_s"] = w
    out["across_value_corr_of_value_means"] = a
    out["r2_direction_value_plus_all_nuisances"] = r2(np.c_[onehot, np.c_[[np.nan_to_num(v) for v in nuis.values()]].T], s)
    out["n_clips"] = int(tr.sum())
    return out


def figure(per_L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = ["unedited", "saddle\nremoved", "random axis\nremoved", "saddle ×2", "random ×2", "saddle ×−1", "random ×−1"]
    colors = ["#8c8c8c", "#c65d3b", "#bdbdbd", "#c65d3b", "#bdbdbd", "#c65d3b", "#bdbdbd"]
    hatch = [None, None, None, "//", "//", "..", ".."]
    for i, L in enumerate(POINTS):
        g = per_L[L]
        vals = [g["U"], g["ed"][("saddle", "x0")]["dir_err"], np.mean([r["dir_err"] for r in g["rnd"]["x0"]], 0),
                g["ed"][("saddle", "x2")]["dir_err"], np.mean([r["dir_err"] for r in g["rnd"]["x2"]], 0),
                g["ed"][("saddle", "xm1")]["dir_err"], np.mean([r["dir_err"] for r in g["rnd"]["xm1"]], 0)]
        x = np.arange(len(vals)) + i * (len(vals) + 1.5)
        for j, v in enumerate(vals):
            s = summarize(v)
            ax[0].bar(x[j], s["mean"], color=colors[j], hatch=hatch[j], edgecolor="k", lw=0.5)
            ax[0].errorbar(x[j], s["mean"], yerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]], color="k", capsize=2)
        for sc, j in (("x0", 2), ("x2", 4), ("xm1", 6)):
            dm = np.array([r["dir_err"].mean() for r in g["rnd"][sc]])
            ax[0].plot([x[j]] * len(dm), dm, "k.", ms=2, alpha=0.5)
        ax[0].text(x[3], ax[0].get_ylim()[1] if i else 0, "")
        ax[0].set_xticks(np.r_[ax[0].get_xticks()] if False else x)
    xt, xl = [], []
    for i, L in enumerate(POINTS):
        x = np.arange(len(labels)) + i * (len(labels) + 1.5)
        xt += list(x)
        xl += [f"{lab}" if j else f"point {L}\n{lab}" for j, lab in enumerate(labels)]
    ax[0].set_xticks(xt)
    ax[0].set_xticklabels(xl, fontsize=6)
    ax[0].set_ylabel("forecast direction error to the clip's true direction (deg)")
    ax[0].set_title("Saddle-axis edits read by the predictor (16 carriers, mean ± 95% CI;\n"
                    "random = 20 variance-matched axes, dots = per-draw means)", fontsize=9)
    t = np.arange(1, 9) / 8
    for L, ls in ((12, "-"), (22, "--")):
        pth = per_L[L]["path"]
        for p, col in (("bent", "#c65d3b"), ("flat", "#4682b4")):
            r = np.array([e["radius"] for e in pth[p]])
            er = np.array([e["err_target"] for e in pth[p]])
            ax[1].plot(t, er.mean(1), ls, color=col, marker="o", ms=3, label=f"{p}, point {L}")
            ax[2].plot(t, r.mean(1), ls, color=col, marker="o", ms=3, label=f"{p}, point {L}")
    ax[1].set_xlabel("path fraction t")
    ax[1].set_ylabel("forecast direction error to the target (deg)")
    ax[1].set_title("Spline path with the saddle bend vs flattened into the ring plane", fontsize=9)
    ax[1].legend(fontsize=7)
    ax[2].set_xlabel("path fraction t")
    ax[2].set_ylabel("forecast radius (norm of the direction probe's (sin, cos))")
    ax[2].set_title("Along-path forecast radius", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_saddle_axis.png", dpi=150)
    print("wrote", FIG / "fig_saddle_axis.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "forward", "score"))
    ap.add_argument("--root", default=str(ART / "forward"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"plan": plan, "forward": lambda: forward(a.root, smoke=a.smoke), "score": lambda: score(a.root)}[a.stage]()

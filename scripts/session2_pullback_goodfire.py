"""Part 2 reverse test (pullback) rerun with Goodfire's recipe (steering paper App. A.8-A.9, causalab
methods/pullback/optimization.py + configs/analysis/pullback.yaml), at point 22, on our data.

What is taken from Goodfire, and where our data forces a choice:
  (a) one shared activation path per (source value, target value) pair, optimised on the MEAN loss over 16 carriers
      whose true direction is the source value (A.8 l.2759-2761). Carriers: every probe/test clip at the source
      value (13), filled to 16 with knot clips (seeded).
  (b) intervention = REPLACEMENT of the top-32 PCA coordinates (PCA-64 of the point-22 meanpool, fit on knot clips
      at the kept values, as run_session2.plan) with the path value; PCs 33-64 and the orthogonal residual stay at
      each carrier's own values (A.8 l.2749-2751). The carrier's own values are those of its context-only
      (frames 1-8) point-22 meanpool, i.e. the activation actually edited; the edit is the resulting meanpool delta
      added to every context token (a uniform per-token delta changes the meanpool by exactly that delta).
  (c) all K = 20 waypoints are free (causalab default trajectory_param: free_points, embedding_optim.n_steps 20,
      fix_start/fix_end false; also what App. C.3 states), not A.8's 10-control-vector spline. Init: the straight
      chord between the source centroid and the target point in the 32-d subspace, sampled at the 20 t's.
  (d) torch L-BFGS, lr 1, strong-Wolfe line search, max_iter 5 per outer step, early stop when the relative loss
      change between consecutive outer steps < 1e-3 (A.8; causalab convergence_window 2). A.8's 50 outer steps are
      capped by a per-pair function-evaluation budget (GPU time); the result records outer steps, evals and
      whether the stop was convergence or the budget. Adam is a fallback only if L-BFGS returns a non-finite loss.
  (e) target = the behaviour-manifold geodesic: our readout is the predictor-native direction probe of
      session2_predictor_native_readout.json (ridge on the step-mean pooled predictor forecast -> (sin, cos), the
      same affine map the old reverse test used). It is not a distribution, so it is turned into one: p_k(P) =
      softmax_k(kappa <P, u_k>) over the 64 direction values u_k = (sin th_k, cos th_k) (a von Mises readout;
      kappa fit by max likelihood of the true value on the test-fold unedited forecasts). Behaviour centroids b_v =
      mean p over all clips at value v (unedited forecasts); M_y = periodic interpolating cubic spline through
      sqrt(b_v) (Hellinger coordinates) over theta; target p_hat(t) = the spline from source to target, squared
      and renormalised, at the 20 t's (causalab _manifold_trace_path). Loss = sum_t mean_n squared Hellinger
      1 - sum_k sqrt(p_k p_hat_k) (A.8 l.2757-2761).
  (f) no norm regulariser (norm_reg_weight 0, path_length_weight 0; A.8 l.2766-2769, weekdays).
  (g) scoring = causalab path_recapitulation_metrics (ported to numpy below): closest-point residual of each
      optimised waypoint to the spline polyline and to the chord, and intrinsic R^2 in the spline's 99%-variance SVD
      basis. The optimised 32-d path is padded to 64-d with the training mean (0 in PCA coordinates), as causalab
      pads dropped dims; a 32-d scoring is also stored. The same closest-point scoring is applied to the old
      reverse-test paths (session2_reverse_path.json / reverse_out.npz).

  prep   CPU, local  -> artifacts/session2/pullback_goodfire/prep.npz
  run    box, GPU    -> artifacts/session2/pullback_goodfire/pair_*.npz
  score  CPU, local  -> results/session2_pullback_goodfire.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "session2"
AP = OUT / "along_path"
RV = AP / "reverse"
PB = OUT / "pullback_goodfire"
RES = PROJECT_ROOT / "results"
L, K, K_OPT, N_CAR = 22, 20, 32, 16
LBFGS = {"steps": 50, "lr": 1.0, "max_iter": 5, "line_search_fn": "strong_wolfe", "tol": 1e-3, "window": 2}


# ================================================================ scoring (causalab optimization.py l.587-683, numpy)
def project_to_polyline(Q, P):
    """Closest point on the polyline through P [m, k] for each row of Q [q, k] (segments clamped)."""
    Q, P = np.atleast_2d(Q), np.atleast_2d(P)
    if len(P) < 2:
        return np.repeat(P[:1], len(Q), 0)
    a, ab = P[:-1], P[1:] - P[:-1]
    ab2 = np.maximum((ab ** 2).sum(-1), 1e-12)
    t = np.clip(((Q[:, None] - a[None]) * ab[None]).sum(-1) / ab2, 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    best = ((Q[:, None] - proj) ** 2).sum(-1).argmin(1)
    return proj[np.arange(len(Q)), best]


def recap_metrics(v, ref, basis_ref=None, var=0.99):
    """causalab path_recapitulation_metrics(v, v_geo=ref): closest-point residual (full space) and intrinsic R^2 in
    the 99%-variance SVD basis of basis_ref (default ref itself; A.9 uses the spline's basis for both comparisons)."""
    v, ref = np.asarray(v, float), np.asarray(ref, float)
    B = ref if basis_ref is None else np.asarray(basis_ref, float)
    resid = np.linalg.norm(v - project_to_polyline(v, ref), axis=-1)
    mu = B.mean(0)
    _, S, Vh = np.linalg.svd(B - mu, full_matrices=False)
    cum = np.cumsum(S ** 2 / max((S ** 2).sum(), 1e-12))
    d = int(max(1, min((cum < var).sum() + 1, len(S))))
    vd, rd = (v - mu) @ Vh[:d].T, (ref - mu) @ Vh[:d].T
    rss = ((vd - project_to_polyline(vd, rd)) ** 2).sum()
    tss = max(((vd - vd.mean(0)) ** 2).sum(), 1e-12)
    arc = lambda x: np.linalg.norm(np.diff(x, axis=0), axis=-1).sum()  # noqa: E731
    return {"r_squared": float(np.clip(1 - rss / tss, 0, 1)), "intrinsic_dim": d,
            "mean_closest_point_residual": float(resid.mean()), "residual_by_t": resid.tolist(),
            "arc_length_ratio": float(arc(v) / arc(ref)) if arc(ref) > 1e-12 else float("nan")}


# ================================================================ behaviour readout (e)
def readout_dist(P, U, kappa):
    """(sin, cos) probe readings P [.., 2] -> distributions over the 64 values [.., 64]."""
    lg = kappa * (np.asarray(P) @ U.T)
    lg -= lg.max(-1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(-1, keepdims=True)


def fit_kappa(P, idx, U):
    from scipy.optimize import minimize_scalar
    nll = lambda lk: -np.log(readout_dist(P, U, np.exp(lk))[np.arange(len(idx)), idx] + 1e-300).mean()  # noqa: E731
    r = minimize_scalar(nll, bounds=(-3, 6), method="bounded")
    return float(np.exp(r.x)), float(r.fun)


def sqrt_spline(values_deg, B):
    """Periodic interpolating cubic spline through sqrt(b_v) over theta (radians)."""
    from scipy.interpolate import CubicSpline
    th = np.radians(values_deg)
    H = np.sqrt(B)
    return CubicSpline(np.append(th, th[0] + 2 * np.pi), np.vstack([H, H[:1]]), bc_type="periodic")


def behaviour_target(spl, src_deg, shift_signed_deg, ts):
    th = np.radians(src_deg + ts * shift_signed_deg) % (2 * np.pi)
    H = np.clip(spl(th), 0, None)
    P = H ** 2
    return P / P.sum(-1, keepdims=True)


# ================================================================ L-BFGS driver (d), shared by the box run and tests
class _Budget(Exception):
    pass


def optimise(V, closure_fn, steps=LBFGS["steps"], max_evals=None, log=print):
    """V: torch leaf tensor [K, k] (requires_grad). closure_fn() -> (loss tensor, per-t list), gradients accumulated
    into V.grad by the caller's backward. Runs torch L-BFGS exactly as causalab _run_lbfgs (lr, max_iter, strong
    Wolfe, stop on relative change < tol over `window` consecutive outer steps). max_evals: a hard cap on loss
    evaluations (GPU budget); when it is hit mid line search the step is abandoned. V is left at the best-loss point
    evaluated (L-BFGS's own last iterate unless a cap interrupted a line search). history = loss at the start of
    each outer step (what torch LBFGS.step returns; history[0] = the init loss). Returns dict(history, per_t,
    eval_losses, n_evals, outer_steps, stop, optimiser, best_loss)."""
    import torch
    opt = torch.optim.LBFGS([V], lr=LBFGS["lr"], max_iter=LBFGS["max_iter"], line_search_fn=LBFGS["line_search_fn"])
    hist, per_t, evals, stop = [], [], [], "max_steps"
    best = {"loss": np.inf, "V": V.detach().clone(), "pt": None}

    def closure():
        if max_evals is not None and len(evals) >= max_evals:
            raise _Budget
        opt.zero_grad()
        loss, pt = closure_fn()
        lv = float(loss)
        evals.append(lv)
        if lv < best["loss"]:
            best.update(loss=lv, V=V.detach().clone(), pt=list(pt))
        return loss

    for step in range(steps):
        try:
            lv = float(opt.step(closure))
        except _Budget:
            stop = "eval_budget"
            break
        if not np.isfinite(lv):
            stop = "non_finite"
            break
        hist.append(lv)
        per_t.append(best["pt"])
        log(f"    L-BFGS step {step:3d}: loss {lv:.6f} evals {len(evals)} best {best['loss']:.6f}")
        if len(hist) >= LBFGS["window"]:
            r = np.array(hist[-LBFGS["window"]:])
            if np.abs(np.diff(r) / (np.abs(r[:-1]) + 1e-8)).max() < LBFGS["tol"]:
                stop = "converged"
                break
    with torch.no_grad():
        V.copy_(best["V"])
    return {"history": hist, "per_t": per_t, "eval_losses": evals, "n_evals": len(evals), "outer_steps": len(hist),
            "stop": stop, "optimiser": "lbfgs_strong_wolfe", "best_loss": best["loss"]}


# ================================================================ prep (CPU)
def signed_wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def geometry():
    from run_session2 import load_plan
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    arrays, info = load_plan(OUT)
    d = load_inputs("direction", L)
    X, y = d["X"].astype(np.float64), d["y"]
    held = np.array(info["held_values"])
    knot = (d["role"] == "knot") & ~np.isin(y, held)
    pca = mf.fit_pca(X[knot], info["k"])
    cent = mf.centroids(pca.project(X[knot]), y[knot])
    choice = mf.choose_angle_source(cent["C"], cent["values"])
    curve = mf.fit_curve(cent, True, angle=choice["angle"], plane=choice["plane"], spline=info["spline"])
    return {"d": d, "X": X, "y": y, "pca": pca, "curve": curve, "mf": mf, "info": info, "knot": knot}


def spline_dir(curve, mf, src, tg):
    """theta direction (+1/-1) the interpolating spline takes from src to tg (the old test's arc_sign)."""
    orient = np.sign(mf.wrap_pi(curve.coord_of_value(src + 1.0) - curve.coord_of_value(src)))
    return np.sign(curve.step(curve.coord_of_value(src), curve.coord_of_value(tg))) * orient


def pair_paths(G, src, tg, n):
    """Spline (activation manifold) and chord between the source centroid and the target's spline point, PCA-64."""
    curve, mf = G["curve"], G["mf"]
    ta, tb = curve.coord_of_value(src), curve.coord_of_value(tg)
    s = np.linspace(0, 1, n)
    sp = curve(ta + s * curve.step(ta, tb))
    ch = sp[:1] + s[:, None] * (sp[-1:] - sp[:1])
    return sp, ch


def prep(n_pairs=36):
    G = geometry()
    X, y, pca, curve, mf, d = G["X"], G["y"], G["pca"], G["curve"], G["mf"], G["d"]
    zr, zp = np.load(RV / "prep.npz"), np.load(AP / "plan.npz")
    W, b = zr["W"], zr["b"]
    # old reverse-test pairs (distinct source value, target value); shift 135-180
    src_c = y[zr["carrier_rows"]]
    tg_c = zp["targets"][zr["pick"][:, None], zr["tj"]]
    pairs = sorted({(float(s), float(t)) for s, tt in zip(src_c, tg_c) for t in tt})
    sh = np.array([abs(signed_wrap(t - s)) for s, t in pairs])
    order = np.lexsort((np.array([p[0] for p in pairs]), sh))
    pri = list(dict.fromkeys([int(i) for i in np.round(np.linspace(0, len(pairs) - 1, 8))]
                             + [int(i) for i in np.round(np.linspace(0, len(pairs) - 1, 16))] + list(range(len(pairs)))))
    pairs = [pairs[order[i]] for i in pri][:n_pairs]
    # behaviour readout
    Zall = np.load(OUT / "native" / "pred_pooled_all.npy").astype(np.float64)
    Pall = Zall.mean(1) @ W + b
    vals = np.unique(y)
    U = np.stack([np.sin(np.radians(vals)), np.cos(np.radians(vals))], 1)
    vidx = np.searchsorted(vals, y)
    test = d["role"] == "test"
    kappa, nll = fit_kappa(Pall[test], vidx[test], U)
    pd_all = readout_dist(Pall, U, kappa)
    Bc = np.stack([pd_all[vidx == i].mean(0) for i in range(len(vals))])
    Bc /= Bc.sum(-1, keepdims=True)
    spl = sqrt_spline(vals, Bc)
    ts = np.linspace(0, 1, K)
    rng = np.random.default_rng(0)
    nat_all = zp["natural_twin_change_norm"]
    sh_all = np.abs(signed_wrap(zp["targets"] - y[zp["carrier_rows"]][:, None]))
    Zfull = pca.project(X)
    recs = {k: [] for k in ("src", "tg", "shift_signed", "rows", "target_p", "spline64", "chord64", "init32",
                            "nat_unit", "unedited_loss_per_t", "zfull_carriers")}
    for s, t in pairs:
        dirn = spline_dir(curve, mf, s, t)
        ss = dirn * abs(signed_wrap(t - s))
        if abs(abs(signed_wrap(t - s)) - 180) > 1e-6:
            assert np.sign(ss) == np.sign(signed_wrap(t - s)), (s, t)
        cand = np.flatnonzero((y == s) & (d["role"] != "knot"))
        kn = np.flatnonzero((y == s) & (d["role"] == "knot"))
        rows = np.sort(np.concatenate([cand, rng.choice(kn, N_CAR - len(cand), replace=False)]))[:N_CAR]
        assert len(rows) == N_CAR
        tp = behaviour_target(spl, s, ss, ts)
        sp, ch = pair_paths(G, s, t, K)
        m = np.abs(sh_all - abs(signed_wrap(t - s))) < 1e-6
        if not m.any():
            m = np.abs(sh_all - abs(signed_wrap(t - s))) == np.abs(sh_all - abs(signed_wrap(t - s))).min()
        ph = readout_dist(Pall[rows], U, kappa)                                     # unedited carriers
        un = 1 - np.sqrt(ph[None] * tp[:, None]).sum(-1)                             # [K, 16]
        for k_, v_ in (("src", s), ("tg", t), ("shift_signed", ss), ("rows", rows), ("target_p", tp),
                       ("spline64", sp), ("chord64", ch), ("init32", ch[:, :K_OPT]), ("nat_unit", nat_all[m].mean()),
                       ("unedited_loss_per_t", un.mean(1)), ("zfull_carriers", Zfull[rows])):
            recs[k_].append(v_)
    PB.mkdir(parents=True, exist_ok=True)
    np.savez(PB / "prep.npz", **{k: np.array(v) for k, v in recs.items()}, W=W, b=b, U=U, kappa=kappa,
             pca_mean=pca.mean, components=pca.components, ts=ts, values=vals, behaviour_centroids=Bc)
    meta = {"kappa": kappa, "kappa_test_nll": nll, "chance_nll": float(np.log(len(vals))), "n_pairs": len(pairs),
            "pairs": pairs, "shift_signed": [float(x) for x in recs["shift_signed"]],
            "target_endpoint_argmax_deg": [[float(vals[p[0].argmax()]), float(vals[p[-1].argmax()])]
                                           for p in recs["target_p"]]}
    (PB / "prep.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "pairs"}, indent=1))


# ================================================================ run (box, GPU)
def run(n_pairs=8, max_evals=48, chunk=16, smoke=0):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import edited_prediction, pool_steps, predict_future
    from wm.propagate import _mean, prefix
    set_precision()
    device = pick_device()
    model = load_model("vjepa2", device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    pf = predict_future.__wrapped__
    enc = model.encoder
    z = np.load(PB / "prep.npz")
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    Wt, bt, Ut, kappa = f32(z["W"]), f32(z["b"]), f32(z["U"]), float(z["kappa"])
    comp = f32(z["components"][:K_OPT])                                            # [32, D]
    comp64, mu = z["components"], z["pca_mean"]
    df = load_table("direction")
    jobs = [(t, n) for t in range(K) for n in range(N_CAR)]
    for i in range(min(n_pairs, len(z["src"]))):
        f = PB / f"pair_{i:02d}.npz"
        if f.exists() and not smoke:
            continue
        t0 = time.time()
        rows = z["rows"][i]
        with torch.no_grad():
            pv = preprocess([decode(df["video"].iloc[int(r)]) for r in rows]).to(device)
            saved, _, _ = prefix(model, pv[:, :8], [L])
        h0 = saved[L]                                                               # [16, 1024, D]
        ctx_mean = _mean(h0).double().cpu().numpy()                                 # [16, D]
        zc64 = (ctx_mean - mu) @ comp64.T                                           # carriers' own PCA-64 coords
        zc = f32(zc64[:, :K_OPT])
        tp = f32(z["target_p"][i])                                                  # [K, 64]
        V = f32(z["init32"][i]).clone().requires_grad_(True)
        nj = len(jobs) if not smoke else chunk

        def closure_fn():
            tot, per_t = 0.0, np.zeros(K)
            for c in range(0, nj, chunk):
                ch = jobs[c:c + chunk]
                ti = torch.as_tensor([j[0] for j in ch], device=device)
                ni = torch.as_tensor([j[1] for j in ch], device=device)
                delta = (V[ti] - zc[ni]) @ comp                                     # replacement of top-32 coords
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = h0[ni] + delta[:, None, :]
                    for layer in enc.layer[L:]:
                        h = layer(h, None, None, False)[0]
                    zf = pf(model, enc.layernorm(h), 0)
                P = zf.float().mean(1) @ Wt + bt
                logp = torch.log_softmax(kappa * (P @ Ut.T), -1)
                dh2 = 1 - (torch.exp(0.5 * logp) * tp[ti].sqrt()).sum(-1)           # squared Hellinger
                (dh2.sum() / N_CAR).backward()
                tot += float(dh2.sum()) / N_CAR
                np.add.at(per_t, ti.cpu().numpy(), dh2.detach().cpu().numpy() / N_CAR)
            return torch.tensor(tot), per_t.tolist()

        with torch.enable_grad():
            res = optimise(V, closure_fn, max_evals=max_evals if not smoke else smoke,
                           log=lambda s: print(s, flush=True))
            if res["stop"] == "non_finite":                                         # Adam fallback (not expected)
                V.data.copy_(f32(z["init32"][i]))
                o = torch.optim.Adam([V], lr=0.5)
                res = {"history": [], "per_t": [], "n_evals": 0, "stop": "adam_fallback", "optimiser": "adam"}
                for _ in range(max_evals):
                    o.zero_grad()
                    lv, pt = closure_fn()
                    o.step()
                    res["history"].append(float(lv))
                    res["per_t"].append(pt)
                    res["n_evals"] += 1
                res["outer_steps"] = res["n_evals"]
        V0 = f32(z["init32"][i])
        # fp32 re-score of the final path and of the chord init, gap-1 forward
        pred = {}
        with torch.no_grad():
            for name, Vx in (("final", V.detach()), ("init", V0)):
                out = np.zeros((K, N_CAR, 4, comp.shape[1]), np.float32)
                for c in range(0, len(jobs), chunk):
                    ch = jobs[c:c + chunk]
                    ti = torch.as_tensor([j[0] for j in ch], device=device)
                    ni = torch.as_tensor([j[1] for j in ch], device=device)
                    zz, _ = edited_prediction(model, h0[ni], L, (Vx[ti] - zc[ni]) @ comp, 0)
                    p = pool_steps(zz).cpu().numpy()
                    for q, (t_, n_) in enumerate(ch):
                        out[t_, n_] = p[q]
                pred[name] = out
        np.savez(f, V=V.detach().cpu().numpy(), zc64=zc64, history=np.array(res["history"]),
                 per_t=np.array([q for q in res["per_t"] if q is not None], float),
                 eval_losses=np.array(res.get("eval_losses", res["history"])),
                 best_loss=res["best_loss"] if "best_loss" in res else min(res["history"]), n_evals=res["n_evals"], outer_steps=res["outer_steps"],
                 stop=res["stop"], optimiser=res["optimiser"], pred_final=pred["final"], pred_init=pred["init"],
                 seconds=time.time() - t0)
        print(f"pair {i} ({float(z['src'][i])}->{float(z['tg'][i])}): loss {res.get('eval_losses', res['history'])[0]:.4f} -> "
              f"{res['best_loss'] if 'best_loss' in res else res['history'][-1]:.4f} steps {res['outer_steps']} evals {res['n_evals']} {res['stop']} "
              f"{time.time() - t0:.0f}s", flush=True)
        if smoke:
            f.unlink()
            return
    (PB / "run_info.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                                                  "autocast": "bf16 (optimisation only; final forecasts fp32)",
                                                  "chunk": chunk, "max_evals_per_pair": max_evals, **LBFGS}, indent=1))


# ================================================================ score (CPU)
def se(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
            "n": int(len(v))}


def paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    dlt = a - b
    t, p = stats.ttest_rel(a, b)
    w = stats.wilcoxon(dlt).pvalue if len(dlt) >= 6 else None
    h = stats.t.ppf(0.975, len(dlt) - 1) * dlt.std(ddof=1) / np.sqrt(len(dlt))
    return {"mean_diff": float(dlt.mean()), "se": float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
            "ci95_t": [float(dlt.mean() - h), float(dlt.mean() + h)], "t": float(t), "p_ttest_rel": float(p),
            "p_wilcoxon": None if w is None else float(w), "n": int(len(dlt)), "n_a_below_b": int((dlt < 0).sum())}


def score_triplet(v, sp, ch, unit):
    """Optimised path v vs the spline and the chord (both R^2 in the spline's basis, per A.9), plus Goodfire's own
    linear baseline (chord scored against the spline)."""
    a, c = recap_metrics(v, sp), recap_metrics(v, ch, basis_ref=sp)
    c_own = recap_metrics(v, ch)
    lin = recap_metrics(ch, sp)
    return {"resid_to_spline": a["mean_closest_point_residual"] / unit,
            "resid_to_chord": c["mean_closest_point_residual"] / unit,
            "r2_spline": a["r_squared"], "r2_chord": c["r_squared"], "r2_chord_own_basis": c_own["r_squared"],
            "intrinsic_dim_spline": a["intrinsic_dim"], "arc_ratio_vs_spline": a["arc_length_ratio"],
            "chord_baseline_resid_to_spline": lin["mean_closest_point_residual"] / unit,
            "chord_baseline_r2_spline": lin["r_squared"],
            "resid_to_spline_by_t": (np.array(a["residual_by_t"]) / unit).tolist(),
            "resid_to_chord_by_t": (np.array(c["residual_by_t"]) / unit).tolist()}


def summarise(per):
    keys = ("resid_to_spline", "resid_to_chord", "r2_spline", "r2_chord", "r2_chord_own_basis",
            "chord_baseline_resid_to_spline", "chord_baseline_r2_spline", "arc_ratio_vs_spline")
    g = lambda k: [p[k] for p in per]  # noqa: E731
    return {"means": {k: se(g(k)) for k in keys},
            "paired_resid_spline_minus_chord": paired(g("resid_to_spline"), g("resid_to_chord")),
            "paired_r2_spline_minus_chord": paired(g("r2_spline"), g("r2_chord")),
            "goodfire_paired_r2_optimised_vs_chord_baseline": paired(g("r2_spline"), g("chord_baseline_r2_spline")),
            "goodfire_paired_resid_optimised_vs_chord_baseline": paired(g("resid_to_spline"),
                                                                      g("chord_baseline_resid_to_spline"))}


def score_old():
    """Closest-point scoring of the old reverse-test paths (deltas in PCA-64 relative to each carrier's full-clip
    coordinates; natural-norm-scaled spline and chord from the gap-1 plan, spline densified to 41 points with the
    same scale factor and checked against the stored 9 waypoints)."""
    G = geometry()
    curve, y = G["curve"], G["y"]
    zr, zp, zo = np.load(RV / "prep.npz"), np.load(AP / "plan.npz"), np.load(RV / "reverse_out.npz")
    pick, tj, rows = zr["pick"], zr["tj"], zr["carrier_rows"]
    w = zo["w"]                                                                     # [N, 2, 8, 64]
    ARMS = [str(a) for a in zp["arms"]]
    ksp, kch = ARMS.index("spline"), ARMS.index("chord")
    dense = np.linspace(0, 1, 41)
    per, par = [], 0.0
    for n in range(len(pick)):
        rec = []
        for j in range(2):
            c, tt = pick[n], tj[n, j]
            s, tg = y[rows[n]], zp["targets"][c, tt]
            sc_sp, sc_ch = zp["scales"][c, tt, ksp], zp["scales"][c, tt, kch]
            ta, tb = curve.coord_of_value(s), curve.coord_of_value(tg)
            on = curve(ta + dense * curve.step(ta, tb))
            sp = sc_sp * (on - on[:1])
            stored = zp["dZ"][c, tt, ksp]                                           # [9, 64]
            par = max(par, float(np.abs(sp[::5] - stored).max() / np.abs(stored).max()))
            ch = zp["dZ"][c, tt, kch][[0, -1]]                                      # straight: its endpoints
            v = np.vstack([np.zeros(64), w[n, j]])                                  # t = 0 (unedited) + 8 waypoints
            unit = zr["nat"][n, j]
            rec.append(score_triplet(v[1:], sp, ch, unit))
        per.append({k: float(np.mean([r[k] for r in rec])) for k in rec[0] if not k.endswith("_by_t")}
                   | {"resid_to_spline_by_t": np.mean([r["resid_to_spline_by_t"] for r in rec], 0).tolist(),
                      "resid_to_chord_by_t": np.mean([r["resid_to_chord_by_t"] for r in rec], 0).tolist()})
    assert par < 1e-6, par
    return per, par


def score():
    from run_session2 import write
    z = np.load(PB / "prep.npz")
    meta = json.loads((PB / "prep.json").read_text())
    files = sorted(PB.glob("pair_*.npz"))
    U, kappa, W, b = z["U"], float(z["kappa"]), z["W"], z["b"]
    per, per32, pairs = [], [], []
    for f in files:
        i = int(f.stem.split("_")[1])
        r = np.load(f)
        V64 = np.hstack([r["V"], np.zeros((K, 64 - K_OPT))])                        # causalab: pad with train mean
        unit = float(z["nat_unit"][i])
        sc = score_triplet(V64, z["spline64"][i], z["chord64"][i], unit)
        sc32 = score_triplet(r["V"], z["spline64"][i][:, :K_OPT], z["chord64"][i][:, :K_OPT], unit)
        # fp32 re-scored losses
        tp = z["target_p"][i]
        lf = {}
        for nm in ("final", "init"):
            pdist = readout_dist(r[f"pred_{nm}"].astype(np.float64).mean(-2) @ W + b, U, kappa)   # [K, 16, 64]
            lf[nm] = float((1 - np.sqrt(pdist * tp[:, None]).sum(-1)).mean(1).sum())
        Pf = r["pred_final"].astype(np.float64).mean(-2) @ W + b
        ang = np.degrees(np.arctan2(Pf[..., 0], Pf[..., 1])) % 360.0
        ideal = (float(z["src"][i]) + z["ts"] * float(z["shift_signed"][i])) % 360.0
        err = np.abs(signed_wrap(ang - ideal[:, None])).mean(1)
        rec = {"pair": f"{float(z['src'][i])}->{float(z['tg'][i])}", "shift_signed_deg": float(z["shift_signed"][i]),
               "nat_unit": unit, **sc, "loss_init_bf16": float(r["eval_losses"][0]),
               "loss_final_bf16": float(r["best_loss"]), "loss_final_fp32": lf["final"], "loss_chord_init_fp32": lf["init"],
               "loss_unedited": float(z["unedited_loss_per_t"][i].sum()), "loss_history": r["history"].round(5).tolist(),
               "eval_losses": r["eval_losses"].round(5).tolist(),
               "outer_steps": int(r["outer_steps"]), "n_evals": int(r["n_evals"]), "stop": str(r["stop"]),
               "optimiser": str(r["optimiser"]), "seconds": float(r["seconds"]),
               "forecast_err_to_ideal_deg_by_t": err.round(2).tolist(),
               "carrier_ctx_vs_fullclip_pca32_offset_over_unit": float(np.linalg.norm(
                   r["zc64"][:, :K_OPT] - z["zfull_carriers"][i][:, :K_OPT], axis=-1).mean() / unit),
               "path_norm32_over_unit_by_t": (np.linalg.norm(r["V"], axis=-1) / unit).round(3).tolist(),
               "spline_norm32_over_unit_by_t": (np.linalg.norm(z["spline64"][i][:, :K_OPT], axis=-1) / unit).round(3).tolist()}
        per.append(rec)
        per32.append(sc32)
        pairs.append(rec["pair"])
    old, par = score_old()
    out = {"point": L, "K": K, "k_opt": K_OPT, "n_carriers_per_pair": N_CAR, "n_pairs": len(per),
           "behaviour_readout": {"kappa": meta["kappa"], "kappa_test_nll": meta["kappa_test_nll"],
                                 "chance_nll": meta["chance_nll"]},
           "run": json.loads((PB / "run_info.json").read_text()) if (PB / "run_info.json").exists() else None,
           "goodfire": {"per_pair": per, "summary_64d": summarise(per), "summary_32d": summarise(per32),
                        "losses": {k: se([p[k] for p in per]) for k in ("loss_unedited", "loss_chord_init_fp32",
                                                                       "loss_final_fp32")},
                        "paired_loss_final_minus_chord_init": paired([p["loss_final_fp32"] for p in per],
                                                                     [p["loss_chord_init_fp32"] for p in per]),
                        "forecast_err_to_ideal_deg_mean_over_t": se([np.mean(p["forecast_err_to_ideal_deg_by_t"])
                                                                     for p in per])},
           "old_reverse_test": {"n_carriers": len(old), "spline_densify_parity_rel_maxabs": par,
                                "per_carrier": old, "summary": summarise(old)},
           "definitions": {
               "resid_to_spline / resid_to_chord": "mean over waypoints of the closest-point distance (full PCA-64 "
               "space) from each optimised waypoint to the polyline of the spline path / to the chord, divided by the "
               "mean natural twin change norm of plan entries at the same |shift| (Goodfire) or the carrier's own "
               "natural twin change (old test)",
               "r2_spline / r2_chord": "causalab intrinsic R^2: 1 - RSS/TSS of the optimised waypoints in the spline's "
               "99%-variance SVD basis, RSS to the closest point on the spline (resp. chord) polyline",
               "r2_chord_own_basis": "same with the chord's own basis (1-d; near 1 for any path spanning the chord)",
               "chord_baseline_*": "Goodfire's linear baseline: the chord itself scored against the spline",
               "summary_64d / summary_32d": "optimised 32-d path padded with the training mean (0) to 64-d, as causalab "
               "pads dropped dims / everything truncated to the top-32 PCs",
               "SE": "over pairs (Goodfire) or over carriers, targets averaged within carrier (old test)",
               "old_reverse_test": "optimised per-carrier waypoint deltas t = 0.125..1 (session2_reverse_path.json); "
               "spline and chord are the gap-1 natural-norm paths (deltas from the carrier), spline densified to 41 "
               "points"}}
    write(RES / "session2_pullback_goodfire.json", out, stage="pullback_goodfire_score", seeds={"carriers": 0},
          cache=str(PB))
    for nm, s in (("goodfire64", out["goodfire"]["summary_64d"]), ("goodfire32", out["goodfire"]["summary_32d"]),
                  ("old", out["old_reverse_test"]["summary"])):
        m = s["means"]
        print(nm, {k: round(v["mean"], 3) for k, v in m.items()}, "paired resid", s["paired_resid_spline_minus_chord"])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "run":
        run(n_pairs=int(sys.argv[2]) if len(sys.argv) > 2 else 8, max_evals=int(sys.argv[3]) if len(sys.argv) > 3 else 48,
            chunk=int(sys.argv[4]) if len(sys.argv) > 4 else 16, smoke=int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        {"prep": prep, "score": score}[a]()

"""Time through the predictor, PATCH instead of PUSH: does the forecast advance when the encoder's time code is
REPLACED token by token with a later slot's, rather than shifted by one pooled vector per step?

Same carriers (artifacts/time_predictor_gpu/plan.npz: 128 speed-set test clips, seed 0), same context-only
predictor readout (frames 1-8 = context slots s = 0..3, blocks L+1..24, final LN, predictor mask token 0) and the same
readers as scripts/time_predictor.py (per-step disk-position probes on the predictor's native forecasts; context time
probe on context-only final-point step pools). Points 22 and 12.

Time subspace T_L: span of the 8 full-clip step centroids C_t (plan.npz C_L, PCA-64 coords, lifted to raw space by
pca_L and centred over t), orthonormal basis Q [D, r], r = 7. For a context token x at slot s, patch position p:
    x' = x - Q Q^T x + Q Q^T y          (y = the source token; every component outside T is x's own)
Sources come from the SAME clip at the same point: the full 16-frame encoding (slots 0..7, all tokens) or the
context-only encoding itself. Arms (per point):
  tp+2      T-component from full-clip slot s+2 (same patch p)             the time code two slots later
  tp_same   T-component from full-clip slot s                              control: full-clip vs context-only encoding
  rp+2      same as tp+2 in a random rank-r subspace (one per carrier)     content control for tp+2
  rp_same   same as tp_same in that random subspace
  tok+2     the whole token from full-clip slot s+2                        upper bound: all of the later slot's content
  tok_same  the whole token from full-clip slot s                          control for tok+2
  tp_rev    T-component from the context's own slot 3-s                    reversed time order (as rev_sp)
  w2:tp-2   on the win+2 context (frames 5-12, baseline = win+2's own forecast): T-component from full-clip slot s,
            i.e. two slots EARLIER in absolute time (the mirrored, backwards patch)
  w2:tp_same  on the win+2 context: T-component from full-clip slot s+2 (same absolute time; control for w2:tp-2)
References recomputed in the same forward: unedited, win+2 (real +2 advance). The pooled PUSH arms (sp+2, ch+2,
null-2, rand+2, rev_sp) are read per clip from artifacts/time_predictor_gpu/forward for paired comparisons.

Stages: plan (CPU) -> ART/patch_plan.npz;  forward (GPU) -> ART/forward/group_*.npz;  score (CPU) -> results json
  python scripts/time_patch_predictor.py plan|forward|score [--out ART] [--max-groups N] [--smoke]
"""
import argparse
import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

_spec = spec_from_file_location("time_predictor", Path(__file__).with_name("time_predictor.py"))
tpm = module_from_spec(_spec)
_spec.loader.exec_module(tpm)

POINTS = (22, 12)
ARMS = ("tp+2", "tp_same", "rp+2", "rp_same", "tok+2", "tok_same", "tp_rev", "w2:tp-2", "w2:tp_same")
ARM_K = {"tp+2": 2, "tp_same": 0, "rp+2": 2, "rp_same": 0, "tok+2": 2, "tok_same": 0, "tp_rev": 0,
         "w2:tp-2": -2, "w2:tp_same": 0}
D = tpm.D
RANK = 7
SRC_PLAN = PROJECT_ROOT / "artifacts" / "time_predictor_gpu"
ART = PROJECT_ROOT / "artifacts" / "time_patch_predictor"
POOLED_ARMS = ("sp+2", "ch+2", "null-2", "rand+2", "rev_sp")


# ---------------------------------------------------------------- pure helpers (tests/test_time_patch_predictor.py)

def time_basis(C, comps, rank=RANK):
    """Orthonormal basis [D, rank] of the span of the centred step centroids C [T, k] lifted by comps [k, D];
    also returns the singular values (all of them)."""
    X = np.asarray(C, np.float64) @ np.asarray(comps, np.float64)
    X = X - X.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:rank].T.copy(), s


def random_bases(n, d, rank, rng):
    """n orthonormal random bases [n, d, rank] (QR of Gaussian matrices)."""
    return np.stack([np.linalg.qr(rng.standard_normal((d, rank)))[0] for _ in range(n)])


def replace_component(tgt, src, Q):
    """Replace the component of tokens tgt [B, N, D] in span(Q) by that of src [B, N, D]; Q is [D, r] or [B, D, r]
    with orthonormal columns. Works for numpy arrays and torch tensors."""
    Qt = Q.swapaxes(-1, -2)
    return tgt + ((src - tgt) @ Q) @ Qt


def slot_sources(n_ctx=4):
    """Source slot index per context slot for each full-clip-sourced arm (for the docs and the test)."""
    s = np.arange(n_ctx)
    return {"tp+2": s + 2, "tp_same": s, "tok+2": s + 2, "tok_same": s, "tp_rev(ctx)": n_ctx - 1 - s,
            "w2:tp-2(abs)": s, "w2:tp_same(abs)": s + 2}


# ---------------------------------------------------------------- plan (CPU)

def plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(SRC_PLAN / "plan.npz")
    rng = np.random.default_rng(args.seed)
    n = len(z["carrier_rows"])
    arrays, info = {"carrier_rows": z["carrier_rows"]}, {"rank": RANK, "seed": args.seed, "n_carriers": int(n),
                                                          "source_plan": str(SRC_PLAN / "plan.npz"), "per_point": {}}
    for L in POINTS:
        Q, s = time_basis(z[f"C_L{L}"], z[f"pca_L{L}"])
        arrays[f"Qt_L{L}"] = Q.astype(np.float32)
        arrays[f"Qr_L{L}"] = random_bases(n, D, RANK, rng).astype(np.float32)
        info["per_point"][str(L)] = {"centroid_singular_values": s.round(4).tolist(),
                                     "variance_share_top_rank": float((s[:RANK] ** 2).sum() / (s ** 2).sum())}
    np.savez(out / "patch_plan.npz", **arrays)
    (out / "patch_plan.json").write_text(json.dumps(info, indent=1))
    print("patch plan", out, n, "carriers", json.dumps(info["per_point"]))


# ---------------------------------------------------------------- forward (GPU)

def forward(args):
    from wm.extract import preprocess
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import prefix
    out = Path(args.out)
    z = np.load(out / "patch_plan.npz")
    rows = z["carrier_rows"]
    df = load_table("speed")
    (out / "forward").mkdir(exist_ok=True)
    model, dev, torch = tpm._model()
    G, done = args.group, 0
    for g0 in range(0, len(rows), G):
        path = out / "forward" / f"group_{g0 // G:03d}.npz"
        if path.exists():
            continue
        if args.max_groups is not None and done >= args.max_groups:
            break
        done += 1
        idx = np.arange(g0, min(g0 + G, len(rows)))
        frames = [decode(df["video"].iloc[rows[i]]) for i in idx]
        with tpm.gpu_lock() as waited, torch.no_grad():
            t0 = time.time()
            pv = preprocess(frames).to(dev)
            B = len(idx)
            sc, fc, _ = prefix(model, pv[:, :8], list(POINTS))
            sf, _, _ = prefix(model, pv, list(POINTS))                                  # full 16 frames, slots 0..7
            sw, fw, _ = prefix(model, pv[:, 4:12], list(POINTS))                        # win+2 context
            res = {"idx": idx,
                   "unedited_pred": pool_steps(predict_future(model, fc, 0)).cpu().numpy(),
                   "unedited_ctx": fc.reshape(B, 4, 256, D).mean(2).cpu().numpy(),
                   "win+2_pred": pool_steps(predict_future(model, fw, 0)).cpu().numpy(),
                   "win+2_ctx": fw.reshape(B, 4, 256, D).mean(2).cpu().numpy()}
            for L in POINTS:
                hc = sc[L].reshape(B, 4, 256, D)
                hf = sf[L].reshape(B, 8, 256, D)
                hw = sw[L].reshape(B, 4, 256, D)
                Qt = torch.as_tensor(z[f"Qt_L{L}"], device=dev)
                Qr = torch.as_tensor(z[f"Qr_L{L}"][idx], device=dev)
                flat = lambda x: x.reshape(B, 1024, D)                                  # noqa: E731
                build = {
                    "tp+2": lambda: replace_component(flat(hc), flat(hf[:, 2:6]), Qt),
                    "tp_same": lambda: replace_component(flat(hc), flat(hf[:, 0:4]), Qt),
                    "rp+2": lambda: replace_component(flat(hc), flat(hf[:, 2:6]), Qr),
                    "rp_same": lambda: replace_component(flat(hc), flat(hf[:, 0:4]), Qr),
                    "tok+2": lambda: flat(hf[:, 2:6]).clone(),
                    "tok_same": lambda: flat(hf[:, 0:4]).clone(),
                    "tp_rev": lambda: replace_component(flat(hc), flat(hc.flip(1)), Qt),
                    "w2:tp-2": lambda: replace_component(flat(hw), flat(hf[:, 0:4]), Qt),
                    "w2:tp_same": lambda: replace_component(flat(hw), flat(hf[:, 2:6]), Qt)}
                pred = np.zeros((B, len(ARMS), 4, D), np.float32)
                ctx = np.zeros((B, len(ARMS), 4, D), np.float32)
                dpool = np.zeros((B, len(ARMS), 4, D), np.float32)
                dtok = np.zeros((B, len(ARMS), 4), np.float32)
                for a, arm in enumerate(ARMS):
                    base = flat(hw) if arm.startswith("w2:") else flat(hc)
                    h = build[arm]()
                    delta = (h - base).reshape(B, 4, 256, D)
                    dpool[:, a] = delta.mean(2).cpu().numpy()
                    dtok[:, a] = delta.norm(dim=-1).mean(-1).cpu().numpy()
                    fe = tpm.finish(model, h, L)
                    pred[:, a] = pool_steps(predict_future(model, fe, 0)).cpu().numpy()
                    ctx[:, a] = fe.reshape(B, 4, 256, D).mean(2).cpu().numpy()
                res[f"edit_pred_L{L}"], res[f"edit_ctx_L{L}"] = pred, ctx
                res[f"delta_pool_L{L}"], res[f"delta_tok_norm_L{L}"] = dpool, dtok
                res[f"tok_norm_L{L}"] = flat(hc).norm(dim=-1).reshape(B, 4, 256).mean(-1).cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            res["device"] = np.array([str(dev)])
            res["gpu_seconds"] = np.array([time.time() - t0])
            res["lock_wait_s"] = np.array([waited])
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        tmp.rename(path)
        print(f"group {g0 // G} {res['gpu_seconds'][0]:.1f}s wait {waited:.1f}s", flush=True)


# ---------------------------------------------------------------- score (CPU)

def readers(native_dir):
    """Position probes (forecast steps 4-7) and the context time probe, exactly as time_predictor.score builds them."""
    from wm import timeman as tm
    from wm.p2_data import load_inputs
    d0 = load_inputs("speed", 0)
    df, role = d0["df"], d0["role"]
    probe = role == "probe"
    Zall = np.load(PROJECT_ROOT / "artifacts/session3_speed/native/pred_pooled_all.npy").astype(np.float64)
    cen = np.load(PROJECT_ROOT / "artifacts/session3_speed/centroids_speed.npy")
    pos_models = []
    for k, t in enumerate(range(4, 8)):
        ok = probe & np.isfinite(cen[:, t]).all(1)
        pos_models.append(tm.ridge(Zall[ok, k], cen[ok, t]))

    def read_pos(P):
        return np.stack([pos_models[k].predict(P[..., k, :].reshape(-1, D)).reshape(P.shape[:-2] + (2,))
                         for k in range(4)], -2)
    shards = [np.load(f) for f in sorted(Path(native_dir).glob("native_ctx_steps_*of*.npz"))]
    have = np.zeros(len(df), bool)
    X25 = np.full((len(df), 4, D), np.nan)
    for sh in shards:
        have[sh["rows"]] = True
        X25[sh["rows"]] = sh["L25"]
    pr_rows = np.flatnonzero(probe & have)
    tprobe = tm.ridge(X25[pr_rows].reshape(-1, D), np.tile(np.arange(4), len(pr_rows)))

    def read_t(X):
        return tprobe.predict(X.reshape(-1, D)).reshape(X.shape[:-1]).mean(-1)
    return df, read_pos, read_t, Zall


def score(args):
    from wm import timeman as tm
    from wm.provenance import provenance, sha256_file
    out = Path(args.out)
    info = json.loads((out / "patch_plan.json").read_text())
    z = np.load(out / "patch_plan.npz")
    files = [f for f in sorted((out / "forward").glob("group_*.npz")) if not f.name.endswith(".tmp.npz")]
    fw = [np.load(f) for f in files]
    F = {k: np.concatenate([f[k] for f in fw]) for k in fw[0].files if k not in ("gpu_seconds", "lock_wait_s", "device")}
    idx = F["idx"]
    rows = z["carrier_rows"][idx]
    df, read_pos, read_t, Zall = readers(SRC_PLAN)
    v, th = df["speed_mps"].to_numpy(float), df["theta_degrees"].to_numpy(float)
    sp = tpm.step_px(v[rows])
    # pooled PUSH run, per clip, same carriers
    pz = [np.load(f) for f in sorted((SRC_PLAN / "forward").glob("group_*.npz")) if not f.name.endswith(".tmp.npz")]
    P = {k: np.concatenate([f[k] for f in pz]) for k in ("idx", "unedited_pred", "win+2_pred", "edit_pred_L12",
                                                          "edit_pred_L22")}
    order = {int(i): j for j, i in enumerate(P["idx"])}
    pj = np.array([order[int(i)] for i in idx])
    parity = {"unedited_vs_pooled_run_rel_maxabs": float(np.abs(F["unedited_pred"] - P["unedited_pred"][pj]).max()
                                                         / np.abs(P["unedited_pred"][pj]).max()),
              "win+2_vs_pooled_run_rel_maxabs": float(np.abs(F["win+2_pred"] - P["win+2_pred"][pj]).max()
                                                      / np.abs(P["win+2_pred"][pj]).max()),
              "unedited_vs_native_cache_rel_maxabs": float(np.abs(F["unedited_pred"] - Zall[rows]).max()
                                                           / np.abs(Zall[rows]).max())}
    base = {"ctx": (read_pos(F["unedited_pred"]), read_t(F["unedited_ctx"])),
            "w2": (read_pos(F["win+2_pred"]), read_t(F["win+2_ctx"]))}
    speed_fc = lambda p: np.linalg.norm(p[..., -1, :] - p[..., 0, :], axis=-1) / 3 / tpm.PX_PER_M / tpm.STEP_S  # noqa
    B = tm.cluster_bootstrap

    def along(p, p0):
        al, ac = tpm.along_across(p - p0, th[rows][:, None])
        return al.mean(-1), np.abs(ac).mean(-1)

    def metrics(pred, ctx, k, which="ctx"):
        p0, t0 = base[which]
        p = read_pos(pred)
        al, ac = along(p, p0)
        dh = np.abs((tpm.heading_deg(p) - tpm.heading_deg(p0) + 180) % 360 - 180)
        m = {"along_px": B(al), "across_abs_px": B(ac), "forecast_speed_change_mps": B(speed_fc(p) - speed_fc(p0)),
             "heading_change_abs_deg": B(dh), "frac_heading_flipped_gt90": float((dh > 90).mean())}
        if ctx is not None:
            m["ctx_time_readout_change_steps"] = B(read_t(ctx) - t0)
        m["advance_fraction"] = B(al / ((k if k else 2) * sp))
        if not k:
            m["advance_fraction_note"] = "k = 0 arm: along_px / (2 x true step px), for comparison with the +2 arms"
        return m, al

    res = {"design": {"doc": __doc__.split("Stages:")[0].strip(), "arms": list(ARMS), "arm_k": ARM_K,
                      "points": list(POINTS), "patch_plan": info, "slot_sources": {k: v.tolist() for k, v in
                                                                                  slot_sources().items()}},
           "n_carriers_scored": int(len(idx)), "parity": parity,
           "readouts": {"true_step_px_carriers_mean": float(sp.mean())}, "references": {}, "arms": {},
           "pooled_push_same_clips": {}, "gaps": {}, "edit_size": {}}
    al_ref = {}
    res["references"]["win+2"], al_ref["win+2"] = metrics(F["win+2_pred"], F["win+2_ctx"], 2)
    res["references"]["unedited"] = {"along_px": B(np.zeros(len(idx))), "advance_fraction": B(np.zeros(len(idx))),
                                     "note": "0 by construction (baseline); see parity for the recomputation check"}
    p0 = base["ctx"][0]
    for L in POINTS:
        S = str(L)
        res["arms"][S], als = {}, {}
        for a, arm in enumerate(ARMS):
            res["arms"][S][arm], als[arm] = metrics(F[f"edit_pred_L{L}"][:, a], F[f"edit_ctx_L{L}"][:, a], ARM_K[arm],
                                                    "w2" if arm.startswith("w2:") else "ctx")
        # pooled PUSH arms, same clips, re-read with the same readers against the unedited forecast
        tp_arms = tpm.ARMS
        res["pooled_push_same_clips"][S] = {}
        for arm in POOLED_ARMS:
            pp = read_pos(P[f"edit_pred_L{L}"][pj][:, tp_arms.index(arm)])
            al, ac = along(pp, p0)
            als["pooled:" + arm] = al
            k = tpm.ARM_K[arm]
            res["pooled_push_same_clips"][S][arm] = {"along_px": B(al), "across_abs_px": B(ac),
                                                     "advance_fraction": B(al / ((k if k else 2) * sp))}
        g = {}
        pairs = [("tp+2", "tp_same"), ("tp+2", "rp+2"), ("tp+2", "pooled:sp+2"), ("tp+2", "win+2"),
                 ("rp+2", "rp_same"), ("tok+2", "tok_same"), ("tok+2", "win+2"), ("w2:tp-2", "w2:tp_same"),
                 ("tp_rev", "tp_same")]
        for a1, a2 in pairs:
            x1 = als[a1]
            x2 = al_ref["win+2"] if a2 == "win+2" else als[a2]
            g[f"{a1}_minus_{a2}_along_px"] = B(x1 - x2)
            g[f"{a1}_minus_{a2}_advance_fraction"] = B((x1 - x2) / (2 * sp))
        res["gaps"][S] = g
        # edit size: per-token change and the pooled per-step change vs the pooled push sp+2 delta
        zp = np.load(SRC_PLAN / "plan.npz")
        sp2 = zp[f"deltas_L{L}"][list(tpm.ARMS).index("sp+2")].astype(np.float64)              # [4, D]
        dp = F[f"delta_pool_L{L}"].astype(np.float64)                                          # [C, A, 4, D]
        tokn = F[f"tok_norm_L{L}"].mean()
        es = {"token_norm_mean": float(tokn), "pooled_sp+2_delta_norm_per_step": np.linalg.norm(sp2, axis=1).round(3).tolist()}
        for a, arm in enumerate(ARMS):
            cos = (dp[:, a] * sp2[None]).sum(-1) / (np.linalg.norm(dp[:, a], axis=-1) * np.linalg.norm(sp2, axis=-1)[None] + 1e-12)
            es[arm] = {"per_token_delta_norm_mean": float(F[f"delta_tok_norm_L{L}"][:, a].mean()),
                       "pooled_step_delta_norm_mean": float(np.linalg.norm(dp[:, a], axis=-1).mean()),
                       "cos_pooled_step_delta_vs_sp+2": float(cos.mean()),
                       "proj_on_sp+2_over_sp+2_norm": float(((dp[:, a] * sp2[None]).sum(-1) / (sp2 ** 2).sum(-1)[None]).mean())}
        res["edit_size"][S] = es
    stored = json.loads((PROJECT_ROOT / "results/p5_time_predictor.json").read_text())
    res["pooled_push_stored"] = {"source": "results/p5_time_predictor.json",
                                 "win+2": {k: stored["references"]["win+2"][k] for k in ("advance_fraction", "along_px")},
                                 **{S: {a: {k: stored["arms"][S][a][k] for k in ("advance_fraction", "along_px",
                                                                                   "across_abs_px", "frac_heading_flipped_gt90")
                                            if k in stored["arms"][S][a]} for a in ("sp+2", "rev_sp", "null-2")}
                                    for S in ("12", "22")}}
    res["gpu"] = {"devices": sorted({str(f["device"][0]) for f in fw}),
                  "forward_gpu_seconds": float(sum(f["gpu_seconds"][0] for f in fw)),
                  "lock_wait_s": float(sum(f["lock_wait_s"][0] for f in fw)), "n_groups": len(fw),
                  "box": args.box}
    res["keys"] = {
        "arms[point][arm]": "as p5_time_predictor.json: along_px = mean over forecast tubelets 4-7 of the forecast "
                            "disk displacement along the clip's motion, edited minus baseline (unedited; win+2's own "
                            "forecast for w2:*); advance_fraction = along_px / (k x true px per tubelet), k = 2 for "
                            "'+2' arms, -2 for w2:tp-2 (1 = moved the full two steps in the aimed direction), 2 for "
                            "k = 0 controls; ctx_time_readout_change_steps: context time probe; CIs: 95% clip bootstrap",
        "gaps[point]": "paired per-clip differences of along_px (and / (2 x step px)); pooled:* = the stored pooled "
                       "PUSH forward of the same clips re-read here",
        "edit_size": "per-token |x' - x| and the per-step pooled change vs the pooled sp+2 push delta"}
    res["provenance"] = provenance(seeds={"patch_plan": info["seed"], "source_plan": 0, "bootstrap": 0},
                                   script_sha256=sha256_file(Path(__file__)), points=list(POINTS),
                                   forward_files=[f.name for f in files],
                                   forward_sha256={f.name: sha256_file(f) for f in files},
                                   source_plan_sha256=sha256_file(SRC_PLAN / "plan.npz"))
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "forward", "score"])
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--result", default=str(PROJECT_ROOT / "results" / "p5_time_patch_predictor.json"))
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--box", default="vast 53235298 (box 2, RTX 4060 Ti)")
    args = ap.parse_args()
    {"plan": plan, "forward": forward, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()

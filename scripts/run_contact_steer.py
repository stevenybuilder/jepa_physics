"""Steer contact: inject a wall bounce into the context-only residual stream of straight clips and read the forecast.

Carriers: the 96 straight_in twins of the in-window set (artifacts/p5_contact_in_context; contact k_b 2..6). For each
carrier i, the context-only encoding (frames 1-8, bf16 autocast / fp32 residual, as run_contact_in_context) is edited
at point P (22, 12) on its POST-CONTACT context tubelets only (tubelet t with 2t >= k_b_i, i.e. the slots where the
bounce clip already moves along theta_out), then the encoder is finished and the predictor forecasts frames 9-16.

Per tubelet t, two natural differences of each pair j (bounce_j minus straight_j at point P):
  pool  dP[j, t] = mean over the 256 tokens of tubelet t (bounce) - same for the straight twin
  disk  dD[j, t] = mean over the bounce clip's disk tokens - mean over the straight clip's disk tokens (tubelet t)
Conditions (vector added to the carrier's tokens of each post-contact tubelet):
  none         unedited carrier
  a_pool_loo   mean dP over donors j != i with the same wall and the same tangential sign of the incoming heading
               (same reflection geometry class; leave-one-pair-out: the carrier's own twin is never a donor),
               donor tubelet t if the donor is post-contact there, else its tubelet 3; added to ALL 256 tokens
  b_disk_loo   same donors, dD, added to the carrier's own disk tokens only
  c_rand_pool / c_rand_disk   random Gaussian direction per (carrier, tubelet), norm matched to a / b, same tokens
  d_far_pool / d_far_disk     mismatched geometry: mean over donors whose theta_out is >= 120 deg from the carrier's
  e_own_pool / e_own_disk     oracle: the carrier's own twin difference dP[i, t] / dD[i, t]
  e_own_tokens                oracle upper bound: the full token-level own difference on post-contact tubelets
                              (= the bounce clip's tokens pasted into those slots; pre-contact slots stay straight)
Readers (scored on the Mac):
  forecast   per-step ridge reader fit on native probe-role forecasts + all stored-set (p5_contact_dynamics) rendered
             forecasts (straight twins and bounces labelled with the true step heading) = the "transfer" reader of
             results/p5_contact_probe_domain.json that reads the in-window bounce forecasts at 0.94; it has never
             seen an in-window clip
  encoder    context-only per-tubelet readers (fit on the 750 velocity clips' context-only encodings,
             artifacts/p5_contact_in_context/out/forward.npz vel/ctx8) at point 25 (post-LN) after the edit, and at
             point 22 (P = 12 only; for P = 22 the point-22 read is the edited stream itself)
turn fraction: 0 = the carrier's incoming heading, 1 = its twin's reflected heading.

  python scripts/run_contact_steer.py plan                    (Mac)
  python scripts/run_contact_steer.py forward --root <out>    (box GPU; flock /tmp/wm_gpu.lock)
  python scripts/run_contact_steer.py score --root <out>      (Mac CPU)
"""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_contact_in_context import trajectory  # noqa: E402
from run_temporal_contact import CT_POINTS, R_PX, tubelet_phase, turn_fraction, wrap_signed  # noqa: E402
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_contact_steer"
IN_DIR = PROJECT_ROOT / "artifacts" / "p5_contact_in_context"
OUT_DIR = PROJECT_ROOT / "artifacts" / "p5_temporal_contact"
RES, FIG = PROJECT_ROOT / "results", PROJECT_ROOT / "figures"
LOCK = "/tmp/wm_gpu.lock"
POINTS = (22, 12)
CONDS = ("none", "a_pool_loo", "b_disk_loo", "c_rand_pool", "c_rand_disk", "d_far_pool", "d_far_disk",
         "e_own_pool", "e_own_disk", "e_own_tokens")
FAR_DEG = 120.0


# ================================================================ pure helpers (tests/test_contact_steer.py)

def disk_token_mask(xy_world, n_tub=4, grid=16, patch=16, radius=R_PX):
    """bool [n_tub, grid*grid]: patch (h, w) of tubelet t is a disk token if the disk (rounded centre, radius) touches
    the patch rectangle in frame 2t or 2t+1."""
    from wm.render_twin import to_pixels
    c = np.round(to_pixels(np.asarray(xy_world)[:2 * n_tub]))          # (col, row) per frame
    lo = np.arange(grid) * patch
    out = np.zeros((n_tub, grid, grid), bool)
    for f in range(2 * n_tub):
        dx = np.maximum(np.maximum(lo - c[f, 0], 0), c[f, 0] - (lo + patch - 1))   # per column
        dy = np.maximum(np.maximum(lo - c[f, 1], 0), c[f, 1] - (lo + patch - 1))   # per row
        out[f // 2] |= (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius ** 2
    return out.reshape(n_tub, grid * grid)


def geometry_class(cfg):
    """(wall, sign of the incoming heading's tangential component): same class = same reflection geometry range."""
    n = np.asarray(cfg["normal"], float)
    tang = np.array([-n[1], n[0]])
    th = np.radians(cfg["theta_in"])
    return cfg["wall"], int(np.sign(np.array([np.cos(th), np.sin(th)]) @ tang))


def donors(cfgs):
    """Leave-one-pair-out donor lists: same-class (never i) and far (theta_out >= FAR_DEG from i's, never i)."""
    cls = [geometry_class(c) for c in cfgs]
    tho = np.array([c["theta_out"] for c in cfgs])
    same, far = [], []
    for i in range(len(cfgs)):
        same.append([j for j in range(len(cfgs)) if j != i and cls[j] == cls[i]])
        far.append([j for j in range(len(cfgs)) if j != i and abs(wrap_signed(tho[j] - tho[i])) >= FAR_DEG])
    return same, far


def donor_mean(D, phase, idx, t):
    """Mean over donors idx of D[j, t] (if donor j is post-contact at t) else D[j, 3]."""
    return np.mean([D[j, t] if phase[j, t] == 1 else D[j, 3] for j in idx], axis=0)


# ================================================================ plan (Mac)

def plan():
    cfgs = json.loads((IN_DIR / "configs.json").read_text())["configs"]
    mS = np.stack([disk_token_mask(trajectory(c, "straight_in")) for c in cfgs])
    mB = np.stack([disk_token_mask(trajectory(c, "bounce")) for c in cfgs])
    same, far = donors(cfgs)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", mask_straight=mS, mask_bounce=mB, kb=np.array([c["k_b"] for c in cfgs]))
    (ART / "donors.json").write_text(json.dumps({"same_class_loo": same, "far": far, "far_deg": FAR_DEG}))
    print("disk tokens per tubelet (straight) mean", mS.sum(-1).mean(), "same-class donors min/mean",
          min(map(len, same)), np.mean(list(map(len, same))), "far donors min/mean", min(map(len, far)),
          np.mean(list(map(len, far))))


# ================================================================ forward (box GPU)

def forward(root, batch=8, smoke=False):
    import torch
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import suffix

    set_precision()
    dev = pick_device()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    FR = np.load(IN_DIR / "out" / "frames.npz")
    Pl = dict(np.load(ART / "plan.npz"))
    dn = json.loads((ART / "donors.json").read_text())
    same, far = dn["same_class_loo"], dn["far"]
    kb = Pl["kb"]
    n = len(kb)
    phase = np.stack([tubelet_phase(k) for k in kb])[:, :4]
    carriers = np.arange(4) if smoke else np.arange(n)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    out, info = {}, {"norms": {}}

    def enc_to(model, px, point):
        enc = model.encoder
        h = enc.embeddings(px[:, :8]).float()
        for layer in enc.layer[:point]:
            h = layer(h, None, None, False)[0]
        return h

    with open(LOCK, "a") as lk:
        tw = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - tw
        print(f"lock after {waited:.0f}s", flush=True)
        t0 = time.time()
        with torch.no_grad():
            model = load_model("vjepa2", dev)
            for P in POINTS:
                hs = {}
                for kind in ("straight_in", "bounce"):
                    F = FR[kind]
                    parts = []
                    for s in range(0, n, batch):
                        with ac():
                            parts.append(enc_to(model, preprocess(list(F[s:s + batch])).to(dev), P).cpu())
                    hs[kind] = torch.cat(parts)                                 # [n, 1024, D] fp32 on CPU
                hS = hs["straight_in"].view(n, 4, 256, -1)
                hB = hs["bounce"].view(n, 4, 256, -1)
                dP = (hB.mean(2) - hS.mean(2)).numpy()                          # [n, 4, D]
                mS = torch.as_tensor(Pl["mask_straight"]).view(n, 4, 256)
                mB = torch.as_tensor(Pl["mask_bounce"]).view(n, 4, 256)

                def dmean(h, m):
                    w = m.float()
                    return (h * w[..., None]).sum(2) / w.sum(2).clamp(min=1)[..., None]
                dD = (dmean(hB, mB) - dmean(hS, mS)).numpy()
                out[f"P{P}/dP"], out[f"P{P}/dD"] = dP.astype(np.float32), dD.astype(np.float32)
                rng = np.random.default_rng(1000 + P)
                res = {c: {"forecast": [], "tp_final": [], "tp_P": []} for c in CONDS}
                norms = {c: [] for c in CONDS}
                for s in range(0, len(carriers), batch):
                    ids = carriers[s:s + batch]
                    base = hS[ids].clone()                                      # [b, 4, 256, D]
                    for c in CONDS:
                        h = base.clone()
                        for bi, i in enumerate(ids):
                            for t in range(4):
                                if phase[i, t] != 1 or c == "none":
                                    continue
                                if c == "e_own_tokens":
                                    add, tok = (hB[i, t] - hS[i, t]), None
                                else:
                                    kind = "disk" if "disk" in c else "pool"
                                    D = dD if kind == "disk" else dP
                                    if c.startswith("a_") or c.startswith("b_") or c.startswith("c_"):
                                        v = donor_mean(D, phase, same[i], t)
                                    elif c.startswith("d_"):
                                        v = donor_mean(D, phase, far[i], t)
                                    else:
                                        v = D[i, t]
                                    if c.startswith("c_"):
                                        g = rng.standard_normal(v.shape)
                                        v = g / np.linalg.norm(g) * np.linalg.norm(v)
                                    add = torch.as_tensor(v, dtype=torch.float32)
                                    tok = mS[i, t] if kind == "disk" else None
                                norms[c].append(float(add.norm(dim=-1).mean()))
                                if tok is None:
                                    h[bi, t] += add
                                else:
                                    h[bi, t, tok] += add
                        h = h.view(len(ids), 1024, -1).to(dev)
                        with ac():
                            r = suffix(model, h, P, None, return_final=True)
                            fc = pool_steps(predict_future(model, r["final"])).float().cpu().numpy()
                        res[c]["forecast"].append(fc)
                        res[c]["tp_final"].append(r["final"].float().view(len(ids), 4, 256, -1).mean(2).cpu().numpy())
                        # point-22 timepool of the edited context (P=22: the edited stream; P=12: after 10 blocks)
                        res[c]["tp_P"].append(h.float().view(len(ids), 4, 256, -1).mean(2).cpu().numpy() if P == 22
                                              else None)
                    print(f"P{P} carriers {s + len(ids)}/{len(carriers)} {time.time() - t0:.0f}s", flush=True)
                for c in CONDS:
                    out[f"P{P}/{c}/forecast"] = np.concatenate(res[c]["forecast"])
                    out[f"P{P}/{c}/tp_final"] = np.concatenate(res[c]["tp_final"])
                    if P == 22:
                        out[f"P{P}/{c}/tp_P"] = np.concatenate(res[c]["tp_P"])
                info["norms"][str(P)] = {c: (float(np.mean(v)) if v else 0.0) for c, v in norms.items()}
                info["token_norm_mean_P" + str(P)] = float(hS.norm(dim=-1).mean())
        secs = time.time() - t0
    info.update({"seconds_gpu_total": secs, "lock_wait_s": waited, "gpu": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "smoke": smoke, "n_carriers": int(len(carriers)),
                 "precision": "context encode to P, suffix and predictor bf16 autocast, fp32 residual/edits"})
    np.savez(root / "forward.npz", carriers=carriers, **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", secs, flush=True)


# ================================================================ score (Mac CPU)

def score(root, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_session2 import write
    from run_token_patching import summarize
    from temporal_contact_score import ang, apply, cerr, fit_probe
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.probes import targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    finfo = json.loads((root / "forward_info.json").read_text())
    car = F["carriers"]
    cfgs = json.loads((IN_DIR / "configs.json").read_text())["configs"]
    th_in = np.array([c["theta_in"] for c in cfgs])[car]
    th_out = np.array([c["theta_out"] for c in cfgs])[car]
    kb = np.array([c["k_b"] for c in cfgs])[car]
    FI = dict(np.load(IN_DIR / "out" / "forward.npz"))
    FO = dict(np.load(OUT_DIR / "out" / "forward.npz"))
    co = json.loads((OUT_DIR / "bounces.json").read_text())["configs"]
    kbo = np.array([c["k_b"] for c in co])
    tio, too = np.array([c["theta_in"] for c in co]), np.array([c["theta_out"] for c in co])
    pho = np.stack([tubelet_phase(k) for k in kbo])
    df = load_table("direction")
    Ydir = targets(df, "direction")[0]
    prow = np.where(load_inputs("direction", 0)["role"] == "probe")[0]
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    vel = np.where(df["motion"].to_numpy() == "velocity")[0]

    def sc(th):
        t = np.radians(th)
        return np.stack([np.sin(t), np.cos(t)], -1)

    # forecast "transfer" reader: native + all stored-set rendered rows (never an in-window clip)
    readers = []
    for s in range(4):
        t = 4 + s
        m = pho[:, t] != 0
        X = np.concatenate([Zall[prow][:, s], FO["ct/straight/forecast"][:, s], FO["ct/bounce/forecast"][m, s]])
        Y = np.concatenate([Ydir[prow], sc(tio), sc(np.where(pho[m, t] == 1, too[m], tio[m]))])
        readers.append(fit_probe(X, Y, "circular"))
    ctx_r = {p: [fit_probe(FI["vel/ctx8"][:, CT_POINTS.index(p), t], Ydir[vel], "circular")[0] for t in range(4)]
             for p in (22, 25)}
    phase = np.stack([tubelet_phase(k) for k in kb])[:, :4]
    subsets = {"all": np.ones(len(car), bool), "early_kb2-3": kb <= 3, "late_kb5-6": kb >= 5}

    def fc_tf(Fc):
        return np.stack([turn_fraction(ang(apply(readers[s][0], Fc[:, s])), th_in, th_out) for s in range(4)], 1)

    def ctx_tf(TP, p):
        """mean turn fraction over the carrier's post-contact context tubelets."""
        tf = np.stack([turn_fraction(ang(apply(ctx_r[p][t], TP[:, t])), th_in, th_out) for t in range(4)], 1)
        return np.nanmean(np.where(phase == 1, tf, np.nan), 1)

    R = {"references": {}, "points": {}}
    ref_b, ref_s = fc_tf(FI["bounce/forecast"][car]), fc_tf(FI["straight_in/forecast"][car])
    R["references"] = {"real_bounce_clips_forecast_step_mean": summarize(ref_b.mean(1)),
                       "unedited_straight_twins_forecast_step_mean": summarize(ref_s.mean(1)),
                       "real_bounce_ctx25_post_tubelets": summarize(ctx_tf(FI["bounce/ctx8"][car][:, CT_POINTS.index(25)], 25)),
                       "straight_ctx25_post_tubelets": summarize(ctx_tf(FI["straight_in/ctx8"][car][:, CT_POINTS.index(25)], 25))}
    curves = {}
    for P in POINTS:
        blk = {"delta_norm_mean": finfo["norms"][str(P)], "token_norm_mean": finfo.get(f"token_norm_mean_P{P}"),
               "conditions": {}}
        par = F[f"P{P}/none/forecast"]
        blk["parity_none_vs_in_context_forecast_rel_maxabs"] = float(
            np.abs(par - FI["straight_in/forecast"][car]).max() / np.abs(FI["straight_in/forecast"][car]).max())
        tf_none = fc_tf(par).mean(1)
        for c in CONDS:
            tf = fc_tf(F[f"P{P}/{c}/forecast"])
            m1 = tf.mean(1)
            e = {"forecast_turn_fraction_step_mean": {k: summarize(m1[m]) for k, m in subsets.items()},
                 "forecast_turn_fraction_by_step": [summarize(tf[:, s]) for s in range(4)],
                 "forecast_minus_none_step_mean": summarize(m1 - tf_none),
                 "forecast_err_to_theta_out_deg_step_mean": summarize(np.mean(
                     [cerr(ang(apply(readers[s][0], F[f"P{P}/{c}/forecast"][:, s])), th_out) for s in range(4)], 0)),
                 "encoder_ctx_post_tubelets_turn_fraction_point25": summarize(ctx_tf(F[f"P{P}/{c}/tp_final"], 25))}
            if P == 22:
                e["encoder_ctx_post_tubelets_turn_fraction_point22_edited_stream"] = summarize(
                    ctx_tf(F[f"P{P}/{c}/tp_P"], 22))
            blk["conditions"][c] = e
            curves[(P, c)] = m1
        R["points"][str(P)] = blk
    res = {"question": "can an injected bounce (twin difference at point 22 / 12, post-contact context tubelets) make "
                       "the predictor forecast the reflected heading, and does it need the disk tokens?",
           "definitions": {"carriers": f"{len(car)} straight_in twins of the in-window set (k_b 2..6)",
                           "conditions": {c: "" for c in CONDS},
                           "doc": __doc__.split("Readers")[0].strip(),
                           "forecast_reader": "per-step ridge on native probe-role forecasts + all p5_contact_dynamics "
                                              "rendered forecasts (true step heading labels); never saw an in-window "
                                              "clip ('transfer' reader of p5_contact_probe_domain.json, 0.94 on "
                                              "in-window bounces)",
                           "encoder_reader": "context-only per-tubelet ridge on the 750 velocity clips' context-only "
                                             "encodings; averaged over the carrier's post-contact context tubelets",
                           "turn_fraction": "0 = carrier's theta_in, 1 = its twin's theta_out",
                           "ci95": "clip bootstrap, 2000 resamples, seed 0"},
           "results": R, "config": {"points": POINTS, "far_deg": FAR_DEG, "forward": finfo}}
    write((ART / "smoke_results.json") if smoke else RES / "p5_contact_steer.json", res,
          seeds={"random_directions": {str(p): 1000 + p for p in POINTS}, "folds": 0, "bootstrap": 0},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz"),
                             "plan_npz": sha256_file(ART / "plan.npz"), "donors_json": sha256_file(ART / "donors.json"),
                             "in_context_forward_npz": sha256_file(IN_DIR / "out" / "forward.npz")},
          script="scripts/run_contact_steer.py", script_sha256=sha256_file(Path(__file__)),
          forward_script_sha256=((root / "forward_script_sha256.txt").read_text().split()[0]
                                 if (root / "forward_script_sha256.txt").exists() else None),
          box="Vast 53255819 (box 4, RTX 4060 Ti)", gpu_seconds={"this_forward": finfo["seconds_gpu_total"], "smoke": 23.4, "discarded_v1_full_forward_b_disk_bug": 75.0, "note": "v1 full run (b_disk_loo mistakenly used the pool placement) kept on box as /workspace/wm_contact_steer/out_v1_bdiskbug, not scored"}, n_carriers=int(len(car)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, P in enumerate(POINTS):
        a = ax[k]
        x = np.arange(len(CONDS))
        m = [R["points"][str(P)]["conditions"][c]["forecast_turn_fraction_step_mean"]["all"]["mean"] for c in CONDS]
        ci = np.array([R["points"][str(P)]["conditions"][c]["forecast_turn_fraction_step_mean"]["all"]["ci95"] for c in CONDS])
        a.bar(x - 0.2, m, 0.4, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], color="C3", label="forecast")
        m2 = [R["points"][str(P)]["conditions"][c]["encoder_ctx_post_tubelets_turn_fraction_point25"]["mean"] for c in CONDS]
        a.bar(x + 0.2, m2, 0.4, color="C0", alpha=.7, label="encoder ctx (pt 25, post tubelets)")
        a.axhline(R["references"]["real_bounce_clips_forecast_step_mean"]["mean"], c="C3", ls="--", lw=1,
                  label="real bounce forecast")
        a.axhline(0, c="k", lw=.5); a.axhline(1, c="k", lw=.5, ls=":")
        a.set_xticks(x, CONDS, rotation=45, ha="right", fontsize=8)
        a.set_ylabel("turn fraction (0 = incoming, 1 = reflected)"); a.set_title(f"edit at point {P} (straight carriers)")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig((ART / "smoke_fig.png") if smoke else FIG / "fig_contact_steer.png", dpi=130)
    plt.close(fig)
    print("figure written")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "forward", "score"])
    ap.add_argument("--root", default=str(ART / "out"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan()
    elif a.cmd == "forward":
        forward(a.root, smoke=a.smoke)
    else:
        score(a.root, smoke=a.smoke)


if __name__ == "__main__":
    main()

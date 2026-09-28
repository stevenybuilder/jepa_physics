"""Contact INSIDE the context window: does the V-JEPA 2 predictor carry an observed wall reflection forward?

Complement of scripts/run_temporal_contact.py (results/p5_contact_dynamics.json), where the contact was in the
forecast window and the forecast kept the incoming heading. Here the elastic wall bounce happens at contact frame
k_b in 2..6 (0-indexed, the same convention as the stored set's k_b in 3..12), so frames k_b..7 of the 8-frame
context already show the reflected motion; the predictor forecasts tubelets 4-7 (frames 8-15), all post-contact.

Stimuli (same renderer and settings as the stored contact set: wm.render_twin RenderParams, MPEG-4 codec round trip,
grey 6 px wall bar, speeds 2/3/4 m/s, incidence 20-65 deg), 96 configurations with k_b stratified over 2..6:
  bounce        incoming theta_in until k_b, reflected theta_out after, wall drawn
  straight_in   straight twin (primary control): same start / speed, theta_in throughout, no wall
  straight_out  reference: constant velocity along theta_out throughout (same positions as the bounce from k_b on),
                no wall -- what the forecast reader gives when the context shows only theta_out motion
Readers (reused from run_temporal_contact / temporal_contact_score):
  forecast   per-future-step ridge direction probes fit on the native fp32 forecast cache of the 480 probe-role
             direction clips (artifacts/session2/native/pred_pooled_all.npy); forecast = bf16 context-only encode of
             frames 1-8 + predictor (mask index 0), exactly as run_temporal_contact.forward.
  enc16      per-tubelet ridge direction probes fit on the stored fp32 16-frame timepool of the 750 constant-velocity
             direction clips, applied to the same tubelet of the rendered clips' 16-frame fp32 encoding
             (points 1/8/12/22/25; 25 = post-LN). Full-clip attention is bidirectional (sees frames 8-15 too).
  ctx8       per-tubelet probes fit on the bf16 context-only (frames 1-8) encoding of the same 750 velocity clips
             (computed here), applied to the rendered clips' context-only encoding = exactly what the predictor reads.
  random     enc16 with the untrained random-init copy (load_model('random'), torch seed 0): probes on the stored
             random-init timepool of the velocity clips. The forecast is NOT run through the random-init encoder:
             the predictor was trained on the trained encoder's outputs, so a trained predictor on untrained features
             has no readout the stored forecast probes could interpret.
turn fraction = signed angular move of the read angle from theta_in toward theta_out over that arc
(0 = incoming heading, 1 = reflected heading), same function as the stored set. 95% CIs: clip bootstrap (2000, seed 0).

  python scripts/run_contact_in_context.py plan                    (Mac: configs + velocity ids)
  python scripts/run_contact_in_context.py render --root <out>     (box CPU: frames npz, before taking the GPU lock)
  python scripts/run_contact_in_context.py forward --root <out>    (box GPU: flock /tmp/wm_gpu.lock)
  python scripts/run_contact_in_context.py score --root <out>      (Mac CPU: results JSON + figure)
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
from run_temporal_contact import (CT_POINTS, NORMALS, PX_PER_M, R_PX, WALL_PX, bounce_trajectory, render_clip,  # noqa: E402
                                  straight_trajectory, tubelet_phase, turn_fraction)
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_contact_in_context"
RES, FIG = PROJECT_ROOT / "results", PROJECT_ROOT / "figures"
LOCK = "/tmp/wm_gpu.lock"
N_CLIPS = 96
KB_RANGE = (2, 6)
KINDS = ("bounce", "straight_in", "straight_out")
N_PARITY = 8


# ================================================================ pure helpers (tests/test_contact_in_context.py)

def straight_out_trajectory(c, theta_out_deg, speed_mps, k_b, n_frames=16, fps=24.0):
    """Constant velocity along theta_out through the whole clip; centre at the contact point c at frame k_b."""
    return straight_trajectory(c, theta_out_deg, speed_mps, k_b, n_frames, fps)


def trajectory(cfg, kind):
    if kind == "bounce":
        return bounce_trajectory(cfg["contact_xy_m"], cfg["theta_in"], cfg["speed_mps"], cfg["k_b"], cfg["normal"])[0]
    if kind == "straight_in":
        return straight_trajectory(cfg["contact_xy_m"], cfg["theta_in"], cfg["speed_mps"], cfg["k_b"])
    if kind == "straight_out":
        return straight_out_trajectory(cfg["contact_xy_m"], cfg["theta_out"], cfg["speed_mps"], cfg["k_b"])
    raise ValueError(kind)


def sample_bounces_ctx(n, seed=0, speeds=(2.0, 3.0, 4.0), kb_range=KB_RANGE, alpha_range=(20.0, 65.0),
                       margin_px=4.0):
    """As run_temporal_contact.sample_bounces (same geometry rules), with k_b stratified over kb_range (config i gets
    k_b = kb_range[0] + i mod span) and all three trajectories (bounce, straight_in, straight_out) inside the frame."""
    rng = np.random.default_rng(seed)
    half = 128.0 / PX_PER_M
    lim = half - (R_PX + margin_px) / PX_PER_M
    span = kb_range[1] - kb_range[0] + 1
    out = []
    while len(out) < n:
        k_b = kb_range[0] + len(out) % span
        wall = ["left", "right", "bottom", "top"][rng.integers(4)]
        nrm = np.array(NORMALS[wall])
        alpha = rng.uniform(*alpha_range) * rng.choice([-1, 1])
        a = np.radians(alpha)
        tang = np.array([-nrm[1], nrm[0]])
        d_in = -np.cos(a) * nrm + np.sin(a) * tang
        th_in = float(np.degrees(np.arctan2(d_in[1], d_in[0])) % 360.0)
        v = float(rng.choice(speeds))
        c = rng.uniform(-2.5, 2.5, 2)
        xy, th_out = bounce_trajectory(c, th_in, v, k_b, nrm)
        xs = straight_trajectory(c, th_in, v, k_b)
        xo = straight_out_trajectory(c, th_out, v, k_b)
        wall_face = float(c @ nrm - R_PX / PX_PER_M)
        if max(np.abs(xy).max(), np.abs(xs).max(), np.abs(xo).max()) > lim:
            continue
        if abs(wall_face) > half - (WALL_PX + 2) / PX_PER_M:
            continue
        out.append({"wall": wall, "normal": nrm.tolist(), "theta_in": th_in, "theta_out": th_out, "speed_mps": v,
                    "k_b": k_b, "contact_xy_m": c.tolist(), "wall_face": wall_face, "incidence_deg": abs(alpha)})
    return out


def render(cfg, kind):
    """uint8 [16, 256, 256, 3]; bounce / straight_in reuse run_temporal_contact.render_clip unchanged."""
    if kind in ("bounce", "straight_in"):
        return render_clip(cfg, "bounce" if kind == "bounce" else "straight")
    from wm.render_twin import RenderParams, codec_roundtrip, render_rgb, to_pixels
    p = RenderParams()
    return codec_roundtrip(render_rgb(to_pixels(trajectory(cfg, kind)), p), p.bit_rate, p.gop)


# ================================================================ plan (Mac)

def plan():
    from wm.data import load_table
    from wm.provenance import git_commit
    df = load_table("direction")
    vel_ids = df["id"].to_numpy()[df["motion"].to_numpy() == "velocity"]
    cfgs = sample_bounces_ctx(N_CLIPS, seed=0)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", vel_ids=vel_ids, parity_ids=vel_ids[:N_PARITY])
    (ART / "configs.json").write_text(json.dumps({"git": git_commit(), "seed": 0, "kb_range": KB_RANGE,
                                                  "configs": cfgs}, indent=1))
    kb = np.array([c["k_b"] for c in cfgs])
    print("configs", len(cfgs), "k_b counts", {int(k): int((kb == k).sum()) for k in np.unique(kb)},
          "velocity clips", len(vel_ids))


def render_all(root, smoke=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfgs = json.loads((ART / "configs.json").read_text())["configs"]
    if smoke:
        cfgs = cfgs[:4]
    t0 = time.time()
    fr = {k: np.stack([render(c, k) for c in cfgs]) for k in KINDS}
    np.savez(root / "frames.npz", **fr)
    print("rendered", {k: v.shape for k, v in fr.items()}, f"{time.time() - t0:.0f}s", flush=True)


# ================================================================ forward (box GPU)

def forward(root, batch=8, smoke=False):
    import torch
    from wm.data import decode, load_table
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future

    set_precision()
    dev = pick_device()
    root = Path(root)
    P = dict(np.load(ART / "plan.npz"))
    FR = dict(np.load(root / "frames.npz"))
    df = load_table("direction")
    vid = dict(zip(df["id"], df["video"]))
    vel_ids, par_ids = P["vel_ids"], P["parity_ids"]
    if smoke:
        vel_ids = vel_ids[:8]
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    out = {}

    def tp16(model, px):
        """fp32 16-frame timepool at CT_POINTS [B, 5, 8, D] (as run_temporal_contact contact pass)."""
        enc = model.encoder
        h = enc.embeddings(px)
        tp = []
        for i, layer in enumerate(enc.layer):
            h = layer(h, None, None, False)[0]
            if i + 1 in CT_POINTS:
                tp.append(h.reshape(len(px), 8, 256, -1).mean(2))
        tp.append(enc.layernorm(h).reshape(len(px), 8, 256, -1).mean(2))
        return torch.stack(tp, 1).float().cpu().numpy()

    def ctx8(model, px, with_forecast=True):
        """bf16 context-only encode of frames 1-8 (as run_temporal_contact): timepool at CT_POINTS [B, 5, 4, D] and
        the step-pooled forecast [B, 4, D]."""
        enc = model.encoder
        tp = []
        with ac():
            e = enc.embeddings(px[:, :8]).float()
            for i, layer in enumerate(enc.layer):
                e = layer(e, None, None, False)[0]
                if i + 1 in CT_POINTS:
                    tp.append(e.float().reshape(len(px), 4, 256, -1).mean(2))
            fin = enc.layernorm(e)
            tp.append(fin.float().reshape(len(px), 4, 256, -1).mean(2))
            fc = pool_steps(predict_future(model, fin)).float().cpu().numpy() if with_forecast else None
        return torch.stack(tp, 1).float().cpu().numpy(), fc

    with open(LOCK, "a") as lk:
        tw = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - tw
        print(f"lock after {waited:.0f}s", flush=True)
        t0 = time.time()
        timing = {}
        with torch.no_grad():
            model = load_model("vjepa2", dev)
            # ---- rendered clips: 16-frame fp32 timepool + bf16 context timepool + forecast
            for kind in KINDS:
                st = {"tp16": [], "ctx8": [], "forecast": []}
                F = FR[kind]
                for s in range(0, len(F), batch):
                    px = preprocess(list(F[s:s + batch])).to(dev)
                    st["tp16"].append(tp16(model, px))
                    c, f = ctx8(model, px)
                    st["ctx8"].append(c)
                    st["forecast"].append(f)
                for k, v in st.items():
                    out[f"{kind}/{k}"] = np.concatenate(v)
                print(f"{kind} {len(F)} {time.time() - t0:.0f}s", flush=True)
            timing["rendered_vjepa2"] = time.time() - t0
            # ---- velocity clips: context-only timepool (probe features for the ctx8 reader)
            st = []
            for s in range(0, len(vel_ids), 16):
                px = preprocess([decode(vid[i]) for i in vel_ids[s:s + 16]]).to(dev)
                st.append(ctx8(model, px, with_forecast=False)[0])
                if (s // 16) % 10 == 0:
                    print(f"velocity ctx {s + len(px)}/{len(vel_ids)} {time.time() - t0:.0f}s", flush=True)
            out["vel/ctx8"] = np.concatenate(st)
            timing["velocity_ctx8"] = time.time() - t0 - timing["rendered_vjepa2"]
            # ---- parity: supplied clips through both paths (vs stored fp32 timepool and native fp32 forecast cache)
            px = preprocess([decode(vid[i]) for i in par_ids]).to(dev)
            out["parity/tp16"] = tp16(model, px)
            c, f = ctx8(model, px)
            out["parity/ctx8"], out["parity/forecast"] = c, f
            del model
            torch.cuda.empty_cache()
            # ---- untrained random-init copy: 16-frame fp32 timepool only (see module docstring)
            t1 = time.time()
            model = load_model("random", dev)
            for kind in KINDS:
                F = FR[kind]
                out[f"{kind}/tp16_random"] = np.concatenate(
                    [tp16(model, preprocess(list(F[s:s + batch])).to(dev)) for s in range(0, len(F), batch)])
            out["parity/tp16_random"] = tp16(model, preprocess([decode(vid[i]) for i in par_ids]).to(dev))
            timing["random"] = time.time() - t1
            print(f"random {time.time() - t0:.0f}s", flush=True)
        secs = time.time() - t0
    info = {"seconds_gpu_total": secs, "timing_s": timing, "lock_wait_s": waited,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "smoke": smoke,
            "precision": "16-frame timepool fp32 (TF32 off); context encode + forecast bf16 autocast, fp32 residual "
                         "(as run_temporal_contact)",
            "n_clips_per_kind": int(len(FR["bounce"])), "n_velocity_ctx": int(len(vel_ids)),
            "parity_ids": [int(i) for i in par_ids]}
    np.savez(root / "forward.npz", **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", secs, flush=True)


# ================================================================ score (Mac CPU)

def score(root, smoke=False):
    res_path = (ART / "smoke_results.json") if smoke else RES / "p5_contact_in_context.json"
    fig_path = (ART / "smoke_fig.png") if smoke else FIG / "fig_contact_in_context.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_session2 import write
    from run_token_patching import summarize
    from temporal_contact_score import ang, apply, cerr, fit_probe
    from wm.data import load_table
    from wm.probes import load_activations, targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    smoke_info = root / "smoke_forward_info.json"
    smoke_s = json.loads(smoke_info.read_text())["seconds_gpu_total"] if smoke_info.exists() else None
    finfo = json.loads((root / "forward_info.json").read_text())
    P = dict(np.load(ART / "plan.npz"))
    cfgs = json.loads((ART / "configs.json").read_text())["configs"]
    n = len(F["bounce/forecast"])
    cfgs = cfgs[:n]
    df = load_table("direction")
    row = {int(i): k for k, i in enumerate(df["id"])}
    Ydir = targets(df, "direction")[0]
    kb = np.array([c["k_b"] for c in cfgs])
    th_in = np.array([c["theta_in"] for c in cfgs])
    th_out = np.array([c["theta_out"] for c in cfgs])
    target = {"bounce": th_out, "straight_in": th_in, "straight_out": th_out}   # true heading after k_b
    early, late = kb <= 3, kb >= 5
    subsets = {"all": np.ones(n, bool), "early_kb2-3": early, "late_kb5-6": late}

    def tf_block(phi, cols=None):
        """phi [n] read angles -> summaries per subset."""
        tf = turn_fraction(phi, th_in, th_out)
        return {nm: {"turn_fraction": summarize(tf[m]), "err_to_theta_in": summarize(cerr(phi, th_in)[m]),
                     "err_to_theta_out": summarize(cerr(phi, th_out)[m]),
                     "frac_closer_to_out": float((cerr(phi, th_out) < cerr(phi, th_in))[m].mean())}
                for nm, m in subsets.items()}, tf

    # ------------------------------------------------ forecast (step probes on the native fp32 forecast cache)
    Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
    from wm.p2_data import load_inputs
    role = load_inputs("direction", 0)["role"]
    prow = np.where(role == "probe")[0]
    fc_step = [fit_probe(Zall[prow][:, s], Ydir[prow], "circular") for s in range(4)]
    fc = {"step_probe_cv_r2": [i["cv_r2"] for _, i in fc_step], "n_probe_clips": int(len(prow)), "steps": {}}
    tf_fc = {k: np.zeros((n, 4)) for k in KINDS}
    for s in range(4):
        blk = {}
        for kind in KINDS:
            phi = ang(apply(fc_step[s][0], F[f"{kind}/forecast"][:, s]))
            blk[kind], tf_fc[kind][:, s] = tf_block(phi)
            blk[kind]["err_to_true_heading"] = summarize(cerr(phi, target[kind]))
        d = tf_fc["bounce"][:, s] - tf_fc["straight_in"][:, s]
        blk["bounce_minus_straight_in_turn_fraction"] = {nm: summarize(d[m]) for nm, m in subsets.items()}
        fc["steps"][f"step{s}_frames{9 + 2 * s}-{10 + 2 * s}"] = blk
    fc["step_mean_turn_fraction"] = {kind: {nm: summarize(tf_fc[kind][m].mean(1)) for nm, m in subsets.items()}
                                     for kind in KINDS}
    dmean = tf_fc["bounce"].mean(1) - tf_fc["straight_in"].mean(1)
    fc["step_mean_bounce_minus_straight_in"] = {nm: summarize(dmean[m]) for nm, m in subsets.items()}
    # bounce forecast relative to the straight_out reference (1 = forecast as reflected as a clip that only ever
    # moved along theta_out; 0 = as the straight twin); ratio of means with clip bootstrap
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, (2000, n))
    b, si, so = tf_fc["bounce"].mean(1), tf_fc["straight_in"].mean(1), tf_fc["straight_out"].mean(1)
    rel = lambda ix: (b[ix].mean() - si[ix].mean()) / (so[ix].mean() - si[ix].mean())   # noqa: E731
    boots = np.array([rel(ix) for ix in idx])
    fc["bounce_relative_to_references_step_mean"] = {
        "value": float(rel(np.arange(n))), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "definition": "(bounce - straight_in) / (straight_out - straight_in) of the step-mean forecast turn fraction"}
    # parity: bf16 forecast of supplied clips vs the native fp32 cache, read by the same step probes
    prr = np.array([row[int(i)] for i in finfo["parity_ids"]])
    fc["parity_forecast_read_angle_diff_deg_vs_native"] = summarize(np.concatenate(
        [cerr(ang(apply(fc_step[s][0], F["parity/forecast"][:, s])), ang(apply(fc_step[s][0], Zall[prr][:, s])))
         for s in range(4)]))
    fc["parity_forecast_rel_maxabs_vs_native"] = float(np.abs(F["parity/forecast"] - Zall[prr]).max()
                                                       / np.abs(Zall[prr]).max())

    # ------------------------------------------------ encoder readers
    vel = np.where(df["motion"].to_numpy() == "velocity")[0]
    assert np.array_equal(df["id"].to_numpy()[vel], P["vel_ids"])
    TPd = load_activations("direction", "timepool")
    TPr = load_activations("direction", "timepool", model="random")
    phase = np.stack([tubelet_phase(k) for k in kb])                  # [n, 8]
    enc = {"enc16": {}, "ctx8": {}, "random_enc16": {}}
    parity = {"enc16": {}, "random_enc16": {}, "ctx8_vs_stored_16frame_note":
              "context-only encoding differs from full-clip encoding by construction; no stored parity reference"}
    curves = {}
    for reader, src_feat, key, ntub in (("enc16", TPd, "tp16", 8), ("random_enc16", TPr, "tp16_random", 8),
                                        ("ctx8", None, "ctx8", 4)):
        for j, p in enumerate(CT_POINTS):
            R = {"probe_cv_r2_by_tubelet": [], "tubelets": {}}
            tfs = {k: np.full((n, ntub), np.nan) for k in KINDS}
            for t in range(ntub):
                X = (np.asarray(src_feat[vel][:, p, t], np.float32) if src_feat is not None
                     else F["vel/ctx8"][:, j, t])
                pr, info = fit_probe(X, Ydir[vel], "circular")
                R["probe_cv_r2_by_tubelet"].append(info["cv_r2"])
                tb = {}
                for kind in KINDS:
                    phi = ang(apply(pr, F[f"{kind}/{key}"][:, j, t]))
                    tfs[kind][:, t] = turn_fraction(phi, th_in, th_out)
                    tb[kind] = {"turn_fraction": summarize(tfs[kind][:, t]),
                                "err_to_true_heading_at_tubelet": summarize(cerr(phi, np.where(
                                    (kind == "bounce") & (phase[:, t] < 1), th_in, target[kind])))}
                ph = phase[:, t]
                tb["bounce_by_phase"] = {nm: summarize(tfs["bounce"][ph == v, t])
                                         for v, nm in ((-1, "pre"), (0, "straddle"), (1, "post"))}
                R["tubelets"][f"t{t}_frames{2 * t + 1}-{2 * t + 2}"] = tb
            # context tubelets 0-3 pooled by phase (bounce), per subset
            ctxb = tfs["bounce"][:, :4]
            R["bounce_context_by_phase"] = {
                nm: {ph_nm: summarize(ctxb[m][phase[m, :4] == v]) for v, ph_nm in
                     ((-1, "pre"), (0, "straddle"), (1, "post"))} for nm, m in subsets.items()}
            R["bounce_last_context_tubelet_t3"] = {nm: summarize(tfs["bounce"][m, 3]) for nm, m in subsets.items()}
            R["straight_in_last_context_tubelet_t3"] = summarize(tfs["straight_in"][:, 3])
            R["straight_out_last_context_tubelet_t3"] = summarize(tfs["straight_out"][:, 3])
            if ntub == 8:
                R["bounce_real_future_t4-7_mean"] = {nm: summarize(tfs["bounce"][m, 4:].mean(1))
                                                     for nm, m in subsets.items()}
            enc[reader][str(p)] = R
            curves[(reader, p)] = tfs
        if src_feat is not None:
            prr_ = np.array([row[int(i)] for i in finfo["parity_ids"]])
            stored = np.asarray(src_feat[prr_][:, list(CT_POINTS)], np.float32)
            got = F["parity/tp16" if reader == "enc16" else "parity/tp16_random"]
            parity[reader] = {str(p): float(np.abs(got[:, j] - stored[:, j]).max() / np.abs(stored[:, j]).max())
                              for j, p in enumerate(CT_POINTS)}

    res = {"question": "when the wall bounce happens inside the context window (frames 1-8 show the reflected "
                       "motion), does the predictor forecast continue the reflected heading or revert to the incoming "
                       "one?",
           "data": {"rendered": f"{n} configurations x 3 kinds (bounce, straight_in, straight_out) with the "
                                "run_temporal_contact renderer (wm.render_twin RenderParams, MPEG-4 codec round trip, "
                                "grey 6 px wall bar on bounce clips only), elastic reflection at k_b in 2..6 "
                                "(0-indexed frame; frames k_b..7 of the context are post-contact), speeds 2/3/4 m/s, "
                                "incidence 20-65 deg from the normal",
                    "configs_file": "artifacts/p5_contact_in_context/configs.json",
                    "k_b_counts": {str(k): int((kb == k).sum()) for k in np.unique(kb)},
                    "n_early_kb2-3": int(early.sum()), "n_late_kb5-6": int(late.sum()),
                    "stored_counterpart": "results/p5_contact_dynamics.json (contact in the forecast window)"},
           "definitions": {
               "turn_fraction": "signed angular move of the read angle from theta_in along the shorter arc toward "
                                "theta_out, over that arc (0 = incoming, 1 = reflected); same function and "
                                "(theta_in, theta_out) pair for all three kinds",
               "straight_in": "primary control: same start, speed, theta_in throughout, no wall (turn fraction "
                              "expected ~0)",
               "straight_out": "reference: constant velocity along theta_out throughout, no wall (what the reader "
                               "gives for pure theta_out motion; expected ~1)",
               "forecast": "bf16 context-only encode (frames 1-8) + predictor (mask index 0), step-pooled; read by "
                           "per-future-step ridge probes fit on the native fp32 forecast cache of the 480 probe-role "
                           "direction clips; steps 0-3 = frames 9-10 ... 15-16 (1-indexed), all post-contact",
               "enc16": "per-tubelet probes on the stored fp32 16-frame timepool of the 750 constant-velocity clips "
                        "(same readers as p5_contact_dynamics (a)); full-clip attention sees frames 9-16 too",
               "ctx8": "per-tubelet probes fit here on the bf16 context-only encoding of the same 750 velocity "
                       "clips, applied to the rendered clips' context-only encoding (what the predictor reads)",
               "random_enc16": "enc16 with the random-init copy (seed 0) and probes on its stored timepool; the "
                               "forecast is not run with the random-init encoder (trained predictor, untrained "
                               "inputs: the stored forecast readers do not apply)",
               "phase": "tubelet t (frames 2t, 2t+1, 0-indexed): pre if 2t+1 < k_b, post if 2t >= k_b, else straddle",
               "ci95": "clip bootstrap, 2000 resamples, seed 0 (run_token_patching.summarize)"},
           "results": {"forecast": fc, "encoder": enc, "parity_fp32_timepool_rel_maxabs_vs_stored": parity},
           "config": {"points": CT_POINTS, "n_clips_per_kind": n, "kb_range": KB_RANGE, "forward": finfo}}
    write(res_path, res, seeds={"configs": 0, "folds": 0, "bootstrap": 0},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz"),
                             "frames_npz": sha256_file(root / "frames.npz"),
                             "configs_json": sha256_file(ART / "configs.json")},
          script="scripts/run_contact_in_context.py",
          script_sha256=sha256_file(Path(__file__)),
          forward_script_sha256=((root / "forward_script_sha256.txt").read_text().split()[0]
                                 if (root / "forward_script_sha256.txt").exists() else None), box="Vast 53255819 (box 4, RTX 4060 Ti)",
          gpu_seconds={"forward": finfo["seconds_gpu_total"], "smoke": smoke_s},
          n_clips={"per_kind": n, "kinds": list(KINDS), "velocity_probe_clips": int(len(vel))},
          contact_frame_range_0idx=list(KB_RANGE))

    # ------------------------------------------------ figure
    cols = {"bounce": "C3", "straight_in": "C0", "straight_out": "C2"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    steps = sorted(fc["steps"])
    for kind in KINDS:
        m = [fc["steps"][k][kind]["all"]["turn_fraction"]["mean"] for k in steps]
        ci = np.array([fc["steps"][k][kind]["all"]["turn_fraction"]["ci95"] for k in steps])
        ax[0].errorbar(range(4), m, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], marker="o", capsize=3,
                       c=cols[kind], label=kind)
    for nm, ls in (("early_kb2-3", ":"), ("late_kb5-6", "--")):
        ax[0].plot(range(4), [fc["steps"][k]["bounce"][nm]["turn_fraction"]["mean"] for k in steps], ls, c="C3",
                   alpha=.7, label=f"bounce {nm}")
    ax[0].set_xticks(range(4), [f"step {s}\nframes {9 + 2 * s}-{10 + 2 * s}" for s in range(4)], fontsize=8)
    ax[0].set_title("predictor forecast (context frames 1-8 contain the bounce)")
    for p_, a, rd in ((22, ax[1], "ctx8"), (22, ax[2], "enc16")):
        T = curves[(rd, p_)]
        nt = T["bounce"].shape[1]
        for kind in KINDS:
            mu = np.nanmean(T[kind], 0)
            a.plot(np.arange(nt), mu, "o-", c=cols[kind], label=kind)
        for nm, ls in (("early_kb2-3", ":"), ("late_kb5-6", "--")):
            a.plot(np.arange(nt), np.nanmean(T["bounce"][subsets[nm]], 0), ls, c="C3", alpha=.7, label=f"bounce {nm}")
        a.set_xticks(range(nt), [f"{2 * t + 1}-{2 * t + 2}" for t in range(nt)], fontsize=8)
        a.set_xlabel("frames (tubelet)")
        if rd == "enc16":
            a.plot(np.arange(nt), np.nanmean(curves[("random_enc16", 22)]["bounce"], 0), "x-", c="grey",
                   label="bounce, random-init (pt 22)")
            a.axvspan(3.5, 7.5, color="grey", alpha=.08)
        a.set_title(f"encoder point {p_}, {'context-only (frames 1-8)' if rd == 'ctx8' else 'full 16-frame clip'}")
    for a in ax:
        a.axhline(0, c="k", lw=.5); a.axhline(1, c="k", lw=.5, ls="--")
        a.set_ylabel("turn fraction (0 = incoming, 1 = reflected)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fig_path, dpi=130); plt.close(fig)
    print("figure written")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "render", "forward", "score"])
    ap.add_argument("--root", default=str(ART / "out"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan()
    elif a.cmd == "render":
        render_all(a.root, smoke=a.smoke)
    elif a.cmd == "forward":
        forward(a.root, smoke=a.smoke)
    else:
        score(a.root, smoke=a.smoke)


if __name__ == "__main__":
    main()

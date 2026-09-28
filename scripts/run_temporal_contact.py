"""Temporal locality and contact dynamics of the V-JEPA 2 direction code (two planned-never-launched Part 2 items).

1. TEMPORAL LOCALITY. (a) Correlational, CPU, local: ridge probes per tubelet t = 0..7 (frames 2t+1, 2t+2 of the
   16-frame clip, i.e. each frame pair; spatial mean of the 256 tokens of that tubelet = the stored `timepool`) at
   points 1, 8, 12, 22; 5-fold CV R^2 for direction (direction set, 1500 clips) and speed (speed set, 1536 clips).
   Caveat: the stored activations are full-clip encodings, and encoder attention is bidirectional, so tubelet t's
   tokens have already attended to every frame; (a) measures where the code is readable, not where it came from.
   (b) Causal, GPU: mean-ablate (replace by the per-position mean over the 480 probe-role clips) the tokens of a
   frame group at the input to block 12 (point 12), finish the encoder and read out:
     - predictor forecast (session-2 readout: frames 1-8 encoded alone = tubelets 0-3, predictor forecasts tubelets
       4-7), groups ctx_first = frames 1-2 (tubelet 0), ctx_middle = frames 3-6 (tubelets 1-2), ctx_last = frames
       7-8 (tubelet 3), ctx_all; the 16-frame {first 4, middle 8, last 4} split scaled to the 8-frame context.
       Reader: step-mean ridge direction probe fit on the fp32 native forecast cache of the probe clips (as in
       scripts/run_token_patching.py).
     - encoder output (post-LN meanpool of the full 16-frame clip), groups first4 = tubelets 0-1, middle8 = 2-5,
       last4 = 6-7, plus each single tubelet. Reader: ridge direction probe fit on the unablated box encoder output
       of the probe clips.
   Eval clips: 200 knot/test-role direction clips (seed 0), disjoint from the probe clips.

2. CONTACT DYNAMICS. The supplied sets have no walls or bounces (constant velocity / constant acceleration only), so
   96 wall-bounce clips are rendered with wm.render_twin (same radius, colours, codec) plus a grey wall bar
   (6 px thick, full image length) and an elastic reflection (angle in = angle out) at contact frame k_b in 3..12;
   96 straight twins continue the incoming direction with no wall. Speeds 2-4 m/s (velocity-clip values).
   (a) per-tubelet readout: per-tubelet direction probes fit on the constant-velocity supplied clips (timepool,
       points 1/8/12/22/25) applied to the same tubelet of the bounce clips; turn fraction = signed angular move of
       the read angle from theta_in toward theta_out over the in->out arc (0 = still reads the incoming direction,
       1 = reads the reflected one), by tubelet phase (pre / straddling / post contact).
   (b) predictor: bounce clips with contact in the forecast window (k_b >= 8, frames 0-7 all pre-contact), per-step
       forecast direction probes (fit per future step on the native cache) -> does the forecast turn to theta_out
       at the steps after contact? Reference: the encoder's own real-future tubelets (from (a), post-LN).

  python scripts/run_temporal_contact.py plan                 (local; writes artifacts/p5_temporal_contact/plan.npz)
  python scripts/run_temporal_contact.py forward --root <out>  (box GPU; flock /tmp/wm_gpu.lock)
  python scripts/run_temporal_contact.py score --root <out>    (local CPU; both results JSONs + figures)
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
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_temporal_contact"
RES, FIG = PROJECT_ROOT / "results", PROJECT_ROOT / "figures"
LOCK = "/tmp/wm_gpu.lock"
POINTS = (1, 8, 12, 22)
CT_POINTS = (1, 8, 12, 22, 25)
ABL_POINT = 12
N_EVAL = 200
N_BOUNCE = 96
PX_PER_M, FPS, R_PX, WALL_PX = 32.0, 24.0, 10.5, 6.0
WALL_RGB = (150, 150, 150)
CTX_GROUPS = {"ctx_first_f1-2": (0,), "ctx_middle_f3-6": (1, 2), "ctx_last_f7-8": (3,), "ctx_all_f1-8": (0, 1, 2, 3)}
FULL_GROUPS = {"first4_f1-4": (0, 1), "middle8_f5-12": (2, 3, 4, 5), "last4_f13-16": (6, 7),
               **{f"tubelet{t}_f{2 * t + 1}-{2 * t + 2}": (t,) for t in range(8)}}
NORMALS = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "bottom": (0.0, 1.0), "top": (0.0, -1.0)}


# ================================================================ pure helpers (tests/test_temporal_contact.py)

def wrap_signed(d):
    d = (np.asarray(d, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def reflect(d, n):
    """Reflect direction vector(s) d [.., 2] off a wall with unit normal n [2]."""
    d, n = np.asarray(d, float), np.asarray(n, float)
    return d - 2.0 * (d @ n)[..., None] * n


def bounce_trajectory(c, theta_in_deg, speed_mps, k_b, normal, n_frames=16, fps=FPS):
    """World positions [n_frames, 2] (m): constant speed along theta_in until the contact frame k_b (the centre is at
    the contact point c exactly at frame k_b), then along the reflected direction. Returns (xy, theta_out_deg)."""
    th = np.radians(theta_in_deg)
    d_in = np.array([np.cos(th), np.sin(th)])
    d_out = reflect(d_in, normal)
    s = (np.arange(n_frames) - k_b) * speed_mps / fps
    xy = np.where((s < 0)[:, None], np.asarray(c)[None] + s[:, None] * d_in, np.asarray(c)[None] + s[:, None] * d_out)
    return xy, float(np.degrees(np.arctan2(d_out[1], d_out[0])) % 360.0)


def straight_trajectory(c, theta_in_deg, speed_mps, k_b, n_frames=16, fps=FPS):
    th = np.radians(theta_in_deg)
    s = (np.arange(n_frames) - k_b) * speed_mps / fps
    return np.asarray(c)[None] + s[:, None] * np.array([np.cos(th), np.sin(th)])[None]


def tubelet_phase(k_b, n_steps=8):
    """Per tubelet t (frames 2t, 2t+1, 0-indexed): -1 = both frames before contact (index < k_b), +1 = both at/after
    contact (index >= k_b), 0 = straddles."""
    f0, f1 = 2 * np.arange(n_steps), 2 * np.arange(n_steps) + 1
    return np.where(f1 < k_b, -1, np.where(f0 >= k_b, 1, 0))


def turn_fraction(phi, th_in, th_out):
    """Signed angular move of phi from th_in along the shorter arc toward th_out, over that arc's length."""
    arc = wrap_signed(np.asarray(th_out, float) - th_in)
    mv = wrap_signed(np.asarray(phi, float) - th_in)
    return mv * np.where(arc >= 0, 1.0, -1.0) / np.maximum(np.abs(arc), 1e-9)


def ablate(h, token_idx, fill):
    """Replace tokens token_idx of h [B, N, D] by fill [N, D] (same positions); returns a copy (numpy or torch)."""
    out = h.clone() if hasattr(h, "clone") else np.array(h, copy=True)
    out[:, token_idx] = fill[token_idx]
    return out


def tubelet_tokens(tubelets, grid=16):
    """Token indices (t, h, w order) of the given tubelets."""
    return np.concatenate([np.arange(t * grid * grid, (t + 1) * grid * grid) for t in tubelets])


def sample_bounces(n, seed=0, speeds=(2.0, 3.0, 4.0), kb_range=(3, 12), alpha_range=(20.0, 65.0), margin_px=4.0):
    """n bounce configurations whose disk stays fully inside the frame; returns a list of dicts."""
    rng = np.random.default_rng(seed)
    half = 128.0 / PX_PER_M
    lim = half - (R_PX + margin_px) / PX_PER_M
    out = []
    while len(out) < n:
        wall = ["left", "right", "bottom", "top"][rng.integers(4)]
        nrm = np.array(NORMALS[wall])
        alpha = rng.uniform(*alpha_range) * rng.choice([-1, 1])
        a = np.radians(alpha)
        tang = np.array([-nrm[1], nrm[0]])
        d_in = -np.cos(a) * nrm + np.sin(a) * tang
        th_in = float(np.degrees(np.arctan2(d_in[1], d_in[0])) % 360.0)
        v = float(rng.choice(speeds))
        k_b = int(rng.integers(kb_range[0], kb_range[1] + 1))
        c = rng.uniform(-2.5, 2.5, 2)
        xy, th_out = bounce_trajectory(c, th_in, v, k_b, nrm)
        xs = straight_trajectory(c, th_in, v, k_b)
        wall_face = float(c @ nrm - R_PX / PX_PER_M)          # wall face: points p with p.n = wall_face
        if np.abs(xy).max() > lim or np.abs(xs).max() > lim:
            continue
        if abs(wall_face) > half - (WALL_PX + 2) / PX_PER_M:
            continue
        out.append({"wall": wall, "normal": nrm.tolist(), "theta_in": th_in, "theta_out": th_out, "speed_mps": v,
                    "k_b": k_b, "contact_xy_m": c.tolist(), "wall_face": wall_face, "incidence_deg": abs(alpha)})
    return out


# ================================================================ rendering (box or local)

def wall_mask(cfg):
    from wm.render_twin import ORIGIN
    col, row = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)
    x, y = (col - 0.5 - ORIGIN) / PX_PER_M, (ORIGIN - (row - 0.5)) / PX_PER_M   # pixel-index -> world (render_twin)
    s = x * cfg["normal"][0] + y * cfg["normal"][1]
    return (s <= cfg["wall_face"]) & (s > cfg["wall_face"] - WALL_PX / PX_PER_M)


def render_clip(cfg, kind):
    from wm.render_twin import RenderParams, codec_roundtrip, render_rgb, to_pixels
    p = RenderParams()
    if kind == "bounce":
        xy, _ = bounce_trajectory(cfg["contact_xy_m"], cfg["theta_in"], cfg["speed_mps"], cfg["k_b"], cfg["normal"])
    else:
        xy = straight_trajectory(cfg["contact_xy_m"], cfg["theta_in"], cfg["speed_mps"], cfg["k_b"])
    fr = render_rgb(to_pixels(xy), p)
    if kind == "bounce":
        fr[:, wall_mask(cfg)] = np.array(WALL_RGB, np.uint8)
    return codec_roundtrip(fr, p.bit_rate, p.gop)


# ================================================================ plan (local)

def plan():
    from wm.data import load_table
    from wm.p2_data import load_inputs
    from wm.provenance import git_commit
    df = load_table("direction")
    role = load_inputs("direction", 0)["role"]
    ids = df["id"].to_numpy()
    rng = np.random.default_rng(0)
    eval_ids = np.sort(rng.choice(ids[role != "probe"], N_EVAL, replace=False))
    cfgs = sample_bounces(N_BOUNCE, seed=0)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", eval_ids=eval_ids, probe_ids=ids[role == "probe"])
    (ART / "bounces.json").write_text(json.dumps({"git": git_commit(), "configs": cfgs}, indent=1))
    kb = np.array([c["k_b"] for c in cfgs])
    print("eval", len(eval_ids), "bounces", len(cfgs), "k_b>=8:", int((kb >= 8).sum()))


# ================================================================ forward (box GPU)

def forward(root, batch=8, smoke=False):
    import torch
    from wm.data import decode, load_table
    from wm.extract import load_model, pick_device, preprocess, set_precision
    from wm.predictor_readout import pool_steps, predict_future
    from wm.propagate import suffix

    set_precision()
    dev = pick_device()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    P = dict(np.load(ART / "plan.npz"))
    cfgs = json.loads((ART / "bounces.json").read_text())["configs"]
    df = load_table("direction")
    vid = dict(zip(df["id"], df["video"]))
    probe_ids, eval_ids = P["probe_ids"], P["eval_ids"]
    if smoke:
        probe_ids, eval_ids, cfgs = probe_ids[:16], eval_ids[:8], cfgs[:8]
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    out = {}

    def enc_to(pv, point):
        enc = model.encoder
        h = enc.embeddings(pv).float()
        for layer in enc.layer[:point]:
            h = layer(h, None, None, False)[0]
        return h

    with open(LOCK, "a") as lk:
        tw = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - tw
        print(f"lock after {waited:.0f}s", flush=True)
        t0 = time.time()
        model = load_model("vjepa2", dev)
        # ---- probe clips: unablated encoder output (16 frames) + point-12 per-position means (8- and 16-frame)
        s8, s16, enc_mp = 0, 0, []
        for s in range(0, len(probe_ids), 16):
            px = preprocess([decode(vid[i]) for i in probe_ids[s:s + 16]]).to(dev)
            with torch.no_grad(), ac():
                h16 = enc_to(px, ABL_POINT)
                h8 = enc_to(px[:, :8], ABL_POINT)
                r = suffix(model, h16, ABL_POINT, None, return_final=True)
            s16 = s16 + h16.double().sum(0)
            s8 = s8 + h8.double().sum(0)
            enc_mp.append(r["final"].float().mean(1).cpu().numpy())
        fill16, fill8 = (s16 / len(probe_ids)).float(), (s8 / len(probe_ids)).float()
        out["probe_enc_mp"] = np.concatenate(enc_mp)
        print(f"probe pass {time.time() - t0:.0f}s", flush=True)
        # ---- eval clips: frame-group mean ablation at point 12
        st = {}
        for s in range(0, len(eval_ids), batch):
            px = preprocess([decode(vid[i]) for i in eval_ids[s:s + batch]]).to(dev)
            with torch.no_grad(), ac():
                h8 = enc_to(px[:, :8], ABL_POINT)
                h16 = enc_to(px, ABL_POINT)
                for name, grp in [("none", ())] + list(CTX_GROUPS.items()):
                    h = ablate(h8, torch.as_tensor(tubelet_tokens(grp), device=dev), fill8) if grp else h8
                    r = suffix(model, h, ABL_POINT, None, return_final=True)
                    st.setdefault(f"ctx/{name}", []).append(pool_steps(predict_future(model, r["final"])).float().cpu().numpy())
                for name, grp in [("none", ())] + list(FULL_GROUPS.items()):
                    h = ablate(h16, torch.as_tensor(tubelet_tokens(grp), device=dev), fill16) if grp else h16
                    r = suffix(model, h, ABL_POINT, None, return_final=True)
                    st.setdefault(f"full/{name}", []).append(r["final"].float().mean(1).cpu().numpy())
            print(f"eval {s + len(px)}/{len(eval_ids)} {time.time() - t0:.0f}s", flush=True)
        for k, v in st.items():
            out[f"abl/{k}"] = np.concatenate(v)
        t_abl = time.time() - t0
        # ---- contact: rendered bounce + straight clips, fp32 timepool at CT_POINTS + bf16 forecast from frames 1-8
        st = {}
        for kind in ("bounce", "straight"):
            for s in range(0, len(cfgs), batch):
                frames = [render_clip(c, kind) for c in cfgs[s:s + batch]]
                if s == 0:
                    np.save(root / f"example_{kind}.npy", np.stack(frames[:4]))
                px = preprocess(frames).to(dev)
                with torch.no_grad():
                    enc = model.encoder
                    h = enc.embeddings(px)
                    tp = []
                    for i, layer in enumerate(enc.layer):
                        h = layer(h, None, None, False)[0]
                        if i + 1 in CT_POINTS:
                            tp.append(h.reshape(len(px), 8, 256, -1).mean(2))
                    fin = enc.layernorm(h)
                    tp.append(fin.reshape(len(px), 8, 256, -1).mean(2))
                    st.setdefault(f"{kind}/timepool", []).append(torch.stack(tp, 1).float().cpu().numpy())
                    with ac():
                        e = enc.embeddings(px[:, :8]).float()
                        for layer in enc.layer:
                            e = layer(e, None, None, False)[0]
                        st.setdefault(f"{kind}/forecast", []).append(
                            pool_steps(predict_future(model, enc.layernorm(e))).float().cpu().numpy())
                print(f"contact {kind} {s + len(px)}/{len(cfgs)} {time.time() - t0:.0f}s", flush=True)
        for k, v in st.items():
            out[f"ct/{k}"] = np.concatenate(v)
        # parity: 8 supplied clips through the same fp32 contact path, compare with stored timepool
        px = preprocess([decode(vid[i]) for i in eval_ids[:8]]).to(dev)
        with torch.no_grad():
            h = model.encoder.embeddings(px)
            tp = []
            for i, layer in enumerate(model.encoder.layer):
                h = layer(h, None, None, False)[0]
                if i + 1 in CT_POINTS:
                    tp.append(h.reshape(len(px), 8, 256, -1).mean(2))
            tp.append(model.encoder.layernorm(h).reshape(len(px), 8, 256, -1).mean(2))
        out["parity_timepool"] = torch.stack(tp, 1).float().cpu().numpy()
        out["parity_ids"] = eval_ids[:8]
        secs = time.time() - t0
    info = {"seconds_gpu_total": secs, "seconds_ablation": t_abl, "lock_wait_s": waited,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "smoke": smoke,
            "precision": "ablation + forecasts: bf16 autocast, fp32 residual; contact timepool: fp32 (TF32 off)",
            "n_probe": int(len(probe_ids)), "n_eval": int(len(eval_ids)), "n_bounce": len(cfgs)}
    np.savez(root / "forward.npz", **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1))
    print("done", secs, flush=True)


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
        from temporal_contact_score import score
        score(a.root)


if __name__ == "__main__":
    main()

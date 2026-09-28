"""Token-source patching: which context tokens does the predictor read direction from, by depth?

The physics paper's future-work hypothesis (refs/physics_paper.txt lines 795-798): velocity information is "bound" to
the object in the middle layers; at the end of the network it is "catered to" next-frame latent prediction. Test at
the predictor: pairs (A, B) of supplied direction clips, same motion type and magnitude, directions >= 90 deg apart.
Session-2 predictor readout (frames 1-8 encoded alone, blocks p+1..24, final LN, predictor with mask token 0,
per-step pooled forecast), bf16 autocast forward with an fp32 residual stream, fp32 scoring. At point p run A but
replace a token subset of A's context residual stream by B's tokens at the same point, then finish the encoder and
the predictor.

Token sets (context tubelets 0-3 = tokens 0..1023, (t, h, w) order; disk mask = wm.data.disk_mask, a patch is
disk if any disk pixel falls in it in either frame of its tubelet; ring = 3x3 spatial dilation per tubelet):
  obj  = disk(A) | disk(B) + ring    (primary: the union, so obj/bg partition the tokens and B's object tokens
                                      are not left in the background set)
  bg   = complement of obj
  objA / bgA = A's disk + ring only / its complement (the literal "A's disk tokens" design)
  objN / bgN = disk(A) | disk(B) without the ring / its complement
  none = A unpatched, all = B unpatched (patching every token at p is B's own forward).
Pair types: 'random' (same bin, >= 90 deg, starts unconstrained) and 'posmatch' (same bin, >= 90 deg, start
positions <= 0.15 m apart = 4.8 px; the supplied direction set has 1500 distinct starts, so the 7-shared-start layout
exists only in the rendered sets, and nearest-start pairs are the position-matched design here).
Pairs use knot + test clips only; the direction probes are fit on the 480 probe-role clips (disjoint).

Readouts: forecast direction probe (session2_native_readout step-mean ridge probe, fit on the fp32 native forecast
cache of the probe clips) and encoder-output direction probe (ridge on the context-only post-LN meanpool of the
same probe clips, point 25). Binding fraction of a condition c = <v_c - v_A, v_B - v_A> / ||v_B - v_A||^2 with v the
probe's (sin, cos) output (per pair; 1 = moved all the way to B, 0 = stayed at A), plus the angle version and the
ratio of pair means Delta_c / Delta_all.

  python scripts/run_token_patching.py plan                           (local CPU)
  python scripts/run_token_patching.py forward --root <out> [--smoke]  (box GPU; takes flock /tmp/wm_gpu.lock)
  python scripts/run_token_patching.py score --root <out>              (local CPU; results + figure)
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

ART = PROJECT_ROOT / "artifacts" / "p5_token_patching"
RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
D = 1024
CTX_STEPS, GRID = 4, 16
N_CTX = CTX_STEPS * GRID * GRID
POINTS = (0, 8, 12, 16, 22)          # 0 = patch embedding (control: bg carries no B-specific content there)
MIN_SEP_DEG = 90.0
POS_MAX_M = 0.15
N_PAIRS = 64
LOCK = "/tmp/wm_gpu.lock"
PAIR_TYPES = ("random", "posmatch")
COND_ALL_POINTS = ("obj", "bg")
COND_EXTRA = ("objA", "bgA", "objN", "bgN")   # at points 8..22 only


def conditions(p):
    return COND_ALL_POINTS + (COND_EXTRA if p > 0 else ())


# ================================================================ pure helpers (tests/test_token_patching.py)

def wrap_signed(d):
    """Signed angle difference in (-180, 180]."""
    d = (np.asarray(d, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def dilate(m):
    """3x3 spatial dilation of bool [..., 16, 16] (per leading index; no wrap-around)."""
    m = np.asarray(m, bool)
    out = m.copy()
    H, W = m.shape[-2:]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            src = m[..., max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
            out[..., max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)] |= src
    return out


def token_sets(mask_a, mask_b):
    """mask_a, mask_b: bool [8, 16, 16] (wm.data.disk_mask) -> dict of bool [1024] context token sets
    (tubelets 0-3 flattened in (t, h, w) order, the token order of the encoder)."""
    a = np.asarray(mask_a, bool)[:CTX_STEPS]
    b = np.asarray(mask_b, bool)[:CTX_STEPS]
    obj = dilate(a | b).reshape(-1)
    objA = dilate(a).reshape(-1)
    objN = (a | b).reshape(-1)
    return {"obj": obj, "bg": ~obj, "objA": objA, "bgA": ~objA, "objN": objN, "bgN": ~objN}


def vec_fraction(v_c, v_a, v_b, eps=1e-12):
    """Projection fraction of the move A -> B produced by condition c (rows = pairs, last dim = probe outputs)."""
    a = np.asarray(v_c, float) - v_a
    b = np.asarray(v_b, float) - v_a
    return (a * b).sum(-1) / np.maximum((b * b).sum(-1), eps)


def ang_fraction(phi_c, phi_a, phi_b):
    """Signed angular displacement of phi_c from phi_a along the shorter arc to phi_b, over that arc's length."""
    arc = wrap_signed(np.asarray(phi_b, float) - phi_a)
    mv = wrap_signed(np.asarray(phi_c, float) - phi_a)
    s = np.where(arc >= 0, 1.0, -1.0)
    return mv * s / np.maximum(np.abs(arc), 1e-9)


def summarize(v, n=2000, seed=0):
    """Mean and pair-bootstrap 95% CI (NaNs dropped)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def ratio_of_means(num, den, n=2000, seed=0):
    """mean(num) / mean(den) with a pair-bootstrap 95% CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].mean(1) / den[idx].mean(1)
    return {"ratio": float(num.mean() / den.mean()), "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "n": int(len(num))}


def pick_pairs(ids, theta, mag_key, xy, kind, n, seed=0, min_sep=MIN_SEP_DEG, pos_max=POS_MAX_M):
    """Disjoint pairs (each clip used once). kind 'random': uniform over admissible partners; 'posmatch': greedy by
    start distance (<= pos_max). Admissible = same mag_key (motion + magnitude), |dtheta| >= min_sep."""
    rng = np.random.default_rng(seed)
    ids, theta, mag_key, xy = map(np.asarray, (ids, theta, mag_key, xy))
    dth = np.abs(wrap_signed(theta[:, None] - theta[None]))
    ok = (mag_key[:, None] == mag_key[None]) & (dth >= min_sep)
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    used, pairs = np.zeros(len(ids), bool), []
    if kind == "posmatch":
        cand = np.argwhere(np.triu(ok & (dist <= pos_max), 1))
        cand = cand[np.argsort(dist[cand[:, 0], cand[:, 1]], kind="stable")]
        for i, j in cand:
            if not used[i] and not used[j]:
                used[[i, j]] = True
                pairs.append((i, j) if rng.random() < 0.5 else (j, i))
    else:
        for i in rng.permutation(len(ids)):
            if used[i]:
                continue
            js = np.where(ok[i] & ~used)[0]
            js = js[js != i]
            if len(js):
                j = rng.choice(js)
                used[[i, j]] = True
                pairs.append((i, j))
    pairs = pairs[:n]
    return np.array([[ids[i], ids[j]] for i, j in pairs], int).reshape(-1, 2), \
        np.array([dist[i, j] for i, j in pairs]), np.array([dth[i, j] for i, j in pairs])


# ================================================================ plan (CPU, local)

def plan():
    from wm.data import decode, disk_mask, load_table
    from wm.p2_data import load_inputs
    from wm.provenance import git_commit

    df = load_table("direction")
    d0 = load_inputs("direction", 0)
    role = d0["role"]
    mag = np.where(df["motion"] == "velocity", df["speed_mps"], df["acceleration_mps2"])
    key = np.array([f"{m}:{v:g}" for m, v in zip(df["motion"], mag)])
    ok = role != "probe"
    ids = df["id"].to_numpy()
    out, meta = {}, {"pair_rule": {"same": "motion type and magnitude (speed or acceleration value)",
                                   "min_sep_deg": MIN_SEP_DEG, "posmatch_max_start_dist_m": POS_MAX_M,
                                   "px_per_m": 32.07, "clips": "knot + test roles (probe role held out for the probes)"},
                     "git": git_commit()}
    for t in PAIR_TYPES:
        pr, dist, dth = pick_pairs(ids[ok], df["theta_degrees"].to_numpy()[ok], key[ok],
                                   df[["start_x", "start_y"]].to_numpy()[ok], t, N_PAIRS, seed=0)
        out[f"pairs_{t}"], out[f"start_dist_{t}"], out[f"dtheta_{t}"] = pr, dist, dth
        meta[t] = {"n_pairs": int(len(pr)), "start_dist_m": [float(dist.min()), float(np.median(dist)), float(dist.max())],
                   "dtheta_deg": [float(dth.min()), float(np.median(dth)), float(dth.max())]}
    need = np.unique(np.concatenate([out[f"pairs_{t}"].ravel() for t in PAIR_TYPES]))
    vid = dict(zip(df["id"], df["video"]))
    masks = np.stack([disk_mask(decode(vid[i]))[0] for i in need])
    out["mask_ids"], out["masks"] = need, masks
    for t in PAIR_TYPES:
        row = {i: k for k, i in enumerate(need)}
        fr, iou, empty = [], [], 0
        for a, b in out[f"pairs_{t}"]:
            s = token_sets(masks[row[a]], masks[row[b]])
            ma, mb = masks[row[a]][:CTX_STEPS].ravel(), masks[row[b]][:CTX_STEPS].ravel()
            empty += int((~masks[row[a]][:CTX_STEPS].any((1, 2))).any() or (~masks[row[b]][:CTX_STEPS].any((1, 2))).any())
            fr.append([s[k].mean() for k in ("obj", "objA", "objN")])
            iou.append((ma & mb).sum() / max((ma | mb).sum(), 1))
        fr, iou = np.array(fr), np.array(iou)
        meta[t].update({"token_frac_mean": dict(zip(("obj", "objA", "objN"), fr.mean(0).round(4).tolist())),
                        "disk_mask_iou_ctx": [float(iou.min()), float(np.median(iou)), float(iou.max())],
                        "pairs_with_an_empty_ctx_tubelet": empty})
    out["probe_ids"] = ids[role == "probe"]
    ART.mkdir(parents=True, exist_ok=True)
    np.savez(ART / "plan.npz", **out)
    (ART / "plan.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


# ================================================================ forward (GPU, box)

def _prefix_f32(model, pv, points):
    """Context encoder pass with an fp32 residual stream (bf16 matmuls under autocast); saves the residual at
    `points` (0 = patch embedding) and returns (saved, post-LN tokens, post-LN meanpool fp32)."""
    enc = model.encoder
    h = enc.embeddings(pv).float()
    saved = {0: h} if 0 in points else {}
    for i, layer in enumerate(enc.layer):
        h = layer(h, None, None, False)[0]
        if i + 1 in points:
            saved[i + 1] = h
    final = enc.layernorm(h)
    return saved, final, final.float().mean(1)


def forward(root, batch=8, smoke=False, model_kind="vjepa2"):
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
    df = load_table("direction")
    vid = dict(zip(df["id"], df["video"]))
    mrow = {int(i): k for k, i in enumerate(P["mask_ids"])}
    probe_ids = P["probe_ids"][:16] if smoke else P["probe_ids"]
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)           # noqa: E731
    out = {}
    with open(LOCK, "a") as lk:
        t_wait = time.time()
        fcntl.flock(lk, fcntl.LOCK_EX)
        waited = time.time() - t_wait
        print(f"lock acquired after {waited:.0f}s", flush=True)
        t0 = time.time()
        model = load_model(model_kind, dev)      # "random" = VJEPA2Config + torch.manual_seed(0) (encoder AND predictor)
        # ---- encoder-output probe inputs + bf16 forecasts of the probe clips
        enc_mp, fc = [], []
        for s in range(0, len(probe_ids), 16):
            pv = preprocess([decode(vid[i]) for i in probe_ids[s:s + 16]]).to(dev)[:, :8]
            with torch.no_grad(), ac():
                _, final, mp = _prefix_f32(model, pv, ())
                fc.append(pool_steps(predict_future(model, final)).float().cpu().numpy())
            enc_mp.append(mp.cpu().numpy())
        out["probe_enc_mp"], out["probe_pred"] = np.concatenate(enc_mp), np.concatenate(fc)
        t_probe = time.time() - t0
        print(f"probe clips {len(probe_ids)} {t_probe:.0f}s", flush=True)
        for t in PAIR_TYPES:
            pairs = P[f"pairs_{t}"][:batch] if smoke else P[f"pairs_{t}"]
            store = {}
            for s in range(0, len(pairs), batch):
                pb = pairs[s:s + batch]
                n = len(pb)
                pv = preprocess([decode(vid[i]) for i in np.concatenate([pb[:, 0], pb[:, 1]])]).to(dev)[:, :8]
                with torch.no_grad(), ac():
                    saved, final, mp = _prefix_f32(model, pv, POINTS)
                    pred = pool_steps(predict_future(model, final)).float().cpu().numpy()
                store.setdefault("none/pred", []).append(pred[:n])
                store.setdefault("all/pred", []).append(pred[n:])
                store.setdefault("none/enc", []).append(mp[:n].cpu().numpy())
                store.setdefault("all/enc", []).append(mp[n:].cpu().numpy())
                sets = [token_sets(P["masks"][mrow[a]], P["masks"][mrow[b]]) for a, b in pb]
                for p in POINTS:
                    hA, hB = saved[p][:n], saved[p][n:]
                    for c in conditions(p):
                        M = torch.as_tensor(np.stack([st[c] for st in sets]), device=dev)[:, :, None]
                        h = torch.where(M, hB, hA)
                        with torch.no_grad(), ac():
                            r = suffix(model, h, p, None, return_final=True)
                            pr = pool_steps(predict_future(model, r["final"])).float().cpu().numpy()
                        store.setdefault(f"{p}/{c}/pred", []).append(pr)
                        store.setdefault(f"{p}/{c}/enc", []).append(r["final"].float().mean(1).cpu().numpy())
                del saved
                torch.cuda.empty_cache()
                print(f"{t} pairs {s + n}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
            for k, v in store.items():
                out[f"{t}/{k}"] = np.concatenate(v)
            out[f"{t}/pairs"] = pairs
        seconds = time.time() - t0
    info = {"seconds_gpu_total": seconds, "seconds_probe_clips": t_probe, "lock_wait_s": waited,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "precision": "bf16 autocast forward, fp32 residual stream, fp32 outputs", "smoke": smoke, "batch": batch,
            "points": POINTS, "conditions": {str(p): conditions(p) for p in POINTS}, "n_probe_clips": int(len(probe_ids)),
            "model": model_kind}
    np.savez(root / "forward.npz", **out)
    (root / "forward_info.json").write_text(json.dumps(info, indent=1, default=float))
    print("done", seconds, flush=True)


# ================================================================ score (CPU, local)

def score(root, model_kind="vjepa2"):
    from run_session2 import angle_of, wrap, write
    from wm.data import load_table
    from wm.probes import Standardizer, cv_select_alpha, fit_ridge, predict, score as pscore, targets
    from wm.provenance import sha256_file

    root = Path(root)
    F = dict(np.load(root / "forward.npz"))
    finfo = json.loads((root / "forward_info.json").read_text())
    P = dict(np.load(ART / "plan.npz"))
    meta = json.loads((ART / "plan.json").read_text())
    df = load_table("direction")
    row_of = {int(i): k for k, i in enumerate(df["id"])}
    theta = df["theta_degrees"].to_numpy(float)
    Y = targets(df, "direction")[0]
    pids = P["probe_ids"][:len(F["probe_enc_mp"])]
    prow = np.array([row_of[int(i)] for i in pids])
    folds5 = np.random.default_rng(0).permutation(np.arange(len(prow)) % 5)
    if model_kind == "vjepa2":
        Zall = np.load(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy").astype(np.float64)
        native_sha = sha256_file(PROJECT_ROOT / "artifacts/session2/native/pred_pooled_all.npy")
    else:   # untrained copy: its forecast reader is refit on ITS OWN (bf16) forecasts of the same probe clips
        Zall, native_sha = None, None

    def fit(X, Yt):
        st = Standardizer().fit(X)
        cv = cv_select_alpha(st.transform(X), Yt, folds5, score_fn=lambda a, b: pscore(a, b, "circular"))
        W, b = fit_ridge(st.transform(X), Yt, cv["alpha"])
        return (st, W, b), {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "n_probe_clips": int(len(X))}

    def apply(pr, X):
        st, W, b = pr
        return predict(st.transform(X), W, b)

    probes, pinfo = {}, {}
    probes["forecast"], pinfo["forecast"] = fit(Zall[prow].mean(1) if Zall is not None
                                                else F["probe_pred"].astype(np.float64).mean(1), Y[prow])
    probes["encoder_output"], pinfo["encoder_output"] = fit(F["probe_enc_mp"].astype(np.float64), Y[prow])
    parity = None if Zall is None else {"bf16_vs_fp32_native_forecast_rel_maxabs":
              float(np.abs(F["probe_pred"] - Zall[prow]).max() / np.abs(Zall[prow]).max()),
              "bf16_vs_fp32_forecast_direction_abs_deg":
              summarize(wrap(angle_of(apply(probes["forecast"], F["probe_pred"].mean(1)))
                             - angle_of(apply(probes["forecast"], Zall[prow].mean(1)))))}

    def feats(key, reader):
        X = F[key].astype(np.float64)
        return X.mean(1) if reader == "forecast" else X

    res = {"model": model_kind,
           **({"control": ("untrained copy: VJEPA2Config + torch.manual_seed(0) encoder and predictor (wm.extract.load_model('random'), "
                           "the copy behind results/p1a_*_random.json); same plan (pairs, token sets); both readers refit on this "
                           "copy's own probe-clip outputs (forecast reader on its bf16 forecasts, not the fp32 native cache)")}
              if model_kind != "vjepa2" else {}),
           "question": "which context tokens does the V-JEPA 2 predictor read motion direction from, by encoder depth?",
           "hypothesis": {"file": "refs/physics_paper.txt", "lines": "795-798",
                          "text": "we hypothesize it may be related to feature object binding, in which velocity information "
                                  "is most \"bound\" to the corresponding object. At the end of the network, information is "
                                  "catered to the optimization objective of predicting the next frame in latent space"},
           "prediction_if_true": "obj (disk) fraction high at points 8-12 and falling by 22 while the bg fraction rises",
           "definitions": {
               "binding_fraction": "per pair <v_c - v_A, v_B - v_A> / ||v_B - v_A||^2, v = the reader probe's (sin, cos) "
                                   "output; mean over pairs with pair-bootstrap 95% CI (1 = patch moves the readout all "
                                   "the way to B, 0 = no move)",
               "ratio_of_means": "mean_pairs <v_c - v_A, u> / mean_pairs <v_B - v_A, u>, u = (v_B - v_A)/||v_B - v_A|| "
                                 "(Delta_c / Delta_all)",
               "angle_fraction": "signed move of the read angle from A's read angle along the shorter arc to B's, over the arc",
               "err_to_A / err_to_B": "circular |read angle - true theta of A / of B| (deg)",
               "token_sets": {"obj": "disk(A) | disk(B), + 1-patch ring (3x3 dilation per tubelet)", "bg": "complement of obj",
                              "objA": "disk(A) + ring only (the literal design; B's disk tokens stay in bgA)",
                              "bgA": "complement of objA", "objN": "disk(A) | disk(B), no ring", "bgN": "complement of objN",
                              "none": "A unpatched", "all": "B unpatched (patching every token = B's forward)"},
               "zone": "paper's physics emergence zone = middle third, points 8-16; our per-patch onset points 8-9"},
           "readers": pinfo, "parity": parity,
           "config": {"points": list(POINTS), "pair_rule": meta["pair_rule"], "plan": {t: meta[t] for t in PAIR_TYPES},
                      "forward": finfo},
           "pair_types": {}}
    curves = {}
    for t in PAIR_TYPES:
        pairs = F[f"{t}/pairs"]
        ra, rb = np.array([row_of[int(a)] for a in pairs[:, 0]]), np.array([row_of[int(b)] for b in pairs[:, 1]])
        T = {"n_pairs": int(len(pairs)), "pairs": pairs.tolist(), "readers": {}}
        for reader in ("forecast", "encoder_output"):
            suf = "pred" if reader == "forecast" else "enc"
            vA = apply(probes[reader], feats(f"{t}/none/{suf}", reader))
            vB = apply(probes[reader], feats(f"{t}/all/{suf}", reader))
            phA, phB = angle_of(vA), angle_of(vB)
            u = (vB - vA) / np.linalg.norm(vB - vA, axis=1, keepdims=True)
            R = {"none": {"err_to_A": summarize(wrap(phA - theta[ra])), "err_to_B": summarize(wrap(phA - theta[rb]))},
                 "all": {"err_to_A": summarize(wrap(phB - theta[ra])), "err_to_B": summarize(wrap(phB - theta[rb]))},
                 "readout_A_to_B_arc_deg": summarize(np.abs(wrap_signed(phB - phA))), "points": {}}
            for p in POINTS:
                Rp = {}
                for c in conditions(p):
                    v = apply(probes[reader], feats(f"{t}/{p}/{c}/{suf}", reader))
                    ph = angle_of(v)
                    f = vec_fraction(v, vA, vB)
                    Rp[c] = {"err_to_A": summarize(wrap(ph - theta[ra])), "err_to_B": summarize(wrap(ph - theta[rb])),
                             "binding_fraction": summarize(f),
                             "ratio_of_means": ratio_of_means(((v - vA) * u).sum(1), ((vB - vA) * u).sum(1)),
                             "angle_fraction": summarize(ang_fraction(ph, phA, phB))}
                    curves[(t, reader, p, c)] = f
                for a, b in (("obj", "bg"), ("objA", "bgA"), ("objN", "bgN")):
                    if a in Rp:
                        Rp[f"{a}_plus_{b}"] = summarize(curves[(t, reader, p, a)] + curves[(t, reader, p, b)])
                        Rp[f"{a}_minus_{b}"] = summarize(curves[(t, reader, p, a)] - curves[(t, reader, p, b)])
                R["points"][str(p)] = Rp
            T["readers"][reader] = R
        # probe-free: projection fraction of the pooled forecast tokens (all 4 steps x 1024)
        zA, zB = F[f"{t}/none/pred"].reshape(len(pairs), -1), F[f"{t}/all/pred"].reshape(len(pairs), -1)
        T["forecast_tokens_probe_free"] = {str(p): {c: summarize(vec_fraction(F[f"{t}/{p}/{c}/pred"].reshape(len(pairs), -1),
                                                                                zA, zB)) for c in conditions(p)}
                                           for p in POINTS}
        res["pair_types"][t] = T
    res["verdict_inputs"] = {t: {r: {str(p): {c: res["pair_types"][t]["readers"][r]["points"][str(p)][c]["binding_fraction"]["mean"]
                                              for c in ("obj", "bg")} for p in POINTS}
                                 for r in ("forecast", "encoder_output")} for t in PAIR_TYPES}
    write(RES / ("p5_token_patching.json" if model_kind == "vjepa2" else f"p5_token_patching_{model_kind}.json"), res, seeds={"pairs": 0, "probe_folds": 0, "bootstrap": 0},
          activation_sha256={"forward_npz": sha256_file(root / "forward.npz"), "native_forecast_cache": native_sha,
                             "plan_npz": sha256_file(ART / "plan.npz")},
          script="scripts/run_token_patching.py")
    figure(res, model_kind)


def figure(res, model_kind="vjepa2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {"random": "Random pairs (disks at different positions)",
              "posmatch": f"Position-matched pairs (starts ≤ {POS_MAX_M} m apart)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    col = {"obj": "#c2410c", "bg": "#2563eb"}
    lab = {"obj": "object tokens (both disks + ring)", "bg": "background tokens"}
    for ax, t in zip(axes, PAIR_TYPES):
        T = res["pair_types"][t]
        ax.axvspan(8, 16, color="#fde68a", alpha=0.45, lw=0, label="paper's emergence zone (middle third)")
        ax.axhline(0, color="#888", lw=0.8)
        ax.axhline(1, color="#888", lw=0.8, ls=":")
        for reader, ls, mk in (("forecast", "-", "o"), ("encoder_output", "--", "s")):
            for c in ("obj", "bg"):
                pts = [res["pair_types"][t]["readers"][reader]["points"][str(p)][c]["binding_fraction"] for p in POINTS]
                m = np.array([q["mean"] for q in pts])
                lo = np.array([q["ci95"][0] for q in pts])
                hi = np.array([q["ci95"][1] for q in pts])
                x = np.array(POINTS) + (0.25 if reader == "encoder_output" else 0)
                ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col[c], ls=ls, marker=mk, ms=4, capsize=2, lw=1.5,
                            alpha=1 if reader == "forecast" else 0.55,
                            label=f"{lab[c]}, {'predictor forecast' if reader == 'forecast' else 'encoder output'} probe")
        ax.set_title(f"{titles[t]}, n = {T['n_pairs']}", fontsize=10)
        ax.set_xlabel("patch point (encoder block; 0 = patch embedding)")
        ax.set_xticks(POINTS)
    axes[0].set_ylabel("fraction of the A→B direction change\n(patch this token set with B's tokens)")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Which context tokens carry motion direction to the V-JEPA 2 predictor?" if model_kind == "vjepa2" else
                 "Untrained V-JEPA 2 copy (random init): which context tokens carry direction to its predictor?", fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    FIG.mkdir(exist_ok=True)
    name = "fig_token_patching.png" if model_kind == "vjepa2" else f"fig_token_patching_{model_kind}.png"
    fig.savefig(FIG / name, dpi=150)
    print("wrote", FIG / name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "forward", "score"))
    ap.add_argument("--root", default=str(ART / "forward"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="vjepa2", choices=("vjepa2", "random"))
    ap.add_argument("--points", type=int, nargs="*", default=None, help="override POINTS (e.g. 8 12 22)")
    a = ap.parse_args()
    if a.points:
        POINTS = tuple(a.points)
    if a.cmd == "plan":
        plan()
    elif a.cmd == "forward":
        forward(a.root, a.batch, a.smoke, a.model)
    else:
        if a.points is None:     # score on the points the forward ran
            POINTS = tuple(json.loads((Path(a.root) / "forward_info.json").read_text())["points"])
        score(a.root, a.model)

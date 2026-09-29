"""Motion-tracking heads: do a few attention heads carry the disk's motion, or is attention diffuse?

V-JEPA 2 ViT-L encoder, 24 blocks x 16 heads (head dim 64); tokens 8 temporal slots x 16 x 16 (slot = 2 frames),
token index t*256 + h*16 + w (wm.extract docstring). Block b = encoder.layer[b-1]; read point p = residual after
block p (REPORT section 3.1). Disk mask = wm.data.disk_mask (a patch is disk if any disk pixel falls in it in either
frame of its tubelet), the same rule run_token_patching.py uses.

Part A (attention geometry). Train-split direction clips (seeded subset, <= 300). fp32 eager attention probabilities
at blocks 6-13. For each head, the queries are the disk's tokens at slot t (t = 1..6, so both neighbours exist and the
disk is present in all three slots); keys are split into (i) disk tokens at t-1, (ii) disk tokens at t, (iii) disk
tokens at t+1, (iv) every other token. Raw mass = summed attention probability per query into the set; density =
mass / |set|. Per clip: query-weighted mean over t. Summary per (b, h): ratio of clip-mean previous-slot density to
clip-mean background density, clip-bootstrap 95% CI. Control "phantom": the same arithmetic with the disk mask rolled
by (8, 8) patches (same trajectory shape, no object there) -- a generic local spatio-temporal head scores high on both,
an object-tracking head only on the real mask. Same on the random-init copy (wm.extract.load_model('random')).

Part B (head ablation read by the Part 1 probe). Zero head h's slice of the input to block b's attention output
projection (forward pre-hook on encoder.layer[b-1].attention.proj; the proj bias stays), finish the encoder to block
22 (bf16 autocast, fp32 residual stream, sdpa attention), meanpool at points 12 and 22, apply the Part 1 pooled ridge
probe (wm.probes: train-standardised, alpha by the split's 5 train folds, refit on all train; fit on the STORED
unablated fp32 train meanpool of artifacts/activations/<dataset>/<model>) and score on the split_v1 test clips
(direction 300, speed 308; probe fitting never sees them). Conditions: every head at blocks 8, 9, 10, 12; per block
the top-K Part-A tracking heads jointly and N_RAND random K-head sets; the global top heads over the four blocks
jointly + count-matched random sets. The random-init copy: per-head at blocks 9 and 12 + its own top-K / random-K.

  python scripts/run_motion_heads.py probes                       (Mac CPU: fit probes -> artifacts/p5_motion_heads/in)
  python scripts/run_motion_heads.py forward --root R [--smoke]   (box GPU; takes flock /tmp/wm_gpu.lock itself)
  python scripts/run_motion_heads.py score --root R               (Mac CPU: results/p5_motion_heads.json + figure)
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "p5_motion_heads"
RES = PROJECT_ROOT / "results"
FIG = PROJECT_ROOT / "figures"
LOCK = "/tmp/wm_gpu.lock"
N_HEADS, HEAD_DIM, N_SLOTS, GRID = 16, 64, 8, 16
N_TOK = N_SLOTS * GRID * GRID
A_BLOCKS = tuple(range(6, 14))
B_BLOCKS = (8, 9, 10, 12)
B_BLOCKS_RANDOM = (9, 12)
READ_POINTS = (12, 22)
LAST_BLOCK_RUN = max(READ_POINTS)
N_A_CLIPS = 300
TOP_K = 3             # per-block joint ablation size
TOP_GLOBAL = 8        # global joint ablation size (over B_BLOCKS)
N_RAND = 4            # random K-head sets per block (and for the global set)
N_RAND_RANDOMINIT = 2
N_SPEED = 154         # seeded half of the 308 speed test clips (GPU budget); direction uses all 300
SEED = 0
PHANTOM_ROLL = (8, 8)


# ================================================================ pure helpers (tests/test_motion_heads.py)

def set_matrix(mask):
    """mask: bool [..., 8, 16, 16] -> float [..., 2048, 8] indicator of 'disk token at slot s' (token order t, h, w)."""
    m = np.asarray(mask, bool)
    lead = m.shape[:-3]
    flat = m.reshape(lead + (N_SLOTS, GRID * GRID))
    S = np.zeros(lead + (N_TOK, N_SLOTS), np.float32)
    for s in range(N_SLOTS):
        S[..., s * GRID * GRID:(s + 1) * GRID * GRID, s] = flat[..., s, :]
    return S


def slot_mass(A, S):
    """A: attention probs [H, N, N] (rows sum to 1); S: [N, 8] indicator (query and key sets are the same disk sets).
    Returns M [H, 8, 8]: M[h, t, s] = sum over queries q in set t of the attention mass q puts on set s; and
    n [8] = set sizes. Works for numpy or torch inputs of matching type."""
    try:
        import torch
        if isinstance(A, torch.Tensor):
            KS = torch.matmul(A, S)                                  # [H, N, 8]: mass of each query on each set
            M = torch.einsum("qt,hqs->hts", S, KS)
            return M, S.sum(0)
    except ImportError:
        pass
    KS = A @ S
    return np.einsum("qt,hqs->hts", S, KS), S.sum(0)


def clip_densities(M, n, n_tok=N_TOK):
    """M [H, 8, 8] summed masses, n [8] set sizes (one clip, one block). Valid query slots: t in 1..6 with
    n[t-1], n[t], n[t+1] > 0. Returns dict of [H] arrays (NaN if no valid slot): raw mean mass per query on
    prev / same / next / other, and densities (mass per key token), query-weighted over valid t. 'far' = the disk's
    tokens >= 2 slots away from t (a subset of 'other'; far-slot baseline for object-attending vs tracking)."""
    M, n = np.asarray(M, float), np.asarray(n, float)
    H = M.shape[0]
    keys = ("prev", "same", "next", "other", "far")
    acc = {f"{k}_{w}": np.zeros(H) for k in keys for w in ("mass", "dens")}
    wsum = 0.0
    for t in range(1, N_SLOTS - 1):
        if not (n[t - 1] > 0 and n[t] > 0 and n[t + 1] > 0):
            continue
        q = n[t]
        mass = {"prev": M[:, t, t - 1], "same": M[:, t, t], "next": M[:, t, t + 1]}
        mass["other"] = q - mass["prev"] - mass["same"] - mass["next"]     # rows sum to 1
        size = {"prev": n[t - 1], "same": n[t], "next": n[t + 1],
                "other": n_tok - n[t - 1] - n[t] - n[t + 1]}
        far = [u for u in range(N_SLOTS) if abs(u - t) >= 2 and n[u] > 0]   # disk tokens >= 2 slots away
        mass["far"] = M[:, t, far].sum(-1) if far else np.full(H, np.nan)   # (a subset of 'other')
        size["far"] = n[far].sum() if far else np.nan
        for k in keys:
            acc[f"{k}_mass"] += mass[k]                                      # summed over queries
            acc[f"{k}_dens"] += mass[k] / size[k]                            # per query: /q below, *q weight
        wsum += q
    if wsum == 0:
        return {k: np.full(H, np.nan) for k in acc}
    return {k: v / wsum for k, v in acc.items()}


def ratio_ci(num, den, n_boot=2000, seed=0):
    """num, den: [n_clips, ...]; ratio of clip means with a clip-bootstrap 95% CI (NaN rows dropped per column)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num).all(axis=tuple(range(1, num.ndim))) & np.isfinite(den).all(axis=tuple(range(1, den.ndim)))
    num, den = num[ok], den[ok]
    r = num.mean(0) / den.mean(0)
    idx = np.random.default_rng(seed).integers(0, len(num), (n_boot, len(num)))
    rb = num[idx].mean(1) / den[idx].mean(1)
    return r, np.percentile(rb, 2.5, axis=0), np.percentile(rb, 97.5, axis=0), int(len(num))


def r2_cols(Y, P):
    """Mean over columns of R^2 (Y, P [..., n, m])."""
    Y, P = np.asarray(Y, float), np.asarray(P, float)
    ss_res = ((Y - P) ** 2).sum(-2)
    ss_tot = ((Y - Y.mean(-2, keepdims=True)) ** 2).sum(-2)
    return (1 - ss_res / ss_tot).mean(-1)


def r2_drop_boot(Y, P_base, P_abl, n_boot=2000, seed=0):
    """R^2(base) - R^2(ablated) on the same clips, with a clip-bootstrap CI (same resample for both)."""
    Y, P_base, P_abl = (np.asarray(a, float).reshape(len(Y), -1) for a in (Y, P_base, P_abl))
    d = r2_cols(Y, P_base) - r2_cols(Y, P_abl)
    idx = np.random.default_rng(seed).integers(0, len(Y), (n_boot, len(Y)))
    db = r2_cols(Y[idx], P_base[idx]) - r2_cols(Y[idx], P_abl[idx])
    return float(d), float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))


def top_heads(score_bh, blocks, k):
    """score_bh: [len(blocks), 16]; returns the k (block, head) with the largest scores (NaN last)."""
    s = np.nan_to_num(np.asarray(score_bh, float), nan=-np.inf)
    order = np.argsort(-s, axis=None, kind="stable")[:k]
    return [(int(blocks[i // N_HEADS]), int(i % N_HEADS)) for i in order]


def conditions(topk_per_block, top_global, blocks, rng, n_rand=N_RAND, per_head=True):
    """List of (name, {block: [heads]}) ablation conditions. Random sets are drawn from all 16 heads of the block,
    matched to the top set's count (global: matched per block)."""
    conds = [("none", {})]
    for b in blocks:
        if per_head:
            conds += [(f"b{b}_h{h}", {b: [h]}) for h in range(N_HEADS)]
        tk = topk_per_block.get(b)
        if tk:
            conds.append((f"b{b}_topk", {b: list(tk)}))
            conds += [(f"b{b}_rand{r}", {b: sorted(rng.choice(N_HEADS, len(tk), replace=False).tolist())})
                      for r in range(n_rand)]
    if top_global:
        g = {}
        for b, h in top_global:
            g.setdefault(b, []).append(h)
        conds.append(("global_top", g))
        for r in range(n_rand):
            conds.append((f"global_rand{r}", {b: sorted(rng.choice(N_HEADS, len(hs), replace=False).tolist())
                                              for b, hs in g.items()}))
    return conds


def test_rows(ds, n):
    """Row positions of the test clips used in Part B (all for direction; a seeded sorted half for speed)."""
    if ds == "speed" and N_SPEED < n:
        return np.sort(np.random.default_rng(SEED + 1).choice(n, N_SPEED, replace=False))
    return np.arange(n)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ probes (Mac CPU)

def probes():
    from wm.data import load_table
    from wm.probes import (ALPHAS, Standardizer, cv_select_alpha, fit_ridge, load_activations, split_rows,
                           targets)
    ART.joinpath("in").mkdir(parents=True, exist_ok=True)
    out, meta = {}, {}
    for model in ("vjepa2", "random"):
        for ds, var in (("direction", "direction"), ("speed", "speed")):
            df = load_table(ds)
            Y, kind, score_fn = targets(df, var)
            tr, te, folds = split_rows(ds, df)
            acts = load_activations(ds, "meanpool", model)
            for p in READ_POINTS:
                X = np.asarray(acts[:, p], np.float64)
                st = Standardizer().fit(X[tr])
                cv = cv_select_alpha(st.transform(X[tr]), Y[tr], folds, ALPHAS, score_fn)
                W, b = fit_ridge(st.transform(X[tr]), Y[tr], cv["alpha"])
                Pte = st.transform(X[te]) @ W + b
                key = f"{model}_{ds}_p{p}"
                out[f"{key}_mean"], out[f"{key}_std"] = st.mean, st.std
                out[f"{key}_W"], out[f"{key}_b"] = W, b
                out[f"{key}_Xte_stored"] = X[te].astype(np.float32)
                meta[key] = {"alpha": cv["alpha"], "cv_r2": cv["cv_mean"], "test_r2_stored": score_fn(Y[te], Pte)["r2"],
                             "n_train": int(len(tr)), "n_test": int(len(te))}
                print(key, meta[key], flush=True)
            out[f"{ds}_test_ids"] = df["id"].to_numpy()[te]
            out[f"{ds}_Yte"] = Y[te]
    tr, te, folds = split_rows("direction")
    df = load_table("direction")
    rng = np.random.default_rng(SEED)
    a_ids = np.sort(rng.choice(df["id"].to_numpy()[tr], N_A_CLIPS, replace=False))
    assert not set(a_ids) & set(out["direction_test_ids"].tolist())
    out["partA_ids"] = a_ids
    np.savez(ART / "in" / "probes.npz", **out)
    (ART / "in" / "probes.json").write_text(json.dumps(meta, indent=1))
    print("wrote", ART / "in" / "probes.npz", sha256(ART / "in" / "probes.npz"))


# ================================================================ forward (box GPU)

def forward(root, smoke=False, batch_a=2, batch_b=16):
    import torch
    from wm.data import decode, disk_mask, load_table
    from wm.extract import load_model, preprocess, set_precision

    set_precision()
    dev = torch.device("cuda")
    root = Path(root)
    out_dir = root / ("smoke" if smoke else "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    P = dict(np.load(root / "in" / "probes.npz"))
    tables = {ds: load_table(ds) for ds in ("direction", "speed")}
    vids = {ds: dict(zip(t["id"], t["video"])) for ds, t in tables.items()}
    n_a = 8 if smoke else len(P["partA_ids"])
    a_ids = P["partA_ids"][:n_a]
    info = {"smoke": smoke, "box": os.environ.get("WM_BOX"), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "timing": {}, "t_start": time.time()}

    lk = open(LOCK, "a")
    t0 = time.time()
    fcntl.flock(lk, fcntl.LOCK_EX)
    info["lock_wait_s"] = time.time() - t0
    print(f"lock acquired after {info['lock_wait_s']:.0f}s", flush=True)
    t_gpu = time.time()

    # ---------------------------------------------------------- Part A
    frames_a = [decode(vids["direction"][int(i)]) for i in a_ids]
    masks_a = np.stack([disk_mask(f)[0] for f in frames_a])
    np.save(out_dir / "partA_masks.npy", masks_a)
    S_real = torch.from_numpy(set_matrix(masks_a)).to(dev)
    S_ph = torch.from_numpy(set_matrix(np.roll(masks_a, PHANTOM_ROLL, axis=(-2, -1)))).to(dev)
    models = ("vjepa2", "random")
    partA = {}
    for mk in models:
        t1 = time.time()
        model = load_model(mk, dev)
        enc = model.encoder
        M_real = np.zeros((n_a, len(A_BLOCKS), N_HEADS, N_SLOTS, N_SLOTS), np.float32)
        M_ph = np.zeros_like(M_real)
        with torch.no_grad():
            for s in range(0, n_a, batch_a):
                pv = preprocess(frames_a[s:s + batch_a]).to(dev)
                h = enc.embeddings(pv).float()
                for i, layer in enumerate(enc.layer[:max(A_BLOCKS)]):
                    b = i + 1
                    if b in A_BLOCKS:
                        h, att = layer(h, None, None, True)
                        for j in range(att.shape[0]):
                            A = att[j].float()
                            M_real[s + j, A_BLOCKS.index(b)] = slot_mass(A, S_real[s + j])[0].cpu().numpy()
                            M_ph[s + j, A_BLOCKS.index(b)] = slot_mass(A, S_ph[s + j])[0].cpu().numpy()
                        del att
                    else:
                        h = layer(h, None, None, False)[0]
        partA[mk] = (M_real, M_ph)
        np.savez(out_dir / f"partA_{mk}.npz", M_real=M_real, M_phantom=M_ph, ids=a_ids,
                 n_real=masks_a.reshape(n_a, N_SLOTS, -1).sum(-1),
                 n_phantom=np.roll(masks_a, PHANTOM_ROLL, axis=(-2, -1)).reshape(n_a, N_SLOTS, -1).sum(-1))
        info["timing"][f"partA_{mk}"] = time.time() - t1
        print(f"part A {mk}: {n_a} clips {time.time() - t1:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    # top heads from Part A (tracking ratio prev density / background density, clip means)
    def tracking_ratio(mk):
        M, _ = partA[mk]
        n = masks_a.reshape(n_a, N_SLOTS, -1).sum(-1)
        prev = np.full((n_a, len(A_BLOCKS), N_HEADS), np.nan)
        bg = np.full_like(prev, np.nan)
        for c in range(n_a):
            for bi in range(len(A_BLOCKS)):
                d = clip_densities(M[c, bi], n[c])
                prev[c, bi], bg[c, bi] = d["prev_dens"], d["other_dens"]
        ok = np.isfinite(prev).all((1, 2))
        return prev[ok].mean(0) / bg[ok].mean(0)                     # [blocks, heads]

    plans = {}
    rng = np.random.default_rng(SEED)
    for mk, blocks in (("vjepa2", B_BLOCKS), ("random", B_BLOCKS_RANDOM)):
        ratio = tracking_ratio(mk)
        rb = ratio[[A_BLOCKS.index(b) for b in blocks]]
        topk = {b: [h for _, h in top_heads(rb[[k]], [b], TOP_K)] for k, b in enumerate(blocks)}
        glob = top_heads(rb, blocks, TOP_GLOBAL) if mk == "vjepa2" else []
        conds = conditions(topk, glob, blocks, rng, n_rand=N_RAND if mk == "vjepa2" else N_RAND_RANDOMINIT)
        if smoke:
            conds = [c for c in conds if c[0] in ("none", f"b{blocks[0]}_h0", f"b{blocks[-1]}_topk", "global_top")]
        plans[mk] = {"tracking_ratio_blocks": list(blocks), "topk": {str(b): v for b, v in topk.items()},
                     "top_global": glob, "conditions": conds}
    (out_dir / "plan_B.json").write_text(json.dumps(plans, indent=1))

    # ---------------------------------------------------------- Part B
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)          # noqa: E731
    for ds, mk in [(d, m) for d in ("direction", "speed") for m in models]:
        model = load_model(mk, dev)
        enc = model.encoder
        conds = plans[mk]["conditions"]
        blocks_needed = sorted({b for _, d in conds for b in d})
        state = {"zero": {}}

        def make_hook(b):
            def hook(mod, args):
                hs = state["zero"].get(b)
                if not hs:
                    return None
                x = args[0].clone()
                for hh in hs:
                    x[..., hh * HEAD_DIM:(hh + 1) * HEAD_DIM] = 0
                return (x,)
            return hook
        handles = [enc.layer[b - 1].attention.proj.register_forward_pre_hook(make_hook(b)) for b in blocks_needed]
        if True:
            t1 = time.time()
            ids = P[f"{ds}_test_ids"][:8] if smoke else P[f"{ds}_test_ids"][test_rows(ds, len(P[f"{ds}_test_ids"]))]
            frames = [decode(vids[ds][int(i)]) for i in ids]
            start_blocks = sorted(set(blocks_needed) | {min(blocks_needed)})
            cache = {b: [] for b in start_blocks}                      # residual INPUT to block b (fp32, CPU)
            with torch.no_grad(), ac():
                for s in range(0, len(ids), batch_b):
                    pv = preprocess(frames[s:s + batch_b]).to(dev)
                    h = enc.embeddings(pv).float()
                    for i, layer in enumerate(enc.layer[:max(start_blocks) - 1]):
                        if i + 1 in cache:
                            cache[i + 1].append(h.cpu())
                        h = layer(h, None, None, False)[0].float()
                    cache[max(start_blocks)].append(h.cpu())
            cache = {b: torch.cat(v) for b, v in cache.items()}
            t_cache = time.time() - t1
            names = [c[0] for c in conds]
            feats_none = None
            preds = {f"p{p}": np.zeros((len(conds), len(ids), P[f"{mk}_{ds}_p{p}_W"].shape[1]), np.float32)
                     for p in READ_POINTS}
            W = {p: torch.from_numpy(P[f"{mk}_{ds}_p{p}_W"]).double().to(dev) for p in READ_POINTS}
            bb = {p: torch.from_numpy(P[f"{mk}_{ds}_p{p}_b"]).double().to(dev) for p in READ_POINTS}
            mu = {p: torch.from_numpy(P[f"{mk}_{ds}_p{p}_mean"]).double().to(dev) for p in READ_POINTS}
            sd = {p: torch.from_numpy(P[f"{mk}_{ds}_p{p}_std"]).double().to(dev) for p in READ_POINTS}
            for ci, (name, abl) in enumerate(conds):
                start = min(abl) if abl else min(start_blocks)
                state["zero"] = abl
                feats = {p: [] for p in READ_POINTS}
                with torch.no_grad(), ac():
                    for s in range(0, len(ids), batch_b):
                        h = cache[start][s:s + batch_b].to(dev, non_blocking=True)
                        for i in range(start - 1, LAST_BLOCK_RUN):
                            h = enc.layer[i](h, None, None, False)[0].float()
                            if i + 1 in READ_POINTS:
                                feats[i + 1].append(h.mean(1).double())
                for p in READ_POINTS:
                    X = torch.cat(feats[p])
                    preds[f"p{p}"][ci] = (((X - mu[p]) / sd[p]) @ W[p] + bb[p]).float().cpu().numpy()
                    if name == "none":
                        feats_none = feats_none or {}
                        feats_none[p] = X.float().cpu().numpy()
                state["zero"] = {}
                if ci % 10 == 0:
                    print(f"B {mk} {ds} cond {ci + 1}/{len(conds)} {name} {time.time() - t1:.0f}s", flush=True)
            np.savez(out_dir / f"partB_{mk}_{ds}.npz", names=np.array(names), ids=ids,
                     **preds, **{f"X_none_p{p}": feats_none[p] for p in READ_POINTS})
            info["timing"][f"partB_{mk}_{ds}"] = {"total_s": time.time() - t1, "cache_s": t_cache,
                                                   "n_conditions": len(conds), "n_clips": int(len(ids))}
            print(f"part B {mk} {ds}: {len(conds)} conds x {len(ids)} clips {time.time() - t1:.0f}s", flush=True)
            del cache
        for hd in handles:
            hd.remove()
        del model
        torch.cuda.empty_cache()

    info["gpu_seconds"] = time.time() - t_gpu
    fcntl.flock(lk, fcntl.LOCK_UN)
    lk.close()
    info["script_sha256"] = sha256(__file__)
    info["probes_sha256"] = sha256(root / "in" / "probes.npz")
    (out_dir / "forward_info.json").write_text(json.dumps(info, indent=1))
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("done", json.dumps(info["timing"]), f"gpu_s={info['gpu_seconds']:.0f}", flush=True)


# ================================================================ score (Mac CPU)

def score(root):
    from wm.provenance import git_commit
    root = Path(root)
    out_dir = root / "out"
    P = dict(np.load(ART / "in" / "probes.npz"))
    pmeta = json.loads((ART / "in" / "probes.json").read_text())
    plans = json.loads((out_dir / "plan_B.json").read_text())
    finfo = json.loads((out_dir / "forward_info.json").read_text())
    res = {"design": __doc__.split("\n\n")[1:4], "part_A": {}, "part_B": {}, "probes": pmeta,
           "framing_and_caveats": {
               "what_this_is": ("our disk-attention test: which heads put attention density on the disk's previous-slot "
                                "tokens, and what zero-ablating them does to the Part 1 pooled direction probe. NOT a "
                                "reproduction of the paper's section 6.3 / App. C.6 attention-distance analysis (C.6 masks "
                                "nearby tokens at distance thresholds and renormalises; 6.3 reports per-head attention "
                                "distance). The paper's per-head spatial/temporal attention distance was NOT computed "
                                "(needs a Part A rerun; did not fit before 21:00 ET)."),
               "selection_criterion": ("top heads = largest ratio of clip-mean previous-slot disk density to clip-mean "
                                       "background density (background = every token outside the disk's slot t-1, t, "
                                       "t+1 sets), queries = disk tokens at slots 1-6 with the disk present in t-1, t, "
                                       "t+1; per-block top-3 and global top-8 over blocks 8/9/10/12 were chosen from "
                                       "Part A on train clips before Part B ran"),
               "labels": ("'object-attending' by default: a head that attends to the object wherever it is, or to the "
                          "same position one slot back (slow disks overlap across slots), also scores high on "
                          "previous-slot density. The phantom control (mask rolled by 8,8 patches) addresses the "
                          "second kind only. A head is labelled 'tracking' only if its previous-slot density beats its "
                          "same-slot density (clip-bootstrap 95% CI of prev/same density ratio entirely above 1). "
                          "far_over_bg_density = the disk's own tokens >= 2 slots away (far-slot baseline)."),
               "ablation_caveat": ("zero-ablation with a probe fit on unablated activations mixes information loss with "
                                   "distribution shift (the probe sees off-distribution residuals); the count-matched "
                                   "random K-head sets are the partial control for that shift"),
               "block_range_caveat": ("attention geometry covers blocks 6-13 only (the emergence zone 8-9 and just "
                                      "after), so there is little outside-the-zone contrast; ablation covers 8/9/10/12"),
               "precision": ("Part A fp32 eager attention; Part B bf16 autocast with fp32 residual stream (sdpa); drops "
                             "are relative to the same pipeline's unablated baseline (see baseline parity vs stored "
                             "fp32 features)")}}

    # ---- Part A
    for mk in ("vjepa2", "random"):
        z = np.load(out_dir / f"partA_{mk}.npz")
        n_clip = len(z["ids"])
        rows = {}
        per = {}
        for tag, M, n in (("real", z["M_real"], z["n_real"]), ("phantom", z["M_phantom"], z["n_phantom"])):
            D = {}
            for c in range(n_clip):
                for bi in range(len(A_BLOCKS)):
                    d = clip_densities(M[c, bi], n[c])
                    for k, v in d.items():
                        D.setdefault(k, np.full((n_clip, len(A_BLOCKS), N_HEADS), np.nan))[c, bi] = v
            per[tag] = D
        tab = []
        for bi, b in enumerate(A_BLOCKS):
            r, lo, hi, nv = ratio_ci(per["real"]["prev_dens"][:, bi], per["real"]["other_dens"][:, bi], seed=bi)
            rp, lop, hip, nvp = ratio_ci(per["phantom"]["prev_dens"][:, bi], per["phantom"]["other_dens"][:, bi], seed=bi)
            rs, _, _, _ = ratio_ci(per["real"]["same_dens"][:, bi], per["real"]["other_dens"][:, bi], seed=bi)
            rn, _, _, _ = ratio_ci(per["real"]["next_dens"][:, bi], per["real"]["other_dens"][:, bi], seed=bi)
            rf, _, _, _ = ratio_ci(per["real"]["far_dens"][:, bi], per["real"]["other_dens"][:, bi], seed=bi)
            rps, lops, hips, _ = ratio_ci(per["real"]["prev_dens"][:, bi], per["real"]["same_dens"][:, bi], seed=bi)
            rpf, lopf, hipf, _ = ratio_ci(per["real"]["prev_dens"][:, bi], per["real"]["far_dens"][:, bi], seed=bi)
            for h in range(N_HEADS):
                row = {"block": b, "head": h, "prev_over_bg_density": float(r[h]), "ci95": [float(lo[h]), float(hi[h])],
                       "same_over_bg_density": float(rs[h]), "next_over_bg_density": float(rn[h]),
                       "far_over_bg_density": float(rf[h]),
                       "prev_over_same_density": float(rps[h]), "prev_over_same_ci95": [float(lops[h]), float(hips[h])],
                       "prev_over_far_density": float(rpf[h]), "prev_over_far_ci95": [float(lopf[h]), float(hipf[h])],
                       "label": ("tracking" if lops[h] > 1 else "object-attending"),
                       "phantom_prev_over_bg_density": float(rp[h]), "phantom_ci95": [float(lop[h]), float(hip[h])],
                       "n_clips_valid": nv}
                for k in ("prev", "same", "next", "other", "far"):
                    row[f"{k}_mass"] = float(np.nanmean(per["real"][f"{k}_mass"][:, bi, h]))
                    row[f"{k}_density"] = float(np.nanmean(per["real"][f"{k}_dens"][:, bi, h]))
                tab.append(row)
        ratio = np.array([[r["prev_over_bg_density"] for r in tab if r["block"] == b] for b in A_BLOCKS])
        ph = np.array([[r["phantom_prev_over_bg_density"] for r in tab if r["block"] == b] for b in A_BLOCKS])
        # concentration: share of the (ratio - 1)+ excess held by the top 5% of heads, per block and overall
        exc = np.clip(ratio - 1, 0, None)
        srt = np.sort(exc.ravel())[::-1]
        res["part_A"][mk] = {
            "n_clips": n_clip, "table": tab,
            "top10_by_prev_ratio": sorted(tab, key=lambda r: -r["prev_over_bg_density"])[:10],
            "block_median_ratio": {str(b): float(np.median(ratio[i])) for i, b in enumerate(A_BLOCKS)},
            "block_max_ratio": {str(b): float(np.max(ratio[i])) for i, b in enumerate(A_BLOCKS)},
            "block_median_phantom_ratio": {str(b): float(np.median(ph[i])) for i, b in enumerate(A_BLOCKS)},
            "top6_of_128_share_of_excess": float(srt[:6].sum() / max(srt.sum(), 1e-12)),
            "n_heads_ratio_gt_2": int((ratio > 2).sum()), "n_heads_ratio_gt_5": int((ratio > 5).sum()),
            "n_heads_labelled_tracking": int(sum(r["label"] == "tracking" for r in tab)),
            "tracking_heads": [(r["block"], r["head"]) for r in tab if r["label"] == "tracking"],
            "corr_real_vs_phantom_ratio_log": float(np.corrcoef(np.log(ratio.ravel()), np.log(ph.ravel()))[0, 1]),
        }
        res["part_A"][mk]["_ratio_grid"] = ratio.tolist()

    # ---- Part B
    for mk in ("vjepa2", "random"):
        res["part_B"][mk] = {"plan": {k: v for k, v in plans[mk].items() if k != "conditions"},
                             "conditions": {n: d for n, d in plans[mk]["conditions"]}}
        for ds in ("direction", "speed"):
            z = np.load(out_dir / f"partB_{mk}_{ds}.npz")
            names = list(z["names"])
            rows_used = test_rows(ds, len(P[f"{ds}_test_ids"]))
            Y = P[f"{ds}_Yte"][rows_used]
            assert np.array_equal(z["ids"], P[f"{ds}_test_ids"][rows_used])
            i0 = names.index("none")
            block = {}
            for p in READ_POINTS:
                pr = z[f"p{p}"]
                key = f"{mk}_{ds}_p{p}"
                Xs = P[f"{key}_Xte_stored"][rows_used]
                Pst = ((Xs - P[f"{key}_mean"]) / P[f"{key}_std"]) @ P[f"{key}_W"] + P[f"{key}_b"]
                base_r2 = float(r2_cols(Y.reshape(len(Y), -1), pr[i0].reshape(len(Y), -1)))
                parity = {"test_r2_box_unablated": base_r2,
                          "test_r2_stored_fp32": float(r2_cols(Y.reshape(len(Y), -1), Pst.reshape(len(Y), -1))),
                          "max_abs_pred_diff": float(np.abs(pr[i0].reshape(len(Y), -1) - Pst.reshape(len(Y), -1)).max()),
                          "feat_rel_err": float(np.linalg.norm(z[f"X_none_p{p}"] - Xs) / np.linalg.norm(Xs))}
                rows = {}
                for ci, n in enumerate(names):
                    if n == "none":
                        continue
                    d, lo, hi = r2_drop_boot(Y, pr[i0], pr[ci], seed=ci)
                    rows[n] = {"r2_drop": d, "ci95": [lo, hi]}
                parity["valid"] = bool(abs(parity["test_r2_box_unablated"] - parity["test_r2_stored_fp32"]) < 0.05)
                if not parity["valid"]:
                    parity["note"] = ("INVALID: the bf16-autocast unablated baseline does not reproduce the stored fp32 "
                                      "probe score (the random-init ridge probe amplifies ~0.5% feature error), so the "
                                      "R2 drops of this model/dataset/point are not interpretable; an fp32 rerun is needed")
                block[f"p{p}"] = {"baseline": parity, "drops": rows}
            res["part_B"][mk][ds] = block
        # summaries: per-(block, head) drop tables, random band, top-k vs random
        summ = {}
        for ds in ("direction", "speed"):
            for p in READ_POINTS:
                dr = res["part_B"][mk][ds][f"p{p}"]["drops"]
                s = {}
                blocks = plans[mk]["tracking_ratio_blocks"]
                for b in blocks:
                    ph = [dr[f"b{b}_h{h}"]["r2_drop"] for h in range(N_HEADS) if f"b{b}_h{h}" in dr]
                    rnd = [v["r2_drop"] for k, v in dr.items() if k.startswith(f"b{b}_rand")]
                    s[str(b)] = {"per_head_drop": ph,
                                 "per_head_ci95": [dr[f"b{b}_h{h}"]["ci95"] for h in range(N_HEADS) if f"b{b}_h{h}" in dr],
                                 "max_head": int(np.argmax(ph)) if ph else None,
                                 "max_head_drop": float(np.max(ph)) if ph else None,
                                 "topk_heads": plans[mk]["topk"].get(str(b)),
                                 "topk_joint_drop": dr.get(f"b{b}_topk"),
                                 "random_k_drops": rnd,
                                 "random_k_range": [float(min(rnd)), float(max(rnd))] if rnd else None}
                if "global_top" in dr:
                    rg = [v["r2_drop"] for k, v in dr.items() if k.startswith("global_rand")]
                    s["global"] = {"heads": plans[mk]["top_global"], "top_joint_drop": dr["global_top"],
                                   "random_matched_drops": rg}
                summ[f"{ds}_p{p}"] = s
        res["part_B"][mk]["summary"] = summ

    # top heads joined across A and B (V-JEPA 2)
    A = {(r["block"], r["head"]): r for r in res["part_A"]["vjepa2"]["table"]}
    joined = []
    for b, h in sorted(A, key=lambda k: -A[k]["prev_over_bg_density"])[:12]:
        row = {"block": b, "head": h, "prev_ratio": A[(b, h)]["prev_over_bg_density"], "prev_ratio_ci95": A[(b, h)]["ci95"],
               "phantom_ratio": A[(b, h)]["phantom_prev_over_bg_density"],
               "same_ratio": A[(b, h)]["same_over_bg_density"], "next_ratio": A[(b, h)]["next_over_bg_density"],
               "far_ratio": A[(b, h)]["far_over_bg_density"], "prev_over_same": A[(b, h)]["prev_over_same_density"],
               "prev_over_same_ci95": A[(b, h)]["prev_over_same_ci95"], "label": A[(b, h)]["label"]}
        for ds in ("direction", "speed"):
            for p in READ_POINTS:
                d = res["part_B"]["vjepa2"][ds][f"p{p}"]["drops"].get(f"b{b}_h{h}")
                row[f"{ds}_p{p}_drop"] = d
        joined.append(row)
    res["top_tracking_heads_joined"] = joined

    ins = {"probes.npz": sha256(ART / "in" / "probes.npz")}
    for f in sorted(out_dir.glob("*.npz")) + [out_dir / "plan_B.json", out_dir / "forward_info.json"]:
        ins[f"out/{f.name}"] = sha256(f)
    res["provenance"] = {"git_commit": git_commit(), "script_sha256": sha256(__file__),
                         "script_sha256_on_box": finfo.get("script_sha256"), "inputs_sha256": ins,
                         "seeds": {"partA_clip_subset": SEED, "random_sets": SEED, "bootstrap": "0 + index",
                                   "random_init_model": "torch.manual_seed(0) (wm.extract.load_model)"},
                         "n_clips": {"partA_train": res["part_A"]["vjepa2"]["n_clips"],
                                     "partB_direction_test": int(len(P["direction_test_ids"])),
                                     "partB_speed_test": int(len(test_rows("speed", len(P["speed_test_ids"])))),
                                     "partB_speed_test_pool": int(len(P["speed_test_ids"]))},
                         "gpu_seconds": finfo["gpu_seconds"], "lock_wait_s": finfo["lock_wait_s"],
                         "timing": finfo["timing"], "box": finfo.get("box"), "gpu": finfo.get("gpu"),
                         "split": "splits/split_v1.json (probes fit on train rows, scored on test rows)",
                         "partA_clip_split": "train rows only (asserted disjoint from test)"}
    RES.mkdir(exist_ok=True)
    grid = {mk: res["part_A"][mk].pop("_ratio_grid") for mk in ("vjepa2", "random")}
    (RES / "p5_motion_heads.json").write_text(json.dumps(res, indent=1, default=float))
    make_figure(grid, res)
    print("wrote", RES / "p5_motion_heads.json")


def make_figure(grid, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1.6]})
    lim = max(np.log2(np.array(grid["vjepa2"])).max(), np.log2(np.array(grid["random"])).max())
    for k, mk in enumerate(("vjepa2", "random")):
        g = np.log2(np.array(grid[mk]))
        im = ax[k].imshow(g.T, aspect="auto", cmap="magma", vmin=0, vmax=lim, origin="lower")
        ax[k].set_xticks(range(len(A_BLOCKS)), A_BLOCKS)
        ax[k].set_yticks(range(0, N_HEADS, 3))
        ax[k].set_xlabel("block")
        ax[k].set_ylabel("head")
        ax[k].set_title(f"{'V-JEPA 2' if mk == 'vjepa2' else 'random-init'}: prev-slot disk / background\n"
                        "attention density (log2)", fontsize=10)
    fig.colorbar(im, ax=ax[1], shrink=0.9, label="log2 ratio")
    a = ax[2]
    s = res["part_B"]["vjepa2"]["summary"]
    cols = {"9": "#1f77b4", "12": "#d62728"}
    for j, (b, off) in enumerate((("9", -0.18), ("12", 0.18))):
        e = s["direction_p12"][b]
        d = np.array(e["per_head_drop"])
        ci = np.array(e["per_head_ci95"])
        if d.size == 0:
            continue
        x = np.arange(N_HEADS) + off
        a.errorbar(x, d, yerr=[d - ci[:, 0], ci[:, 1] - d], fmt="o", ms=4, color=cols[b], label=f"block {b}, read at point 12")
        e22 = s["direction_p22"][b]
        a.plot(x, e22["per_head_drop"], "x", color=cols[b], alpha=0.6, label=f"block {b}, read at point 22")
        rr = e["random_k_range"]
        if rr:
            a.axhspan(rr[0], rr[1], color=cols[b], alpha=0.12,
                      label=f"block {b}: random {TOP_K}-head sets (pt 12)")
    a.axhline(0, color="k", lw=0.6)
    a.set_xticks(range(N_HEADS))
    a.set_xlabel("head ablated (zeroed output)")
    a.set_ylabel("direction R² drop (test split)")
    a.set_title("Single-head ablation read by the Part 1 pooled ridge direction probe", fontsize=10)
    a.set_ylim(top=max(a.get_ylim()[1], 0.4) * 1.35)
    a.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_motion_heads.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probes", "forward", "score"))
    ap.add_argument("--root", default=str(ART))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "probes":
        probes()
    elif a.cmd == "forward":
        forward(a.root, a.smoke)
    else:
        score(a.root)

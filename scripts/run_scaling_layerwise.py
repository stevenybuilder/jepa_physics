"""Model scaling of the Part 1 layer-wise probe curves: V-JEPA 2 ViT-L (ours, 24 blocks) vs ViT-H (32) [vs ViT-g (40)].

Same pipeline as Part 1 (wm.extract + wm.probes.layer_sweep): 16 decoded frames -> /255 -> ImageNet mean/std, no
resize/crop; all 2048 tokens; mean pool over tokens after every block. Points per model with L blocks (L + 2 total):
[emb, block1..blockL (pre-LN; block L via a forward hook), final post-LN], the same layout as the ViT-L files.
Extraction is wm.extract.load_model / encode_meanpool (parametrised by size; token means pooled on the fly by hooks).
Only difference: the new encoders run under bf16 autocast (the ViT-L features were extracted in fp32); the mean pool is
taken in fp32. `parity` measures what bf16 does to ViT-L meanpool features against the stored fp32 ones.

Probes: wm.probes recipe (train-standardised features, closed-form ridge, alpha from the 13-value grid by fold-mean R^2
over the 5 split_v1 train folds, refit on all train, score the held-out test split once). Test R^2 CI: 1000-draw
bootstrap over test clips of the fixed test predictions. Onset = first point over 0..L with CV R^2 >= 90% of the CV
max (wm.probes.availability, post-LN point excluded), with the 200-draw train-clip bootstrap of OOF predictions.
Test clips are used only for the final scores.

  python scripts/run_scaling_layerwise.py extract --model vith --init pretrained --out OUT [--limit 16]
  python scripts/run_scaling_layerwise.py parity  --out OUT --limit 16       (ViT-L bf16 vs stored fp32)
  python scripts/run_scaling_layerwise.py probe   --feat OUT/vith_pretrained --tag vith_pretrained --json OUT/probe_vith_pretrained.json
  python scripts/run_scaling_layerwise.py merge   --probes P1.json P2.json ... --out results/p5_scaling_layerwise.json --fig figures/fig_scaling_layerwise.png
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = {"vitl": ("facebook/vjepa2-vitl-fpc64-256", 24, 1024),
          "vith": ("facebook/vjepa2-vith-fpc64-256", 32, 1280),
          "vitg": ("facebook/vjepa2-vitg-fpc64-256", 40, 1408)}
TARGETS = {"direction": ["direction"], "speed": ["speed", "vxvy"], "acceleration": ["acceleration", "axay"]}   # dataset -> variables (wm.probes.VARIABLES)
LOCK = "/tmp/wm_gpu.lock"
N_BOOT_TEST = 1000
SEED = 0


# ---------------------------------------------------------------- pure arithmetic (unit-tested)

def depth_fraction(point, n_blocks):
    """x-axis: block index / n_blocks; the post-LN point (n_blocks + 1) also sits at 1.0."""
    return min(point, n_blocks) / n_blocks


def curve_summary(curve, n_blocks):
    """Peak and onset of a per-point curve over points 0..n_blocks (the post-LN point, if present, is dropped).

    onset = first point >= 0.9 * max (None if max <= 0), exactly wm.probes.availability's rule."""
    c = np.asarray(curve, float)[:n_blocks + 1]
    assert len(c) == n_blocks + 1, (len(c), n_blocks)
    peak = int(np.argmax(c))
    onset = int(np.argmax(c >= 0.9 * c[peak])) if c[peak] > 0 else None
    return {"peak": peak, "peak_score": float(c[peak]), "peak_frac": depth_fraction(peak, n_blocks),
            "onset": onset, "onset_frac": None if onset is None else depth_fraction(onset, n_blocks),
            "final_block_score": float(c[n_blocks]), "decline": float(c[peak] - c[n_blocks])}


def halves_zone(points, cross, n_blocks):
    """First run of consecutive sampled points with cross-half R2 < 0 (transfer collapse) and the first point after it
    with cross-half R2 > 0 (recovery). Returns absolute points and depth fractions (None if no collapse)."""
    neg = [i for i, c in enumerate(cross) if c < 0]
    if not neg:
        return {"collapse_points": [], "recovery_point": None, "collapse_frac": None, "recovery_frac": None}
    i0 = neg[0]
    i1 = i0
    while i1 + 1 < len(cross) and cross[i1 + 1] < 0:
        i1 += 1
    run = [points[i] for i in range(i0, i1 + 1)]
    rec = points[i1 + 1] if i1 + 1 < len(points) else None
    return {"collapse_points": run, "recovery_point": rec,
            "collapse_frac": [depth_fraction(run[0], n_blocks), depth_fraction(run[-1], n_blocks)],
            "recovery_frac": None if rec is None else depth_fraction(rec, n_blocks),
            "min_cross_point": points[int(np.argmin(cross))], "min_cross_r2": float(min(cross))}


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- GPU side

def load_encoder(model_key, init, device):
    """wm.extract.load_model, parametrised by encoder size (same random-init recipe: torch.manual_seed(0))."""
    from wm.extract import load_model
    mid, n_blocks, width = MODELS[model_key]
    return load_model("vjepa2" if init == "pretrained" else "random", device, model_id=mid, n_blocks=n_blocks, width=width)


def encode_meanpool(model, pixel_values, n_blocks, amp):
    """wm.extract.encode_meanpool (per-block token means pooled on the fly by hooks)."""
    from wm.extract import encode_meanpool as enc
    X = enc(model, pixel_values, amp=amp)
    assert X.shape[1] == n_blocks + 2, X.shape
    return X


def model_revision(model_key):
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download(MODELS[model_key][0], allow_patterns=["*.json", "*.safetensors"], local_files_only=True))
    files = {f.name: sha256_file(f) for f in sorted(p.glob("*.safetensors"))}
    return {"hf_id": MODELS[model_key][0], "revision": p.name, "safetensors_sha256": files}


def run_extract(args):
    import torch
    from wm.data import decode, frame_hash, load_table
    from wm.extract import preprocess, set_precision
    set_precision()
    device = torch.device("cuda")
    _, n_blocks, width = MODELS[args.model]
    out_root = Path(args.out) / f"{args.model}_{args.init}"
    info = {"model": args.model, "init": args.init, "amp_bf16": not args.fp32, "batch_size": args.batch_size,
            "n_blocks": n_blocks, "width": width, "datasets": {}, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "random_init_seed": 0 if args.init == "random" else None}
    info["hf"] = model_revision(args.model)   # the random init reads only config.json from this revision
    lock_f = open(LOCK, "w")
    t_wait = time.time()
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    info["lock_wait_s"] = time.time() - t_wait
    try:
        t0 = time.time()
        model = load_encoder(args.model, args.init, device)
        info["load_s"] = time.time() - t0
        for dataset in args.datasets:
            df = load_table(dataset)
            if args.limit:
                df = df.head(args.limit)
            ids = [int(i) for i in df["id"]]
            feats, hashes = [], {}
            t1 = time.time()
            for s in range(0, len(ids), args.batch_size):
                rows = df.iloc[s:s + args.batch_size]
                frames = [decode(v) for v in rows["video"]]
                for i, f in zip(rows["id"], frames):
                    hashes[str(int(i))] = frame_hash(f)
                feats.append(encode_meanpool(model, preprocess(frames).to(device), n_blocks, not args.fp32))
                if s % (args.batch_size * 20) == 0:
                    print(f"{args.model}/{args.init}/{dataset}: {s + len(rows)}/{len(ids)} {time.time() - t1:.0f}s",
                          flush=True)
            torch.cuda.synchronize()
            dt = time.time() - t1
            X = np.concatenate(feats).astype(np.float32)
            assert X.shape == (len(ids), n_blocks + 2, width) and np.isfinite(X).all(), X.shape
            d = out_root / dataset
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "meanpool.npy", X)
            (d / "ids.json").write_text(json.dumps(ids))
            (d / "frame_hash.json").write_text(json.dumps(hashes))
            info["datasets"][dataset] = {"n": len(ids), "gpu_seconds": dt, "meanpool_sha256": sha256_file(d / "meanpool.npy"),
                                         "ids_sha256": sha256_json(ids), "frame_hash_sha256": sha256_json(hashes)}
            print(f"{args.model}/{args.init}/{dataset}: done {len(ids)} clips in {dt:.0f}s", flush=True)
        info["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
        lock_f.close()
    info["total_gpu_seconds"] = sum(v["gpu_seconds"] for v in info["datasets"].values()) + info["load_s"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "datasets"}, indent=1))


def run_parity(args):
    """ViT-L pretrained under bf16 autocast on the first --limit clips of each dataset vs the stored fp32 meanpool."""
    import torch
    from wm.data import decode, load_table
    from wm.extract import preprocess, set_precision
    set_precision()
    device = torch.device("cuda")
    res = {}
    lock_f = open(LOCK, "w")
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        model = load_encoder("vitl", "pretrained", device)
        for dataset in args.datasets:
            df = load_table(dataset).head(args.limit)
            ref = np.load(Path(args.ref_root) / dataset / "vjepa2" / "meanpool.npy", mmap_mode="r")[:args.limit]
            X = np.concatenate([encode_meanpool(model, preprocess([decode(v) for v in df["video"].iloc[s:s + 8]]).to(device), 24, True)
                                for s in range(0, len(df), 8)])
            ref = np.asarray(ref, np.float64)
            rel = np.linalg.norm(X - ref, axis=2) / np.linalg.norm(ref, axis=2)          # [n, 26]
            cos = (X * ref).sum(2) / np.linalg.norm(X, axis=2) / np.linalg.norm(ref, axis=2)
            res[dataset] = {"n": len(df), "rel_l2_err_max_per_point": rel.max(0).round(6).tolist(),
                            "rel_l2_err_median": float(np.median(rel)), "rel_l2_err_max": float(rel.max()),
                            "cos_min": float(cos.min())}
            print(dataset, res[dataset]["rel_l2_err_median"], res[dataset]["rel_l2_err_max"], res[dataset]["cos_min"], flush=True)
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "parity_vitl_bf16.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- probes (CPU)

def boot_ci(Y, P, score_fn, n_boot=N_BOOT_TEST, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(Y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(score_fn(Y[i], P[i])["r2"])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_probe(args):
    from wm.data import load_table
    from wm.probes import (ALPHAS, Standardizer, bootstrap_onset, cv_select_alpha, fit_ridge, predict, split_rows,
                           targets)
    from wm.splits import SPLIT_PATH
    n_blocks = args.n_blocks
    out = {"tag": args.tag, "feat_dir": args.feat, "n_blocks": n_blocks, "n_points": n_blocks + 2, "variables": {}}
    for dataset, variable in [(d, v) for d, vs in TARGETS.items() for v in vs]:
        if args.datasets and dataset not in args.datasets:
            continue
        if args.variables and variable not in args.variables:
            continue
        t0 = time.time()
        df = load_table(dataset)
        ids = df["id"].tolist()
        fdir = Path(args.feat.format(dataset=dataset)) if "{dataset}" in args.feat else Path(args.feat) / dataset
        path = fdir / "meanpool.npy"
        acts = np.load(path, mmap_mode="r")
        stored_ids = json.loads((fdir / "ids.json").read_text())
        assert stored_ids == ids, "features not in manifest order / not the full dataset"
        assert acts.shape[:2] == (len(ids), n_blocks + 2), acts.shape
        Y, kind, score_fn = targets(df, variable)
        tr, te, folds = split_rows(dataset, df)
        Ytr, Yte = Y[tr], Y[te]
        rows, oofs = [], []
        for point in range(n_blocks + 2):
            X = np.asarray(acts[:, point], np.float64)
            st = Standardizer().fit(X[tr])
            Xtr, Xte = st.transform(X[tr]), st.transform(X[te])
            cv = cv_select_alpha(Xtr, Ytr, folds, ALPHAS, score_fn)
            W, b = fit_ridge(Xtr, Ytr, cv["alpha"])
            P = predict(Xte, W, b)
            test = score_fn(Yte, P)
            rows.append({"point": point, "frac": depth_fraction(point, n_blocks), "post_ln": point == n_blocks + 1,
                         "alpha": cv["alpha"], "cv_mean": cv["cv_mean"], "cv_sd": cv["cv_sd"],
                         "test_r2": test["r2"], "test_r2_ci95": boot_ci(Yte, P, score_fn), "test_mae": test["mae"]})
            oofs.append(cv["oof"])
            print(f"{args.tag} {dataset} pt {point:2d}: cv {cv['cv_mean']:.4f} test {test['r2']:.4f}", flush=True)
        cvc = [r["cv_mean"] for r in rows]
        tec = [r["test_r2"] for r in rows]
        onset_ci, n_none = bootstrap_onset(Ytr, oofs[:n_blocks + 1], score_fn, 200, SEED)
        out["variables"][variable] = {
            "dataset": dataset, "kind": kind, "n_train": int(len(tr)), "n_test": int(len(te)),
            "features_sha256": sha256_file(path), "layers": rows,
            "summary_cv": {**curve_summary(cvc, n_blocks), "onset_ci": onset_ci, "onset_boot_draws_without_onset": n_none},
            "summary_test": curve_summary(tec, n_blocks),
            "cpu_seconds": time.time() - t0}
    out["split_file"] = str(SPLIT_PATH.relative_to(ROOT)) if str(SPLIT_PATH).startswith(str(ROOT)) else str(SPLIT_PATH)
    out["split_sha256"] = sha256_file(SPLIT_PATH)
    Path(args.json).write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- merge + figure (Mac)

def run_merge(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    probes = [json.loads(Path(p).read_text()) for p in args.probes]
    by = {}
    for p in probes:   # one tag may come in several files (direction first, speed/acceleration later)
        if p["tag"] in by:
            assert by[p["tag"]]["n_blocks"] == p["n_blocks"] and by[p["tag"]]["split_sha256"] == p["split_sha256"]
            assert not set(by[p["tag"]]["variables"]) & set(p["variables"]), p["tag"]
            by[p["tag"]]["variables"].update(p["variables"])
        else:
            by[p["tag"]] = json.loads(json.dumps(p))
    extract_info = {}
    for p in args.info or []:
        d = json.loads(Path(p).read_text())
        extract_info[f"{d['model']}_{d['init']}_{'+'.join(d['datasets'])}"] = d
    extras = {}
    for p in args.extra or []:
        extras[Path(p).stem] = json.loads(Path(p).read_text())
    # ViT-L cross-check against the Part 1 files (same recipe; the CV numbers must match to ~1e-9)
    xcheck = {}
    for tag, suffix in (("vitl_pretrained", ""), ("vitl_random", "_random")):
        if tag not in by:
            continue
        for ds, vs in TARGETS.items():
          for var in vs:
            f = ROOT / "results" / f"p1a_{ds}_{var}_meanpool{suffix}.json"
            if not f.exists() or var not in by[tag]["variables"]:
                continue
            old = json.loads(f.read_text())["layers"]
            new = by[tag]["variables"][var]["layers"]
            xcheck[f"{tag}/{var}"] = {"file": str(f.relative_to(ROOT)),
                                      "max_abs_diff_cv_mean": float(max(abs(a["cv_mean"] - b["cv_mean"]) for a, b in zip(old, new))),
                                      "max_abs_diff_test_r2": float(max(abs(a["test_r2"] - b["test_r2"]) for a, b in zip(old, new)))}
    zone = None
    if args.pp:
        ph = json.loads(Path(args.pp).read_text())
        lf = ROOT / "results" / "p1a_perpatch_direction_vjepa2_hard.json"
        lj = json.loads(lf.read_text())
        lpts = [r["point"] for r in lj["layers"]]
        lcr = [r["halves"]["cross_r2_mean"] for r in lj["layers"]]
        hpts = [r["point"] for r in ph["layers"]]
        hcr = [r["cross_r2_mean"] for r in ph["layers"]]
        zone = {
            "readout": "hard-render set (artifacts/stimuli/hard, render seed 0, splits/split_hard.json), per-patch time-pooled "
                       "features, pooled ridge fit on one half-frame (8 columns), read on the other half; cross_r2 = mean of "
                       "left->right and right->left (scripts/p1a_perpatch.py halves recipe)",
            "vitl": {"source": str(lf.relative_to(ROOT)), "n_blocks": 24, "points": lpts,
                     "cross_half_r2": lcr, "within_half_r2": [r["halves"]["within_r2_mean"] for r in lj["layers"]],
                     "summary": halves_zone(lpts, lcr, 24)},
            "vith": {"source": "artifacts/p5_scaling/box2/pp_halves_vith_pretrained_hard.json", "n_blocks": 32,
                     "points": hpts, "cross_half_r2": hcr, "cross_half_r2_ci95": [r["cross_r2_ci95"] for r in ph["layers"]],
                     "within_half_r2": [r["within_r2_mean"] for r in ph["layers"]],
                     "extract_gpu_seconds": ph["extract_gpu_seconds"], "summary": halves_zone(hpts, hcr, 32)},
            "vitg": "not run",
            "untrained_vith": "not run",
            "statement": ("On the hard render (seed 0 only; ViT-H only, ViT-g not run), the half-frame sign flip back to positive "
                          "transfer stays at absolute block 9 for ViT-L and ViT-H, so as a fraction of depth it moves earlier "
                          "(0.375 -> 0.28). ViT-H also shows a second negative dip (block 13) that ViT-L's sampled points do not."),
        }
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {
        "experiment": "p5 model scaling of the Part 1 layer-wise mean-pooled ridge probes",
        "models": {tag: {"n_blocks": p["n_blocks"], "variables": p["variables"]} for tag, p in by.items()},
        "definitions": {
            "x_axis": "depth fraction = block index / n_blocks (L = 24 ViT-L, 32 ViT-H, 40 ViT-g); point 0 = patch embedding at 0.0; post-LN point also at 1.0 but excluded from onset/peak",
            "curves": "layers[].test_r2 = held-out split_v1 test R2 (probe refit on all train at the CV alpha), test_r2_ci95 = 1000-draw test-clip bootstrap; layers[].cv_mean = 5-fold train CV R2",
            "summary_cv": "onset = first point with CV R2 >= 0.9 x max CV R2 over points 0..L (wm.probes.availability), onset_ci = 200-draw train-clip bootstrap of OOF predictions; peak = argmax CV R2; *_frac = point / L",
            "summary_test": "same rule applied to the test R2 curve (no CI)",
            "readability_not_zone": "on the supplied clips ViT-L direction is 0.875 CV R2 at block 1, so the 90%-of-max onset is a readability onset vs depth, not a Physics Emergence Zone; only the hard-render per-patch half-frame readout (pp_halves_*) carries a zone, and it was run for ViT-H only (ViT-L reference: results/p1a_perpatch_direction_vjepa2_hard.json)",
            "acceleration": "target |a| on the supplied set; every clip starts at rest, so it is mean speed in disguise (REPORT 3.1); axay / vxvy are the Cartesian targets (paper Fig. 2b)",
            "precision": "ViT-L features: stored fp32 forward (Part 1). ViT-H / ViT-g: bf16 autocast forward, token means in fp32. bf16 parity measured on ViT-L only (parity_vitl_bf16: 16 clips/dataset, median rel. L2 err ~0.5%, cosine >= 0.9957)",
            "untrained": "random init = VJEPA2Config of the same checkpoint + torch.manual_seed(0) (wm.extract.load_model)",
        },
        "vitl_crosscheck_vs_part1_files": xcheck,
        "zone_halfframe_hard": zone,
        "extract_info": extract_info,
        **extras,
        "provenance": {"git_commit": git, "script_sha256": sha256_file(Path(__file__)),
                       "split_file": probes[0]["split_file"], "split_sha256": probes[0]["split_sha256"],
                       "seeds": {"random_init": 0, "test_bootstrap": SEED, "onset_bootstrap": SEED, "folds": "split_v1 (seed 0)"},
                       "notes": args.note or []},
    }
    Path(args.out).write_text(json.dumps(result, indent=1))

    colors = {"vitl": "#1f77b4", "vith": "#d62728", "vitg": "#2ca02c"}
    names = {"vitl": "ViT-L (24)", "vith": "ViT-H (32)", "vitg": "ViT-g (40)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, var in zip(axes, ("direction", "speed", "acceleration")):
        for tag, p in sorted(by.items()):
            m, init = tag.split("_")
            v = p["variables"].get(var)
            if v is None:
                continue
            L = [r for r in v["layers"] if not r["post_ln"]]
            x = [r["frac"] for r in L]
            y = [r["test_r2"] for r in L]
            lo = [r["test_r2_ci95"][0] for r in L]
            hi = [r["test_r2_ci95"][1] for r in L]
            ls = "-" if init == "pretrained" else "--"
            ax.plot(x, y, ls, color=colors[m], lw=1.6, label=f"{names[m]}{'' if init == 'pretrained' else ' untrained'}")
            ax.fill_between(x, lo, hi, color=colors[m], alpha=0.15, lw=0)
            if init == "pretrained":
                on = v["summary_cv"]["onset"]
                ax.axvline(on / p["n_blocks"], color=colors[m], lw=0.8, ls=":")
        ax.set_title({"direction": "direction (sin, cos)", "speed": "speed |v|",
                      "acceleration": "acceleration |a| (mean speed on this set)"}[var])
        ax.set_xlabel("block / depth")
        ax.set_ylim(-0.1, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out test R² (mean-pooled ridge)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("V-JEPA 2 mean-pooled readability vs fraction of depth, by model size. Lines: held-out test R² with 95% "
                 "clip-bootstrap band (dashed = untrained); dotted verticals: CV 90%-of-max onset (trained)", fontsize=9)
    fig.text(0.5, 0.005, "x = block / L, embedding at 0. ViT-L fp32; ViT-H / ViT-g bf16 autocast (parity measured on ViT-L only: "
             "median rel. err 0.5%, cosine ≥ 0.9957). Supplied clips: a readability onset, not an emergence zone.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=150)


# ---------------------------------------------------------------- per-patch half-frame transfer (zone test)

def run_pp_extract(args):
    """Time-pooled per-position features (mean over the 8 token time steps, 16x16 positions kept, index h*16 + w) at
    every point in --points, pooled on the fly by hooks, bf16 autocast -> fp16 memmap [N, P, 256, width].
    Same layout as scripts/p1a_perpatch.py extract (which is fp32 ViT-L only)."""
    import torch
    from wm.data import PROJECT_ROOT, decode, frame_hash, load_table
    from wm.extract import preprocess, set_precision
    set_precision()
    device = torch.device("cuda")
    _, n_blocks, width = MODELS[args.model]
    df = load_table("direction", root=PROJECT_ROOT / args.set_root)
    ids = [int(i) for i in df["id"]]
    points = args.points
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(len(ids), len(points), 256, width))
    hashes = {}
    lock_f = open(LOCK, "w")
    t_wait = time.time()
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    wait = time.time() - t_wait
    try:
        model = load_encoder(args.model, args.init, device)
        enc = model.encoder
        mods = {0: enc.embeddings, **{k + 1: enc.layer[k] for k in range(n_blocks)}, n_blocks + 1: enc.layernorm}
        t0 = time.time()
        for s in range(0, len(ids), args.batch_size):
            rows = df.iloc[s:s + args.batch_size]
            frames = [decode(v) for v in rows["video"]]
            for i, f in zip(rows["id"], frames):
                hashes[str(int(i))] = frame_hash(f)
            got = {}
            B = len(frames)
            hooks = [mods[p].register_forward_hook(
                lambda m, i, o, p=p: got.__setitem__(p, (o[0] if isinstance(o, tuple) else o).float()
                                                      .reshape(B, 8, 256, width).mean(1).half().cpu()))
                for p in points]
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(pixel_values_videos=preprocess(frames).to(device), skip_predictor=True)
            finally:
                for h in hooks:
                    h.remove()
            mm[s:s + B] = torch.stack([got[p] for p in points], 1).numpy()
        torch.cuda.synchronize()
        gpu_s = time.time() - t0
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    mm.flush()
    info = {"model": args.model, "init": args.init, "set_root": args.set_root, "ids": ids, "points": points,
            "shape": list(mm.shape), "gpu_seconds": gpu_s, "lock_wait_s": wait, "amp_bf16": True,
            "frame_hash_sha256": sha256_json(hashes), "hf": model_revision(args.model),
            "pooling": "mean over the 8 token time steps; positions kept (16x16, index h*16 + w, w = column)"}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "ids"}))


def run_pp_halves(args):
    """Half-frame transfer exactly as scripts/p1a_perpatch.py probe's 'halves' (its pooled_probe): one pooled ridge fit
    on the left 8 columns (every train clip x position a sample, folds by clip, alpha by fold-mean R2), read on the
    left (within) and right (cross) halves of test clips, and the reverse; 200-draw test-clip bootstrap CI."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import p1a_perpatch as pp
    from wm.data import PROJECT_ROOT, load_table
    from wm.probes import ALPHAS, split_rows, targets
    info = json.loads(Path(args.memmap).with_suffix(".json").read_text())
    mm = np.load(args.memmap, mmap_mode="r")
    df = load_table("direction", root=PROJECT_ROOT / info["set_root"])
    assert [int(i) for i in df["id"]] == info["ids"]
    Y, _, score_fn = targets(df, "direction")
    split_file = PROJECT_ROOT / args.split
    tr, te, folds = split_rows("direction", df, split_file)
    Ytr, Yte = Y[tr], Y[te]
    left = np.array([p for p in range(256) if p % 16 < 8])
    right = np.array([p for p in range(256) if p % 16 >= 8])
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(te), len(te)) for _ in range(200)]
    rows = []
    for pi, point in enumerate(info["points"]):
        X = np.asarray(mm[:, pi], np.float32)
        aL, _, pL = pp.pooled_probe(X[tr][:, left], Ytr, folds, {"L": X[te][:, left], "R": X[te][:, right]}, score_fn, ALPHAS)
        aR, _, pR = pp.pooled_probe(X[tr][:, right], Ytr, folds, {"R": X[te][:, right], "L": X[te][:, left]}, score_fn, ALPHAS)
        cross = lambda i: (pp.r2_pooled(Yte[i], pL["R"][i]) + pp.r2_pooled(Yte[i], pR["L"][i])) / 2   # noqa: E731
        within = lambda i: (pp.r2_pooled(Yte[i], pL["L"][i]) + pp.r2_pooled(Yte[i], pR["R"][i])) / 2  # noqa: E731
        allidx = np.arange(len(te))
        cb = [cross(b) for b in boots]
        rows.append({"point": point, "frac": depth_fraction(point, info["n_blocks"] if "n_blocks" in info else MODELS[info["model"]][1]),
                     "alpha_left": aL, "alpha_right": aR, "cross_r2_mean": cross(allidx), "within_r2_mean": within(allidx),
                     "cross_r2_ci95": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))]})
        print(f"pt {point:2d}: cross {rows[-1]['cross_r2_mean']:.3f} within {rows[-1]['within_r2_mean']:.3f}", flush=True)
    out = {"model": info["model"], "init": info["init"], "set_root": info["set_root"], "split": args.split,
           "split_sha256": sha256_file(split_file), "n_train": int(len(tr)), "n_test": int(len(te)),
           "memmap_sha256_of_sidecar": sha256_file(Path(args.memmap).with_suffix(".json")),
           "extract_gpu_seconds": info["gpu_seconds"], "layers": rows}
    Path(args.json).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", choices=list(MODELS), required=True)
    e.add_argument("--init", choices=["pretrained", "random"], required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--datasets", nargs="+", default=list(TARGETS))
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--fp32", action="store_true")
    pa = sub.add_parser("parity")
    pa.add_argument("--out", required=True)
    pa.add_argument("--ref-root", default=str(ROOT / "artifacts" / "activations"))
    pa.add_argument("--datasets", nargs="+", default=list(TARGETS))
    pa.add_argument("--limit", type=int, default=16)
    p = sub.add_parser("probe")
    p.add_argument("--feat", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n-blocks", type=int, required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--variables", nargs="+", default=None)
    m = sub.add_parser("merge")
    m.add_argument("--probes", nargs="+", required=True)
    m.add_argument("--info", nargs="*")
    m.add_argument("--extra", nargs="*")
    m.add_argument("--note", nargs="*")
    m.add_argument("--pp", default=None)
    m.add_argument("--out", required=True)
    m.add_argument("--fig", required=True)
    q = sub.add_parser("pp-extract")
    q.add_argument("--model", choices=list(MODELS), default="vith")
    q.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    q.add_argument("--set-root", default="artifacts/stimuli/hard")
    q.add_argument("--points", type=int, nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--batch-size", type=int, default=8)
    h = sub.add_parser("pp-halves")
    h.add_argument("--memmap", required=True)
    h.add_argument("--split", default="splits/split_hard.json")
    h.add_argument("--json", required=True)
    a = ap.parse_args()
    {"extract": run_extract, "parity": run_parity, "probe": run_probe, "merge": run_merge,
     "pp-extract": run_pp_extract, "pp-halves": run_pp_halves}[a.cmd](a)


if __name__ == "__main__":
    main()

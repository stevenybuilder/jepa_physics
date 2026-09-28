"""Frame-shuffle test of the within-clip time code: does the decoded step follow the SLOT (token time position, RoPE)
or the CONTENT (which frames sit there)?

Speed set. The 16 frames form 8 tubelets of 2 frames; a shuffle permutes whole tubelets (the frame order inside a
tubelet is kept): slot s of the shuffled clip shows the content of original tubelet perm[s]. Every test clip is
encoded unshuffled and under 3 slot derangements (no tubelet stays in its slot): `rev` (perm = 7..0) and two random
derangements `d1`, `d2` drawn per clip (numpy default_rng(seed)); rev alone makes slot and content collinear
(content = 7 - slot), the random ones decorrelate them.

Reader (the same as run_time_manifold `decode`, test half): ridge (StandardScaler + RidgeCV, tm.ridge) for the step
index t = 0..7 on per-step timepool features (mean over the 256 spatial tokens of a tubelet) of ALL speed-set train
clips (knot + probe, split_v1), fit per model and point on unshuffled clips; read on the 308 test clips (role test,
disjoint from knot/probe). V-JEPA 2's reader is fit on the stored timepool (artifacts/activations/speed/vjepa2);
the random-init copy (torch.manual_seed(0), same init as every other random-init array) has no stored speed-set
timepool on the Mac, so its train clips are re-encoded here, unshuffled, in the same pass.

Static twins (artifacts/time_static/static, 96 speed-0 renders made by scripts/time_static.py; local copy only):
encoded unshuffled; every tubelet has the same content, so any advance of the decoded step with the slot is
positional. Their reader excludes the 52 train rows the twins were rendered from.

Stages:  plan (Mac)  -> ART/plan.json;  extract (box GPU) -> ART/feat_<model>_<set>.npz;  score (Mac) ->
results/p5_time_shuffle.json (+ figures/fig_time_shuffle.png)
  python scripts/run_time_shuffle.py plan|extract|score [--out ART] [--limit 4]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, load_table  # noqa: E402

T, D = 8, 1024
SAVE_POINTS = (0, 1, 4, 8, 12, 16, 19, 22, 25)
REPORT_POINTS = (1, 12, 22)
PERMS = ("rev", "d1", "d2")
ART = PROJECT_ROOT / "artifacts" / "time_shuffle"
STATIC = PROJECT_ROOT / "artifacts" / "time_static"


# ---------------------------------------------------------------- pure helpers (tests/test_time_shuffle.py)

def derangement(rng, n=T):
    """Uniform random permutation of range(n) with no fixed point (rejection sampling)."""
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).all():
            return p


def shuffle_tubelets(frames, perm, tubelet=2):
    """frames [16, H, W, C] -> the same frames with tubelet slot s showing original tubelet perm[s]."""
    n = len(perm)
    x = frames.reshape((n, tubelet) + frames.shape[1:])
    return x[np.asarray(perm)].reshape(frames.shape)


def slot_content_scores(pred, perms, n_boot=1000, seed=0):
    """pred [n_clips, P, 8] decoded step of the tubelet in slot s; perms [n_clips, P, 8] content index of slot s.
    Returns nearest-slot / nearest-content fractions and an OLS of pred on (1, slot, content) with a clip bootstrap."""
    pred, perms = np.asarray(pred, float), np.asarray(perms, float)
    n = len(pred)
    slot = np.broadcast_to(np.arange(pred.shape[-1], dtype=float), pred.shape)
    ds, dc = np.abs(pred - slot), np.abs(pred - perms)
    near_slot = (ds < dc).reshape(n, -1).mean(1)
    near_content = (dc < ds).reshape(n, -1).mean(1)

    def ols(ix):
        y = pred[ix].reshape(-1)
        X = np.stack([np.ones_like(y), slot[ix].reshape(-1), perms[ix].reshape(-1)], 1)
        return np.linalg.lstsq(X, y, rcond=None)[0]

    full = ols(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([ols(rng.integers(0, n, n)) for _ in range(n_boot)])
    rng_f = np.random.default_rng(seed + 1)
    boots_f = np.array([[near_slot[i].mean(), near_content[i].mean()] for i in
                        (rng_f.integers(0, n, n) for _ in range(n_boot))])
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    y = pred.reshape(-1)
    X = np.stack([np.ones_like(y), slot.reshape(-1), perms.reshape(-1)], 1)
    resid = y - X @ full
    return {"n_clips": int(n), "n_tubelets": int(pred.size),
            "frac_nearer_slot": {"mean": float(near_slot.mean()), "ci95": ci(boots_f[:, 0])},
            "frac_nearer_content": {"mean": float(near_content.mean()), "ci95": ci(boots_f[:, 1])},
            "frac_tie": float(1 - near_slot.mean() - near_content.mean()),
            "coef_slot": {"mean": float(full[1]), "ci95": ci(boots[:, 1])},
            "coef_content": {"mean": float(full[2]), "ci95": ci(boots[:, 2])},
            "intercept": {"mean": float(full[0]), "ci95": ci(boots[:, 0])},
            "r2": float(1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "mae_to_slot": float(ds.mean()), "mae_to_content": float(dc.mean()),
            "slot_content_corr": float(np.corrcoef(slot.reshape(-1), perms.reshape(-1))[0, 1])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- plan (Mac)

def plan(args):
    from wm.p2_data import load_inputs
    d0 = load_inputs("speed", 0)
    df, role, train = d0["df"], d0["role"], d0["is_train"]
    test_rows = np.flatnonzero(role == "test")
    train_rows = np.flatnonzero(train)
    assert not np.intersect1d(test_rows, train_rows).size
    rng = np.random.default_rng(args.seed)
    perms = {"rev": [list(range(T - 1, -1, -1))] * len(test_rows),
             "d1": [derangement(rng).tolist() for _ in test_rows],
             "d2": [derangement(rng).tolist() for _ in test_rows]}
    st = json.loads((STATIC / "plan.json").read_text()) if (STATIC / "plan.json").exists() else None
    info = {"seed": args.seed, "test_rows": test_rows.tolist(), "test_ids": df["id"].to_numpy()[test_rows].tolist(),
            "train_rows": train_rows.tolist(), "train_ids": df["id"].to_numpy()[train_rows].tolist(),
            "perms": perms, "save_points": list(SAVE_POINTS),
            "static": None if st is None else {"n": st["n"], "rows": st["rows"], "ids": st["ids"],
                                               "frame_hash": st["frame_hash"]["static"]}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(info))
    print("plan", len(test_rows), "test", len(train_rows), "train", "static" if st else "no static")


# ---------------------------------------------------------------- extract (box GPU; caller holds flock /tmp/wm_gpu.lock)

def extract(args):
    import torch
    from wm.extract import encode, load_model, pick_device, preprocess, set_precision
    from wm.data import frame_hash
    out = Path(args.out)
    info = json.loads((out / "plan.json").read_text())
    lim = args.limit
    df = load_table("speed")
    set_precision()
    dev = pick_device()
    test_rows = np.array(info["test_rows"])[:lim]
    video = df["video"].to_numpy()
    test_frames = [decode(video[r]) for r in test_rows]
    sets = {"test_orig": lambda: test_frames}
    for p in PERMS:
        sets[f"test_{p}"] = (lambda p=p: [shuffle_tubelets(f, info["perms"][p][i]) for i, f in enumerate(test_frames)])
    if info["static"] is not None and (out / "static").exists():
        def static_frames():
            fr = [decode(out / "static" / f"clip_{i:04d}.mp4") for i in range(info["static"]["n"])][:lim]
            assert [frame_hash(f) for f in fr] == info["static"]["frame_hash"][:len(fr)]
            return fr
        sets["static"] = static_frames
    train_rows = np.array(info["train_rows"])[:lim]
    pts = list(SAVE_POINTS)
    for model_kind in args.models:
        model = load_model(model_kind, dev)
        todo = list(sets) + (["train"] if model_kind == "random" else [])
        for name in todo:
            path = out / f"feat_{model_kind}_{name}.npz"
            if path.exists():
                continue
            frames = (lambda: [decode(video[r]) for r in train_rows]) if name == "train" else sets[name]
            fl = frames()
            res = np.zeros((len(fl), len(pts), T, D), np.float16)
            t0 = time.time()
            with torch.no_grad():
                for s in range(0, len(fl), args.batch):
                    points, _ = encode(model, preprocess(fl[s:s + args.batch]).to(dev))
                    for k, p in enumerate(pts):
                        x = points[p]
                        res[s:s + len(x), k] = x.reshape(len(x), T, -1, D).mean(2).float().cpu().numpy()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
            secs = time.time() - t0
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, timepool=res, points=np.array(pts), gpu_seconds=secs, device=str(dev), n=len(fl),
                     gpu_name=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
            tmp.rename(path)
            print(f"{model_kind} {name} n={len(fl)} {secs:.1f}s", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    (out / "DONE").write_text(time.strftime("%FT%T%z"))


# ---------------------------------------------------------------- score (Mac)

def _score_point(out, model, k, L):
    from wm import timeman as tm
    out = Path(out)
    info = json.loads((out / "plan.json").read_text())
    test_rows, train_rows = np.array(info["test_rows"]), np.array(info["train_rows"])
    perms = np.stack([np.array(info["perms"][p]) for p in PERMS], 1)
    st = info["static"]
    static_src = np.array(st["rows"]) if st else np.array([], int)
    tt = np.tile(np.arange(T), len(train_rows))
    stored_v = np.load(PROJECT_ROOT / "artifacts/activations/speed/vjepa2/timepool.npy", mmap_mode="r")
    feat = {s: np.load(out / f"feat_{model}_{s}.npz")["timepool"][:, k] for s in
            ["test_orig", *[f"test_{p}" for p in PERMS]] + (["static"] if st else [])}
    Xtr = (np.asarray(stored_v[train_rows, L], np.float64) if model == "vjepa2"
           else np.load(out / "feat_random_train.npz")["timepool"][:, k].astype(np.float64))
    reader = tm.ridge(Xtr.reshape(-1, D), tt)
    rd = lambda X: reader.predict(np.asarray(X, np.float64).reshape(-1, D)).reshape(X.shape[:-1])  # noqa: E731
    p0 = rd(feat["test_orig"])
    m = {"unshuffled_test": {"t_r2": tm.r2(np.tile(np.arange(T), len(p0)), p0.reshape(-1)),
                             "t_mae_steps": float(np.abs(p0 - np.arange(T)).mean())}}
    if model == "vjepa2":
        ref = np.asarray(stored_v[test_rows, L], np.float64)
        m["parity_gpu_vs_stored_rel"] = float(np.linalg.norm(feat["test_orig"] - ref) / np.linalg.norm(ref))
    P = np.stack([rd(feat[f"test_{p}"]) for p in PERMS], 1)                                # [n, P, 8]
    m["shuffled_pooled"] = slot_content_scores(P, perms)
    m["shuffled_random_derangements"] = slot_content_scores(P[:, 1:], perms[:, 1:])
    m["per_perm_fractions"] = {p: {q: slot_content_scores(P[:, i:i + 1], perms[:, i:i + 1], n_boot=200)[q]
                                   for q in ("frac_nearer_slot", "frac_nearer_content", "mae_to_slot",
                                             "mae_to_content")} for i, p in enumerate(PERMS)}
    m["mean_decoded_by_slot_shuffled"] = P.reshape(-1, T).mean(0).tolist()
    m["mean_decoded_by_content_shuffled"] = [float(P[perms == c].mean()) for c in range(T)]
    if st:
        keep = np.setdiff1d(np.arange(len(train_rows)), np.flatnonzero(np.isin(train_rows, static_src)))
        r_s = tm.ridge(Xtr[keep].reshape(-1, D), np.tile(np.arange(T), len(keep)))
        Ps = r_s.predict(feat["static"].astype(np.float64).reshape(-1, D)).reshape(-1, T)
        m["static"] = {"n_clips": int(len(Ps)), "slope_decoded_on_slot": tm.cluster_bootstrap(tm.per_clip_slope(Ps)),
                       "mae_to_slot": float(np.abs(Ps - np.arange(T)).mean()),
                       "mean_decoded_by_slot": Ps.mean(0).tolist(), "reader_train_clips": int(len(keep))}
    return m, P.astype(np.float32), p0.astype(np.float32)


def score(args):
    from wm.provenance import provenance
    out = Path(args.out)
    info = json.loads((out / "plan.json").read_text())
    test_rows, train_rows = np.array(info["test_rows"]), np.array(info["train_rows"])
    perms = np.stack([np.array(info["perms"][p]) for p in PERMS], 1)                      # [n, P, 8]
    st = info["static"]
    res = {"design": {"n_test_clips": int(len(test_rows)), "n_train_clips_reader": int(len(train_rows)),
                      "perms": list(PERMS), "perm_seed": info["seed"], "points": list(SAVE_POINTS),
                      "static_set": "artifacts/time_static/static (local copy, 96 speed-0 twins)" if st else "absent"},
           "models": {}, "gpu": {}}
    files = sorted(out.glob("feat_*.npz"))
    gpu = {f.name: float(np.load(f)["gpu_seconds"]) for f in files}
    res["gpu"] = {"files_seconds": gpu, "total_gpu_seconds": float(sum(gpu.values())),
                  "device": str(np.load(files[0])["gpu_name"]),
                  "box": args.box}
    res["artifact_sha256"] = {f.name: sha256(f) for f in files}
    from joblib import Parallel, delayed
    jobs = [(model, k, L) for model in ("vjepa2", "random") for k, L in enumerate(SAVE_POINTS)]
    outs = Parallel(n_jobs=args.jobs)(delayed(_score_point)(str(out), model, k, L) for model, k, L in jobs)
    fig_rows, preds = {}, {}
    for (model, k, L), (m, P, p0) in zip(jobs, outs):
        res["models"].setdefault(model, {})[str(L)] = m
        fig_rows.setdefault(model, []).append((L, m))
        preds[f"{model}_L{L}_shuffled"], preds[f"{model}_L{L}_unshuffled"] = P, p0
    np.savez(out / "decoded_steps.npz", perms=perms, perm_names=np.array(PERMS), **preds)
    res["keys"] = {
        "models[m][point].shuffled_pooled": "all 3 derangements (rev, d1, d2) x 8 slots x test clips; pred = reader's "
            "decoded step of the tubelet in slot s. frac_nearer_slot / frac_nearer_content: |pred - slot| < |pred - "
            "content| (and vice versa; content = original tubelet index); coef_slot / coef_content: OLS of pred on "
            "(1, slot, content) pooled over tubelets; CIs: 1000 clip-bootstrap resamples. slot/content = positional / "
            "content-derived (slot 1, content 0 = pure positional).",
        "models[m][point].shuffled_random_derangements": "the same on d1 + d2 only (slot-content correlation ~ -0.14)",
        "models[m][point].per_perm_fractions": "per derangement; rev has content = 7 - slot",
        "models[m][point].unshuffled_test": "reader on the unshuffled test clips (as decode.test.t_r2)",
        "models[m][point].static": "96 speed-0 twins, unshuffled; slope of decoded step on slot per clip (1 = frame "
            "counter from position, 0 = no time signal without motion); reader refit without the 52 train source rows",
        "parity_gpu_vs_stored_rel": "V-JEPA 2 unshuffled test re-encoding vs stored timepool, relative L2"}
    res["provenance"] = provenance(seeds={"perms": info["seed"], "bootstrap": 0}, points=list(SAVE_POINTS),
                                   script_sha256=sha256(Path(__file__)), box=args.box,
                                   n_test_clips=int(len(test_rows)), n_static_clips=int(st["n"]) if st else 0)
    Path(args.result).write_text(json.dumps(res, indent=1))
    print("wrote", args.result)
    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, model in zip(axes, ("vjepa2", "random")):
            Ls = [L for L, _ in fig_rows[model]]
            for key, lab, c in (("coef_slot", "slot (position)", "C0"), ("coef_content", "content (original step)", "C3")):
                y = [m["shuffled_pooled"][key]["mean"] for _, m in fig_rows[model]]
                lo = [m["shuffled_pooled"][key]["ci95"][0] for _, m in fig_rows[model]]
                hi = [m["shuffled_pooled"][key]["ci95"][1] for _, m in fig_rows[model]]
                ax.plot(Ls, y, "o-", color=c, label=lab)
                ax.fill_between(Ls, lo, hi, color=c, alpha=0.2)
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(1, color="k", lw=0.6, ls=":")
            ax.set_title({"vjepa2": "V-JEPA 2", "random": "random-init copy"}[model] + ": decoded step on shuffled clips",
                         fontsize=9)
            ax.set_xlabel("read point")
        axes[0].set_ylabel("regression coefficient")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "extract", "score"])
    ap.add_argument("--out", default=str(ART))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="extract: first N clips per set (smoke)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=9, help="score: parallel (model, point) workers")
    ap.add_argument("--models", nargs="+", default=["vjepa2", "random"])
    ap.add_argument("--box", default="vast 53252443 (box 3, RTX 4060 Ti)")
    ap.add_argument("--result", default=str(PROJECT_ROOT / "results" / "p5_time_shuffle.json"))
    ap.add_argument("--fig", default=str(PROJECT_ROOT / "figures" / "fig_time_shuffle.png"))
    args = ap.parse_args()
    {"plan": plan, "extract": extract, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()

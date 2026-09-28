"""Object-binding test on the relational clips (physics paper App. C.1 hypothesis: velocity is most "bound" to its object
in the middle layers, re-formatted for latent prediction at the end).

Per model (V-JEPA 2, random-init), point (26) and pool (disk 1 tokens, disk 2 tokens, background tokens;
scripts/extract_relational_objpool.py): ridge probes (wm.probes recipe via run_relational_motion.cv_multi; stored split
splits/split_relational.json, alpha by fold-mean R^2 over the 5 train folds) for v1, |v1|, v2, |v2| (velocity along x:
signed velocity carries direction, |v| is speed). v1 and v2 are near-independent by design (corr 0.067), so reading a
disk's velocity from the OTHER disk's tokens measures cross-object mixing, not label correlation.

Binding index per point = (own-pool R^2 - other-pool R^2) / own-pool R^2 on pooled out-of-fold R^2 of the 509 train
clips, with a 1000-draw paired clip-bootstrap 95% CI; test R^2 (disjoint 128 clips) reported beside it.
Appends key "binding" to results/p5_relational_motion.json; figure figures/fig_relational_binding.png.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT  # noqa: E402
from wm.metrics import r2  # noqa: E402
from wm.probes import Standardizer, fit_ridge, predict, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("rrm", Path(__file__).with_name("run_relational_motion.py"))
rrm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrm)

ACT = PROJECT_ROOT / "artifacts" / "activations" / "stimuli_relational_objpool"
POOLS = ["disk1", "disk2", "background"]
TGT = {"v1": "v1", "speed1": "abs_v1", "v2": "v2", "speed2": "abs_v2"}
OWN = {"v1": ("disk1", "disk2"), "speed1": ("disk1", "disk2"), "v2": ("disk2", "disk1"), "speed2": ("disk2", "disk1")}
EMERGENCE = [8, 9]
LATE = list(range(20, 25))
CACHE = PROJECT_ROOT / "artifacts" / "relational" / "binding.json"


def main():
    df = rrm.table()
    df["abs_v1"], df["abs_v2"] = df["v1"].abs(), df["v2"].abs()
    split = json.loads(rrm.SPLIT.read_text())["relational"]
    pos = {int(i): r for r, i in enumerate(df["id"])}
    tr = np.array([pos[i] for i in split["train_ids"]])
    te = np.array([pos[i] for i in split["test_ids"]])
    folds = np.array([split["fold"][str(i)] for i in split["train_ids"]])
    Y = df[list(TGT.values())].to_numpy(float)
    idx = rrm.boot_idx(len(tr))
    out = {"design": "pools = token means over disk 1 (orange) / disk 2 (blue) / background tokens, all 8 time steps",
           "binding_index": "(own R^2 - other R^2) / own R^2, pooled OOF R^2 on train, paired 1000-draw clip bootstrap",
           "emergence_zone_points": EMERGENCE, "late_points": LATE, "models": {}}
    for model in ("vjepa2", "random"):
        folder = ACT / model
        P = np.load(folder / "objpool.npy", mmap_mode="r")
        assert json.loads((folder / "ids.json").read_text()) == df["id"].tolist()
        ntok = np.load(folder / "ntok.npy")
        shared = np.load(folder / "n_shared_tokens.npy")
        rows = []
        for p in range(P.shape[1]):
            oof, rec = {}, {"point": p, "r2": {}}
            for k, pool in enumerate(POOLS):
                X = np.asarray(P[:, p, k], np.float64)
                st = Standardizer().fit(X[tr])
                Xtr, Xte = st.transform(X[tr]), st.transform(X[te])
                cvs = rrm.cv_multi(Xtr, Y[tr], folds)
                for j, t in enumerate(TGT):
                    W, b = fit_ridge(Xtr, Y[tr, j], cvs[j]["alpha"])
                    oof[(pool, t)] = cvs[j]["oof"]
                    rec["r2"][f"{t}|{pool}"] = {"oof_r2": r2(Y[tr, j], cvs[j]["oof"]),
                                                "oof_r2_ci": rrm.ci(rrm.boot_r2(Y[tr, j], cvs[j]["oof"], idx)),
                                                "cv_mean": cvs[j]["cv_mean"], "alpha": cvs[j]["alpha"],
                                                "test_r2": r2(Y[te, j], predict(Xte, W, b)[:, 0])}
            rec["binding_index"] = {}
            for j, t in enumerate(TGT):
                own, oth = OWN[t]
                bo = rrm.boot_r2(Y[tr, j], oof[(own, t)], idx)
                bx = rrm.boot_r2(Y[tr, j], oof[(oth, t)], idx)
                a, b = rec["r2"][f"{t}|{own}"]["oof_r2"], rec["r2"][f"{t}|{oth}"]["oof_r2"]
                rec["binding_index"][t] = {"value": (a - b) / a, "ci": rrm.ci((bo - bx) / bo),
                                           "own_minus_other": a - b, "own_minus_other_ci": rrm.ci(bo - bx)}
            rows.append(rec)
            print(model, p, " ".join(f"{t}:{rec['binding_index'][t]['value']:.2f}" for t in TGT),
                  " own v1 %.3f other v1 %.3f bg v1 %.3f" % tuple(rec["r2"][f"v1|{q}"]["oof_r2"] for q in POOLS),
                  flush=True)
        prof = {}
        for t in TGT:
            bi = np.array([r["binding_index"][t]["value"] for r in rows])
            prof[t] = {"argmax_point_0_24": int(np.argmax(bi[:25])), "max": float(bi[:25].max()),
                       "mean_emergence_8_9": float(bi[EMERGENCE].mean()), "mean_late_20_24": float(bi[LATE].mean()),
                       "point_1": float(bi[1]), "point_25": float(bi[25])}
        out["models"][model] = {"points": rows, "profile": prof,
                                "tokens_per_pool_mean": ntok.mean(0).tolist(), "n_clips_shared_tokens": int((shared > 0).sum()),
                                "index_json_sha256": hashlib.sha256((folder / "index.json").read_bytes()).hexdigest(),
                                "objpool_sha256": hashlib.sha256((folder / "objpool.npy").read_bytes()).hexdigest()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out))
    merge_into_results(out)
    figure(out, PROJECT_ROOT / "figures" / "fig_relational_binding.png")
    print(json.dumps({m: out["models"][m]["profile"] for m in out["models"]}, indent=1))


def merge_into_results(out):
    """Add key "binding" to results/p5_relational_motion.json (run_relational_motion.py also merges CACHE)."""
    path = PROJECT_ROOT / "results" / "p5_relational_motion.json"
    if not path.exists():
        print("results file not written yet; binding kept in", CACHE)
        return
    res = json.loads(path.read_text())
    res.pop("provenance", None)
    res["binding"] = out
    res["keys"]["binding"] = ("per-object pools (disk1/disk2/background) x points x {v1, speed1, v2, speed2}: r2[t|pool] "
                              "(oof_r2 + CI, cv_mean, test_r2), binding_index[t] (value, CI, own_minus_other + CI); "
                              "profile = argmax / emergence (8-9) / late (20-24) means of the index")
    write_json(path, res, rrm.SPLIT)


def figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    pts = np.arange(26)
    col = {"v1": "#d68910", "v2": "#2471a3", "speed1": "#d68910", "speed2": "#2471a3"}
    for model, ls in (("vjepa2", "-"), ("random", ":")):
        rows = out["models"][model]["points"]
        for t in ("v1", "v2"):
            own = OWN[t][0]
            for pool, mk in ((own, "o"), (OWN[t][1], "x"), ("background", "")):
                y = [r["r2"][f"{t}|{pool}"]["oof_r2"] for r in rows]
                ax[0 if t == "v1" else 1].plot(pts, y, ls, marker=mk, ms=3, color={"disk1": "#d68910", "disk2": "#2471a3",
                                               "background": "grey"}[pool], label=f"{model}: {pool} tokens")
        for t in TGT:
            v = [r["binding_index"][t]["value"] for r in rows]
            lo = [r["binding_index"][t]["ci"][0] for r in rows]
            hi = [r["binding_index"][t]["ci"][1] for r in rows]
            ax[2].plot(pts, v, ls, color=col[t], lw=1.5 if t.startswith("v") else 0.8,
                       label=f"{model}: {t}")
            if model == "vjepa2":
                ax[2].fill_between(pts, lo, hi, color=col[t], alpha=0.1)
    for a, ttl in zip(ax, ("disk 1 (orange) velocity v1 read from each pool", "disk 2 (blue) velocity v2 read from each pool",
                           "binding index (own - other) / own")):
        a.axvspan(7.5, 9.5, color="#f5b7b1", alpha=0.3)
        a.axvspan(19.5, 24.5, color="#d5dbdb", alpha=0.4)
        a.set(xlabel="layer point (shaded: emergence 8-9, late 20-24)", title=ttl)
        a.legend(fontsize=6)
    ax[0].set(ylabel="pooled OOF R$^2$", ylim=(-0.1, 1.02))
    ax[1].set(ylim=(-0.1, 1.02))
    ax[2].set(ylim=(-0.2, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

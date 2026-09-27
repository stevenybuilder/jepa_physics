"""Object permanence on the direction set (spec §6 extra): direction and speed decodability from time-step tokens
after the disk has left the frame. See wm.permanence for the protocol.

  python scripts/run_object_permanence.py [--points 1 4 8 12 16 22 25] [--n-boot 1000] [--n-perm 1000]

Writes results/p1a_object_permanence.json and figures/fig6_object_permanence.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, load_table  # noqa: E402
from wm.permanence import LAYERS, run  # noqa: E402
from wm.probes import split_rows, targets, write_json  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--points", type=int, nargs="+", default=list(LAYERS))
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--n-perm", type=int, default=1000)
parser.add_argument("--act-root", type=Path, default=PROJECT_ROOT / "artifacts" / "activations")
parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures")
args = parser.parse_args()

DATASET = "direction"
df = load_table(DATASET)
tr, te, folds = split_rows(DATASET, df)
folder = args.act_root / DATASET / "vjepa2"
timepool = np.load(folder / "timepool.npy", mmap_mode="r")
diskmask = np.load(folder / "diskmask.npy")
if (folder / "ids.json").exists():
    assert json.loads((folder / "ids.json").read_text()) == df["id"].tolist(), "activations not in manifest order"
qa_path = args.results / "qa_direction.json"
qa_any_frame = json.loads(qa_path.read_text()).get("n_clips_any_frame_disk_absent") if qa_path.exists() else None

out = {"dataset": DATASET, "pool": "timepool", "model": "vjepa2", "points": args.points,
       "protocol": ("ridge (sin, cos) / scalar probe fit on VISIBLE time-step tokens of train clips only; per-time-step "
                    "standardiser fit on train visible tokens; alpha by 5-fold CV over those tokens with the split's "
                    "clip-level folds. 'test' = the all-train probe read on test clips (spec: test only). 'pooled' = "
                    "test + train clips, train clips read by 5-fold cross-fitted probes (no clip in the fit that "
                    "scores it; test labels never enter a fit). CIs: 1000 clip-bootstrap draws. absent_shuffled: "
                    "clip-level label permutation within the absent set (1000 draws); p_mae = P(null MAE <= observed)."),
       "absence_rule": ("a time step is absent when its diskmask [16x16 token grid] is empty: no disk pixel in either "
                        "frame of the tubelet. results/qa_direction.json counts clips with ANY absent FRAME "
                        f"({qa_any_frame}); clips whose only missing frame shares a tubelet with a visible frame have "
                        "no absent token step and are not counted here."),
       "caveat": ("the V-JEPA 2 encoder attends over all 8 time steps with no causal mask, so an absent-step token can "
                  "read the same clip's visible steps directly. Above-null decoding at absent steps shows the variable "
                  "is carried into tokens whose frames hold no disk, not that a memory mechanism keeps it. The "
                  "random-init timepool, which would separate 'any attention mixing' from 'learned carrying', was not "
                  "stored (random-init extraction kept meanpool only), so that control is skipped."),
       "variables": {}}

random_tp = args.act_root / DATASET / "random" / "timepool.npy"
out["random_init"] = ("present: not run by this script version" if random_tp.exists()
                      else f"skipped: {random_tp.relative_to(PROJECT_ROOT) if random_tp.is_relative_to(PROJECT_ROOT) else random_tp} "
                           "does not exist (only meanpool was stored for the random-init model)")
print(out["random_init"])

# direction: every clip; speed: the constant-speed (motion == velocity) clips only
Y_dir, kind_dir, _ = targets(df, "direction")
print("== direction")
out["variables"]["direction"] = {"rows": "all 1500 direction-set clips", "target": "[sin theta, cos theta]",
                                 **run(timepool, diskmask, Y_dir, kind_dir, tr, te, folds, args.points,
                                       args.n_boot, args.n_perm)}

vel = df["motion"].to_numpy() == "velocity"
keep_tr, keep_te = vel[tr], vel[te]
Y_sp, kind_sp, _ = targets(df, "speed")
print("== speed (velocity clips)")
out["variables"]["speed"] = {"rows": f"motion == velocity clips only (train {keep_tr.sum()}, test {keep_te.sum()})",
                             "target": "speed_mps (m/s)",
                             **run(timepool, diskmask, Y_sp, kind_sp, tr[keep_tr], te[keep_te], folds[keep_tr],
                                   args.points, args.n_boot, args.n_perm)}
out["speed_note"] = ("every clip that loses the disk is a velocity clip, and they are the faster ones: the absent "
                     "set's speed range is narrower than the probe's, so R2 on it is taken against a smaller variance; "
                     "compare MAE to the shuffled-label null rather than reading R2 alone.")
write_json(args.results / "p1a_object_permanence.json", out)
print(f"wrote {args.results / 'p1a_object_permanence.json'}")


# ------------------------------------------------------------------ figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def err(r, key):
    lo, hi = r.get(f"{key}_ci", [np.nan, np.nan])
    v = r.get(key, np.nan)
    return v, v - lo, hi - v


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
units = {"direction": "circular MAE (deg)", "speed": "MAE (m/s)"}
for row, var in enumerate(("direction", "speed")):
    res = out["variables"][var]
    pts = [r["point"] for r in res["points"]]
    ax = axes[row, 0]
    series = [("test", "visible", "C0", "o", "test, visible steps"),
              ("pooled", "visible_time_matched", "C2", "s", "pooled, visible steps at absent-step times"),
              ("pooled", "absent", "C3", "D", "pooled, absent steps"),
              ("test", "absent", "C1", "v", "test, absent steps")]
    for j, (where, key, color, marker, label) in enumerate(series):
        vals = [err(r[where][key], "mae") for r in res["points"]]
        x = np.array(pts) + (j - 1.5) * 0.18
        ax.errorbar(x, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], color=color,
                    marker=marker, ms=4, lw=1.2, capsize=2, label=label)
    null = [r["pooled"].get("absent_shuffled") or {} for r in res["points"]]
    ax.fill_between(pts, [n.get("mae_95", [np.nan, np.nan])[0] for n in null],
                    [n.get("mae_95", [np.nan, np.nan])[1] for n in null], color="0.7", alpha=0.5,
                    label="pooled absent, shuffled labels (95%)")
    ax.set_xlabel("layer point")
    ax.set_ylabel(units[var])
    ax.set_title(f"{var}: readout of a probe fit on visible steps")
    ax.legend(fontsize=7, frameon=False)
    ax = axes[row, 1]
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(res["points"]):
        per = r["pooled"]["persistence"]
        lags = sorted(int(k) for k in per if per[k].get("n_clips", 0))
        vals = [err(per[str(L)], "mae") for L in lags]
        ax.errorbar(np.array(lags) + (i - 3) * 0.04, [v[0] for v in vals],
                    yerr=[[v[1] for v in vals], [v[2] for v in vals]], color=cmap(i / max(1, len(pts) - 1)),
                    marker="o", ms=4, lw=1.2, capsize=2, label=f"point {r['point']}")
    n_by_lag = {int(k): v["n_clips"] for k, v in res["points"][0]["pooled"]["persistence"].items()}
    ax.set_xticks(sorted(n_by_lag))
    ax.set_xticklabels([f"{L}\n(n={n_by_lag[L]})" for L in sorted(n_by_lag)])
    ax.set_xlabel("steps after the last visible step (0 = last visible; pooled clips)")
    ax.set_ylabel(units[var])
    ax.set_title(f"{var}: persistence after the disk leaves")
    ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
args.figures.mkdir(parents=True, exist_ok=True)
fig.savefig(args.figures / "fig6_object_permanence.png", dpi=150)
print(f"wrote {args.figures / 'fig6_object_permanence.png'}")

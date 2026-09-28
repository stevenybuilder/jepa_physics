"""Merge results/p5_motion_geometry_parts/*.json into results/p5_motion_geometry.json (+ provenance) and draw
figures/fig_motion_subspace_angles.png, fig_motion_leakage.png, fig_motion_coordinate_grid.png."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wm.provenance import git_commit  # noqa: E402

PARTS = ROOT / "results" / "p5_motion_geometry_parts"
FIG = ROOT / "figures"
POINTS = ["1", "4", "8", "12", "16", "19", "22"]


def load(name):
    p = PARTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    parts = {k: load(k) for k in ("exp1", "exp1w", "exp2", "exp3", "exp4")}
    extra = {k: load(k) for k in ("chord_leak", "mlp_calibration")}
    gc = git_commit()
    out = {"provenance": {"commit": gc.get("commit") if isinstance(gc, dict) else str(gc),
                          "dirty": gc.get("dirty") if isinstance(gc, dict) else None,
                          "script": "scripts/run_motion_geometry.py (exp1 exp1w exp2 exp3 exp4) + scripts/fig_motion_geometry.py",
                          "module": "src/wm/motiongeom.py (tests/test_motiongeom.py)",
                          "split": "splits/split_v1.json", "activations": "artifacts/activations/{speed,acceleration,direction}/{vjepa2,random}/{meanpool,timepool}.npy",
                          "cpu_only": True},
           "exp1_subspace_atlas": parts["exp1"], "exp1_whitened_metric": parts["exp1w"],
           "exp2_interference_leakage": parts["exp2"], "exp3_within_vs_between_and_fourier": parts["exp3"],
           "exp4_coordinate_search_and_exp5_shortcuts": parts["exp4"],
           "exp2b_chord_leak_mechanism": extra["chord_leak"], "mlp_readout_calibration": extra["mlp_calibration"]}
    (ROOT / "results" / "p5_motion_geometry.json").write_text(json.dumps(out, indent=1))
    FIG.mkdir(exist_ok=True)
    # --- fig 1: overlap heatmaps by point
    names = ["dir", "dir2", "speed", "acc", "pos", "time"]
    rows = []
    if parts["exp1"]:
        rows.append(("raw metric, V-JEPA 2", parts["exp1"]["points"].get("vjepa2", {})))
    if parts["exp1w"]:
        rows.append(("noise-whitened, V-JEPA 2", parts["exp1w"]["points"].get("vjepa2", {})))
        rows.append(("noise-whitened, random init", parts["exp1w"]["points"].get("random", {})))
    if rows:
        fig, axes = plt.subplots(len(rows), len(POINTS), figsize=(2.3 * len(POINTS), 2.5 * len(rows)), squeeze=False)
        for i, (lab, d) in enumerate(rows):
            for j, p in enumerate(POINTS):
                ax = axes[i, j]
                M = np.full((6, 6), np.nan)
                if p in d:
                    for a in range(6):
                        M[a, a] = 1
                        for b in range(6):
                            k1, k2 = f"{names[a]}|{names[b]}", f"{names[b]}|{names[a]}"
                            v = d[p]["pairs"].get(k1) or d[p]["pairs"].get(k2)
                            if v and a != b:
                                M[a, b] = v["overlap"]
                ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
                for a in range(6):
                    for b in range(6):
                        if np.isfinite(M[a, b]) and a != b:
                            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=5.5,
                                    color="w" if M[a, b] < 0.6 else "k")
                ax.set_xticks(range(6)); ax.set_yticks(range(6))
                ax.set_xticklabels(names, rotation=90, fontsize=6); ax.set_yticklabels(names, fontsize=6)
                if i == 0:
                    ax.set_title(f"point {p}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=7)
        fig.suptitle("Subspace overlap (mean cos$^2$ of principal angles) between motion codes, by point", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "fig_motion_subspace_angles.png", dpi=160)
        plt.close(fig)
    # --- fig 2: leakage matrix
    e2 = parts["exp2"]
    if e2 and "sets" in e2:
        rlab = ["dir", "dir2", "speed", "acc", "time", "pos"]
        clab = ["direction", "speed", "acceleration", "time", "position"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, p in zip(axes, ("12", "22")):
            M = np.full((6, 5), np.nan)
            sp = e2["sets"].get("speed", {}).get(p)
            ac = e2["sets"].get("acceleration", {}).get(p)
            if sp:
                cmap_ = {"dir": 0, "s": 1, "other_scalar": 2, "time": 3, "pos": 4}
                for r, e in (("dir", "dir"), ("dir2", "dir2"), ("speed", "s"), ("time", "time"), ("pos", "pos")):
                    for g, c in cmap_.items():
                        M[rlab.index(r), c] = sp["matched"][e][g]["mean"]
            if ac:
                cmap_ = {"dir": 0, "other_scalar": 1, "s": 2, "time": 3, "pos": 4}
                for g, c in cmap_.items():
                    M[rlab.index("acc"), c] = ac["matched"]["s"][g]["mean"]
            im = ax.imshow(np.log10(M), cmap="magma", vmin=-2, vmax=2)
            for a in range(6):
                for b in range(5):
                    if np.isfinite(M[a, b]):
                        ax.text(b, a, f"{M[a, b]:.2g}", ha="center", va="center", fontsize=7,
                                color="w" if np.log10(M[a, b]) < 0.5 else "k")
            ax.set_xticks(range(5)); ax.set_xticklabels(clab, rotation=30, fontsize=7)
            ax.set_yticks(range(6)); ax.set_yticklabels([f"edit {r}" for r in rlab], fontsize=7)
            ax.set_title(f"point {p}: |readout change| / natural spread\n(all edits norm-matched to the direction edit)", fontsize=8)
        fig.colorbar(im, ax=axes, label="log10 ratio", shrink=0.8)
        fig.savefig(FIG / "fig_motion_leakage.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    # --- fig 3: coordinate search grid
    e4 = parts["exp4"]
    if e4 and e4.get("points"):
        coords = e4["registration"]["candidates"]
        pts = [p for p in POINTS if p in e4["points"]]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        panels = [("transfer encoding R$^2$ (registered primary)", lambda r: r["encoding_r2_transfer_to_dirset"], "viridis"),
                  ("within-set decode θ MAE (deg)", lambda r: r["decode_cv"]["theta_mae_deg"], "viridis_r"),
                  ("straight-in-C steer: θ endpoint err (deg)", lambda r: r.get("steer_straight_in_C", {}).get("theta_err_deg", {}).get("mean", np.nan), "viridis_r")]
        for ax, (title, fn, cm) in zip(axes, panels):
            M = np.array([[fn(e4["points"][p]["per_coordinate"][c]) for p in pts] for c in coords])
            ax.imshow(M, aspect="auto", cmap=cm)
            for i in range(len(coords)):
                for j, p in enumerate(pts):
                    w = e4["points"][p]["winner"] == coords[i] and "R" in title
                    ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if w else ""), ha="center", va="center", fontsize=6.5,
                            color="w", fontweight="bold" if w else "normal")
            ax.set_xticks(range(len(pts))); ax.set_xticklabels(pts, fontsize=7); ax.set_xlabel("point", fontsize=7)
            ax.set_yticks(range(len(coords))); ax.set_yticklabels(coords, fontsize=7)
            ax.set_title(title, fontsize=8)
        fig.suptitle("Model-native coordinate search (speed set; transfer to direction-set constant-velocity clips); * = winner", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "fig_motion_coordinate_grid.png", dpi=160)
        plt.close(fig)
    print("merged", {k: v is not None for k, v in parts.items()})


if __name__ == "__main__":
    main()

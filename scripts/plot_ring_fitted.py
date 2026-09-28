"""Ring figure from the FITTED steering curve (spline validator, issue #213 follow-up).

Reads results/p2_spline_validation_fig.npz (validator figure pass: headline contiguous arc, seed 0, knot folds at kept
values; the steering curve = count-weighted smoothing spline, sampled at 4,000 points in its own angle-coordinate
order) and writes figures/fig_ring_fitted_L{12,22}.png. Display plane: the ring plane, i.e. the orthonormalised
[cos th, sin th] regression plane of the kept centroids (the labels are used for display only; the top-2 PC plane of
the centroids at point 22 is one ring axis plus the cos 2th fold and draws the ring as a figure 8). Both figures share
axis limits (PCA units).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLINE, CHORD, HELD = "#C8553D", "#3D6A9E", "#9AA1AB"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 24,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2.5,
                     "xtick.major.width": 2, "ytick.major.width": 2, "xtick.major.size": 7, "ytick.major.size": 7,
                     "legend.frameon": False})


def main():
    z = np.load(ROOT / "results" / "p2_spline_validation_fig.npz")
    g = lambda L, k: z[f"L{L}__{k}"]
    lim = max(np.abs(np.concatenate([g(L, "curve_xy"), g(L, "knot_xy"), g(L, "held_xy")])).max()
              for L in (12, 22)) * 1.15
    for L in (12, 22):
        fig, ax = plt.subplots(figsize=(10, 10))
        cx = g(L, "curve_xy")
        ax.plot(*np.vstack([cx, cx[:1]]).T, color=SPLINE, lw=3.5, zorder=2)
        kv, kxy = g(L, "knot_val"), g(L, "knot_xy")
        ax.scatter(*kxy.T, c=kv, cmap="twilight", vmin=0, vmax=360, s=150, edgecolor="#272727", lw=0.8, zorder=3)
        hxy = g(L, "held_xy")
        ax.scatter(*hxy.T, marker="D", s=150, facecolor="white", edgecolor=HELD, lw=3, zorder=4)
        ends = g(L, "chord_ends_xy")
        ax.plot(*ends.T, color=CHORD, lw=3.5, zorder=3)
        ax.scatter(*g(L, "raw_chord_xy").T, color=CHORD, s=45, zorder=5)
        ax.scatter(*g(L, "target_xy").T, color=SPLINE, s=45, zorder=5)
        # direct labels
        c0 = kxy.mean(0)
        for ang in (0.0, 90.0, 180.0, 270.0):
            if ang in kv:
                p = kxy[list(kv).index(ang)]
                u = (p - c0) / np.linalg.norm(p - c0)
                ax.annotate(f"{ang:.0f}°", p, p + 0.16 * lim * u, ha="center", va="center", fontsize=22)
        hm = hxy.mean(0)
        u = (hm - c0) / np.linalg.norm(hm - c0)
        ax.annotate("held-out 45° arc\n(not in the fit)", hm, hm + 0.42 * lim * u, ha="center", va="center",
                    color="#5E646C", fontsize=22, arrowprops=dict(arrowstyle="-", color=HELD, lw=1.5))
        em = ends.mean(0)
        ax.text(*(em - 0.3 * lim * (em - c0) / np.linalg.norm(em - c0)), "raw-centroid chord", ha="center",
                va="center", color=CHORD, fontsize=22)
        ax.text(*(c0 + 0.08 * lim * (c0 - em) / np.linalg.norm(em - c0)), "fitted\nsmoothing spline", ha="center",
                va="center", color=SPLINE, fontsize=22)
        ax.text(0.02, 0.98, f"point {L} · ring plane · centroids coloured by true angle", transform=ax.transAxes, fontsize=20, va="top",
                color="#4D4D4D")
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="ring-plane axis 1 (PCA units)",
               ylabel="ring-plane axis 2 (PCA units)")
        ax.set_aspect("equal")
        fig.tight_layout(pad=1)
        out = ROOT / "figures" / f"fig_ring_fitted_L{L}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()

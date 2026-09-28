"""Figure: per-patch direction probes, VideoMAE vs V-JEPA 2 vs random-init ViT-L (supplied direction set).
Reads results/p1a_perpatch_direction_{vjepa2,random,videomae}.json -> figures/fig1k_perpatch_videomae.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("vjepa2", "V-JEPA 2", "#1f77b4"), ("videomae", "VideoMAE-L (v1)", "#d62728"), ("random", "random-init ViT-L", "#7f7f7f")]
PANELS = [("meanpool_r2", "mean-pool ridge (test R²)"), ("perpos_mean_r2", "per-position probes: mean R²"),
          ("pooled_mean_r2", "pooled-patch probe: mean R² per position"), ("cross_half_r2", "half-frame transfer (cross-half R²; clipped at −0.1)")]


def main():
    data = {m: json.loads((ROOT / "results" / f"p1a_perpatch_direction_{m}.json").read_text()) for m, _, _ in MODELS}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=False)
    for ax, (key, title) in zip(axes, PANELS):
        for m, label, c in MODELS:
            d = data[m]
            ax.plot(d["points"], d["curves"][key], "o-", ms=3.5, lw=1.6, color=c, label=label)
        ax.axvspan(8, 9, color="#f2c14e", alpha=0.25, lw=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("layer point (0 = embedding, 25 = final LN)")
        ax.set_ylim(-0.1, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("direction (sin, cos) test R²")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-patch direction probes on the supplied clips: VideoMAE vs V-JEPA 2 vs random init "
                 "(VideoMAE 14×14 grid at 224 px, V-JEPA 16×16 at 256 px; shaded = paper's points 8→9)\n"
                 "VideoMAE adds an absolute sinusoidal position embedding, V-JEPA 2 uses RoPE: pooled-patch and half-frame "
                 "comparisons confound training with position encoding; per-position probes do not. VideoMAE cross-half is −24.1 / −18.7 at points 1 / 2", fontsize=9)
    fig.tight_layout()
    out = ROOT / "figures" / "fig1k_perpatch_videomae.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()

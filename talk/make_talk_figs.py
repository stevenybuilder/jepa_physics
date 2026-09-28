"""Talk figures (deck v2): one panel per file, drawn from results/*.json in the deck palette.

Style follows figures4papers (github.com/ChenLiu-1996/figures4papers, references/design-theory.md): top/right spines
off, thick axes, large type, frameless legends replaced by direct labels, explicit limits, tight_layout(pad=2).
Colour carries one meaning on every figure:
  terracotta = V-JEPA 2 / the spline, grey = random init / nulls / unedited, ochre = the paper's value,
  steel blue = the straight chord.
Sizes: figsize in inches = display px / 96, rendered at dpi 192 (2x), so a 22 pt label reads as ~29 px on the slide.

  PYTHONPATH=src .venv/bin/python talk/make_talk_figs.py [--font /path/IBMPlexSans.ttf]
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "talk" / "figures_v2"

TERRA, GREY, OCHRE, BLUE, INK, SOFT = "#C8553D", "#9AA1AB", "#C9A227", "#3D6A9E", "#14213D", "#5B6472"
TICK, LABEL, NOTE = 22, 24, 22
DPI = 192


def style(font=None):
    fam = ["DejaVu Sans"]
    if font and Path(font).exists():
        fm.fontManager.addfont(font)
        fam = ["IBM Plex Sans", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": fam, "font.size": TICK, "axes.labelsize": LABEL, "xtick.labelsize": TICK,
        "ytick.labelsize": TICK, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2.5,
        "xtick.major.width": 2.5, "ytick.major.width": 2.5, "xtick.major.size": 8, "ytick.major.size": 8,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 4, "lines.solid_capstyle": "round",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def fig(w_px, h_px):
    return plt.subplots(figsize=(w_px / 96, h_px / 96))


def save(f, name):
    OUT.mkdir(parents=True, exist_ok=True)
    f.tight_layout(pad=2)
    f.savefig(OUT / name, dpi=DPI)
    plt.close(f)
    print("wrote", OUT / name)


def J(name):
    return json.loads((RES / name).read_text())


def label_end(ax, x, y, text, color, dx=0.4, dy=0.0, ha="left", va="center", size=NOTE, weight="normal"):
    ax.text(x + dx, y + dy, text, color=color, ha=ha, va=va, fontsize=size, fontweight=weight)


# ------------------------------------------------------------------ slide 2: the stimulus
def stimulus(scene=100, frames=(0, 5, 10, 15)):
    video = ROOT / "vjepa-physics-takehome-4E00" / "data" / "direction" / "videos" / f"scene_{scene:04d}" / "video.mp4"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), str(Path(td) / "f%02d.png")], check=True)
        imgs = [plt.imread(Path(td) / f"f{i + 1:02d}.png") for i in frames]
    f, axes = plt.subplots(1, len(imgs), figsize=(1600 / 96, 400 / 96))
    for ax, im in zip(axes, imgs):
        ax.imshow(im, interpolation="nearest")
        ax.set_axis_off()
    f.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.04)
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / "stimulus_strip.png", dpi=DPI)
    plt.close(f)
    print("wrote", OUT / "stimulus_strip.png")


# ------------------------------------------------------------------ slide 4: pooled probes
def pooled_layers():
    f, ax = fig(1100, 550)
    pts = lambda d: [l["point"] for l in d["layers"]][:25]
    cv = lambda d: [l["cv_mean"] for l in d["layers"]][:25]
    v = J("p1a_direction_direction_meanpool.json")
    r = J("p1a_direction_direction_meanpool_random.json")
    s = J("p1a_speed_speed_meanpool.json")
    a = J("p1a_acceleration_acceleration_meanpool.json")
    ax.plot(pts(s), cv(s), color=TERRA, lw=2.5, alpha=0.45)
    ax.plot(pts(a), cv(a), color=TERRA, lw=2.5, alpha=0.45, ls=(0, (4, 3)))
    ax.plot(pts(v), cv(v), color=TERRA)
    ax.plot(pts(r), cv(r), color=GREY)
    label_end(ax, 24, cv(v)[-1] + 0.035, "V-JEPA 2, direction", TERRA, dx=0, ha="right", va="bottom")
    label_end(ax, 24, cv(r)[-1] - 0.04, "random init, direction", SOFT, dx=0, ha="right", va="top")
    label_end(ax, 2.5, 0.72, "light lines: speed, acceleration", TERRA, dx=0, ha="left", va="center")
    ax.set_xlim(0, 24.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("block")
    ax.set_ylabel("pooled probe R²")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    save(f, "pooled_layers.png")


# ------------------------------------------------------------------ slide 5: per-patch, supplied clips
def perpatch_supplied():
    f, ax = fig(1100, 550)
    v = J("p1a_perpatch_direction_vjepa2.json")
    r = J("p1a_perpatch_direction_random.json")
    ax.plot(v["points"], v["curves"]["meanpool_r2"], color=TERRA, lw=2.5, alpha=0.5, ls=(0, (4, 3)))
    ax.plot(r["points"], r["curves"]["meanpool_r2"], color=GREY, lw=2.5, ls=(0, (4, 3)))
    ax.text(12, 0.62, "dashed: pooled probe", color=SOFT, ha="center", va="center", fontsize=NOTE)
    ax.plot(r["points"], r["curves"]["perpos_mean_r2"], color=GREY)
    ax.plot(v["points"], v["curves"]["perpos_mean_r2"], color=TERRA)
    label_end(ax, 24, 1.04, "V-JEPA 2: solid per patch", TERRA, dx=0, ha="right",
              va="bottom")
    label_end(ax, 24, r["curves"]["meanpool_r2"][-1] - 0.03, "random init, pooled", SOFT, dx=0, ha="right", va="top")
    label_end(ax, 24, r["curves"]["perpos_mean_r2"][-1] + 0.03, "random init, per patch", SOFT, dx=0, ha="right",
              va="bottom")
    ax.set_xlim(0, 24.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("block")
    ax.set_ylabel("direction probe R²")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    save(f, "perpatch_supplied.png")


# ------------------------------------------------------------------ slide 6: per-patch, hard render
def perpatch_hard():
    f, ax = fig(1100, 550)
    ax.axvspan(8, 9, color=OCHRE, alpha=0.35, lw=0)
    ax.text(8.5, 0.28, "paper's\ntransition", color=INK, ha="center", va="bottom", fontsize=NOTE)
    for name in ("p1a_perpatch_direction_random_hard.json", "p1a_perpatch_direction_random_hard_seed1.json"):
        d = J(name)
        ax.plot(d["points"], d["curves"]["perpos_mean_r2"], color=GREY, lw=3)
    for name in ("p1a_perpatch_direction_vjepa2_hard.json", "p1a_perpatch_direction_vjepa2_hard_seed1.json",
                 "p1a_perpatch_direction_vjepa2_hard_seed2.json"):
        d = J(name)
        ax.plot(d["points"], d["curves"]["perpos_mean_r2"], color=TERRA, lw=3)
    label_end(ax, 22, 0.99, "V-JEPA 2, three render seeds", TERRA, dx=0, ha="right", va="bottom")
    label_end(ax, 22, 0.66, "random init, two seeds", SOFT, dx=0, ha="right", va="bottom")
    ax.set_xlim(0, 23)
    ax.set_ylim(0.25, 1.08)
    ax.set_xlabel("block")
    ax.set_ylabel("mean per-patch R²")
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    save(f, "perpatch_hard.png")


# ------------------------------------------------------------------ slide 7: INLP at the paper's layer
def inlp_direction():
    f, ax = fig(1100, 550)
    d = J("p1b_direction_direction_meanpool_L9.json")
    x = [r["dims_removed"] for r in d["rounds"]]
    y = [r["cv_r2"] for r in d["rounds"]]
    rx = [r["dims_removed"] for r in d["random"]["rows"]]
    ry = [r["cv_r2"] for r in d["random"]["rows"]]
    ax.plot(rx, ry, color=GREY)
    ax.plot(x, y, color=TERRA)
    ax.axhline(0.1, color=SOFT, lw=2, ls=(0, (2, 3)))
    k = 2 * d["K"]
    ax.plot([k], [y[x.index(k)]], "o", color=TERRA, ms=14)
    ax.text(k + 2, y[x.index(k)] + 0.06, f"{d['K']} probes", color=TERRA, fontsize=NOTE, fontweight="normal")
    label_end(ax, 92, ry[-1] - 0.04, "random directions removed", SOFT, dx=0, ha="right", va="top")
    label_end(ax, 30, 0.72, "probe directions removed", TERRA, dx=0, ha="left")
    ax.set_xlim(0, 95)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("dimensions removed (2 per probe)")
    ax.set_ylabel("direction probe R²")
    save(f, "inlp_direction.png")


# ------------------------------------------------------------------ slide 9: multi-probe steering
def steer_direction():
    f, ax = fig(1100, 550)
    learned = J("p1c_direction_L9.json")["single"]
    adam = J("p1c_direction_L9_adam_basis.json")["single"]
    rows = J("p1c_direction_L9_rankmatched.json")["rank_matched_null"]["rows"]
    n = [r["n"] for r in rows]
    ax.fill_between(n, [r["rank_matched"]["mae_to_target"]["p05"] for r in rows],
                    [r["rank_matched"]["mae_to_target"]["p95"] for r in rows], color=GREY, alpha=0.18, lw=0)
    ax.plot(n, [r["rank_matched"]["mae_to_target"]["mean"] for r in rows], color=GREY)
    ax.plot([s["n"] for s in adam if 1 <= s["n"] <= 40], [s["mae_to_target"] for s in adam if 1 <= s["n"] <= 40],
            color=TERRA, lw=3, ls=(0, (4, 3)))
    ax.plot([s["n"] for s in learned if s["n"] >= 1], [s["mae_to_target"] for s in learned if s["n"] >= 1],
            color=TERRA)
    ax.plot([20], [12], marker="D", color=OCHRE, ms=16, mec=INK, mew=1.5)          # REPORT §1: paper ≈ 12° at ~20
    ax.text(21, 16, "Joseph et al.", color=INK, fontsize=NOTE, va="bottom")
    label_end(ax, 38, 88, "random basis, same rank", SOFT, dx=0, ha="right", va="bottom")
    label_end(ax, 5.5, 6, "ridge probes", TERRA, dx=0, va="bottom")
    label_end(ax, 10.5, 32, "Adam probes", TERRA, dx=0, va="bottom")
    ax.set_xlim(0, 38)
    ax.set_ylim(0, 100)
    ax.set_xlabel("probes used for the edit")
    ax.set_ylabel("error to target (°)")
    save(f, "steer_direction.png")


# ------------------------------------------------------------------ slides 12-13: the ring at point 12
def ring_data(point=12):
    from wm import ellipse as el
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    d = load_inputs("direction", point)
    X, y = d["X"][d["is_train"]], d["y"][d["is_train"]]
    pca = mf.fit_pca(X, 64)
    cent = mf.centroids(pca.project(X), y)
    C, values = cent["C"], cent["values"]
    basis = np.linalg.svd(el.chart_fit(C, np.radians(values))["A"], full_matrices=False)[0][:, :2]
    mu = C.mean(0)
    return cent, basis, mu


def ring(point=12):
    from wm import manifold as mf
    cent, basis, mu = ring_data(point)
    C, values = cent["C"], cent["values"]
    curve = mf.fit_curve(cent, True, angle="labels", spline="smooth")
    _, dense = curve.dense(720)                                   # the fitted periodic curve, sampled in angle order
    P, D = (C - mu) @ basis, (dense - mu) @ basis
    f, ax = fig(1100, 540)
    ax.plot(*np.vstack([D, D[:1]]).T, color=TERRA, lw=4, zorder=2)
    ax.scatter(*P.T, s=70, color=INK, zorder=3)
    for v in (0, 90, 180, 270):
        i = int(np.argmin(np.abs(values - v)))
        u = P[i] / np.linalg.norm(P[i])
        ax.text(*(P[i] + 1.3 * u), f"{v}°", ha="center", va="center", fontsize=NOTE, color=INK)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(f, f"ring_L{point}.png")


def heldout_arc(point=12):
    """The headline held-out arc (results/p2_steer_direction_direction_L12_contiguous_rawchord.json): smoothing spline
    fitted without the 8 held-out values, and the raw-centroid chord between the arc's end knots."""
    from wm import manifold as mf
    cent, basis, mu = ring_data(point)
    C, values = cent["C"], cent["values"]
    held = J("p2_steer_direction_direction_L12_contiguous_rawchord.json")["held_out_values"]
    keep = ~np.isin(values, held)
    curve = mf.fit_curve(cent, True, keep=keep, angle="labels", spline="smooth")
    lo, hi = np.radians(held[0] - 360 / 64), np.radians(held[-1] + 360 / 64)
    t = np.linspace(lo, hi, 200)
    arc = (curve(t) - mu) @ basis
    _, dense = curve.dense(720)
    D = (dense - mu) @ basis
    P = (C - mu) @ basis
    a, b = P[np.argmin(np.abs(values - (held[0] - 360 / 64)))], P[np.argmin(np.abs(values - (held[-1] + 360 / 64)))]
    f, ax = fig(1100, 550)
    ax.plot(*np.vstack([D, D[:1]]).T, color=GREY, lw=2.5, zorder=1)
    ax.scatter(*P[keep].T, s=160, color=INK, zorder=3)
    ax.scatter(*P[~keep].T, s=260, facecolor="white", edgecolor=INK, linewidth=2.5, zorder=3)
    ax.plot(*arc.T, color=TERRA, lw=6, zorder=2)
    ax.plot(*np.array([a, b]).T, color=BLUE, lw=6, zorder=2)
    box = np.vstack([arc, a, b, P[~keep]])                        # zoom on the held-out arc
    lo_, hi_ = box.min(0), box.max(0)
    span = (hi_ - lo_).max()
    ax.set_xlim(lo_[0] - 0.35 * span, hi_[0] + 0.35 * span)
    ax.set_ylim(lo_[1] - 0.35 * span, hi_[1] + 0.35 * span)
    ax.set_aspect("equal", adjustable="datalim")
    mid = arc[len(arc) * 3 // 4]
    ax.text(mid[0], mid[1] + 0.08 * span, "spline", color=TERRA, fontsize=NOTE, ha="right", va="bottom")
    ax.text(a[0] + 0.04 * span, a[1], "chord", color=BLUE, fontsize=NOTE, ha="left", va="center")
    lowest = P[~keep][np.argmin(P[~keep][:, 1])]
    ax.text(lowest[0], lowest[1] - 0.06 * span, "held-out class means", color=INK, fontsize=NOTE, ha="center",
            va="top")
    ax.set_axis_off()
    save(f, f"heldout_arc_L{point}.png")


# ------------------------------------------------------------------ slide 14: readout radius along the path
def arc_radius():
    d = J("p2_steer_direction_direction_L12_contiguous_rawchord.json")["waypoint_readout"]
    b = 3                                                          # shift bin 135-180 degrees
    sp, ch = d["manifold"][b]["radius"], d["linear_raw"][b]["radius"]
    s = np.linspace(0, 1, len(sp))
    f, ax = fig(1100, 550)
    ax.plot(s, ch, color=BLUE)
    ax.plot(s, sp, color=TERRA)
    label_end(ax, 0.5, max(sp) + 0.03, "spline", TERRA, dx=0, ha="center", va="bottom")
    i = int(np.argmin(ch))
    label_end(ax, s[i], ch[i] - 0.04, "chord", BLUE, dx=0, ha="center", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("fraction of the path")
    ax.set_ylabel("readout radius")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["source", "midpoint", "target"])
    save(f, "arc_radius_L12.png")


# ------------------------------------------------------------------ slide 16: forecast error by edit point
def forecast_by_layer():
    d = J("session2_predictor_native_readout.json")["per_layer"]
    Ls = [2, 8, 12, 22]
    f, ax = fig(1100, 550)
    base = d["22"]["zero"]["dir_err_to_target"]["mean"]
    ax.axhline(base, color=GREY, lw=2.5, ls=(0, (4, 3)))
    ax.text(2, base + 3, "unedited", color=SOFT, fontsize=NOTE, va="bottom")
    for arm, col, lab in (("random_matched", GREY, "random"), ("chord", BLUE, "chord"), ("spline", TERRA, "spline")):
        yv = [d[str(L)][arm]["dir_err_to_target"]["mean"] for L in Ls]
        ax.plot(Ls, yv, "-o", color=col, ms=12)
        label_end(ax, 22.4, yv[-1], lab, col if col != GREY else SOFT)
    ax.set_xlim(1, 25.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(Ls)
    ax.set_xlabel("block where the edit is made")
    ax.set_ylabel("forecast error to target (°)")
    save(f, "forecast_by_layer.png")


# ------------------------------------------------------------------ claim 1: the zone is where the code becomes position-invariant
def zone_invariance():
    d = J("p1a_perpatch_hard_seeds.json")["by_point"]
    pts = [b["point"] for b in d]
    f, ax = fig(1100, 550)
    ax.axvspan(8, 9, color=OCHRE, alpha=0.35, lw=0)
    ax.axhline(0, color=SOFT, lw=1.5)
    for key, col in (("perpos_mean_r2", TERRA), ("cross_half_r2", INK)):
        seeds = sorted(d[0][key]["per_seed"])
        for sd in seeds:
            ax.plot(pts, [b[key]["per_seed"][sd] for b in d], color=col, lw=1.5, alpha=0.3)
        ax.plot(pts, [b[key]["mean"] for b in d], "-o", color=col, ms=9)
    ax.text(22, 1.06, "decodable per patch", color=TERRA, ha="right", va="bottom", fontsize=NOTE)
    ax.text(22, -0.15, "transfers across the frame", color=INK, ha="right", va="top", fontsize=NOTE)
    ax.text(9.4, -1.75, "paper's zone", color=INK, ha="left", va="center", fontsize=NOTE)
    ax.set_xlim(0, 23)
    ax.set_ylim(-2.3, 1.35)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks([1, 4, 8, 12, 16, 20])
    ax.set_xlabel("block")
    ax.set_ylabel("test R²")
    save(f, "zone_invariance.png")


# ------------------------------------------------------------------ claim 4: readability falls as usability rises
def read_vs_use():
    v = J("p1a_perpatch_direction_vjepa2.json")
    nat = J("session2_predictor_native_readout.json")["per_layer"]
    Ls = [2, 8, 12, 22]
    base = nat["22"]["zero"]["dir_err_to_target"]["mean"]
    shift = [base - nat[str(L)]["spline"]["dir_err_to_target"]["mean"] for L in Ls]
    extra = RES / "p5_repair_attribution_L12.json"                 # adds points 16 and 19 when it lands
    if extra.exists():
        fr = json.loads(extra.read_text()).get("forecast_response_by_edit_point", {})
        for k, val in fr.items():
            if isinstance(val, (int, float)) and int(k) not in Ls:
                Ls.append(int(k))
                shift.append(float(val))
        order = np.argsort(Ls)
        Ls, shift = [Ls[i] for i in order], [shift[i] for i in order]
    f, ax = fig(1100, 550)
    ax.plot(v["points"], v["curves"]["perpos_mean_r2"], color=TERRA)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("per-patch R²", color=TERRA)
    ax.set_xlabel("block")
    ax.set_xlim(0, 25)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(Ls, shift, "-o", color=INK, ms=12)
    ax2.set_ylim(-5, 90)
    ax2.set_ylabel("forecast moved to target (°)", color=INK)
    ax.text(9, 0.99, "readable per patch", color=TERRA, fontsize=NOTE, va="bottom")
    ax2.text(15.5, 6, "used by the predictor", color=INK, fontsize=NOTE, ha="left", va="center")
    save(f, "read_vs_use.png")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--font", default=None)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args(argv)
    style(a.font)
    jobs = {"stimulus": stimulus, "pooled": pooled_layers, "perpatch": perpatch_supplied, "hard": perpatch_hard,
            "inlp": inlp_direction, "steer": steer_direction, "ring": ring, "heldout": heldout_arc,
            "radius": arc_radius, "forecast": forecast_by_layer, "zone": zone_invariance, "readuse": read_vs_use}
    for k, fn in jobs.items():
        if a.only is None or k in a.only:
            fn()


if __name__ == "__main__":
    main()

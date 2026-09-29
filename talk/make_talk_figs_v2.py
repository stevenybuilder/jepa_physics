"""Headline talk figures v2 (deck rewrite, 2026-09-28): five PNGs drawn from results/*.json in the deck palette.

Style: figures4papers conventions (github.com/ChenLiu-1996/figures4papers): top/right spines off, no grid, no titles,
direct labels instead of legends, one accent colour, one sans font. Display box 1600x900 px, rendered at 2x
(figsize = px / 96, dpi 192 -> 3200x1800). Colour meanings: terracotta = V-JEPA 2 / the spline / the point,
grey = unedited / null / background, steel blue = chord / probe, ochre = the paper's reference, navy = ink.

  .venv/bin/python talk/make_talk_figs_v2.py [--font /path/IBMPlexSans-Regular.ttf]
Every plotted number is printed with its JSON key path.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "talk" / "figures_v2"
TERRA, GREY, OCHRE, BLUE, INK, SOFT, BG = "#C8553D", "#9AA1AB", "#C9A227", "#3D6A9E", "#14213D", "#5B6472", "#FBFBF8"
TICK, LABEL, NOTE = 22, 24, 24
W, H, DPI = 1600, 900, 192


def style(font):
    fam = ["DejaVu Sans"]
    if font and Path(font).exists():
        fm.fontManager.addfont(font)
        med = Path(font).with_name(Path(font).name.replace("Regular", "Medium"))
        if med.exists():
            fm.fontManager.addfont(str(med))
        fam = ["IBM Plex Sans", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": fam, "font.size": TICK, "axes.labelsize": LABEL, "xtick.labelsize": TICK,
        "ytick.labelsize": TICK, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2.0,
        "xtick.major.width": 2.0, "ytick.major.width": 2.0, "xtick.major.size": 7, "ytick.major.size": 7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "lines.linewidth": 4, "lines.solid_capstyle": "round", "axes.grid": False,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    })


def J(name):
    return json.loads((RES / name).read_text())


def log(path, val):
    print(f"  {path} = {val}")


def save(f, name):
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / name, dpi=DPI)
    plt.close(f)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. read vs use
def read_vs_use():
    print("fig_read_vs_use")
    p = J("p1a_perpatch_direction_vjepa2.json")
    pts, r2 = p["points"], p["curves"]["perpos_mean_r2"]
    for x, y in zip(pts, r2):
        log(f"p1a_perpatch_direction_vjepa2.json curves.perpos_mean_r2[point {x}]", round(y, 4))
    s = J("session2_predictor_native_readout.json")
    layers = sorted(int(k) for k in s["per_layer"])
    sp = [s["per_layer"][str(L)]["spline"]["dir_err_to_target"] for L in layers]
    z = s["per_layer"][str(layers[0])]["zero"]["dir_err_to_target"]["mean"]
    for L, d in zip(layers, sp):
        log(f"session2_predictor_native_readout.json per_layer.{L}.spline.dir_err_to_target.{{mean,ci95}}",
            (round(d["mean"], 2), [round(c, 2) for c in d["ci95"]]))
    for L in layers:
        log(f"session2_predictor_native_readout.json per_layer.{L}.zero.dir_err_to_target.mean",
            round(s["per_layer"][str(L)]["zero"]["dir_err_to_target"]["mean"], 2))

    f, (a1, a2) = plt.subplots(2, 1, figsize=(W / 96, H / 96), sharex=True, gridspec_kw={"hspace": 0.28})
    a1.plot(pts, r2, color=INK, marker="o", ms=9)
    a1.set_ylim(0, 1.08)
    a1.set_yticks([0, 0.5, 1.0])
    a1.set_ylabel("R²")
    a1.text(24.3, r2[-1] - 0.10, "heading readable from each patch (R²)", color=INK, ha="right", va="top", fontsize=NOTE)
    m = [d["mean"] for d in sp]
    lo = [d["mean"] - d["ci95"][0] for d in sp]
    hi = [d["ci95"][1] - d["mean"] for d in sp]
    a2.axhline(z, color=GREY, lw=3, ls=(0, (5, 4)))
    a2.text(0.3, z + 6, f"unedited forecast {z:.1f}°", color=SOFT, ha="left", va="bottom", fontsize=NOTE)
    a2.errorbar(layers, m, yerr=[lo, hi], color=TERRA, marker="o", ms=11, lw=4, capsize=0, elinewidth=3)
    a2.text(layers[-1] - 0.6, m[-1], f"{m[-1]:.1f}°", color=TERRA, ha="right", va="center", fontsize=NOTE,
            fontweight="medium")
    a2.text(0.3, 45, "forecast error after an edit here (°)", color=TERRA, ha="left", va="center", fontsize=NOTE)
    a2.set_ylim(0, 115)
    a2.set_yticks([0, 45, 90])
    a2.set_ylabel("error (°)")
    a2.set_xlabel("block")
    a2.set_xlim(-0.5, 24.8)
    a2.set_xticks([0, 4, 8, 12, 16, 20, 24])
    f.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.12)
    save(f, "fig_read_vs_use.png")


# ---------------------------------------------------------------- 2. binding by depth
def binding_by_depth():
    print("fig_binding_by_depth")
    t = J("p5_token_patching.json")
    P = t["pair_types"]["random"]["readers"]["forecast"]["points"]
    pts = [0, 8, 12, 16, 22]
    share = t["config"]["plan"]["random"]["token_frac_mean"]["obj"]
    log("p5_token_patching.json config.plan.random.token_frac_mean.obj", share)
    log("background token share = 1 - obj (derived)", round(1 - share, 4))
    f, ax = plt.subplots(figsize=(W / 96, H / 96))
    ax.axvspan(8, 9, color=OCHRE, alpha=0.22, lw=0)
    ax.text(8.5, 1.1, "paper's zone", color=INK, ha="center", va="bottom", fontsize=NOTE)
    for key, col, lab in [("obj", TERRA, "disk tokens"), ("bg", GREY, "background tokens")]:
        m = [P[str(p)][key]["binding_fraction"]["mean"] for p in pts]
        ci = [P[str(p)][key]["binding_fraction"]["ci95"] for p in pts]
        for p, mm, c in zip(pts, m, ci):
            log(f"p5_token_patching.json pair_types.random.readers.forecast.points.{p}.{key}.binding_fraction.{{mean,ci95}}",
                (round(mm, 3), [round(x, 3) for x in c]))
        ax.errorbar(pts, m, yerr=[[mm - c[0] for mm, c in zip(m, ci)], [c[1] - mm for mm, c in zip(m, ci)]],
                    color=col, marker="o", ms=11, lw=4, capsize=0, elinewidth=3)
        ax.text(22.5, m[-1], lab, color=col if key == "obj" else SOFT, ha="left", va="center", fontsize=NOTE,
                fontweight="medium")
    ax.axhline(share, color=TERRA, lw=2, ls=(0, (5, 4)), alpha=0.8)
    ax.axhline(1 - share, color=GREY, lw=2, ls=(0, (5, 4)))
    ax.text(0.2, share + 0.03, f"disk's share of tokens {share:.0%}", color=TERRA, ha="left", va="bottom", fontsize=NOTE)
    ax.text(0.2, 1 - share - 0.03, f"background's share {1 - share:.0%}", color=SOFT, ha="left", va="top",
            fontsize=NOTE)
    ax.set_xlim(-0.5, 28.5)
    ax.set_xticks(pts)
    ax.set_ylim(-0.03, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.spines["bottom"].set_bounds(0, 22)
    ax.set_xlabel("block where tokens are swapped")
    ax.set_ylabel("share of the forecast's change")
    f.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.12)
    save(f, "fig_binding_by_depth.png")


# ---------------------------------------------------------------- 3. chord vs spline
def chord_vs_spline():
    print("fig_chord_vs_spline")
    b = J("p2_bakeoff_direction_direction_L12_contiguous_rawchord.json")
    arms = [("probe_qr", "the paper's probes", BLUE), ("chord_raw", "straight edit", BLUE), ("spline", "curved edit", TERRA)]
    vals = [b["arms"][k]["err_probe_unmatched"] for k, _, _ in arms]
    for (k, _, _), v in zip(arms, vals):
        log(f"p2_bakeoff_direction_direction_L12_contiguous_rawchord.json arms.{k}.err_probe_unmatched", round(v, 3))
    w = J("p2_steer_direction_direction_L12_contiguous_rawchord.json")["waypoint_readout"]
    f, (a1, a2) = plt.subplots(1, 2, figsize=(W / 96, H / 96), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    y = range(len(arms))
    a1.barh(list(y), vals, color=[c for _, _, c in arms], height=0.6)
    for i, v in zip(y, vals):
        a1.text(v + 0.25, i, f"{v:.1f}°", va="center", ha="left", fontsize=NOTE, color=arms[i][2], fontweight="medium")
    a1.set_yticks(list(y))
    a1.set_yticklabels([l for _, l, _ in arms])
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    a1.set_xlim(0, 12)
    a1.set_xticks([0, 5, 10])
    a1.set_xlabel("error at an unseen heading (°)")
    _path_panel(a2, w, small=True)
    f.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.14)
    save(f, "fig_chord_vs_spline.png")


def _path_panel(ax, w, small=False):
    b3 = 3
    rm = w["manifold"][b3]["radius"]
    rl = w["linear_raw"][b3]["radius"]
    log(f"p2_steer_direction_direction_L12_contiguous_rawchord.json waypoint_readout.manifold[{b3}] "
        f"(shift {w['manifold'][b3]['shift_lo']}-{w['manifold'][b3]['shift_hi']}, n={w['manifold'][b3]['n']}).radius "
        f"(50 waypoints) min", round(min(rm), 4))
    log(f"p2_steer_direction_direction_L12_contiguous_rawchord.json waypoint_readout.linear_raw[{b3}].radius (50 waypoints) min",
        round(min(rl), 4))
    log("  endpoints manifold[0], [-1]", (round(rm[0], 3), round(rm[-1], 3)))
    log("  endpoints linear_raw[0], [-1]", (round(rl[0], 3), round(rl[-1], 3)))
    x = [i / (len(rm) - 1) for i in range(len(rm))]
    ax.axhline(1.0, color=GREY, lw=2, ls=(0, (5, 4)))
    ax.plot(x, rm, color=TERRA)
    ax.plot(x, rl, color=BLUE)
    im = min(range(len(rl)), key=lambda i: rl[i])
    ax.text(x[im], rl[im] - 0.05, f"{rl[im]:.2f}", color=BLUE, ha="center", va="top", fontsize=NOTE, fontweight="medium")
    jm = min(range(len(rm)), key=lambda i: rm[i])
    ax.text(0.30, 1.06, "curved edit", color=TERRA, ha="center", va="bottom", fontsize=NOTE, fontweight="medium")
    ax.text(0.72, 0.30, "straight edit", color=BLUE, ha="left", va="center", fontsize=NOTE, fontweight="medium")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["source", "halfway", "target"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("how much heading code is left")
    ax.set_xlabel("source → target")


# ---------------------------------------------------------------- 3b. heading path (p2-radius)
def heading_path():
    print("fig_heading_path")
    w = J("p2_steer_direction_direction_L12_contiguous_rawchord.json")["waypoint_readout"]
    f, ax = plt.subplots(figsize=(W / 96, H / 96))
    _path_panel(ax, w)
    f.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.13)
    save(f, "fig_heading_path.png")


# ---------------------------------------------------------------- 4. velocity sheet
def sheet():
    print("fig_sheet")
    v = J("p5_velocity_sheet.json")["layers"]["22"]["steer"]
    d = v["direction"]["summary"]
    jn = v["joint"]["summary"]
    base = "p5_velocity_sheet.json layers.22.steer"
    rows_dir = [("sheet", "one joint edit", TERRA), ("chord_raw", "straight edit", BLUE),
                ("ring1d", "curved edit, direction only", GREY)]
    rows_joint = [("sheet", "one joint edit", TERRA), ("seq_dir_then_speed", "two edits in turn", GREY)]
    f, axes = plt.subplots(1, 3, figsize=(W / 96, H / 96), gridspec_kw={"wspace": 0.12})
    panels = [
        (axes[0], [(k, l, c, d[k]["endpoint_err"], f"direction.summary.{k}.endpoint_err") for k, l, c in rows_dir],
         "heading error (°)", "{:.1f}°", 10),
        (axes[1], [(k, l, c, d[k]["offtarget_change"], f"direction.summary.{k}.offtarget_change") for k, l, c in rows_dir],
         "speed leaked (m/s)", "{:.2f}", 0.1),
        (axes[2], [(k, l, c, jn[k]["endpoint_err_dir"], f"joint.summary.{k}.endpoint_err_dir") for k, l, c in rows_joint],
         "heading error, both changed (°)", "{:.1f}°", 10),
    ]
    for ax, rows, xl, fmt, xmax in panels:
        ys = list(range(len(rows)))[::-1]
        for yy, (k, l, c, val, path) in zip(ys, rows):
            log(f"{base}.{path}", round(val, 4))
            ax.barh(yy, val, color=c, height=0.62)
            ax.text(val + xmax * 0.03, yy, fmt.format(val), va="center", ha="left", fontsize=NOTE,
                    color=c if c != GREY else SOFT, fontweight="medium")
            ax.text(0, yy + 0.42, l, va="bottom", ha="left", fontsize=NOTE - 2, color=c if c != GREY else SOFT)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, xmax * 1.02)
        ax.set_ylim(-0.6, 3.1)
        ax.set_xlabel(xl)
    log(f"{base}.joint.summary.seq_speed_then_dir.endpoint_err_dir (same as dir_then_speed)",
        round(jn["seq_speed_then_dir"]["endpoint_err_dir"], 4))
    axes[1].set_xticks([0, 0.05, 0.1])
    axes[0].set_xticks([0, 5, 10])
    axes[2].set_xticks([0, 5, 10])
    f.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.14)
    save(f, "fig_sheet.png")


# ---------------------------------------------------------------- 6. coordinate frames (Finding 2)
def coordinates():
    print("fig_coordinates")
    e4 = J("p5_motion_geometry.json")["exp4_coordinate_search_and_exp5_shortcuts"]
    P, U = e4["points"]["12"]["per_coordinate"], e4["random_init"]["12"]
    base = "p5_motion_geometry.json exp4_coordinate_search_and_exp5_shortcuts"
    names = {"cartesian": "x–y", "polar": "angle, speed", "log_polar": "angle, log speed",
             "fourier_polar": "cos, sin, speed", "fourier_logpolar": "cos, sin, log speed",
             "fourier2_polar": "cos, sin, cos 2θ, sin 2θ, speed"}
    order = ["cartesian", "polar", "log_polar", "fourier_polar", "fourier_logpolar", "fourier2_polar"]
    f, ax = plt.subplots(figsize=(W / 96, H / 96))
    ys = list(range(len(order)))[::-1]
    for yy, c in zip(ys, order):
        v, u = P[c]["encoding_r2_transfer_to_dirset"], U[c]["encoding_r2_transfer_to_dirset"]
        log(f"{base}.points.12.per_coordinate.{c}.encoding_r2_transfer_to_dirset", round(v, 4))
        log(f"{base}.random_init.12.{c}.encoding_r2_transfer_to_dirset", round(u, 4))
        col = TERRA if c == "fourier2_polar" else (INK if c == "cartesian" else GREY)
        ax.plot([0, v], [yy, yy], color=col, lw=6, solid_capstyle="butt")
        ax.plot(v, yy, "o", ms=18, color=col)
        ax.plot(u, yy, "o", ms=14, mfc=BG, mec=GREY, mew=3)
        ax.text(max(v, u) + 0.03, yy, f"{v:.2f}", va="center", fontsize=NOTE + 2,
                color=col if col != GREY else SOFT, fontweight="medium")
    ax.set_yticks(ys)
    ax.set_yticklabels([names[c] for c in order])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("variance of a new clip set explained (block 12)")
    yc = ys[order.index("cartesian")]
    ax.annotate("untrained copy (hollow)", xy=(U["cartesian"]["encoding_r2_transfer_to_dirset"], yc),
                xytext=(0.66, yc - 0.45), fontsize=NOTE - 2, color=SOFT, arrowprops=dict(arrowstyle="-", color=GREY))
    f.subplots_adjust(left=0.3, right=0.97, top=0.97, bottom=0.14)
    save(f, "fig_coordinates.png")


# ---------------------------------------------------------------- 7. velocity sheet v2 (Finding 3, what it buys)
def sheet_v2():
    print("fig_sheet_v2")
    R = J("p5_velocity_sheet_v2.json")["results"]["22"]["block2"]
    base = "p5_velocity_sheet_v2.json results.22.block2"
    d, jn = R["direction"]["own"]["summary"], R["joint"]["own"]["summary"]
    floor = R["reader_floor"]["real"]["summary"]["real_target_clips"]["err_dir"]["mean"]
    log(f"{base}.reader_floor.real.summary.real_target_clips.err_dir.mean", round(floor, 4))
    rows_dir = [("sheet", "one joint fit", TERRA), ("ring1d_band_reinschcv", "ring, cross-validated", GREY),
                ("ring1d_band_interp", "ring, paper's spline", GREY)]
    panels = [([(l, c, d[k]["err_dir"]["mean"], f"direction.own.summary.{k}.err_dir.mean") for k, l, c in rows_dir],
               "heading error, direction only (°)", "{:.1f}°", 10, False),
              ([(l, c, d[k]["leak_spd"]["mean"], f"direction.own.summary.{k}.leak_spd.mean") for k, l, c in rows_dir],
               "speed leaked (m/s)", "{:.2f}", 0.1, False),
              ([("one joint edit", TERRA, jn["sheet"]["err_dir"]["mean"], "joint.own.summary.sheet.err_dir.mean"),
                ("two 1-D curves in turn", GREY, jn["seq_global_interp"]["err_dir"]["mean"],
                 "joint.own.summary.seq_global_interp.err_dir.mean")],
               "heading error, both changed (°)", "{:.1f}°", 10, True)]
    f, axes = plt.subplots(1, 3, figsize=(W / 96, H / 96), gridspec_kw={"wspace": 0.15})
    for ax, (rows, xl, fmt, xmax, fl) in zip(axes, panels):
        ys = list(range(len(rows)))[::-1]
        for yy, (l, c, val, path) in zip(ys, rows):
            log(f"{base}.{path}", round(val, 4))
            ax.barh(yy, val, color=c, height=0.5)
            ax.text(val + xmax * 0.03, yy, fmt.format(val), va="center", fontsize=NOTE, color=c if c != GREY else SOFT)
            ax.text(0, yy + 0.36, l, fontsize=NOTE - 2, color=c if c != GREY else SOFT)
        if fl:
            ax.axvline(floor, color=OCHRE, ls="--", lw=2)
            ax.text(floor, len(rows) - 0.35, "probe floor", fontsize=NOTE - 4, color=INK, ha="center")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.5, len(rows))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xl)
    f.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.14)
    save(f, "fig_sheet_v2.png")


# ---------------------------------------------------------------- 8. unified 16-arc bake-off (Finding 3)
def _arc_means(pt, arm):
    import statistics as st
    per = []
    for part in ("a", "b"):
        for rec in json.loads((RES / "endpoint_diagnosis_raw" / f"unified_L{pt}_{part}.json").read_text()):
            per.append(st.mean(rec["arms"][arm]["err_end"]))
    m = st.mean(per); h = 2 * st.stdev(per) / len(per) ** 0.5
    return m, h, len(per)


def bakeoff_unified():
    print("fig_bakeoff_unified")
    d = J("p2_bakeoff_unified_16arc.json")
    T = d["table"]
    base = "p2_bakeoff_unified_16arc.json table"
    arms = [("fourier2_4d", "straight, polar frame", BLUE), ("probe_qr", "probe steer (our refit)", BLUE),
            ("chord_raw", "straight, raw centroids", BLUE), ("causalab_lam_cv", "curve, cross-validated*", TERRA),
            ("interp_causalab_lam0", "curve, paper's spline", TERRA), ("fitpack_smooth", "curve, our first smoother", TERRA)]
    XMAX, FS = 10, 20
    f, axes = plt.subplots(1, 2, figsize=(W / 96, (W * 520 / 1616) / 96), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = list(range(len(arms)))[::-1]
    for ax, pt in zip(axes, ["22", "12"]):
        for yy, (k, lab, c) in zip(ys, arms):
            e = T[k]["err_end"][pt]; r = T[k]["radius_min"][pt]["mean"]
            am, ah, na = _arc_means(pt, k)
            log(f"{base}.{k}.err_end.{pt}", (round(e["mean"], 3), e["n_arcs"]))
            log(f"endpoint_diagnosis_raw/unified_L{pt}_{{a,b}}.json arms.{k}.err_end arc-level mean ± 2SE", (round(am, 3), round(ah, 3), na))
            log(f"{base}.{k}.radius_min.{pt}.mean", round(r, 3))
            m = e["mean"]
            filled = r >= 0.8
            if m > XMAX:
                ax.annotate("", xy=(XMAX, yy), xytext=(XMAX - 1.2, yy), arrowprops=dict(arrowstyle="-|>", color=c, lw=3))
                ax.text(XMAX - 1.4, yy, f"{m:.1f}°", ha="right", va="center", fontsize=FS, color=c)
                continue
            ax.plot([am - ah, am + ah], [yy, yy], color=c, lw=3)
            ax.plot(m, yy, "o", ms=14, color=c, mfc=c if filled else BG, mew=3)
            ax.text(am + ah + 0.25, yy, f"{m:.1f}°", va="center", fontsize=FS, color=c)
        ax.set_xlim(0, XMAX)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(labelsize=FS)
        ax.set_xlabel(f"endpoint error (°), block {pt}", fontsize=FS)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([l for _, l, _ in arms], fontsize=FS)
    axes[1].tick_params(axis="y", length=0)
    f.text(0.98, 0.02, "whiskers: across 16 arcs · filled: keeps ≥0.8 of the heading on the way · *exploratory",
           ha="right", fontsize=FS - 2, color=SOFT)
    f.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.27)
    save(f, "fig_bakeoff_unified.png")


# ---------------------------------------------------------------- 9. contact (wall bounce)
def contact():
    print("fig_contact")
    R = J("p5_contact_dynamics.json")["results"]
    base = "p5_contact_dynamics.json results"
    by = R["per_point"]["22"]["bounce"]["turn_fraction_by_offset"]
    offs = sorted(by, key=int)
    xs = [int(o) for o in offs]
    ys = [by[o]["mean"] for o in offs]
    lo = [by[o]["ci95"][0] for o in offs]
    hi = [by[o]["ci95"][1] for o in offs]
    for o in offs:
        log(f"{base}.per_point.22.bounce.turn_fraction_by_offset.{o}.mean", (round(by[o]["mean"], 3), by[o]["n"]))
    steps = R["predictor_forecast"]["steps"]
    keys = sorted(steps, key=lambda s: int(s.split("_")[0][4:]))
    fc = [steps[k]["forecast_turn_fraction_post"]["mean"] for k in keys]
    rf = [steps[k]["real_future_encoder_turn_fraction_post_point25"]["mean"] for k in keys]
    for k, a, b in zip(keys, fc, rf):
        log(f"{base}.predictor_forecast.steps.{k}.forecast_turn_fraction_post.mean / real_future_encoder_turn_fraction_post_point25.mean",
            (round(a, 3), round(b, 3), steps[k]["n_post_contact"]))
    f, (a1, a2) = plt.subplots(1, 2, figsize=(W / 96, H / 96), gridspec_kw={"wspace": 0.25, "width_ratios": [1.3, 1]})
    a1.fill_between(xs, lo, hi, color=TERRA, alpha=0.18, lw=0)
    a1.plot(xs, ys, "-o", color=TERRA, ms=10)
    a1.axvline(0, color=GREY, ls="--", lw=2)
    a1.text(0.2, 1.02, "contact", fontsize=28, color=SOFT)
    a1.set_ylim(0, 1.1); a1.set_yticks([0, 0.5, 1]); a1.set_xticks([-6, -3, 0, 3, 6])
    a1.set_xlabel("frame pair relative to contact", fontsize=30)
    a1.set_ylabel("share of the turn decoded", fontsize=30)
    a1.text(-6, 0.95, "encoder, block 22", fontsize=30, color=TERRA)
    sx = list(range(1, len(keys) + 1))
    ic = J("p5_contact_in_context.json")["results"]["forecast"]["steps"]
    ick = sorted(ic, key=lambda s: int(s.split("_")[0][4:]))
    seen = [ic[k]["bounce"]["all"]["turn_fraction"]["mean"] for k in ick]
    for k, v in zip(ick, seen):
        log(f"p5_contact_in_context.json results.forecast.steps.{k}.bounce.all.turn_fraction.mean", round(v, 3))
    a2.plot(sx, rf, "o", ms=16, color=INK)
    a2.plot(sx, seen, "s", ms=16, color=TERRA)
    a2.plot(sx, fc, "o", ms=16, color=TERRA, mfc=BG, mew=3)
    a2.text(0.6, 1.05, "real future (encoder)", fontsize=26, color=INK)
    a2.text(0.6, 0.62, "forecast, bounce seen", fontsize=26, color=TERRA)
    a2.text(0.6, 0.25, "forecast, bounce unseen", fontsize=26, color=TERRA)
    a2.set_ylim(0, 1.1); a2.set_yticks([0, 0.5, 1])
    a2.set_xticks(sx); a2.set_xlim(0.5, len(keys) + 0.5)
    a2.set_xlabel("future step after the context", fontsize=30)
    for ax in (a1, a2):
        ax.tick_params(labelsize=28)
    f.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.17)
    save(f, "fig_contact.png")


# ---------------------------------------------------------------- 10. Part 1 in three panels (README Part 1)
def part1_panels():
    print("fig_part1_panels")
    f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W / 96, (W * 560 / 1616) / 96), gridspec_kw={"wspace": 0.32})
    cols = {"direction": TERRA, "speed": INK, "acceleration": BLUE}
    for v, c in cols.items():
        for suf, ls in (("", "-"), ("_random", "--")):
            L = J(f"p1a_{v}_{v}_meanpool{suf}.json")["layers"]
            xs = [l["point"] for l in L]; ys = [l["cv_mean"] for l in L]
            log(f"p1a_{v}_{v}_meanpool{suf}.json layers[1].cv_mean", round(ys[1], 3))
            a1.plot(xs, ys, ls=ls, color=c, lw=3 if ls == "-" else 2.5)
    a1.set_ylim(0, 1.05); a1.set_yticks([0, 0.5, 1]); a1.set_xticks([0, 8, 16, 24])
    a1.set_xlabel("block"); a1.set_ylabel("decoded (R²)")
    a1.text(24, 0.12, "dashed: untrained", ha="right", fontsize=20, color=SOFT)
    a1.text(24, 0.30, "direction", ha="right", fontsize=20, color=TERRA)
    a1.text(24, 0.44, "speed", ha="right", fontsize=20, color=INK)
    a1.text(24, 0.58, "acceleration", ha="right", fontsize=20, color=BLUE)
    b = J("p1b_direction_direction_meanpool_L9.json")
    rr = b["rounds"]; K = b["K"]
    log("p1b_direction_direction_meanpool_L9.json K", K)
    a2.plot([r["dims_removed"] for r in rr], [r["cv_r2"] for r in rr], color=TERRA)
    rn = b["random"]["rows"]
    a2.plot([r["dims_removed"] for r in rn], [r["cv_r2"] for r in rn], color=GREY, ls="--", lw=2.5)
    a2.axhline(0.1, color=GREY, lw=1.5, ls=":")
    a2.set_ylim(0, 1.05); a2.set_yticks([0, 0.5, 1]); a2.set_xlim(0, 90)
    a2.set_xlabel("dimensions erased, block 9"); a2.set_ylabel("heading decoded (R²)")
    a2.text(88, 0.88, "random", ha="right", fontsize=20, color=SOFT)
    a2.text(88, 0.6, f"{K} probes = {2 * K} dims", ha="right", fontsize=20, color=TERRA)
    B = J("p1c_direction_L9_nulls200.json")["bases"]
    rt, at = B["ridge"]["table"], B["adam"]["table"]
    a3.plot([r["n"] for r in rt], [r["rank_matched"]["median"] for r in rt], color=GREY, ls="--", lw=2.5)
    a3.plot([r["n"] for r in at if r["n"] <= 37], [r["learned_mae_to_target"] for r in at if r["n"] <= 37],
            color=TERRA, lw=2.5, ls=(0, (4, 3)))
    a3.plot([r["n"] for r in rt], [r["learned_mae_to_target"] for r in rt], color=TERRA)
    a3.text(36, 74, "random, same rank", ha="right", fontsize=20, color=SOFT)
    a3.text(11, 7, "ridge probes", fontsize=20, color=TERRA)
    a3.text(13, 34, "Adam probes", fontsize=20, color=TERRA)
    a3.set_ylim(0, 95); a3.set_yticks([0, 45, 90]); a3.set_xlim(0, 37)
    a3.set_xlabel("probes used to steer"); a3.set_ylabel("heading error (°)")
    for ax in (a1, a2, a3):
        ax.tick_params(labelsize=20)
        ax.xaxis.label.set_size(20); ax.yaxis.label.set_size(20)
    f.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.2)
    save(f, "fig_part1_panels.png")


# ---------------------------------------------------------------- 11. coordinate competition, rank 2 (Finding 2, rescoped)
def coordinate_competition():
    print("fig_coordinate_competition_deck")
    E = J("p5_coordinate_competition.json")["encoding"]
    base = "p5_coordinate_competition.json encoding"
    FS = 28
    f, ax = plt.subplots(figsize=(W / 96, (W * 536 / 972) / 96))
    ax.axvspan(7.5, 9.5, color=OCHRE, alpha=0.18, lw=0)
    ax.text(8.5, 0.036, "zone", ha="center", fontsize=FS - 4, color=INK)
    for m, c, lab in (("vjepa2", TERRA, "V-JEPA 2"), ("random", GREY, "untrained copy")):
        pts = sorted(E[m], key=int)
        xs = [int(p) for p in pts]
        pc = [E[m][p]["primary_speedset"]["per_candidate"]["polar2"] for p in pts]
        ys = [x["minus_cartesian"] for x in pc]
        lo = [x["minus_cartesian_ci95"][0] for x in pc]
        hi = [x["minus_cartesian_ci95"][1] for x in pc]
        for p, y in zip(pts, ys):
            if p in ("9", "12", "22"):
                log(f"{base}.{m}.{p}.primary_speedset.per_candidate.polar2.minus_cartesian", round(y, 4))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(xs, ys, color=c, lw=4)
        ax.text(25.4, ys[-1], lab, va="center", fontsize=FS - 2, color=c if c != GREY else SOFT)
    ax.axhline(0, color=INK, lw=1.5)
    ax.set_xlim(0, 25); ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_ylim(-0.03, 0.04); ax.set_yticks([-0.02, 0, 0.02, 0.04])
    ax.tick_params(labelsize=FS - 2)
    ax.set_xlabel("block", fontsize=FS)
    ax.set_ylabel("polar minus x–y (R², rank 2)", fontsize=FS)
    f.subplots_adjust(left=0.16, right=0.8, top=0.95, bottom=0.17)
    save(f, "fig_coordinate_competition_deck.png")


# ---------------------------------------------------------------- 12. the three splines (README Part 2 minimum)
def splines3():
    """Knot-clip class means in PCA-64, shown on the top two principal axes of the kept centroids; the curve is fit on
    the kept knots only (count-weighted smoothing spline as stored in the steer files; periodic for direction, shown in its ring plane), held-out class means hollow.
    Held-out values and blocks from results/p2_steer_{speed_speed_L19,acceleration_acceleration_L21,
    direction_direction_L22}_contiguous.json (held_out_values)."""
    print("fig_splines3")
    import numpy as np
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from wm import manifold as mf
    from wm.p2_data import load_inputs
    specs = [("speed", 19, "p2_steer_speed_speed_L19_contiguous.json", False, "speed, block 19"),
             ("acceleration", 21, "p2_steer_acceleration_acceleration_L21_contiguous.json", False, "acceleration, block 21"),
             ("direction", 22, "p2_steer_direction_direction_L22_contiguous.json", True, "direction, block 22")]
    FS = 26
    f, axes = plt.subplots(1, 3, figsize=(W / 96, (W * 560 / 1616) / 96), gridspec_kw={"wspace": 0.12})
    for ax, (ds, L, fn, per, lab) in zip(axes, specs):
        d = load_inputs(ds, L)
        knot = d["role"] == "knot"
        held = np.array(J(fn)["held_out_values"])
        kept_rows = knot & ~np.isclose(d["y"][:, None], held[None, :]).any(1)
        pca = mf.fit_pca(d["X"][kept_rows], 64)          # plane fit on kept knots only, as in the experiment
        cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
        keep = ~np.isclose(cent["values"][:, None], held[None, :]).any(1)
        log(f"{fn} held_out_values", [round(float(h), 3) for h in held])
        log(f"{ds} L{L}: knot clips / values kept / held out", (int(knot.sum()), int(keep.sum()), int((~keep).sum())))
        kw = {"angle": "labels"} if per else {}
        curve = mf.fit_curve(cent, per, keep=keep, spline="smooth", **kw)
        _, dense = curve.dense(1000)
        C = cent["C"]
        mu = C[keep].mean(0)
        if per:
            from wm import ellipse as el
            basis = np.linalg.svd(el.chart_fit(C[keep], np.radians(cent["values"][keep]))["A"], full_matrices=False)[0][:, :2]
        else:
            basis = np.linalg.svd(C[keep] - mu, full_matrices=False)[2][:2].T
        P, Dn = (C - mu) @ basis, (dense - mu) @ basis
        if per:
            Dn = np.vstack([Dn, Dn[:1]])
        ax.plot(*Dn.T, color=TERRA if per else INK, lw=3, zorder=2)
        ax.scatter(*P[keep].T, s=36, color=GREY, zorder=3)
        ax.scatter(*P[~keep].T, s=110, facecolor=BG, edgecolor=TERRA, linewidth=2.5, zorder=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(lab, fontsize=FS, color=INK, loc="left")
    f.text(0.99, 0.03, "grey: knot class means · hollow: held-out values · line: spline fit on knots", ha="right",
           fontsize=FS - 4, color=SOFT)
    f.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.1)
    save(f, "fig_splines3.png")


# ---------------------------------------------------------------- 13. Part 1, one figure per README step
def part1_single():
    print("fig_p1_{layers,nullspace,steer}")
    FS = 28
    size = (W / 96, (W * 536 / 972) / 96)

    def finish(f, ax, name, xl, yl):
        ax.tick_params(labelsize=FS - 2)
        ax.set_xlabel(xl, fontsize=FS); ax.set_ylabel(yl, fontsize=FS)
        f.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.17)
        save(f, name)

    f, ax = plt.subplots(figsize=size)
    cols = {"direction": TERRA, "speed": INK, "acceleration": BLUE}
    ypos = {"direction": 0.30, "speed": 0.44, "acceleration": 0.58}
    for v, c in cols.items():
        for suf, ls in (("", "-"), ("_random", "--")):
            L = J(f"p1a_{v}_{v}_meanpool{suf}.json")["layers"]
            ax.plot([l["point"] for l in L], [l["cv_mean"] for l in L], ls=ls, color=c, lw=4 if ls == "-" else 3)
            log(f"p1a_{v}_{v}_meanpool{suf}.json layers[1].cv_mean", round(L[1]["cv_mean"], 3))
        ax.text(24, ypos[v], v, ha="right", fontsize=FS - 2, color=c)
    ax.text(24, 0.14, "dashed: untrained copy", ha="right", fontsize=FS - 4, color=SOFT)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([0, 8, 16, 24])
    finish(f, ax, "fig_p1_layers.png", "block", "decoded (R²)")

    f, ax = plt.subplots(figsize=size)
    b = J("p1b_direction_direction_meanpool_L9.json")
    rr, rn, K = b["rounds"], b["random"]["rows"], b["K"]
    ax.plot([r["dims_removed"] for r in rr], [r["cv_r2"] for r in rr], color=TERRA, lw=4)
    ax.plot([r["dims_removed"] for r in rn], [r["cv_r2"] for r in rn], color=GREY, ls="--", lw=3)
    ax.axhline(0.1, color=GREY, lw=1.5, ls=":")
    ax.text(88, 0.88, "random directions", ha="right", fontsize=FS - 2, color=SOFT)
    ax.text(88, 0.3, f"{K} probes = {2 * K} dimensions", ha="right", fontsize=FS - 2, color=TERRA)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(0, 90)
    finish(f, ax, "fig_p1_nullspace.png", "dimensions erased (2 per probe), block 9", "heading decoded (R²)")

    f, ax = plt.subplots(figsize=size)
    B = J("p1c_direction_L9_nulls200.json")["bases"]
    rt, at = B["ridge"]["table"], B["adam"]["table"]
    ax.plot([r["n"] for r in rt], [r["rank_matched"]["median"] for r in rt], color=GREY, ls="--", lw=3)
    ax.plot([r["n"] for r in at if r["n"] <= 37], [r["learned_mae_to_target"] for r in at if r["n"] <= 37],
            color=TERRA, lw=3, ls=(0, (4, 3)))
    ax.plot([r["n"] for r in rt], [r["learned_mae_to_target"] for r in rt], color=TERRA, lw=4)
    for r in rt + at:
        if r["n"] in (5, 18):
            log(f"p1c_direction_L9_nulls200.json bases.*.table[n={r['n']}] learned / rank_matched median / p",
                (round(r["learned_mae_to_target"], 2), round(r["rank_matched"]["median"], 2), round(r["rank_matched"]["p"], 3)))
    ax.text(36, 76, "random, same rank", ha="right", fontsize=FS - 2, color=SOFT)
    ax.text(11, 8, "ridge probes", fontsize=FS - 2, color=TERRA)
    ax.text(12, 36, "Adam probes", fontsize=FS - 2, color=TERRA)
    ax.set_ylim(0, 95); ax.set_yticks([0, 45, 90]); ax.set_xlim(0, 37)
    finish(f, ax, "fig_p1_steer.png", "probes used to steer, block 9", "heading error (°)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run only these figure functions")
    a = ap.parse_args()
    style(a.font)
    FIGS = [read_vs_use, binding_by_depth, chord_vs_spline, heading_path, sheet, coordinates, sheet_v2, bakeoff_unified, contact, part1_panels, coordinate_competition, splines3, part1_single]
    for fn in FIGS:
        if a.only is None or fn.__name__ in a.only:
            fn()

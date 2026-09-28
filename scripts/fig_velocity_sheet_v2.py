"""Figure for results/p5_velocity_sheet_v2.json -> figures/fig_velocity_sheet_v2.png (point 22 by default)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
full = json.loads((ROOT / "results" / "p5_velocity_sheet_v2.json").read_text())
L = sys.argv[1] if len(sys.argv) > 1 else "22"
accel = len(sys.argv) > 2 and sys.argv[2] == "acceleration"
R = (full["acceleration"]["results"] if accel else full["results"])[L]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))


def bar(a, arms, getter, title, ylabel, designs=("block2", "block3", "cross")):
    w = 0.8 / len(designs)
    for j, de in enumerate(designs):
        m, lo, hi = [], [], []
        for arm in arms:
            s = getter(de, arm)
            m.append(s["mean"] if s else np.nan)
            lo.append(s["mean"] - s["ci95"][0] if s else 0)
            hi.append(s["ci95"][1] - s["mean"] if s else 0)
        a.bar(np.arange(len(arms)) + (j - (len(designs) - 1) / 2) * w, m, w, yerr=[lo, hi], capsize=2, label=de)
    a.set_xticks(range(len(arms)))
    a.set_xticklabels(arms, rotation=35, ha="right", fontsize=7)
    a.set_title(title, fontsize=9)
    a.set_ylabel(ylabel)
    a.legend(fontsize=7)


def g(task, cond, q):
    def f(de, arm):
        try:
            return R[de][task][cond]["summary"][arm][q]
        except KeyError:
            return None
    return f


arms = ["sheet", "sheet_tps_interp", "abl_shape_frozen", "abl_radius_frozen", "abl_global_speed_axis", "abl_cylinder_from_sheet",
        "seq_global_interp", "seq_global_reinschcv_ring_interp_line", "seq_global", "chord_cells_ORACLE", "chord_cells_idw"]
bar(ax[0], arms, g("joint", "matched", "err_dir"), f"joint steer to a held-out cell, point {L}: direction error\n"
    "(all arms at the sheet's per-clip norm; ORACLE uses held-out clips)", "direction-probe error (deg)")
bar(ax[1], arms, g("joint", "matched", "err_spd"), "joint steer: speed error (matched norm)", "speed-probe error (m/s)")
arms_d = ["sheet", "sheet_tps_interp", "sheet_pooled_over_speed", "ring1d_band_reinschcv", "ring1d_band_interp",
          "ring_global_reinschcv", "ring1d_band"]
bar(ax[2], arms_d, g("direction", "own", "err_dir"), "direction-only steer at fixed speed (OWN norm; *_interp = paper spline,\n"
    "*_reinschcv = causalab Reinsch-CV, ring1d_band = our FITPACK smoother)\n"
    "direction error; sheet − pooled = speed-specific slice", "direction-probe error (deg)", designs=("block2", "block3"))
a = ax[3]
try:
    D = R["block2"]["dose_direction"]
    for arm, c in (("sheet", "C0"), ("ring1d_band", "C1"), ("ring_global", "C2"), ("sheet_pooled_over_speed", "C3")):
        xs = sorted(float(k[1:]) for k in D)
        m = [D[f"x{x}"]["summary"][arm]["leak_spd"]["mean"] for x in xs]
        lo = [D[f"x{x}"]["summary"][arm]["leak_spd"]["ci95"][0] for x in xs]
        hi = [D[f"x{x}"]["summary"][arm]["leak_spd"]["ci95"][1] for x in xs]
        a.plot(xs, m, "o-", color=c, label=arm)
        a.fill_between(xs, lo, hi, color=c, alpha=0.2)
    a.set_xlabel("dose (× the edit)"); a.set_ylabel("speed leakage (m/s)")
    a.set_title("dose response, direction-only steer (block2)", fontsize=9); a.legend(fontsize=7)
except KeyError:
    pass
fig.tight_layout()
if accel:
    for a in ax[:2]:
        a.set_title(a.get_title().replace("speed", "acceleration"), fontsize=9)
    ax[1].set_ylabel("acceleration-probe error (m/s^2)")
    ax[3].set_ylabel("acceleration leakage (m/s^2)")
out = ROOT / "figures" / (f"fig_accel_sheet_v2_L{L}.png" if accel else f"fig_velocity_sheet_v2{'' if L == '22' else '_L' + L}.png")
fig.savefig(out, dpi=130)
print("wrote", out)

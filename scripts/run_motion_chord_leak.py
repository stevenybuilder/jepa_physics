"""Why does the raw-centroid direction chord leak into the speed readout while the joint-encoding (Fourier) edit does not?
Hypothesis: centroid imbalance. A per-value centroid averages whatever nuisance values its clips happen to have; the chord
c(target) - c(source) then carries (nuisance imbalance) x (nuisance encoding direction). Direction set, meanpool, knot
folds 0-2 at kept values (the exp3 fourier setup, held-out arc 303.75-343.125). For each carrier the chord's signed
speed-readout change (speed-set ridge probe, transferred) is compared with the imbalance in TRUE mean speed and in the
share of constant-velocity clips between the clips that build the target (interpolated) and source centroids.
Writes results/p5_motion_geometry_parts/chord_leak.json."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_motion_geometry import ARC, ARC_NEIGH, PARTS, load_mp, meta  # noqa: E402
from wm import motiongeom as mg  # noqa: E402

out = {"method": __doc__, "points": {}}
df, f = meta("direction"); th = df.theta_degrees.to_numpy(float)
spd = df.speed_mps.to_numpy(float); vel = (df.motion == "velocity").to_numpy(float); acc = df.acceleration_mps2.to_numpy(float)
dfs, fs = meta("speed")
held = np.isin(np.round(th, 4), np.round(ARC, 4))
knot = np.isin(f, (0, 1, 2)) & ~held
test = (f < 0) & ~held
for p in (12, 22):
    X = load_mp("direction", "vjepa2", p)
    Xs = load_mp("speed", "vjepa2", p)
    sp = mg.SVDRidge(Xs[fs >= 0]).fit(dfs.speed_mps[fs >= 0].to_numpy(float))
    vals = np.unique(th[knot])
    c = {v: X[knot & np.isclose(th, v)].mean(0) for v in vals}
    ms = {v: spd[knot & np.isclose(th, v)].mean() for v in vals}
    mv = {v: vel[knot & np.isclose(th, v)].mean() for v in vals}
    ma = {v: acc[knot & np.isclose(th, v)].mean() for v in vals}
    rng = np.random.default_rng(p)
    rc = np.repeat(np.where(test)[0], 3); tgt = rng.choice(ARC, len(rc)); src = th[rc]
    a, b = ARC_NEIGH; w = (tgt - a) / (b - a)
    D = (1 - w)[:, None] * c[a] + w[:, None] * c[b] - np.array([c[v] for v in src])
    leak = sp.predict(X[rc] + D) - sp.predict(X[rc])
    d_s = (1 - w) * ms[a] + w * ms[b] - np.array([ms[v] for v in src])
    d_v = (1 - w) * mv[a] + w * mv[b] - np.array([mv[v] for v in src])
    d_a = (1 - w) * ma[a] + w * ma[b] - np.array([ma[v] for v in src])
    Z = np.column_stack([d_s, d_v, d_a, np.ones_like(d_s)])
    coef, *_ = np.linalg.lstsq(Z, leak, rcond=None)
    r2 = 1 - ((leak - Z @ coef) ** 2).sum() / ((leak - leak.mean()) ** 2).sum()
    out["points"][str(p)] = {"mean_abs_leak_mps": float(np.abs(leak).mean()),
                             "pearson_leak_vs_true_speed_imbalance": [float(x) for x in pearsonr(d_s, leak)],
                             "pearson_leak_vs_velocity_share_imbalance": [float(x) for x in pearsonr(d_v, leak)],
                             "pearson_leak_vs_accel_imbalance": [float(x) for x in pearsonr(d_a, leak)],
                             "r2_leak_on_imbalances": float(r2), "coef_speed_vel_acc_const": [float(x) for x in coef],
                             "n_rows": int(len(rc))}
    print(p, out["points"][str(p)], flush=True)
(PARTS / "chord_leak.json").write_text(json.dumps(out, indent=1))

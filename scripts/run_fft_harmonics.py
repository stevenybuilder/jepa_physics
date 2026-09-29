"""Harmonic content of the direction code (Kantamneni & Tegmark 2502.00873: FFT over values finds periods).

At every stored point: PCA-64 on train clips, the 64 direction-class centroids (wm.manifold.centroids, as in
run_part2.build), FFT of each PCA coordinate along the angle axis, power summed over coordinates, reported as the
fraction of non-constant power in harmonics k = 1, 2, 3, >= 4. CI: stratified clip bootstrap (resample train clips
within each class, PCA fixed). Controls: the random-init encoder, and a label-shuffle noise floor (centroid noise
only). The speed set gets the same FFT along its value axis; a straight line there gives the sawtooth spectrum
(power ~ 1/k^2), reported as the reference.

Writes results/p2_fft_harmonics.json and figures/fig_fft_harmonics.png.
  python scripts/run_fft_harmonics.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))  # run without installing wm
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs

BANDS = ("k1", "k2", "k3", "k4plus")


def harmonic_fractions(C):
    """C [m, k] centroids ordered along the value axis (equally spaced). Returns the fraction of non-constant power
    in k = 1, 2, 3, >= 4 (two-sided power: rfft bins 1..m/2-1 count twice, the Nyquist bin once)."""
    C = np.asarray(C, dtype=float)
    m = len(C)
    F = np.fft.rfft(C - C.mean(0), axis=0)
    p = (np.abs(F) ** 2).sum(1)
    w = np.full(len(p), 2.0)
    w[0] = 0.0
    if m % 2 == 0:
        w[-1] = 1.0
    p = p * w
    tot = p.sum()
    if tot <= 0:
        return {b: float("nan") for b in BANDS}
    return {"k1": p[1] / tot, "k2": p[2] / tot, "k3": p[3] / tot, "k4plus": p[4:].sum() / tot}


def spectrum_point(X, y, n_boot=200, n_shuffle=20, seed=0, k=64):
    """Fractions at one point for train rows X [n, D], labels y [n]; bootstrap CI and label-shuffle floor."""
    pca = mf.fit_pca(X, k)
    Z = pca.project(X)
    cent = mf.centroids(Z, y)
    est = harmonic_fractions(cent["C"])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == v) for v in cent["values"]]
    boots = []
    for _ in range(n_boot):
        C = np.stack([Z[rng.choice(g, len(g))].mean(0) for g in groups])
        boots.append([harmonic_fractions(C)[b] for b in BANDS])
    boots = np.array(boots)
    tot_real = _total_power(cent["C"])
    shuf_tot, shuf_frac = [], []
    for _ in range(n_shuffle):
        c = mf.centroids(Z, rng.permutation(y))["C"]
        shuf_tot.append(_total_power(c))
        shuf_frac.append([harmonic_fractions(c)[b] for b in BANDS])
    return {**{b: float(est[b]) for b in BANDS},
            "ci95": {b: [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]
                     for i, b in enumerate(BANDS)},
            "shuffle_noise_power_over_real": float(np.mean(shuf_tot) / tot_real),
            "shuffle_fractions": {b: float(np.mean(np.array(shuf_frac)[:, i])) for i, b in enumerate(BANDS)}}


def _total_power(C):
    C = np.asarray(C, float)
    return float(((C - C.mean(0)) ** 2).sum())


def sweep(dataset, variable, act_dir, points, n_boot, seed):
    out = {}
    for p in points:
        d = load_inputs(dataset, p, variable, act_dir)
        tr = d["is_train"]
        out[str(p)] = spectrum_point(d["X"][tr].astype(np.float64), d["y"][tr], n_boot=n_boot, seed=seed)
    return out


def ramp_reference(m=64):
    return harmonic_fractions(np.arange(m, dtype=float)[:, None])


def plot(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = {"k1": "C0", "k2": "C3", "k3": "C2", "k4plus": "C7"}
    for ax, (key, title) in zip(axes, (("direction_vjepa2", "direction, V-JEPA 2 (trained)"),
                                       ("direction_random", "direction, random-init encoder"),
                                       ("speed_vjepa2", "speed, V-JEPA 2 (trained)"))):
        r = res[key]
        pts = sorted(int(p) for p in r)
        for b in BANDS:
            v = np.array([r[str(p)][b] for p in pts])
            lo = np.array([r[str(p)]["ci95"][b][0] for p in pts])
            hi = np.array([r[str(p)]["ci95"][b][1] for p in pts])
            ax.plot(pts, v, "o-", ms=3, color=cols[b], label={"k4plus": "k >= 4"}.get(b, f"k = {b[1]}"))
            ax.fill_between(pts, lo, hi, color=cols[b], alpha=0.2)
        if key.startswith("speed"):
            for b in ("k1", "k2"):
                ax.axhline(res["ramp_reference"][b], color=cols[b], ls=":", lw=1)
        ax.set(title=title, xlabel="point (block)", ylim=(0, 1))
    axes[0].set_ylabel("fraction of non-constant centroid power")
    axes[0].legend(fontsize=8)
    axes[2].text(0.02, 0.97, "dotted: straight line (sawtooth) k=1, k=2", transform=axes[2].transAxes, fontsize=7,
                 va="top")
    fig.suptitle("FFT of the 64 class centroids (PCA-64, train clips) along the value axis; bands: 95% clip bootstrap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--points", default="0-25")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures"))
    a = p.parse_args(argv)
    lo, hi = (int(s) for s in a.points.split("-"))
    pts = range(lo, hi + 1)
    act = PROJECT_ROOT / "artifacts" / "activations"
    res = {"method": "PCA-64 on train clips per point; 64 class centroids (wm.manifold.centroids); rfft of each PCA "
                     "coordinate along the sorted value axis; two-sided power summed over coordinates; fractions of "
                     "non-constant (k >= 1) power. CI: 95% percentile, stratified clip bootstrap (PCA fixed). "
                     "shuffle: labels permuted among train clips (centroid-noise floor), mean of 20.",
           "source": "Kantamneni & Tegmark 2025, arXiv 2502.00873 (FFT over values finds periods)",
           "n_boot": a.n_boot, "seed": a.seed,
           "direction_vjepa2": sweep("direction", "direction", act / "direction" / "vjepa2", pts, a.n_boot, a.seed),
           "direction_random": sweep("direction", "direction", act / "direction" / "random", pts, a.n_boot, a.seed),
           "speed_vjepa2": sweep("speed", "speed", act / "speed" / "vjepa2", pts, a.n_boot, a.seed),
           "ramp_reference": {b: float(v) for b, v in ramp_reference().items()},
           "ramp_note": "a perfectly straight, evenly sampled line is a sawtooth along a non-periodic axis: its FFT has "
                        "power ~ 1/k^2 (k = 1 0.61, k = 2 0.15), so a line is NOT all k = 1 under a periodic FFT"}
    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    Path(a.figures_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.results_dir) / "p2_fft_harmonics.json").write_text(json.dumps(res, indent=1))
    plot(res, Path(a.figures_dir) / "fig_fft_harmonics.png")
    return res


if __name__ == "__main__":
    main()

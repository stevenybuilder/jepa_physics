"""Across-arc summary of one arc directory, written into results/p2_coast_faithful_L{L}.json under a key per directory:
  results/arcs_coast_faithful/       -> "arcs"       (pass 1: full + pca16, selected alpha, stride 8, no tokens/random)
  results/arcs_coast_faithful_full/  -> "arcs_full"  (pass 2: full, pca16, pca64, full_jaegerR, alpha sweep, random,
                                                       stride 4, no tokens)
Every arm present in all arcs of that directory; per arc = that arc's mean over its steers. The directory and each
arc's own config are recorded; arcs whose configs differ are refused.
  python scripts/aggregate_coast_faithful.py 12 results/arcs_coast_faithful arcs
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("probe_own", "probe_chordnorm", "probe_splinenorm", "probe_interpnorm", "mlp_own", "mlp_chordnorm",
           "mlp_interpnorm", "R_own", "R_chordnorm", "R_interpnorm", "norm_ratio_to_chord")


def aggregate(L, arcs_dir):
    arcs_dir = Path(arcs_dir)
    res = {s: json.loads(p.read_text()) for s in range(1, 17) if (p := arcs_dir / f"L{L}_s{s}.json").exists()}
    if not res:
        return None
    strip = lambda c: {k: v for k, v in c.items() if k not in ("target_condition_knots",)}
    cfgs = {json.dumps(strip(r["config"]), sort_keys=True) for r in res.values()}
    assert len(cfgs) == 1, f"{arcs_dir}: arcs with different configs: {cfgs}"
    common = set.intersection(*(set(r["arms"]) for r in res.values()))
    out = {"arcs_dir": str(arcs_dir.relative_to(ROOT)), "seeds": sorted(res), "n_arcs": len(res),
           "angle_source": {str(s): r["angle_source"] for s, r in res.items()},
           "alpha_selected": {v: {str(s): r["aperture"][v]["selected"] for s, r in res.items()}
                              for v in next(iter(res.values()))["aperture"]},
           "any_alpha_in_band": {v: any(r["aperture"][v]["in_band"] for r in res.values())
                                 for v in next(iter(res.values()))["aperture"]},
           "mean_overlap_range": {v: [float(min(min(r["aperture"][v]["mean_overlap"].values()) for r in res.values())),
                                      float(max(max(r["aperture"][v]["mean_overlap"].values()) for r in res.values()))]
                                  for v in next(iter(res.values()))["aperture"]},
           "config": next(iter(res.values()))["config"], "arms": {}}
    out["config"].pop("target_condition_knots", None)
    for arm in sorted(common):
        o = {}
        for q in METRICS:
            v = np.array([r["arms"][arm][q] for r in res.values()])
            o[q] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
        g = np.array([r["arms"][arm]["gap_vs_chord_probe_chordnorm"]["mean"] for r in res.values()])
        ci = np.array([r["arms"][arm]["gap_vs_chord_probe_chordnorm"]["ci95"] for r in res.values()])
        o["gap_vs_chord_probe_chordnorm"] = {"mean": float(g.mean()), "sd": float(g.std(ddof=1)) if len(g) > 1 else None,
                                             "n_arcs_ci_above_zero": int((ci[:, 0] > 0).sum()),
                                             "n_arcs_ci_below_zero": int((ci[:, 1] < 0).sum())}
        go = np.array([r["arms"][arm]["gap_vs_chord_probe_own"]["mean"] for r in res.values()])
        cio = np.array([r["arms"][arm]["gap_vs_chord_probe_own"]["ci95"] for r in res.values()])
        o["gap_vs_chord_probe_own"] = {"mean": float(go.mean()), "n_arcs_ci_above_zero": int((cio[:, 0] > 0).sum()),
                                       "n_arcs_ci_below_zero": int((cio[:, 1] < 0).sum())}
        for key in ("gap_vs_interp_probe_interpnorm", "gap_vs_interp_probe_own"):
            gi = np.array([r["arms"][arm][key]["mean"] for r in res.values()])
            cii = np.array([r["arms"][arm][key]["ci95"] for r in res.values()])
            o[key] = {"mean": float(gi.mean()), "sd": float(gi.std(ddof=1)) if len(gi) > 1 else None,
                      "n_arcs_ci_above_zero": int((cii[:, 0] > 0).sum()), "n_arcs_ci_below_zero": int((cii[:, 1] < 0).sum())}
        out["arms"][arm] = o
    return out


if __name__ == "__main__":
    L, d, key = int(sys.argv[1]), ROOT / sys.argv[2], sys.argv[3]
    hp = ROOT / "results" / f"p2_coast_faithful_L{L}.json"
    h = json.loads(hp.read_text())
    h[key] = aggregate(L, d)
    hp.write_text(json.dumps(h, indent=1))
    print(f"{hp}: {key} n = {h[key]['n_arcs'] if h[key] else 0}")

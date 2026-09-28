"""16-arc aggregate of the conceptor arms (run_conceptor.py --lite per arc) beside the stored spline and raw-chord
16-arc means (results/arcs_rawchord, run_part2 rows). Arcs = the 16 contiguous held-out blocks of REPORT section 4.3:
point 12 seeds 1-16 with the default (label-free) angle rule, run with --aim arc (issue #266: with the legacy coord aim
the conceptor target condition sat on foreign knots on seeds 1, 4, 5, 7, 8, 9, 11, 13); point 22 seeds 1-16 with the
labels angle (the arcs_rawchord L22 runs, labels reruns for seeds 4, 8, 9, 10). CI = 95% bootstrap resampling clips
within each arc (a clip keeps all its targets), arcs fixed (wm.makelov.arc_clip_bootstrap).

  python scripts/aggregate_conceptor_16arc.py --layer 12 [--arcs-dir results/arcs_conceptor16]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT  # noqa: E402
from wm.makelov import arc_clip_bootstrap  # noqa: E402
from wm.provenance import git_commit, sha256_file  # noqa: E402

ARMS = ("manifold", "linear_raw", "coast_a", "coast_a_b0.3", "coast_a_centered", "coast_b", "coast_b_dose_matched")
LABELS22 = (4, 8, 9, 10)
MISAIMED12 = (1, 4, 5, 7, 8, 9, 11, 13)
RES = PROJECT_ROOT / "results"


def stored_path(L, s, fixed):
    if L == 12 and fixed and s in MISAIMED12:
        return RES / "arcs_rawchord_fixed" / f"L12_s{s}" / "p2_steer_direction_direction_L12_contiguous.json"
    sub = f"L22_s{s}_labels" if L == 22 and s in LABELS22 else f"L{L}_s{s}"
    return RES / "arcs_rawchord" / sub / f"p2_steer_direction_direction_L{L}_contiguous.json"


def stored_block(L, fixed):
    out, per = {}, {}
    for arm in ("manifold", "linear_raw"):
        v, g, means = [], [], []
        for s in range(1, 17):
            d = json.loads(stored_path(L, s, fixed).read_text())
            rows = [r for r in d["rows"] if r["arm"] == arm]
            v.append(np.array([r["probe_err_to_target"] for r in rows]))
            g.append(np.array([r["id"] for r in rows]))
            means.append(d["summary"][arm]["overall"]["probe_err_to_target"])
            per.setdefault(arm, {})[s] = means[-1]
        out[arm] = {**arc_clip_bootstrap(v, g, 2000, 0), "mean_of_stored_summary_means": float(np.mean(means))}
    return out, per


def main(L, arcs_dir):
    arcs = {}
    for s in range(1, 17):
        p = Path(arcs_dir) / f"L{L}_s{s}" / f"p2_conceptor_direction_L{L}.json"
        if p.exists():
            arcs[s] = json.loads(p.read_text())
    seeds = sorted(arcs)
    res = {"layer": L, "n_arcs": len(seeds), "seeds": seeds, "arms": {}}
    chord = {s: np.array(arcs[s]["per_clip"]["linear_raw"]["probe_err_to_target"]) for s in seeds}
    for arm in ARMS:
        v = [np.array(arcs[s]["per_clip"][arm]["probe_err_to_target"]) for s in seeds]
        g = [np.array(arcs[s]["per_clip"][arm]["id"]) for s in seeds]
        for s, gi in zip(seeds, g):
            assert (gi == np.array(arcs[s]["per_clip"]["linear_raw"]["id"])).all()
        gap = [vi - chord[s] for vi, s in zip(v, seeds)]
        am = np.array([vi.mean() for vi in v])
        gm = np.array([x.mean() for x in gap])
        res["arms"][arm] = {
            "endpoint_err_deg": arc_clip_bootstrap(v, g, 2000, 0),
            "endpoint_err_sd_across_arcs": float(am.std(ddof=1)),
            "gap_vs_raw_chord_deg": arc_clip_bootstrap(gap, g, 2000, 0),
            "n_arcs_better_than_raw_chord": int((gm < 0).sum()),
            "per_arc_endpoint_err_deg": {str(s): float(x) for s, x in zip(seeds, am)},
            "unsteered_err_deg_mean": float(np.mean([arcs[s]["arms"][arm]["unsteered_err_deg"] for s in seeds])),
            "nearest_real_R_mean": float(np.mean([arcs[s]["arms"][arm]["nearest_real_R"]["mean"] for s in seeds])),
            "norm_ratio_to_raw_chord_mean": float(np.mean([arcs[s]["arms"][arm]["delta_norm_ratio_to_raw_chord"]
                                                           ["ratio_of_means"] for s in seeds])),
            "ring_plane_energy_frac_mean": float(np.mean([arcs[s]["arms"][arm]["off_target"]
                                                          ["ring_plane_energy_frac"]["mean"] for s in seeds]))}
    st_legacy, per_legacy = stored_block(L, fixed=False)
    res["stored_16arc"] = {"legacy_coord_aim": st_legacy}
    per_fixed = per_legacy
    if L == 12:
        st_fixed, per_fixed = stored_block(L, fixed=True)
        res["stored_16arc"]["aim_fixed_on_8_misaimed_arcs"] = st_fixed
    res["stored_16arc"]["source"] = ("results/arcs_rawchord/L{L}_s{1..16}[_labels for L22 s4,8,9,10]/"
                                     "p2_steer_direction_direction_L{L}_contiguous.json rows; point 12 fixed variant "
                                     "swaps in results/arcs_rawchord_fixed/ for seeds 1,4,5,7,8,9,11,13")
    ref = per_fixed
    res["reproduction_check"] = {arm: {"max_abs_diff_arc_mean_deg": float(max(
        abs(res["arms"][arm]["per_arc_endpoint_err_deg"][str(s)] - ref[arm][s]) for s in seeds)),
        "vs": "stored arc summary (aim-fixed where applicable)"} for arm in ("manifold", "linear_raw")}
    res["aperture"] = {str(s): {"alpha": arcs[s]["aperture"]["selected"], "in_band": arcs[s]["aperture"]["in_band"]}
                       for s in seeds}
    res["held_out_aim"] = {str(s): arcs[s].get("held_out_aim") for s in seeds}
    res["angle_source"] = {str(s): arcs[s]["angle_source"] for s in seeds}
    res["notes"] = {"coast_faithful": "coast_a_b0.3 is the COAST-retained beta; coast_a runs beta to 1 (outside COAST)",
                    "coast_b": "target-aimed update is our variant, not COAST's",
                    "recipe": "current src/wm/conceptor.py (a separate COAST-faithful audit is pending); --lite, "
                              "4 random-projector draws per arc (the random null is not aggregated here)",
                    "ci": "95% bootstrap, clips resampled within each arc, arcs fixed, 2000 draws"}
    res["provenance"] = {"aggregated_at": git_commit(),
                         "arc_files": {str(s): {"sha256": sha256_file(Path(arcs_dir) / f"L{L}_s{s}" /
                                                                      f"p2_conceptor_direction_L{L}.json"),
                                                "git": arcs[s]["provenance"].get("git_commit") or
                                                arcs[s]["provenance"].get("git")} for s in seeds}}
    out = RES / f"p2_conceptor_direction_16arc_L{L}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    for arm in ARMS:
        a = res["arms"][arm]
        print(f"L{L} {arm:22s} {a['endpoint_err_deg']['mean']:7.2f} {a['endpoint_err_deg']['ci95']} gap "
              f"{a['gap_vs_raw_chord_deg']['mean']:+7.2f} {a['gap_vs_raw_chord_deg']['ci95']}")
    print(json.dumps(res["stored_16arc"], indent=0, default=str)[:1500])
    print(res["reproduction_check"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--arcs-dir", default=str(RES / "arcs_conceptor16"))
    a = p.parse_args()
    main(a.layer, a.arcs_dir)

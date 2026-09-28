"""Label-free angle computed as Goodfire's code does (mf.goodfire_periodic_angle: variance-scaled atan2 on activation
PC1/PC2 of the knot centroids, near-equal-variance periodicity test with tolerance 0.45, no centroid-plane fallback),
on exactly the knot centroids run_part2.build fits (knot folds 0-2 at kept values, PCA-64 on them), for the contiguous
design at points 8 / 12 / 22 (seed 0) and the 16 arc seeds at points 12 / 22. No steering rerun: per run, whether
their test passes, circular correlation / max deviation against the labels (mf.compare_angles), the same numbers for
our unscaled atan2 in the activation plane, and the angle source our pipeline used (choose_angle_source recomputed
here, and the stored steering JSON's angle_source where one exists).

  PYTHONPATH=src python scripts/run_angle_goodfire.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wm import geometry_checks as gc
from wm import manifold as mf
from wm.data import PROJECT_ROOT
from wm.p2_data import load_inputs
from wm.provenance import provenance

RES = PROJECT_ROOT / "results"


def angle_row(d, seed, design="contiguous", k=64):
    """Goodfire-method and pipeline angle checks on the knot centroids run_part2.build uses for (design, seed)."""
    values = np.unique(d["y"])
    mask, info = mf.heldout_design(values, design, True, seed=seed)
    knot = (d["role"] == "knot") & ~np.isin(d["y"], values[mask])
    pca = gc.fit_subspace(d["X"][knot], d["y"][knot], "pca", k, None)
    cent = mf.centroids(pca.project(d["X"][knot]), d["y"][knot])
    g = mf.goodfire_periodic_angle(cent["C"])
    cg = mf.compare_angles(g["angle"], cent["values"])
    ch = mf.choose_angle_source(cent["C"], cent["values"])
    ca = ch["checks"]["activation"]
    keep = ("circular_corr", "max_dev_deg", "mean_dev_deg", "order_coarse_preserved")
    return {"held_out": [info["block_first_value"], info["block_last_value"]],
            "goodfire_passes_periodicity_test": g["passes"], "goodfire_eigenvalues_pc1_pc2": g["eigenvalues"],
            "goodfire_rel_diff": g["rel_diff"], "goodfire_angle_vs_labels": {q: cg[q] for q in keep},
            "ours_unscaled_activation_plane_vs_labels": {q: ca[q] for q in keep},
            "pipeline_source": ("labels_angle" if ch["angle"] == "labels" else f"unsupervised_angle ({ch['plane']} "
                                                                               "plane)")}


def stored_source(L, seed):
    p = (RES / f"p2_steer_direction_direction_L{L}_contiguous.json" if seed == 0 else
         RES / "arcs" / f"L{L}_s{seed}" / f"p2_steer_direction_direction_L{L}_contiguous.json")
    return json.loads(p.read_text())["angle_source"] if p.exists() else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(RES / "p2_angle_goodfire_method.json"))
    a = p.parse_args(argv)
    runs = {8: [0], 12: list(range(17)), 22: list(range(17))}
    out = {"provenance": provenance(None, seeds={"designs": "contiguous seed 0 at 8; seeds 0-16 at 12 and 22"}),
           "method": mf.goodfire_periodic_angle.__doc__.split("Returns")[0].strip(),
           "note": ("seed 0 = the headline contiguous arc (303.75-343.125); seeds 1-16 = results/arcs; "
                    "stored_source = the stored steering JSON's angle_source (none stored at point 8)"),
           "points": {}}
    for L, seeds in runs.items():
        d = load_inputs("direction", L, "direction")
        out["points"][str(L)] = {str(s): {**angle_row(d, s), "stored_source": stored_source(L, s)} for s in seeds}
        rows = out["points"][str(L)].values()
        print(f"L{L}: goodfire test passes {sum(r['goodfire_passes_periodicity_test'] for r in rows)}/{len(seeds)}",
              flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

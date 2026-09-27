"""Step 2 extra (spec §6 item 7, paper C.4 method): direction-vs-speed and direction-vs-acceleration
subspace angles from the saved INLP bases, per layer point where both bases exist.

  python scripts/run_step2_angles.py [--pool meanpool] [--inlp-dir artifacts/inlp]

Writes results/step2_subspace_angles.json. The bases come from different datasets (direction set,
speed set, acceleration set) at the same layer point, each in its own dataset's train-standardised
coordinates. The primary numbers map the speed/acceleration readout directions into the direction
set's coordinates (covectors_into, needs the activations for the per-feature train SDs); the
"as_stored" numbers compare the stored bases directly and are reported for reference.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.inlp import basis_path, covectors_into, load_basis, subspace_overlap  # noqa: E402
from wm.probes import (N_POINTS, RESULTS, Standardizer, layer_fraction, layer_matrix, load_activations,  # noqa: E402
                       split_rows, write_json)

parser = argparse.ArgumentParser()
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--random-draws", type=int, default=20)
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
args = parser.parse_args()


def train_std(dataset, point, cache={}):
    """Per-feature train SD at one layer point (the standardiser of steps 1-3), or None without activations."""
    if dataset not in cache:
        try:
            cache[dataset] = (load_activations(dataset, args.pool, act_root=args.act_root), split_rows(dataset)[0])
        except FileNotFoundError:
            cache[dataset] = None
    if cache[dataset] is None:
        return None
    acts, tr = cache[dataset]
    return Standardizer().fit(layer_matrix(acts, point)[tr]).std


out = {"pool": args.pool, "A": "direction (direction set)",
       "note": ("bases for different variables come from different datasets (direction set vs speed set vs "
                "acceleration set) but the same layer point; primary numbers map B's readout directions into the "
                "direction set's train-standardised coordinates (w * sd_direction / sd_B); as_stored compares the "
                "stored bases directly. Paper C.4 metrics: mean principal angle, Grassmann distance sqrt(sum theta^2) "
                "(radians), overlap A<-B = ||Q_A^T Q_B||_F^2 / dim B (random k_A/d) and B<-A (/ dim A, random k_B/d)."),
       "pairs": {}}
for other in ("speed", "acceleration"):
    rows = []
    for point in range(N_POINTS):
        pa, pb = (basis_path(v, v, point, args.pool, args.inlp_dir) for v in ("direction", other))
        if not (pa.exists() and pb.exists()):
            continue
        QA, QB = load_basis(pa)["Q"], load_basis(pb)["Q"]
        row = {"point": point, "frac": layer_fraction(point), "post_ln": point == N_POINTS - 1}
        if QA.shape[1] == 0 or QB.shape[1] == 0:
            row["skipped"] = f"empty basis (K_direction={QA.shape[1] // 2}, K_{other}={QB.shape[1]})"
            rows.append(row)
            continue
        row["as_stored"] = subspace_overlap(QA, QB, args.random_draws)
        sd_a, sd_b = train_std("direction", point), train_std(other, point)
        if sd_a is not None and sd_b is not None:
            row["mapped"] = subspace_overlap(QA, covectors_into(QB, sd_b, sd_a), args.random_draws)
        rows.append(row)
        main = row.get("mapped", row["as_stored"])
        print(f"direction vs {other} point {point:2d}: k_A={main['k_A']} k_B={main['k_B']} "
              f"mean angle {main['mean_angle_deg']:.1f} Grassmann {main['grassmann_distance_rad']:.2f} "
              f"overlap A<-B {main['overlap_A_from_B']:.3f} (random {main['random_expectation_A_from_B']:.3f}) "
              f"B<-A {main['overlap_B_from_A']:.3f} (random {main['random_expectation_B_from_A']:.3f})")
    out["pairs"][f"direction_vs_{other}"] = rows
write_json(Path(args.results or RESULTS) / "step2_subspace_angles.json", out)

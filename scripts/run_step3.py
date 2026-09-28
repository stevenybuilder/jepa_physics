"""Step 3: multi-probe subspace steering with a held-out evaluation probe (paper App. C.12).
Needs step 2's basis at the same layer; runs at the step-1 onset and peak layers and the paper's layer (point 9 =
the paper's layer 8) by default. Writes results/p1c_{dataset}_L{layer}.json.

  python scripts/run_step3.py --dataset direction [--layer 12]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import VARIABLES, load_sweep  # noqa: E402
from wm.provenance import step_layers  # noqa: E402
from wm.steer import run_rank_matched_null, run_steering  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None, help="default: the dataset's primary variable")
parser.add_argument("--layer", type=int, default=None, help="layer point 0..25; overrides --layer-role")
parser.add_argument("--layer-role", default="all", choices=["peak", "onset", "both", "paper", "all"],
                    help="step-1 CV peak, availability onset, both, the paper's layer (point 9), or onset + paper + "
                         "peak (default)")
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
parser.add_argument("--model", default="vjepa2", choices=["vjepa2", "random"],
                    help="random = the random-init ViT-L control; outputs carry _random")
parser.add_argument("--strict-eval", action="store_true",
                    help="extra: also fit the evaluation probe on one half of test and steer the other half")
parser.add_argument("--null-draws", type=int, default=20, help="draws per random null (>= 20 per spec 5.3)")
parser.add_argument("--n-max", type=int, default=None,
                    help="cap the probe-count sweep at this N (default K); N grid = 1..25, 30, 40, 55, 75, 100, "
                         "140, 200, top")
parser.add_argument("--rank-matched-null", action="store_true",
                    help="extra arm only: random-basis null of rank m*N (the learned basis' rank at N) beside the "
                         "rank-K*m null; writes p1c_{dataset}_L{layer}_rankmatched.json, leaves p1c files untouched")
args = parser.parse_args()

for dataset in [args.dataset] if args.dataset else VARIABLES:
    variable = args.variable or dataset
    roles = ({args.layer: "given"} if args.layer is not None
             else step_layers(load_sweep(dataset, variable, args.pool, args.results, args.model), args.layer_role,
                              alt=False))
    for point, role in roles.items():
        if args.rank_matched_null:
            run_rank_matched_null(dataset, args.variable, point, args.pool, args.act_root, args.results,
                                  args.inlp_dir, args.null_draws, role, args.model, args.n_max)
            continue
        run_steering(dataset, args.variable, point, args.pool, args.act_root, args.results, args.inlp_dir,
                     args.null_draws, role, args.model, args.strict_eval, args.n_max)

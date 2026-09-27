"""Step 3: multi-probe subspace steering with a held-out evaluation probe (paper App. C.12).
Needs step 2's basis at the same layer; runs at the step-1 peak and onset layers by default. Writes results/p1c_{dataset}_L{layer}.json.

  python scripts/run_step3.py --dataset direction [--layer 12]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import VARIABLES, chosen_layers, load_sweep  # noqa: E402
from wm.steer import run_steering  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None, help="default: the dataset's primary variable")
parser.add_argument("--layer", type=int, default=None, help="layer point 0..25; overrides --layer-role")
parser.add_argument("--layer-role", default="both", choices=["peak", "onset", "both"],
                    help="step-1 CV peak, availability onset (the paper's emergence layer), or both (default)")
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
parser.add_argument("--model", default="vjepa2", choices=["vjepa2", "random"],
                    help="random = the random-init ViT-L control; outputs carry _random")
parser.add_argument("--null-draws", type=int, default=20, help="draws per random null (>= 20 per spec 5.3)")
args = parser.parse_args()

for dataset in [args.dataset] if args.dataset else VARIABLES:
    variable = args.variable or dataset
    roles = ({args.layer: "given"} if args.layer is not None
             else chosen_layers(load_sweep(dataset, variable, args.pool, args.results, args.model), args.layer_role))
    for point, role in roles.items():
        run_steering(dataset, args.variable, point, args.pool, args.act_root, args.results, args.inlp_dir,
                     args.null_draws, role, args.model)

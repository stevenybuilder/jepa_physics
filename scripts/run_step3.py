"""Step 3: multi-probe subspace steering with a held-out evaluation probe (paper App. C.12).
Needs step 2's basis at the same layer. Writes results/p1c_{dataset}_L{layer}.json.

  python scripts/run_step3.py --dataset direction [--layer 12]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import VARIABLES  # noqa: E402
from wm.steer import run_steering  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=list(VARIABLES), default=None)
parser.add_argument("--variable", default=None, help="default: the dataset's primary variable")
parser.add_argument("--layer", type=int, default=None, help="layer point 0..25; default: step-1 CV peak")
parser.add_argument("--pool", default="meanpool", choices=["meanpool", "timepool", "diskpool"])
parser.add_argument("--act-root", type=Path, default=None)
parser.add_argument("--results", type=Path, default=None)
parser.add_argument("--inlp-dir", type=Path, default=None)
args = parser.parse_args()

for dataset in [args.dataset] if args.dataset else VARIABLES:
    run_steering(dataset, args.variable, args.layer, args.pool, args.act_root, args.results, args.inlp_dir)

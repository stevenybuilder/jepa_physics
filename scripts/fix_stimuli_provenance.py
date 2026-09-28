"""Correct the split recorded in the stimulus-set step-1 results: they were run with --split splits/split_{set}.json but
result_provenance recorded splits/split_v1.json. Only provenance.split_file / split_sha256 change.

  python scripts/fix_stimuli_provenance.py [--results results]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.probes import RESULTS  # noqa: E402
from wm.provenance import rewrite_split_provenance  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, default=RESULTS)
args = parser.parse_args()
for name in ("paper_layout", "hard"):
    for f in sorted((args.results / "stimuli" / name).glob("*.json")):
        old, new = rewrite_split_provenance(f, f"splits/split_{name}.json")
        print(f"{f}: {old[0]} {old[1][:12]} -> {new[0]} {new[1][:12]}")

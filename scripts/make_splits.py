"""Write splits/split_v1.json for all three datasets and validate it.

Frame hashes come from results/qa_{dataset}.json (run scripts/run_qa.py first); a few are
re-decoded here as a spot check.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import DATASETS, RESULTS_DIR, decode, frame_hash, load_table  # noqa: E402
from wm.splits import SPLIT_PATH, make_split, validate_split  # noqa: E402

out = {}
for dataset in DATASETS:
    df = load_table(dataset)
    qa = json.loads((RESULTS_DIR / f"qa_{dataset}.json").read_text())
    hashes = {c["id"]: c["frame_hash"] for c in qa["per_clip"]}
    assert set(hashes) == set(df["id"]), dataset
    for row in df.iloc[[0, len(df) // 2, len(df) - 1]].itertuples():
        assert frame_hash(decode(row.video)) == hashes[row.id], (dataset, row.id)
    out[dataset] = make_split(df, hashes)
    validate_split(dataset, out[dataset])
    print(dataset, out[dataset]["counts"])

SPLIT_PATH.parent.mkdir(exist_ok=True)
SPLIT_PATH.write_text(json.dumps(out, indent=1))
for dataset in DATASETS:
    validate_split(dataset)
print("wrote", SPLIT_PATH)

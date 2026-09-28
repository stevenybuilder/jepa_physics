"""Write splits/split_{set}.json for the stimulus sets (artifacts/stimuli/{set}): same make_split (80/20, seed 0,
stratified by theta, identical-frame units merged), frame hashes from decoding every clip. Key = "direction".

  python scripts/make_stimulus_splits.py [paper_layout hard]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wm.data import PROJECT_ROOT, decode, frame_hash, load_table  # noqa: E402
from wm.splits import make_split, validate_split  # noqa: E402

for name in sys.argv[1:] or ["paper_layout", "hard"]:
    df = load_table("direction", root=PROJECT_ROOT / "artifacts" / "stimuli" / name)
    hashes = {int(r.id): frame_hash(decode(r.video)) for r in df.itertuples()}
    split = make_split(df, hashes)
    validate_split("direction", split, df=df)
    out = PROJECT_ROOT / "splits" / f"split_{name}.json"
    out.write_text(json.dumps({"direction": split}, indent=1))
    print(name, split["counts"], "->", out)

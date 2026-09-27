"""Provenance block for every results JSON (split file + sha256, git commit, seeds, layer role, timestamp)."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from wm.data import PROJECT_ROOT
from wm.splits import SPLIT_PATH

PAPER_LAYER = 8          # the physics paper's layer for these experiments


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    """HEAD commit and whether tracked files differ from it (None outside a git checkout)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def layer_role(dataset, layer, variable=None, model="vjepa2", results_dir=None):
    """peak | onset | paper_layer | exploratory, from step 1's results/p1a_{dataset}_{variable}_meanpool*.json."""
    variable = variable or dataset
    name = f"p1a_{dataset}_{variable}_meanpool" + ("" if model == "vjepa2" else f"_{model}") + ".json"
    path = Path(results_dir or PROJECT_ROOT / "results") / name
    roles = []
    if path.exists():
        av = json.loads(path.read_text()).get("availability", {})
        roles += [r for r in ("peak", "onset") if av.get(r) == layer]
    if layer == PAPER_LAYER:
        roles.append("paper_layer")
    return {"layer_role": roles[0] if roles else "exploratory", "all_roles": roles,
            "source": str(path) if path.exists() else f"missing: {path}"}


def provenance(split=None, seeds=None, **fields):
    """Provenance dict. split: path of the split file actually read (default splits/split_v1.json). fields: layer,
    pool, holdout, spline, k, ... anything that defines the run."""
    split = Path(split) if split else SPLIT_PATH
    return {"split_file": str(split), "split_sha256": sha256_file(split) if split.exists() else None,
            "seeds": seeds, **git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields}

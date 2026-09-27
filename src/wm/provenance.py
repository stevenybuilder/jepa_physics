"""Provenance block for every results JSON (split file + sha256, git commit, seeds, layer role, timestamp).
The single provenance module: wm.common re-exports result_provenance for the Part 1 writers."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from wm.data import PROJECT_ROOT
from wm.splits import SEED, SPLIT_PATH

PAPER_LAYER = 8          # the physics paper's layer for these experiments


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=1)
def git_commit():
    """HEAD commit and uncommitted-change flags, under both key conventions: commit / dirty (tracked files anywhere)
    and git_commit / git_dirty_src_or_scripts (src/ or scripts/ only; the Part 1 writers' names)."""
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    head = run("rev-parse", "HEAD") or None
    dirty = bool(run("status", "--porcelain", "--untracked-files=no")) if head else None
    dirty_src = bool(run("status", "--porcelain", "--", "src", "scripts")) if head else None
    return {"commit": head, "dirty": dirty, "git_commit": head or "unknown", "git_dirty_src_or_scripts": dirty_src}


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


def result_provenance(obj=None):
    """Part 1 form (probes.write_json): split file + sha256, seeds, git commit, UTC timestamp, and the pool / model /
    layer of the result when it carries them. Carries both commit key conventions."""
    obj = obj if isinstance(obj, dict) else {}
    return {"split_file": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
            "split_sha256": sha256_file(SPLIT_PATH) if SPLIT_PATH.exists() else "missing",
            "seeds": {"split": SEED, "other": "all other RNG seeds are 0 (bootstrap, folds, null draws, shuffles); "
                                                "random-removal controls use seeds 0..n-1"},
            **git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pool": obj.get("pool"), "model": obj.get("model", "vjepa2" if "pool" in obj else None),
            "layer": obj.get("point")}

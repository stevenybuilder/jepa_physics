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

PAPER_LAYER = 9          # the physics paper's "layer 8" (0-23, block outputs) = output of block 9 = our point 9
PAPER_LAYER_ALT = 8      # our point 8 (output of block 8), run beside it in case the paper's index means the input
LAYER_NOTE = ("paper layers are 0-23 and each probes a block's output, so the paper's layer L = our point L+1 "
              "(point 0 = embedding, point k = output of block k, point 25 = post-LN). paper_layer = point 9 "
              "(the paper's layer 8); paper_layer_alt = point 8.")


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
    """peak | onset | paper_layer | paper_layer_alt | exploratory, from step 1's
    results/p1a_{dataset}_{variable}_meanpool*.json."""
    variable = variable or dataset
    name = f"p1a_{dataset}_{variable}_meanpool" + ("" if model == "vjepa2" else f"_{model}") + ".json"
    path = Path(results_dir or PROJECT_ROOT / "results") / name
    roles = []
    if path.exists():
        av = json.loads(path.read_text()).get("availability", {})
        roles += [r for r in ("peak", "onset") if av.get(r) == layer]
    if layer == PAPER_LAYER:
        roles.append("paper_layer")
    if layer == PAPER_LAYER_ALT:
        roles.append("paper_layer_alt")
    return {"layer_role": roles[0] if roles else "exploratory", "all_roles": roles,
            "source": str(path) if path.exists() else f"missing: {path}"}


def step_layers(sweep, role="all", alt=True):
    """{point: role label} for steps 2-3. role: peak | onset | both (probes.chosen_layers), paper (point 9, plus
    point 8 as paper_layer_alt if alt), or all (onset, peak, paper [, paper_alt]). Coinciding points are run once
    with joined labels."""
    from wm.probes import chosen_layers
    out = {} if role == "paper" else dict(chosen_layers(sweep, "both" if role == "all" else role))
    if role in ("paper", "all"):
        extra = [(PAPER_LAYER, "paper_layer")] + ([(PAPER_LAYER_ALT, "paper_layer_alt")] if alt else [])
        for point, name in extra:
            out[point] = f"{out[point]}+{name}" if point in out else name
    return out


def provenance(split=None, seeds=None, **fields):
    """Provenance dict. split: path of the split file actually read (default splits/split_v1.json). fields: layer,
    pool, holdout, spline, k, ... anything that defines the run."""
    split = Path(split) if split else SPLIT_PATH
    return {"split_file": str(split), "split_sha256": sha256_file(split) if split.exists() else None,
            "seeds": seeds, **git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields}


def split_record(split_path=None):
    """(split_file, split_sha256) of the split file actually read: path relative to the project root when inside it,
    sha256 of its bytes ("missing" if absent). Default: splits/split_v1.json."""
    split = Path(split_path) if split_path else SPLIT_PATH
    if not split.is_absolute():
        split = PROJECT_ROOT / split
    rel = split.relative_to(PROJECT_ROOT) if split.is_relative_to(PROJECT_ROOT) else split
    return str(rel), sha256_file(split) if split.exists() else "missing"


def result_provenance(obj=None, split_path=None):
    """Part 1 form (probes.write_json): split file + sha256, seeds, git commit, UTC timestamp, and the pool / model /
    layer of the result when it carries them. Carries both commit key conventions. split_path: the split file the
    run read (default splits/split_v1.json)."""
    obj = obj if isinstance(obj, dict) else {}
    split_file, split_sha = split_record(split_path)
    return {"split_file": split_file,
            "split_sha256": split_sha,
            "seeds": {"split": SEED, "other": "all other RNG seeds are 0 (bootstrap, folds, null draws, shuffles); "
                                                "random-removal controls use seeds 0..n-1"},
            **git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pool": obj.get("pool"), "model": obj.get("model", "vjepa2" if "pool" in obj else None),
            "layer": obj.get("point")}


def rewrite_split_provenance(path, split_path):
    """Point the provenance block of an existing results JSON at the split file actually used: only the values of
    provenance.split_file and provenance.split_sha256 change, every other byte stays. Returns (old, new) records."""
    path = Path(path)
    text = path.read_text()
    prov = json.loads(text)["provenance"]
    old = (prov["split_file"], prov["split_sha256"])
    new = split_record(split_path)
    start = text.rindex('"provenance": {')
    head, block = text[:start], text[start:]
    for key, o, n in (("split_file", *[json.dumps(v) for v in (old[0], new[0])]),
                      ("split_sha256", *[json.dumps(v) for v in (old[1], new[1])])):
        pat = f'"{key}": {o}'
        assert block.count(pat) == 1, f"{path}: {pat} not unique in the provenance block"
        block = block.replace(pat, f'"{key}": {n}')
    path.write_text(head + block)
    return old, new

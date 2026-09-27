"""Provenance stamped into every results JSON written through probes.write_json."""
import hashlib
import subprocess
from datetime import datetime, timezone
from functools import lru_cache

from wm.data import PROJECT_ROOT
from wm.splits import SEED, SPLIT_PATH


@lru_cache(maxsize=1)
def _git():
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    commit = run("rev-parse", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain", "--", "src", "scripts")) if commit != "unknown" else None
    return commit, dirty


@lru_cache(maxsize=1)
def _split_sha():
    try:
        return hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def provenance(obj=None):
    """Split file + sha256, seeds, git commit (and whether src/ or scripts/ had uncommitted changes),
    UTC timestamp, and the pool / model / layer of the result when the result carries them."""
    obj = obj if isinstance(obj, dict) else {}
    commit, dirty = _git()
    return {"split_file": str(SPLIT_PATH.relative_to(PROJECT_ROOT)), "split_sha256": _split_sha(),
            "seeds": {"split": SEED, "other": "all other RNG seeds are 0 (bootstrap, folds, null draws, shuffles); "
                                                "random-removal controls use seeds 0..n-1"},
            "git_commit": commit, "git_dirty_src_or_scripts": dirty,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pool": obj.get("pool"), "model": obj.get("model", "vjepa2" if "pool" in obj else None),
            "layer": obj.get("point")}

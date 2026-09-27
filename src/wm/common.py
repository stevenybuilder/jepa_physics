"""Backward-compatible alias: provenance lives in wm.provenance (single source)."""
from wm.provenance import git_commit, result_provenance as provenance  # noqa: F401

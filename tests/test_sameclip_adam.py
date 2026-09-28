"""run_sameclip_adam restricts to exactly run_sameclip's clips and split (real metadata; skipped if absent)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sameclip_adam_restriction_matches_ridge():
    from wm.probes import load_table, split_rows
    try:
        df = load_table("direction")
    except (FileNotFoundError, OSError):
        pytest.skip("direction metadata not available")
    ridge, adam = load("run_sameclip"), load("run_sameclip_adam")
    keep = (df["motion"] == "velocity").to_numpy()
    tr0, te0, f0 = ridge.restrict(*split_rows("direction", df), keep)
    _, keep1, tr1, te1, f1 = adam.sameclip_rows(df)
    assert keep1.sum() == 750 and len(tr1) == 596 and len(te1) == 154
    ids = df["id"].to_numpy()
    assert set(ids[keep1]) == set(ids[keep]) == set(ids[np.concatenate([tr1, te1])])
    np.testing.assert_array_equal(tr1, tr0)
    np.testing.assert_array_equal(te1, te0)
    np.testing.assert_array_equal(f1, f0)

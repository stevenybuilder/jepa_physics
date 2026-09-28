"""Token sets and doses of scripts/session2_disk_token_sweep.py (QA #338: the src set must not use the twin's disk)."""
import importlib.util
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("dts", Path(__file__).resolve().parents[1] / "scripts" / "session2_disk_token_sweep.py")
dts = importlib.util.module_from_spec(_s)
_s.loader.exec_module(dts)


def _masks():
    a = np.zeros((8, 16, 16), bool)
    b = np.zeros((8, 16, 16), bool)
    a[:, 2, 2] = True          # source disk stays top-left
    b[:, 12, 12] = True        # twin disk far away
    return a, b


def test_src_set_ignores_twin_location():
    a, b = _masks()
    m = dts.token_masks(a, b)
    m2 = dts.token_masks(a, np.zeros_like(b))
    assert (m["src"] == m2["src"]).all()                 # changing the twin's disk leaves the src set unchanged
    assert not (m["src"] & m["twinonly"]).any()
    assert ((m["src"] | m["twinonly"]) == m["union"]).all()
    assert m["src"].sum() == 4 * 9                       # 3x3 ring, 4 context tubelets
    assert m2["twinonly"].sum() == 0


def test_dose_kinds():
    n = np.arange(1024, dtype=float)
    mask = np.zeros(1024, bool)
    mask[[3, 5, 10]] = True
    p = dts.dose(n, mask, "pertoken", 2.0)
    assert np.allclose(p[mask], 2 * n[mask]) and (p[~mask] == 0).all()
    u = dts.dose(n, mask, "uniform", 1.0)
    assert np.allclose(u[mask], n[mask].mean()) and (u[~mask] == 0).all()
    assert (dts.dose(n, np.zeros(1024, bool), "pertoken", 1.0) == 0).all()


def test_arm_table():
    names = [a[0] for a in dts.ARMS]
    assert len(set(names)) == len(names)
    assert all(a[1] in dts.DIRS and a[2] in ("union", "src", "twinonly") for a in dts.ARMS)
    assert {dts.DIR_POINT[a[1]] for a in dts.ARMS} == set(dts.POINTS)

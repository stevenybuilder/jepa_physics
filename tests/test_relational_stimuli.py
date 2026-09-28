"""Invariants of the relational-motion stimulus generator (no rendering): balanced (c, v_rel) grid, both disks fully
in frame for all 16 frames, disks never overlap, identity carried by colour (top/bottom not tied to a disk)."""
import importlib.util
from pathlib import Path

import numpy as np

from wm.data import SIZE

spec = importlib.util.spec_from_file_location(
    "render_relational", Path(__file__).resolve().parents[1] / "scripts" / "render_relational_stimuli.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def test_layout_invariants():
    metas, rejected = rr.layout()
    assert len(metas) == len(rr.COMMON) * len(rr.VREL) * rr.N_PER_CELL == 637
    assert [m["id"] for m in metas] == list(range(637))
    v1 = np.array([m["v1_mps"] for m in metas])
    v2 = np.array([m["v2_mps"] for m in metas])
    c, vr = (v1 + v2) / 2, v1 - v2
    cells = {}
    for m, ci, vi in zip(metas, c, vr):
        assert m["common_mps"] == ci and m["v_rel_mps"] == vi and m["cell"] == rr.cell_of(ci, vi)
        cells[(ci, vi)] = cells.get((ci, vi), 0) + 1
    assert set(cells.values()) == {rr.N_PER_CELL} and len(cells) == 91            # balanced grid
    assert abs(np.corrcoef(c, vr)[0, 1]) < 1e-12
    tops = []
    for m in metas:
        tr = rr.tracks_px(m)                                                      # [2, 16, 2] rounded (col, row)
        assert ((tr >= rr.LO_PX) & (tr <= rr.HI_PX)).all()                         # fully inside, every frame
        assert (tr[..., 0] - 10.5 >= 0).all() and (tr[..., 0] + 0.5 + 10.5 <= SIZE).all()
        gap = np.hypot(*(tr[0] - tr[1]).T)
        assert gap.min() > 2 * 10.5 + 2                                           # never overlap
        tops.append(m["disk1_on_top"])
    assert 0.3 < np.mean(tops) < 0.7
    assert rejected > 0

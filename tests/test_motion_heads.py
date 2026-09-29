"""Unit tests for the mask / attention-density arithmetic of scripts/run_motion_heads.py (synthetic tensors)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_motion_heads as mh  # noqa: E402

N = mh.N_TOK


def _mask(cells):
    """cells: {slot: [(h, w), ...]} -> bool [8, 16, 16]."""
    m = np.zeros((8, 16, 16), bool)
    for t, hw in cells.items():
        for h, w in hw:
            m[t, h, w] = True
    return m


def _tok(t, h, w):
    return t * 256 + h * 16 + w


def test_set_matrix_token_order():
    m = _mask({2: [(3, 4)], 5: [(0, 15), (15, 0)]})
    S = mh.set_matrix(m)
    assert S.shape == (N, 8)
    assert S[_tok(2, 3, 4), 2] == 1 and S[:, 2].sum() == 1
    assert S[_tok(5, 0, 15), 5] == 1 and S[_tok(5, 15, 0), 5] == 1 and S[:, 5].sum() == 2
    assert S.sum() == 3


def test_slot_mass_and_densities_uniform_attention():
    # uniform attention: every density equals 1/N, so every ratio is 1
    m = _mask({t: [(t, t), (t, t + 1)] for t in range(8)})
    S = mh.set_matrix(m)
    A = np.full((2, N, N), 1.0 / N)
    M, n = mh.slot_mass(A, S)
    assert M.shape == (2, 8, 8) and np.allclose(n, 2)
    assert np.allclose(M, 2 * 2 / N)          # 2 queries x 2 keys x 1/N
    d = mh.clip_densities(M, n)
    for k in ("prev", "same", "next", "other", "far"):
        assert np.allclose(d[f"{k}_dens"], 1.0 / N)
    # far slots for t=1..6: all slots >= 2 away, each with 2 disk tokens
    assert np.allclose(d["far_mass"], np.mean([2 * sum(abs(u - t) >= 2 for u in range(8)) for t in range(1, 7)]) / N)
    assert np.allclose(d["prev_mass"], 2.0 / N)
    assert np.allclose(d["other_mass"], 1 - 6.0 / N)


def test_prev_slot_tracking_head():
    # head 0 puts all mass of each disk query on the previous slot's disk tokens (split evenly); head 1 on itself
    cells = {t: [(4, 4 + t)] for t in range(8)}
    cells[3] = [(4, 7), (5, 7)]                     # slot 3 has two disk tokens
    m = _mask(cells)
    S = mh.set_matrix(m)
    A = np.full((2, N, N), 1.0 / N)
    for t in range(1, 8):
        prev = np.where(S[:, t - 1] > 0)[0]
        for q in np.where(S[:, t] > 0)[0]:
            A[0, q] = 0
            A[0, q, prev] = 1.0 / len(prev)
            A[1, q] = 0
            A[1, q, q] = 1.0
    M, n = mh.slot_mass(A, S)
    d = mh.clip_densities(M, n)
    assert np.isclose(d["prev_mass"][0], 1.0) and np.isclose(d["other_mass"][0], 0.0)
    assert np.isclose(d["same_mass"][1], 1.0) and np.isclose(d["prev_mass"][1], 0.0)
    assert np.isclose(d["far_mass"][0], 0.0) and np.isclose(d["far_mass"][1], 0.0)
    # prev density per query = 1 / |disk_{t-1}|, query-weighted over t = 1..6 (queries: 1,1,2,1,1,1 -> 7)
    sizes = {t: len(cells[t]) for t in range(8)}
    want = sum(len(cells[t]) * (1.0 / sizes[t - 1]) for t in range(1, 7)) / sum(len(cells[t]) for t in range(1, 7))
    assert np.isclose(d["prev_dens"][0], want)


def test_invalid_slots_are_skipped():
    m = _mask({0: [(1, 1)], 1: [(1, 2)], 2: [(1, 3)]})     # only t=1 has both neighbours
    S = mh.set_matrix(m)
    A = np.full((1, N, N), 1.0 / N)
    M, n = mh.slot_mass(A, S)
    d = mh.clip_densities(M, n)
    assert np.isclose(d["prev_dens"][0], 1.0 / N)
    empty = mh.clip_densities(M[:, :, :] * 0, np.zeros(8))
    assert np.isnan(empty["prev_dens"]).all()


def test_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    m = rng.random((8, 16, 16)) < 0.02
    S = mh.set_matrix(m)
    logits = rng.normal(size=(3, N, N)).astype(np.float32)
    A = np.exp(logits - logits.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    Mn, nn = mh.slot_mass(A, S)
    Mt, nt = mh.slot_mass(torch.from_numpy(A), torch.from_numpy(S))
    assert np.allclose(Mn, Mt.numpy(), atol=1e-5) and np.allclose(nn, nt.numpy())


def test_ratio_ci_and_r2_drop():
    num = np.array([[2.0, 1.0], [4.0, 1.0], [np.nan, 1.0]])
    den = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    r, lo, hi, n = mh.ratio_ci(num, den, n_boot=200)
    assert n == 2 and np.allclose(r, [3.0, 1.0]) and lo[0] <= 3 <= hi[0]
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(200, 2))
    d, lo, hi = mh.r2_drop_boot(Y, Y, Y, n_boot=100)
    assert d == 0 and lo == 0 and hi == 0
    noisy = Y + rng.normal(size=Y.shape)
    d, lo, hi = mh.r2_drop_boot(Y, Y, noisy, n_boot=200)
    assert d > 0.5 and lo < d < hi


def test_top_heads_and_conditions():
    s = np.zeros((2, 16))
    s[0, 3], s[1, 7], s[1, 2] = 5, 9, 4
    assert mh.top_heads(s, [9, 12], 3) == [(12, 7), (9, 3), (12, 2)]
    conds = mh.conditions({9: [3, 1], 12: [7, 2]}, [(12, 7), (9, 3)], [9, 12], np.random.default_rng(0), n_rand=2)
    names = [c[0] for c in conds]
    assert names[0] == "none" and "b9_h15" in names and "b12_topk" in names and "global_top" in names
    d = dict(conds)
    assert d["global_top"] == {12: [7], 9: [3]}
    for r in range(2):
        assert len(d[f"b9_rand{r}"][9]) == 2 and len(set(d[f"b9_rand{r}"][9])) == 2
        assert {b: len(v) for b, v in d[f"global_rand{r}"].items()} == {12: 1, 9: 1}

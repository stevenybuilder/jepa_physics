"""scripts/run_token_patching.py: token-set bookkeeping (disk mask -> context tokens, ring, partition) and the
binding-fraction arithmetic on a toy."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tp", ROOT / "scripts" / "run_token_patching.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


def test_dilate_ring_is_3x3_without_wraparound():
    m = np.zeros((2, 16, 16), bool)
    m[0, 5, 5] = True
    m[1, 0, 15] = True
    d = tp.dilate(m)
    assert d[0].sum() == 9 and d[0, 4:7, 4:7].all()
    assert d[1].sum() == 4 and d[1, 0:2, 14:16].all()          # corner: clipped, no wrap to column 0 / row 15
    assert not d[1, :, 0].any() and not d[1, 15].any()


def test_token_sets_order_partition_and_context_only():
    a = np.zeros((8, 16, 16), bool)
    b = np.zeros((8, 16, 16), bool)
    a[2, 3, 7] = True                       # context tubelet 2, row 3, col 7
    b[0, 10, 10] = True
    a[6, 0, 0] = True                       # future tubelet: must not enter the context token sets
    s = tp.token_sets(a, b)
    assert all(v.shape == (1024,) for v in s.values())
    tok = 2 * 256 + 3 * 16 + 7              # (t, h, w) order, as the encoder's tokens
    assert s["objN"][tok] and s["objN"][10 * 16 + 10] and s["objN"].sum() == 2
    assert s["objA"][tok] and not s["objA"][10 * 16 + 10] and s["objA"].sum() == 9
    assert s["obj"].sum() == 18 and not s["obj"][0]
    for o, g in (("obj", "bg"), ("objA", "bgA"), ("objN", "bgN")):
        assert not (s[o] & s[g]).any() and (s[o] | s[g]).all()
    assert (s["objN"] <= s["obj"]).all() and (s["objA"] <= s["obj"]).all()


def test_vec_fraction_toy():
    vA = np.array([[1.0, 0.0], [0.0, 1.0]])
    vB = np.array([[-1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(tp.vec_fraction(vA, vA, vB), 0)
    assert np.allclose(tp.vec_fraction(vB, vA, vB), 1)
    mid = 0.25 * vA + 0.75 * vB + np.array([[0.0, 5.0], [0.0, 0.0]])  # off-axis component on row 0 ignored
    assert np.allclose(tp.vec_fraction(mid, vA, vB), [0.75, 0.75])
    # additive toy: disk moves 0.7 of the way, background 0.3 -> fractions sum to 1
    fd, fb = tp.vec_fraction(vA + 0.7 * (vB - vA), vA, vB), tp.vec_fraction(vA + 0.3 * (vB - vA), vA, vB)
    assert np.allclose(fd + fb, 1)


def test_angle_fraction_wraps_and_signs():
    assert np.allclose(tp.ang_fraction(350.0, 330.0, 10.0), 0.5)      # crosses 0 going +40
    assert np.allclose(tp.ang_fraction(310.0, 330.0, 10.0), -0.5)     # moved away from B
    assert np.allclose(tp.ang_fraction(100.0, 120.0, 20.0), 0.2)      # negative arc
    assert np.allclose(tp.wrap_signed([180.0, -180.0, 190.0, -10.0]), [180, 180, -170, -10])


def test_ratio_of_means_and_summary():
    num, den = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
    r = tp.ratio_of_means(num, den)
    assert abs(r["ratio"] - 0.5) < 1e-12 and r["ci95"][0] <= 0.5 <= r["ci95"][1]
    s = tp.summarize([1.0, np.nan, 3.0])
    assert s["n"] == 2 and s["mean"] == 2.0


def test_pick_pairs_rules():
    rng = np.random.default_rng(1)
    n = 60
    ids = np.arange(100, 100 + n)
    theta = rng.choice(np.arange(64) * 5.625, n)
    key = rng.choice(["v:1", "v:2"], n)
    xy = rng.uniform(-2, 2, (n, 2))
    for kind in ("random", "posmatch"):
        pr, dist, dth = tp.pick_pairs(ids, theta, key, xy, kind, 50, pos_max=1.0)
        assert len(np.unique(pr)) == pr.size                               # disjoint
        ia, ib = pr[:, 0] - 100, pr[:, 1] - 100
        assert (key[ia] == key[ib]).all() and (dth >= 90).all()
        assert np.allclose(np.abs(tp.wrap_signed(theta[ia] - theta[ib])), dth)
        if kind == "posmatch":
            assert (dist <= 1.0).all() and np.allclose(np.linalg.norm(xy[ia] - xy[ib], axis=1), dist)

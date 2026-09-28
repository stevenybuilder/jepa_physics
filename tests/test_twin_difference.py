"""scripts/session2_twin_difference.py: rank-r reconstruction and kNN mean difference on a synthetic difference matrix."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("td", ROOT / "scripts" / "session2_twin_difference.py")
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)


def synthetic(n=400, d=64, k=3, noise=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(d, k)))[0].T               # [k, d] orthonormal
    coef = np.linalg.qr(rng.normal(size=(n, k)))[0] * np.array([10.0, 5.0, 2.0])[:k]   # orthogonal columns
    return coef @ basis + noise * rng.normal(size=(n, d)), basis


def test_rank_r_recovers_planted_subspace():
    X, basis = synthetic()
    for r in (1, 2, 3):
        V = td.svd_basis(X, r)
        assert np.allclose(V @ V.T, np.eye(r), atol=1e-10)
        # the top-r singular subspace equals the span of the r strongest planted directions
        P_true = basis[:r].T @ basis[:r]
        assert np.abs(V.T @ V - P_true).max() < 1e-3
    V3 = td.svd_basis(X, 3)
    rec = td.rank_reduce(X, V3)
    assert np.linalg.norm(X - rec) / np.linalg.norm(X) < 1e-3
    # rank-1 residual is exactly what the lower directions carry (Eckart-Young)
    s = np.linalg.svd(X, compute_uv=False)
    assert np.isclose(np.linalg.norm(X - td.rank_reduce(X, td.svd_basis(X, 1))) ** 2, (s[1:] ** 2).sum())
    assert np.array_equal(td.rank_reduce(X, None), X)


def test_knn_mean_difference_and_rescale():
    rng = np.random.default_rng(1)
    pair_src = np.repeat(np.arange(0, 360, 45.0), 2)                  # 16 pairs, 8 sources x 2 targets
    pair_tgt = np.tile([90.0, 270.0], 8)
    diffs = rng.normal(size=(16, 8))
    src = np.array([10.0, 350.0])
    tgt = np.array([[90.0, 270.0], [270.0, 90.0]])
    out, used = td.knn_mean_difference(diffs, pair_src, pair_tgt, src, tgt, k=2)
    # carrier 0 (source 10): nearest sources to 10 with target 90 are 0 and 45 -> pairs 0, 2
    assert set(used[0, 0]) == {0, 2} and np.allclose(out[0, 0], diffs[[0, 2]].mean(0))
    # carrier 1 (source 350): nearest with target 270 are 0 and 315 (wrap-around) -> pairs 1, 15
    assert set(used[1, 0]) == {1, 15}
    arms, spec_, _ = td.twin_deltas(diffs, pair_src, pair_tgt, src, tgt, ranks=(1, 8), k=2)
    assert np.allclose(arms["twin_r8"], arms["twin_full"])            # rank = d reconstructs exactly
    ref = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert np.allclose(np.linalg.norm(td.rescale(arms["twin_r1"], ref), axis=-1), ref)

"""scripts/run_object_vs_scene.py: the background pool derived from stored per-step pools equals the direct mean of the
non-disk tokens; the cross-pool transfer matrix is symmetric / invariant where it must be; the binding index and the
Gram-matrix PCA behave."""
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from wm import manifold as mf
from wm import probes as pr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_object_vs_scene", ROOT / "scripts" / "run_object_vs_scene.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def synthetic_tokens(n=6, T=8, G=4, D=5, seed=0, empty_step=True):
    rng = np.random.default_rng(seed)
    tok = rng.standard_normal((n, T, G * G, D))
    mask = rng.random((n, T, G, G)) < 0.2
    mask[:, :, 0, 0] = True
    if empty_step:
        mask[0, 3] = False                               # a step with the disk out of frame
    m = mask.reshape(n, T, G * G)
    timepool = tok.mean(2)
    with np.errstate(invalid="ignore"):
        diskpool = (tok * m[..., None]).sum(2) / m.sum(2)[..., None]
    diskpool[m.sum(2) == 0] = np.nan                     # wm.extract.pool stores NaN at empty steps
    obj = np.stack([tok[i][m[i]].mean(0) for i in range(n)])
    bg = np.stack([tok[i][~m[i]].mean(0) for i in range(n)])
    return tok, mask, timepool, diskpool, obj, bg


def test_pool_derivation_identity(mod):
    tok, mask, tp, dp, obj, bg = synthetic_tokens()
    out = mod.derive_pools(tp, dp, mask)
    np.testing.assert_allclose(out["object"], obj, atol=1e-12)
    np.testing.assert_allclose(out["background"], bg, atol=1e-12)
    np.testing.assert_allclose(out["scene_recon"], tok.mean((1, 2)), atol=1e-12)
    n_tok = tok.shape[1] * tok.shape[2]
    recon = (out["n_obj"][:, None] * out["object"] + out["n_bg"][:, None] * out["background"]) / n_tok
    np.testing.assert_allclose(recon, out["scene_recon"], atol=1e-12)


def test_pool_derivation_fp16_storage(mod):
    """Storing timepool/diskpool as fp16 (as extracted) keeps the derived background within fp16 rounding."""
    tok, mask, tp, dp, obj, bg = synthetic_tokens(seed=1)
    out = mod.derive_pools(tp.astype(np.float16), dp.astype(np.float16), mask)
    assert np.abs(out["background"] - bg).max() < 5e-3
    assert np.abs(out["object"] - obj).max() < 5e-3


def test_no_disk_clip_is_nan(mod):
    tok, mask, tp, dp, obj, bg = synthetic_tokens()
    mask[2] = False
    m = mask.reshape(mask.shape[0], mask.shape[1], -1)
    dp[2] = np.nan
    out = mod.derive_pools(tp, dp, mask)
    assert np.isnan(out["object"][2]).all()
    np.testing.assert_allclose(out["background"][2], tok[2].mean((0, 1)), atol=1e-12)
    assert out["n_obj"][2] == 0 and m[2].sum() == 0


def direction_problem(n=240, D=12, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 360, n)
    Y = np.stack([np.sin(np.radians(th)), np.cos(np.radians(th))], 1)
    X = rng.standard_normal((n, D))
    X[:, :2] += 3 * Y
    tr, te = np.arange(0, 180), np.arange(180, n)
    folds = np.arange(180) % 5
    return X, Y, tr, te, folds


def test_transfer_identical_pools_symmetric(mod):
    X, Y, tr, te, folds = direction_problem()
    sf = partial(pr.score, kind="circular")
    M, _ = mod.transfer_matrix({"a": X, "b": X.copy()}, tr, te, Y[tr], Y[te], folds, sf)
    vals = np.array([[M[a][b]["r2"] for b in M[a]] for a in M])
    np.testing.assert_allclose(vals, vals.T, atol=1e-12)
    np.testing.assert_allclose(vals, vals[0, 0], atol=1e-12)


def test_transfer_invariant_to_per_pool_affine(mod):
    """Per-pool standardisation: a pool that is a per-feature affine copy of another transfers perfectly both ways."""
    X, Y, tr, te, folds = direction_problem(seed=2)
    rng = np.random.default_rng(3)
    Xb = X * rng.uniform(0.5, 4, X.shape[1]) + rng.normal(0, 10, X.shape[1])
    sf = partial(pr.score, kind="circular")
    M, _ = mod.transfer_matrix({"a": X, "b": Xb}, tr, te, Y[tr], Y[te], folds, sf)
    assert M["a"]["b"]["r2"] == pytest.approx(M["a"]["a"]["r2"], abs=1e-9)
    assert M["b"]["a"]["r2"] == pytest.approx(M["b"]["b"]["r2"], abs=1e-9)
    assert M["a"]["b"]["r2"] == pytest.approx(M["b"]["a"]["r2"], abs=1e-6)


def test_transfer_fails_on_disjoint_axes(mod):
    """Direction carried on different axes in two pools: each decodes itself, neither transfers."""
    X, Y, tr, te, folds = direction_problem(seed=4)
    Xb = X.copy()
    Xb[:, :2] -= 3 * Y
    Xb[:, 2:4] += 3 * Y
    sf = partial(pr.score, kind="circular")
    M, _ = mod.transfer_matrix({"a": X, "b": Xb}, tr, te, Y[tr], Y[te], folds, sf)
    assert M["a"]["a"]["r2"] > 0.6 and M["b"]["b"]["r2"] > 0.6
    assert M["a"]["b"]["r2"] < 0.2 and M["b"]["a"]["r2"] < 0.2


def test_binding_index(mod):
    assert mod.binding_index(0.9, 0.9) == 0
    assert mod.binding_index(0.8, 0.0) == 1
    X, Y, tr, te, folds = direction_problem(seed=5)
    sf = partial(pr.score, kind="circular")
    P = Y[te] + np.random.default_rng(6).normal(0, 0.3, Y[te].shape)
    b = mod.paired_binding(Y[te], P, P.copy(), sf, n_boot=20)
    assert b["binding_r2"] == pytest.approx(0) and b["binding_r2_ci"] == pytest.approx([0, 0])
    b = mod.paired_binding(Y[te], P, np.zeros_like(P) + P.mean(0), sf, n_boot=20)
    assert b["binding_r2"] > 0.9 and b["binding_err"] > 0.3


def test_ring_far_masks(mod):
    m = np.zeros((1, 1, 5, 5), bool)
    m[0, 0, 2, 2] = True
    ring, far = mod.ring_far_masks(m)
    assert ring.sum() == 8 and not ring[0, 0, 2, 2] and far.sum() == 25 - 9
    assert not (ring & far).any() and not (ring & m).any() and (ring | far | m).all()


def test_token_pools_match_derived(mod):
    """Token-level CPU pools agree with derive_pools on the per-step pools that wm.extract.pool would have stored."""
    import torch
    tok, mask, tp, dp, obj, bg = synthetic_tokens(seed=7, empty_step=False)
    n, T, GG, D = tok.shape
    pools, counts = mod.token_pools(torch.from_numpy(tok.reshape(n, T * GG, D)), mask)
    der = mod.derive_pools(tp, dp, mask)
    np.testing.assert_allclose(pools["object"], der["object"], atol=1e-5)
    np.testing.assert_allclose(pools["background"], der["background"], atol=1e-5)
    np.testing.assert_allclose(pools["scene"], tok.mean((1, 2)), atol=1e-5)
    recon = (counts["ring"][:, None] * pools["ring"] + counts["far"][:, None] * pools["far"]) / counts["background"][:, None]
    np.testing.assert_allclose(recon, pools["background"], atol=1e-5)


def test_fast_pca_matches_fit_pca(mod):
    X = np.random.default_rng(6).standard_normal((80, 200)) @ np.diag(np.linspace(3, 0.1, 200))
    a, b = mf.fit_pca(X, 16), mod.fast_pca(X, 16)
    np.testing.assert_allclose(a.explained, b.explained, rtol=1e-9)
    np.testing.assert_allclose(np.abs((a.components * b.components).sum(1)), 1, atol=1e-9)
    np.testing.assert_allclose(np.abs(a.project(X)), np.abs(b.project(X)), atol=1e-8)

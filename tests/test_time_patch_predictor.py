"""Pure-helper tests for scripts/time_patch_predictor.py (patch = replace a subspace component token by token)."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_spec = spec_from_file_location("tpp", Path(__file__).resolve().parents[1] / "scripts" / "time_patch_predictor.py")
tpp = module_from_spec(_spec)
_spec.loader.exec_module(tpp)


def test_time_basis_orthonormal_and_spans_centroid_differences():
    rng = np.random.default_rng(0)
    comps = rng.standard_normal((64, 32))
    C = rng.standard_normal((8, 64))
    Q, s = tpp.time_basis(C, comps, rank=7)
    assert Q.shape == (32, 7)
    assert np.allclose(Q.T @ Q, np.eye(7), atol=1e-10)
    X = C @ comps
    d = np.diff(X, axis=0)
    assert np.allclose(Q @ (Q.T @ d.T), d.T, atol=1e-8)          # every step difference lies in span(Q)
    assert len(s) == 8 and s[-1] < 1e-8                          # centring removes one dimension


def test_replace_component_semantics():
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.standard_normal((16, 3)))[0]
    x, y = rng.standard_normal((2, 5, 16)), rng.standard_normal((2, 5, 16))
    out = tpp.replace_component(x, y, Q)
    assert np.allclose(out @ Q, y @ Q)                            # the subspace component is the source's
    P = np.eye(16) - Q @ Q.T
    assert np.allclose(out @ P, x @ P)                            # everything else is the target's own
    assert np.allclose(tpp.replace_component(x, x, Q), x)         # self-patch is the identity


def test_replace_component_per_batch_basis_and_torch():
    import torch
    rng = np.random.default_rng(2)
    Qb = tpp.random_bases(2, 16, 3, rng)
    assert np.allclose(np.einsum("bdr,bds->brs", Qb, Qb), np.eye(3)[None], atol=1e-10)
    x, y = rng.standard_normal((2, 5, 16)), rng.standard_normal((2, 5, 16))
    out = tpp.replace_component(x, y, Qb)
    for b in range(2):
        assert np.allclose(out[b], tpp.replace_component(x[b:b + 1], y[b:b + 1], Qb[b])[0])
    ot = tpp.replace_component(torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(Qb)).numpy()
    assert np.allclose(ot, out)


def test_slot_sources():
    s = tpp.slot_sources()
    assert s["tp+2"].tolist() == [2, 3, 4, 5] and s["tp_rev(ctx)"].tolist() == [3, 2, 1, 0]
    assert tpp.ARM_K["w2:tp-2"] == -2 and set(tpp.ARM_K) == set(tpp.ARMS)

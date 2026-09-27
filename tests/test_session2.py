"""GPU session 2 code on CPU: zero-edit reproduction, a planted same-layer edit, renderer validation, predictor smoke.

Model tests load V-JEPA 2 ViT-L on CPU (~10 s per clip forward on the Intel Mac); the predictor test is marked slow.
  pytest tests/test_session2.py -s            # everything
  pytest tests/test_session2.py -m "not slow"
"""
import json

import numpy as np
import pytest
import torch

from wm.data import PROJECT_ROOT, decode, load_table
from wm.extract import encode, load_model, pool, preprocess, set_precision
from wm.predictor_readout import (N_CTX, context_prefix, pool_steps, predict_future, real_future, recovery)
from wm.probes import Standardizer, fit_ridge, predict, targets
from wm.propagate import prefix, propagate, suffix, to_raw_delta
from wm.render_twin import load_meta, render, twin_mask, validate

ACT = PROJECT_ROOT / "artifacts" / "activations" / "direction" / "vjepa2" / "meanpool.npy"
IDS = [0, 777]
needs_acts = pytest.mark.skipif(not ACT.exists(), reason="stored direction activations not present")


@pytest.fixture(scope="module")
def model():
    set_precision()
    torch.set_num_threads(max(torch.get_num_threads(), 4))
    return load_model("vjepa2", torch.device("cpu"))


@pytest.fixture(scope="module")
def clips():
    df = load_table("direction")
    return df, [decode(df["video"].iloc[i]) for i in IDS]


@pytest.fixture(scope="module")
def zero_pass(model, clips):
    _, frames = clips
    x = preprocess(frames)
    saved, final, mp = prefix(model, x, [22])
    return x, saved, final, mp


@needs_acts
def test_zero_delta_reproduces_stored_meanpool(model, clips, zero_pass):
    x, saved, _, mp = zero_pass
    points, _ = encode(model, x)                                           # the extraction path itself
    mp_ref, _, _ = pool(points, torch.zeros(len(IDS), 8, 16, 16, dtype=torch.bool))
    assert np.array_equal(mp, mp_ref), "block-by-block pass must equal extract.encode + pool bit for bit"
    r = suffix(model, saved[22], 22, np.zeros(1024, np.float32))
    assert np.array_equal(r["meanpool"], mp[:, 22:]), "a zero delta must not change anything downstream"
    stored = np.load(ACT, mmap_mode="r")[IDS]
    rel = np.abs(mp - stored).max(axis=(0, 2)) / np.abs(stored).max(axis=(0, 2))
    print(f"\nlocal CPU (batch 2) vs stored GPU batch-16 meanpool: per-layer max|d|/max|x| max {rel.max():.2e} "
          f"at point {rel.argmax()}; points 0-16 max {rel[:17].max():.2e}")
    # Same-device reproduction is exact (above). Across devices / batch sizes fp32 accumulation differs: the spec's
    # revised rule (§2) is 1e-3; gpu_session1 measured 1.0e-4 for CPU vs GPU batch 16 on speed clips. The 1e-4 check
    # at the extraction's own batch composition runs on the box (run_session2.py parity).
    assert rel.max() < 1e-3
    assert rel[:17].max() < 1e-4


@needs_acts
def test_planted_delta_moves_same_layer_readout(model, clips, zero_pass):
    """Plant the minimal-norm edit that makes a layer-22 direction probe read theta* = source + 90 deg, propagate it,
    and check (1) the edited point's meanpool is exactly the zero-edit meanpool + delta, (2) the same probe reads
    theta* there, (3) a zero delta leaves it at the source angle."""
    df, frames = clips
    L = 22
    X = np.load(ACT, mmap_mode="r")[:, L].astype(np.float64)
    Y, _, _ = targets(df, "direction")
    rows = np.setdiff1d(np.arange(len(df)), IDS)
    st = Standardizer().fit(X[rows])
    W, b = fit_ridge(st.transform(X[rows]), Y[rows], 10.0)
    _, saved, _, mp = zero_pass
    theta = df["theta_degrees"].to_numpy()[IDS]
    tgt = (theta + 90.0) % 360.0
    y_star = np.stack([np.sin(np.radians(tgt)), np.cos(np.radians(tgt))], 1)
    z = st.transform(mp[:, L].astype(np.float64))
    d_std = (y_star - predict(z, W, b)) @ np.linalg.pinv(W)                # minimal-norm solve in standardised space
    delta = to_raw_delta(d_std, st.std).astype(np.float32)
    r = suffix(model, saved[L], L, delta)
    assert np.allclose(r["meanpool"][:, 0], mp[:, L] + delta, atol=1e-4 * np.abs(mp[:, L]).max())
    read = lambda v: np.degrees(np.arctan2(*predict(st.transform(v.astype(np.float64)), W, b).T)) % 360.0  # noqa: E731
    err = lambda a, t: np.minimum(np.abs(a - t) % 360, 360 - np.abs(a - t) % 360)                         # noqa: E731
    assert err(read(r["meanpool"][:, 0]), tgt).max() < 1.0
    assert err(read(mp[:, L]), theta).max() < 20.0
    # propagate() on one clip with [E, D] and waypoint [E, K, D] deltas returns matching shapes
    out = propagate(model, frames[0], 24, np.stack([np.zeros(1024), delta[0]]).astype(np.float32), batch_size=2)
    assert out["meanpool"].shape == (2, 2, 1024)
    wp = propagate(model, frames[0], 24, np.zeros((1, 3, 1024), np.float32))
    assert wp["meanpool"].shape == (1, 3, 2, 1024)


def test_renderer_validation_20_clips():
    rep = validate(n=20)
    print("\n", json.dumps({k: rep[k] for k in ("codec", "no_codec", "mask_twin_iou_mean")}, indent=1))
    assert rep["n_clips"] == 20
    assert rep["codec"]["frac_frames_iou_gt_0.95"] >= 0.95
    assert rep["codec"]["frame_mae_mean"] < 0.5
    assert rep["mask_twin_iou_mean"] > 0.95


def test_twin_changes_only_direction():
    m = load_meta("direction", 0)                                           # theta 0, 1 m/s, velocity
    a, b = twin_mask(m), twin_mask(m, theta_deg=90.0)
    assert np.array_equal(a[0], b[0]), "twin starts at the same position"
    ya, xa = np.nonzero(a[-1])
    yb, xb = np.nonzero(b[-1])
    dx_a, dy_b = xa.mean() - np.nonzero(a[0])[1].mean(), np.nonzero(b[0])[0].mean() - yb.mean()
    assert abs(dx_a - dy_b) < 1.0 and dx_a > 15, (dx_a, dy_b)              # same displacement, rotated 90 deg
    f = render(m, theta_deg=90.0)
    assert f.shape == (16, 256, 256, 3) and f.dtype == np.uint8


def test_load_table_root_override():
    root = PROJECT_ROOT / "artifacts" / "stimuli" / "paper_layout"
    if not (root / "manifest.jsonl").exists():
        pytest.skip("stimuli not rendered (scripts/render_hard_stimuli.py)")
    df = load_table("direction", root=root)
    assert len(df) == 392 and sorted(df["label"].unique()) == [float(d) for d in range(0, 360, 45)]


@pytest.mark.slow
def test_predictor_smoke_one_clip(model, clips, zero_pass):
    _, frames = clips
    x, _, final, _ = zero_pass
    x1, final1 = x[:1], final[:1]
    saved, ctx_final, _ = context_prefix(model, x1, [12])
    assert ctx_final.shape == (1, N_CTX, 1024)
    leak = ((final1[:, :N_CTX] - ctx_final).norm() / ctx_final.norm()).item()
    assert leak > 0.05, f"context tokens of the full encoding should differ (they saw the future): {leak}"
    cm = [torch.arange(N_CTX).unsqueeze(0)]
    tm = [torch.arange(N_CTX, 2 * N_CTX).unsqueeze(0)]
    hf = model.predictor(encoder_hidden_states=ctx_final, context_mask=cm, target_mask=tm).last_hidden_state
    assert torch.allclose(predict_future(model, ctx_final, mask_index=1), hf, atol=1e-5), "re-implementation != HF"
    z0 = predict_future(model, ctx_final)
    assert z0.shape == (1, N_CTX, 1024) and torch.isfinite(z0).all()
    assert model.predictor.embeddings.mask_tokens[1:].abs().max() == 0, "checkpoint: mask tokens 1..9 are zero"
    r = suffix(model, saved[12], 12, np.zeros(1024, np.float32), return_final=True)
    assert torch.equal(r["final"], ctx_final)
    rf = real_future(final1)
    assert pool_steps(rf).shape == (1, 4, 1024)
    assert np.isclose(recovery(rf, z0, rf)[0], 1.0) and np.isclose(recovery(z0, z0, rf)[0], 0.0)
    mse_own = ((z0 - rf) ** 2).mean().item()
    print(f"\ncontext leak (rel diff) {leak:.3f}; predictor MSE to own real future {mse_own:.3f}")


@needs_acts
def test_stored_basis_frame_check():
    """--part1-basis paper: a basis is accepted only with the standardiser of the rows it was fit on (all train)."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_session2 import check_stored_basis
    from wm.inlp import inlp
    from wm.p2_data import load_inputs
    d = load_inputs("direction", 22)
    Y, kind, score_fn = targets(d["df"], "direction")
    tr = d["is_train"]
    st = Standardizer().fit(d["X"][tr].astype(np.float64))
    Xtr = st.transform(d["X"][tr].astype(np.float64))
    _, Q, W, b = inlp(Xtr, Y[tr], Xtr[:5], Y[tr][:5], d["fold"][tr], 10.0, score_fn, kind, max_rounds=2)
    probes = {"Q": Q, "W": W, "b": b, "alpha": 10.0, "point": 22, "kind": kind}
    assert check_stored_basis(probes, Xtr, Y[tr], 22, 10.0) < 1e-6
    sub = tr & (d["fold"] != 0)                                             # a different row set / standardiser
    st2 = Standardizer().fit(d["X"][sub].astype(np.float64))
    with pytest.raises(AssertionError):
        check_stored_basis(probes, st2.transform(d["X"][sub].astype(np.float64)), Y[sub], 22, 10.0)


def test_basis_cache_keyed_by_alpha(tmp_path):
    """A refit cached at one step-1 α is never reused after the sweep picks another α."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from run_session2 import basis_cache_path
    a = basis_cache_path(tmp_path, 12, "design", "arc", "abc", 10.0)
    assert a == basis_cache_path(tmp_path, 12, "design", "arc", "abc", 10.0)
    assert a != basis_cache_path(tmp_path, 12, "design", "arc", "abc", 31.622776601683793)

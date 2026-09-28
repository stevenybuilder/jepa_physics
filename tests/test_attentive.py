"""Synthetic checks for wm.attentive: the probe must (1) read a target carried by ONE token whose position varies by
clip (which a mean-pool readout dilutes), (2) put its attention on that token, and (3) score ~0 on a shuffled target."""
from functools import partial

import numpy as np
import torch

from wm.attentive import AttentiveProbe, cross_validate, fit_predict
from wm.probes import score


def _data(n=400, T=16, d=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, d)).astype(np.float32)
    y = rng.uniform(-1, 1, size=n)
    pos = rng.integers(0, T, size=n)
    X[np.arange(n), pos, 0] = 6.0                  # marker channel: which token is "the object"
    X[np.arange(n), pos, 1] = 3.0 * y              # target lives only on the marked token
    return X, y[:, None], pos


def test_shapes_and_attention_normalised():
    m = AttentiveProbe(d=8, out_dim=2, heads=3, hidden=16)
    x = torch.randn(5, 7, 8)
    assert m(x).shape == (5, 2)
    a = m.attention(x)
    assert a.shape == (5, 3, 7)
    assert torch.allclose(a.sum(-1), torch.ones(5, 3), atol=1e-5)


def test_reads_single_token_target_and_attends_to_it():
    X, Y, pos = _data()
    tr, te = np.arange(300), np.arange(300, 400)
    (P,), model = fit_predict(X[tr], Y[tr], [X[te]], epochs=80, lr=3e-3, weight_decay=1e-4, heads=2, hidden=32, dropout=0.0,
                              return_model=True)
    r2 = score(Y[te], P, kind="scalar")["r2"]
    assert r2 > 0.9, r2
    # mean-pool ridge baseline is much weaker on this construction
    from wm.probes import fit_ridge, predict
    W, b = fit_ridge(X[tr].mean(1).astype(np.float64), Y[tr], 1.0)
    r2_mean = score(Y[te], predict(X[te].mean(1).astype(np.float64), W, b), kind="scalar")["r2"]
    assert r2 > r2_mean + 0.2, (r2, r2_mean)
    # attention mass on the marked token (after the same standardisation the probe used)
    Xf = torch.from_numpy(X[tr])
    mu, sd = Xf.reshape(-1, X.shape[-1]).mean(0), Xf.reshape(-1, X.shape[-1]).std(0)
    with torch.no_grad():
        a = model.attention((torch.from_numpy(X[te]) - mu) / sd).numpy()          # [n, H, T]
    on_target = a[np.arange(len(te)), :, pos[te]].max(-1).mean()
    T = X.shape[1]
    assert on_target > 3.0 / T, on_target            # uniform attention would put 1/T = 0.0625 on it


def test_cv_shuffled_target_near_zero():
    X, Y, _ = _data(n=250)
    Ys = np.random.default_rng(1).permutation(Y)
    folds = np.arange(len(Y)) % 5
    cv = cross_validate(X, Ys, folds, partial(score, kind="scalar"), epochs=20, heads=2, hidden=16)
    assert cv["cv_r2_mean"] < 0.1, cv["cv_r2_mean"]
    assert cv["oof"].shape == Ys.shape and len(cv["folds"]) == 5

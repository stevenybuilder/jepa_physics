"""Attentive-MLP probe (the physics paper's §3.2 patch-preserving readout, run here as an EXTENSION: the paper reports
attentive-MLP results on IntPhys only). Learned queries, softmax attention over a clip's tokens, then a small MLP head;
trained with AdamW on train rows only; K-fold CV over the given folds, with the learning rate / weight decay chosen by
the CV mean (as the ridge baseline chooses alpha on the same folds).

Architecture: each head h has one learned d-dim query q_h applied directly to the (standardised) tokens,
a_h = softmax_t(q_h . x_t), pooled_h = sum_t a_h,t x_t; the H pooled vectors are concatenated -> LayerNorm -> Dropout ->
Linear -> GELU -> Linear. Without the LayerNorm this would equal a single-query attention block with linear keys /
values (q . W_k x = (W_k^T q) . x, and W_v commutes with the weighted sum, folding into the first Linear); the
LayerNorm between pooling and the first Linear breaks that equivalence, so this is a close relative of, not the same
function class as, a linear-K/V attention block. The simplification avoids [n, T, d] x [d, d] projections.

Tokens as used by scripts/run_attentive_probe.py are per spatial position averaged over the 8 token time steps (not the
2,048 space-time tokens), which removes the temporal structure that attentive probes are meant to preserve.
"""
import numpy as np
import torch
from torch import nn


class AttentiveProbe(nn.Module):
    def __init__(self, d, out_dim, heads=4, hidden=128, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(heads * d), nn.Dropout(dropout), nn.Linear(heads * d, hidden),
                                  nn.GELU(), nn.Linear(hidden, out_dim))

    def attention(self, x):
        """x [B, T, d] -> attention weights [B, H, T] (softmax over tokens; no 1/sqrt(d) so logits scale with q)."""
        return torch.softmax(torch.matmul(x, self.q.t()).transpose(1, 2), dim=-1)

    def forward(self, x):
        a = self.attention(x)                                   # [B, H, T]
        pooled = torch.bmm(a, x).reshape(x.shape[0], -1)         # [B, H*d]
        return self.head(pooled)


DEFAULT = dict(heads=4, hidden=128, dropout=0.1, lr=1e-3, weight_decay=0.01, epochs=60, batch=32, seed=0)


def _standardise(Xfit, Xevals, device):
    Xf = torch.from_numpy(np.ascontiguousarray(Xfit, dtype=np.float32)).to(device)
    flat = Xf.reshape(-1, Xf.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp(min=1e-6)
    Xf = (Xf - mu) / sd
    Xe = [(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device) - mu) / sd for x in Xevals]
    return Xf, Xe


def _train_predict(Xf, Yfit, Xe, cfg):
    """Xf [n, T, d] standardised torch tensor, Yfit numpy [n, m]; Xe list of standardised tensors. Returns predictions."""
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
    dev = Xf.device
    Y = np.asarray(Yfit, np.float64).reshape(len(Yfit), -1)
    ym, ys = Y.mean(0), Y.std(0) + 1e-12
    Yt = torch.from_numpy(((Y - ym) / ys).astype(np.float32)).to(dev)
    model = AttentiveProbe(Xf.shape[-1], Y.shape[1], cfg["heads"], cfg["hidden"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(Xf)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, cfg["batch"]):
            idx = perm[s:s + cfg["batch"]]
            loss = torch.mean((model(Xf[idx]) - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    outs = []
    with torch.no_grad():
        for x in Xe:
            P = torch.cat([model(x[s:s + 256]) for s in range(0, len(x), 256)]).cpu().numpy()
            outs.append(P.astype(np.float64) * ys + ym)
    return outs, model


def fit_predict(Xfit, Yfit, Xevals, device="cpu", return_model=False, **cfg):
    """Train on Xfit [n, T, d] (numpy), Yfit [n, m]; predict each array in Xevals. Tokens standardised per dimension
    with the fit tokens' mean / SD, targets with the fit rows' mean / SD; MSE; AdamW; fixed epoch budget."""
    Xf, Xe = _standardise(Xfit, Xevals, torch.device(device))
    outs, model = _train_predict(Xf, Yfit, Xe, cfg)
    return (outs, model) if return_model else outs


def cv_sweep(X, Y, folds, score_fn, configs, device="cpu"):
    """K-fold CV of every config over the given fold labels (train rows only). Standardisation is re-fit per fold.
    Returns, per config, per-fold R2 / MAE, fold mean / SD, and the out-of-fold predictions."""
    Y = np.asarray(Y, np.float64).reshape(len(Y), -1)
    res = [{"config": c, "folds": [], "oof": np.zeros_like(Y)} for c in configs]
    for k in np.unique(folds):
        ev = folds == k
        Xf, (Xe,) = _standardise(X[~ev], [X[ev]], torch.device(device))
        for r in res:
            (P,), _ = _train_predict(Xf, Y[~ev], [Xe], r["config"])
            r["oof"][ev] = P
            s = score_fn(Y[ev], P)
            r["folds"].append({"fold": int(k), "r2": float(s["r2"]), "mae": float(s["mae"]), "n": int(ev.sum())})
        del Xf, Xe
    for r in res:
        r2 = np.array([f["r2"] for f in r["folds"]])
        mae = np.array([f["mae"] for f in r["folds"]])
        r.update(cv_r2_mean=float(r2.mean()), cv_r2_sd=float(r2.std(ddof=1)), cv_mae_mean=float(mae.mean()),
                 cv_mae_sd=float(mae.std(ddof=1)))
    return res


def cross_validate(X, Y, folds, score_fn, device="cpu", **cfg):
    """Single-config CV (cv_sweep with one config)."""
    return cv_sweep(X, Y, folds, score_fn, [cfg], device)[0]

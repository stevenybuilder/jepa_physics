import importlib.util
from pathlib import Path

import numpy as np

from wm.probes import ALPHAS, Standardizer, cv_select_alpha, fit_ridge, predict, score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p1a_perpatch", ROOT / "scripts" / "p1a_perpatch.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


def _data(n=60, d=40, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    return rng.standard_normal((n, d)) + Y @ rng.standard_normal((2, d)), Y


def test_ridge_dual_predict_matches_primal():
    X, Y = _data()
    W, b = fit_ridge(X[:45], Y[:45], 3.0)
    assert np.allclose(pp.ridge_dual_predict(X[:45], Y[:45], X[45:], 3.0), predict(X[45:], W, b), atol=1e-8)


def test_perpos_fold_matches_nested_primal_recipe(tmp_path):
    X, Y = _data()
    mm = np.lib.format.open_memmap(tmp_path / "m.npy", mode="w+", dtype=np.float16, shape=(60, 1, 2, 40))
    mm[:, 0, 1] = X
    mm.flush()
    tr, folds = np.arange(60), np.arange(60) % 5
    fn = lambda a, b: score(a, b, "circular")  # noqa: E731
    pp._G.clear()
    pp._G.update(memmap=str(tmp_path / "m.npy"), tr=tr, Ytr=Y, folds=folds, score_fn=fn)
    _, _, k, alpha, P = pp._perpos_fold((0, 1, 2))
    Xh = np.asarray(mm[:, 0, 1], np.float64)
    fit, ev = folds != 2, folds == 2
    st = Standardizer().fit(Xh[fit])
    cv = cv_select_alpha(st.transform(Xh[fit]), Y[fit], folds[fit], ALPHAS, fn)
    W, b = fit_ridge(st.transform(Xh[fit]), Y[fit], cv["alpha"])
    assert k == 2 and alpha == cv["alpha"] and np.allclose(P, predict(st.transform(Xh[ev]), W, b), atol=1e-5)


def test_fold_curves_perfect_predictions_and_onset():
    _, Y = _data(20)
    good = np.repeat(Y[:, None, :], 4, 1)
    bad = np.zeros_like(good) + Y.mean(0)
    c, on = pp.fold_curves([3, 8, 9], Y, [bad, good, good], [bad, bad, good], [good] * 3, [bad, bad, good],
                           [good] * 3, [bad, bad, good], [Y, Y, Y])
    assert np.allclose(c["perpos_mean_r2"], [0, 1, 1]) and on["perpos_mean_r2"] == 8 and on["cross_half_r2"] == 9
    assert np.allclose(c["meanpool_r2"], 1) and on["pooled_mean_r2"] == 9

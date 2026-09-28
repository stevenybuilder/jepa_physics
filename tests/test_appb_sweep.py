"""App. B sweep (scripts/run_appb_sweep.py): on synthetic layers whose direction code appears at point 2, the cv_adam
sweep + the stored onset rule (availability, bootstrap_onset) put the onset at 2; fold merging keeps folds whole."""
import importlib.util
from functools import partial
from pathlib import Path

import numpy as np

from wm.adam_probe import cv_adam
from wm.probes import availability, bootstrap_onset, score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_appb_sweep", ROOT / "scripts" / "run_appb_sweep.py")
appb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appb)


def test_appb_sweep_onset_on_planted_emergence():
    rng = np.random.default_rng(0)
    n, d = 300, 10
    th = rng.uniform(0, 2 * np.pi, n)
    Y = np.stack([np.sin(th), np.cos(th)], 1)
    folds = np.arange(n) % 5
    sf = partial(score, kind="circular")
    fits = []
    for signal in (0.0, 0.3, 1.0, 1.0):      # points 0..3: none, weak, full, full
        X = np.hstack([signal * Y + 0.15 * rng.standard_normal((n, 2)), rng.standard_normal((n, d - 2))])
        X = (X - X.mean(0)) / X.std(0)                       # train-standardised, as the real pipeline
        fits.append(cv_adam(X, Y, folds, sf, epochs=100, lrs=(1e-3, 5e-3), wds=(0.01, 0.8)))
    curve = [f["cv_mean"] for f in fits]
    assert curve[0] < 0.1 < curve[1] < 0.9 * max(curve) <= curve[2]
    assert availability(curve)["onset"] == 2
    ci, n_none = bootstrap_onset(Y, [f["oof"] for f in fits], sf, 50, 0)
    assert ci == [2, 2] and n_none == 0

    merged = appb.merge_folds(folds * 7 + 3, 3)   # arbitrary fold labels -> 3 folds, each old fold kept whole
    assert set(merged) == {0, 1, 2}
    for f in np.unique(folds):
        assert len(set(merged[folds == f])) == 1

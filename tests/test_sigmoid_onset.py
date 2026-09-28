import numpy as np

from wm.sigmoid_onset import criterion, fit_logistic4, logistic4, onset90


def test_recovers_planted_logistic():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 19, 22, 24], float)
    y = logistic4(x, 0.05, 0.95, 1.2, 8.5) + np.random.default_rng(0).normal(0, 0.005, len(x))
    f = fit_logistic4(x, y)
    assert abs(f["x0"] - 8.5) < 0.2 and f["r2"] > 0.99


def test_criterion_flags():
    x = np.arange(25, dtype=float)
    late = criterion(x, logistic4(x, 0.0, 0.9, 2.0, 18.0))
    assert late["pass"]["r2_gt_0.9"] and not late["pass"]["inflection_le_50pct_depth"] and not late["accept_rebuttal_rule"]
    early = criterion(x, logistic4(x, 0.0, 0.9, 2.0, 6.0))
    assert early["accept_rebuttal_rule"] and abs(early["inflection_frac"] - 0.25) < 0.01
    flat = criterion(x, 0.05 + 0.01 * np.sin(x))
    assert not flat["pass"]["peak_ge_15pp_above_chance"]
    assert onset90(x, logistic4(x, 0.0, 1.0, 2.0, 6.0)) == 8

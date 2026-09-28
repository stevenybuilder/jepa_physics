"""Sigmoid onset criterion (the paper authors' OpenReview rebuttal, as summarised in sonia_joseph.md l.153-154):
fit a 4-parameter logistic in layer index, accept a transition if fit R² > 0.9, inflection ≤ 50% depth and the peak
≥ 15 pp above chance."""
import numpy as np
from scipy.optimize import curve_fit

LAST_BLOCK = 24          # depth fraction = point / 24 (wm.probes.layer_fraction)


def logistic4(x, lo, hi, k, x0):
    return lo + (hi - lo) / (1.0 + np.exp(np.clip(-k * (np.asarray(x, float) - x0), -60, 60)))


def fit_logistic4(x, y, x0_bounds=(0.0, float(LAST_BLOCK)), k_max=50.0):
    """Least-squares 4PL fit (rising: k > 0, x0 in x0_bounds). Profile grid first (x0 step 0.05, 80 log-spaced k; lo, hi
    solved in closed form for each (x0, k)), then scipy curve_fit refined from the best grid point.
    Returns dict(lo, hi, k, x0, r2, sse); r2 = 1 - SSE / SS_tot on the fitted points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x0g = np.arange(x0_bounds[0], x0_bounds[1] + 1e-9, 0.05)
    kg = np.logspace(-2, np.log10(k_max), 80)
    S = 1.0 / (1.0 + np.exp(-np.clip(kg[:, None, None] * (x[None, None, :] - x0g[None, :, None]), -60, 60)))
    Sc = S - S.mean(-1, keepdims=True)
    yc = y - y.mean()
    c = (Sc * yc).sum(-1) / np.maximum((Sc ** 2).sum(-1), 1e-300)        # hi - lo
    c = np.maximum(c, 0.0)                                                # rising
    a = y.mean() - c * S.mean(-1)                                         # lo
    sse_g = ((a[..., None] + c[..., None] * S - y) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse_g), sse_g.shape)
    lo_b, hi_b = min(y.min(), 0.0) - 1.0, max(y.max(), 1.0) + 1.0
    p0 = [float(np.clip(a[i, j], lo_b, hi_b)), float(np.clip(a[i, j] + c[i, j], lo_b, hi_b)), kg[i], x0g[j]]
    best = (np.array(p0), float(sse_g[i, j]))
    try:
        p, _ = curve_fit(logistic4, x, y, p0=p0, bounds=([lo_b, lo_b, 1e-3, x0_bounds[0]],
                                                         [hi_b, hi_b, k_max, x0_bounds[1]]), maxfev=20000)
        sse = float(((logistic4(x, *p) - y) ** 2).sum())
        if sse < best[1]:
            best = (p, sse)
    except (RuntimeError, ValueError):
        pass
    p, sse = best
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"lo": float(p[0]), "hi": float(p[1]), "k": float(p[2]), "x0": float(p[3]),
            "r2": 1.0 - sse / ss_tot if ss_tot > 0 else float("nan"), "sse": sse,
            "x0_at_bound": bool(np.isclose(p[3], x0_bounds[0], atol=1e-3) or np.isclose(p[3], x0_bounds[1], atol=1e-3)),
            "k_at_bound": bool(np.isclose(p[2], k_max, rtol=1e-3))}


def criterion(x, y, chance=0.0, r2_min=0.9, depth_max=0.5, pp_min=0.15):
    """Fit + the three tests. 'peak_above_chance' (the rebuttal's wording: peak >= 15 pp above chance) uses the
    observed maximum; 'rise_fit' (fitted curve at the last sampled point minus at the first: hi - lo is not used, it
    is unidentified when the step falls between two samples) and 'rise_obs' (max - min over sampled points) are
    reported beside it, and 'rise_ge_15pp' tests the fitted rise. inflection_bracket = the sampled points on either
    side of x0: when the rise is a step between two samples, x0 is only localised to that gap."""
    f = fit_logistic4(x, y)
    y = np.asarray(y, float)
    frac = f["x0"] / LAST_BLOCK
    peak = float(y.max() - chance)
    xs = np.sort(np.asarray(x, float))
    rise_fit = float(logistic4(xs[-1], f["lo"], f["hi"], f["k"], f["x0"])
                     - logistic4(xs[0], f["lo"], f["hi"], f["k"], f["x0"]))
    below, above = xs[xs <= f["x0"]], xs[xs >= f["x0"]]
    bracket = [float(below.max()) if len(below) else None, float(above.min()) if len(above) else None]
    out = {"inflection_point": f["x0"], "inflection_frac": frac, "fit_r2": f["r2"], "fit": f, "inflection_bracket": bracket,
           "rise_fit_pp": 100 * rise_fit, "rise_obs_pp": 100 * float(y.max() - y.min()),
           "peak_above_chance_pp": 100 * peak,
           "inflection_before_first_sample": bool(f["x0"] < min(x)),
           "pass": {"r2_gt_0.9": bool(f["r2"] > r2_min), "inflection_le_50pct_depth": bool(frac <= depth_max),
                    "peak_ge_15pp_above_chance": bool(peak >= pp_min), "rise_ge_15pp": bool(rise_fit >= pp_min)}}
    p = out["pass"]
    out["accept_rebuttal_rule"] = p["r2_gt_0.9"] and p["inflection_le_50pct_depth"] and p["peak_ge_15pp_above_chance"]
    out["accept_rise_variant"] = p["r2_gt_0.9"] and p["inflection_le_50pct_depth"] and p["rise_ge_15pp"]
    return out


def onset90(x, y):
    """First sampled point with y >= 90% of max over the sampled points (the stored onset rule)."""
    y = np.asarray(y, float)
    return int(np.asarray(x)[np.argmax(y >= 0.9 * y.max())])

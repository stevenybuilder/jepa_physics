"""Scores used throughout: circular MAE for direction, R^2 for everything."""
import numpy as np


def circular_mae_deg(theta_true_deg, sin_pred, cos_pred):
    """Mean absolute angular error in degrees, in [0, 180], from a predicted (sin, cos) pair.
    The pair need not be unit norm: atan2 only uses its direction."""
    theta_pred = np.degrees(np.arctan2(sin_pred, cos_pred))
    diff = (np.asarray(theta_pred) - np.asarray(theta_true_deg) + 180.0) % 360.0 - 180.0
    return float(np.mean(np.abs(diff)))


def r2(y_true, y_pred):
    """Coefficient of determination. For 2-D targets (e.g. sin, cos) the residual and total sums
    of squares are pooled over both columns."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def readout_radius(P):
    """Per-clip radius ‖(ŝ, ĉ)‖ of a direction readout P [n, 2]. Ridge readouts shrink below 1, and
    atan2 of a near-zero readout is noise, so the radius is reported beside every decoded angle."""
    P = np.asarray(P, dtype=float)
    return np.hypot(P[:, 0], P[:, 1])


def radius_summary(P):
    """{'mean', 'median', 'p05', 'p95'} of the per-clip readout radius."""
    rad = readout_radius(P)
    return {"mean": float(rad.mean()), "median": float(np.median(rad)),
            "p05": float(np.percentile(rad, 5)), "p95": float(np.percentile(rad, 95))}

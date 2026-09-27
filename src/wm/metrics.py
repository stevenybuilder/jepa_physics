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

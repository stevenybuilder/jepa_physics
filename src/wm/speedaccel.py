"""Pure helpers for scripts/run_speed_accel_angles.py: how speed and acceleration lines are represented and steered.

All vectors live in the raw activation space of one layer. A "line vector" u is the OLS slope of the features on
the label (mean feature shift per unit of the label, e.g. per m/s): the direction and step size along which the
centroids move, i.e. the natural steering vector for a straight variable. It is not a readout direction (a probe's
weights are rotated by the noise covariance); both are reported where it matters.
"""
import numpy as np


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def cos(a, b):
    return float(unit(a) @ unit(b))


def line_vector(X, y):
    """OLS slope of every feature on y: u = sum (x - xbar)(y - ybar) / sum (y - ybar)^2. X [n, D], y [n] -> [D]."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    yc = y - y.mean()
    return (X - X.mean(0)).T @ yc / (yc @ yc)


def joint_line_vectors(X, cols):
    """Multiple OLS of every feature on the columns of `cols` [n, p] (an intercept is added). Returns B [p, D]:
    B[j] is the feature shift per unit of column j with the other columns held fixed."""
    X, C = np.asarray(X, float), np.asarray(cols, float)
    A = np.column_stack([np.ones(len(C)), C])
    return np.linalg.lstsq(A, X, rcond=None)[0][1:]


def disattenuated_cos(c_ab, rel_a, rel_b):
    """Cosine between two noisy vector estimates corrected by their split-half reliabilities (Spearman's formula,
    reliabilities = split-half cosines; the half-sample cosine understates full-sample reliability, so the corrected
    value is conservative-high). Returns nan if a reliability is <= 0."""
    if rel_a <= 0 or rel_b <= 0:
        return float("nan")
    return float(c_ab / np.sqrt(rel_a * rel_b))


def ols_slope(x, y):
    """(slope, intercept, r2) of y on x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(b), float(a), float(r2)


def rate_vs_displacement(U, tau):
    """Per-step line vectors U [T, D] (feature shift per m/s at step t). A rate code shifts every step by the same
    vector (U_t = b); a code for distance travelled s_t = v tau_t shifts step t by tau_t d. Fits U_t = b + tau_t d
    by least squares over t and returns the energy of each part over the steps, relative to sum_t ||U_t||^2:
    rate_frac = T ||b + tau_bar d||^2 / E (the step-constant part), growth_frac = ||d||^2 sum (tau_t - tau_bar)^2 / E
    (the part growing with tau), resid_frac = the rest. They sum to 1."""
    U, tau = np.asarray(U, float), np.asarray(tau, float)
    E = (U ** 2).sum()
    tc = tau - tau.mean()
    d = tc @ U / (tc @ tc)
    mean = U.mean(0)
    fit = mean[None] + tc[:, None] * d[None]
    return {"rate_frac": float(len(U) * (mean ** 2).sum() / E), "growth_frac": float((d ** 2).sum() * (tc @ tc) / E),
            "resid_frac": float(((U - fit) ** 2).sum() / E), "norm_by_step": np.linalg.norm(U, axis=1).tolist()}


def spacing_r2(coord, values):
    """R^2 of a line coordinate against the label (linear) and against log(label)."""
    coord, values = np.asarray(coord, float), np.asarray(values, float)
    return {"r2_linear": ols_slope(values, coord)[2], "r2_log": ols_slope(np.log(values), coord)[2]}


def cumulative_chord(C):
    """Cumulative chord length along centroid rows C [m, D] (ordered by value)."""
    C = np.asarray(C, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])


def knn_ratio(Q, R, k=5, R_self=True, chunk=512):
    """Mean distance of each query row to its k nearest rows of R, divided by the median of the same quantity for
    R's own rows (leave-self-out). >1 means the queries sit in sparser territory than real clips."""
    Q, R = np.asarray(Q, np.float32), np.asarray(R, np.float32)
    r2 = (R ** 2).sum(1)

    def kd(A, self_rows):
        out = np.empty(len(A))
        for i in range(0, len(A), chunk):
            a = A[i:i + chunk]
            d2 = (a ** 2).sum(1)[:, None] + r2[None] - 2 * a @ R.T
            d = np.sqrt(np.maximum(d2, 0))
            d.sort(1)
            out[i:i + chunk] = d[:, 1:k + 1].mean(1) if self_rows else d[:, :k].mean(1)
        return out
    ref = np.median(kd(R, True)) if R_self else 1.0
    return kd(Q, False) / ref


def nearest_real_R(X_edit, X_orig, X_target):
    """Row-wise 1 - |edit - target| / |orig - target| (target: the real clip, e.g. a pixel twin)."""
    a = np.linalg.norm(np.asarray(X_edit) - X_target, axis=-1)
    b = np.linalg.norm(np.asarray(X_orig) - X_target, axis=-1)
    return 1.0 - a / np.maximum(b, 1e-12)


def along_line_gain(delta, u, dv):
    """How far a real change `delta` [n, D] moves along the line vector u per the label change dv [n]: the
    projection delta . u / (|u|^2 dv) (1 = exactly the linear prediction), and the fraction of the change's energy
    along u, (delta . u_hat)^2 / |delta|^2."""
    delta, u = np.asarray(delta, float), np.asarray(u, float)
    p = delta @ u
    return p / ((u @ u) * np.asarray(dv, float)), (p / np.linalg.norm(u)) ** 2 / (delta ** 2).sum(1)


def angle_bins(theta_deg, n_bins):
    """Bin index with bins centred on 0, 360/n, ... degrees."""
    w = 360.0 / n_bins
    return (np.floor(((np.asarray(theta_deg, float) % 360.0) + w / 2) / w).astype(int)) % n_bins


def circ_diff_deg(a, b):
    return (np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0
